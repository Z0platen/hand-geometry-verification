import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate_palm_transfer import (
    _distances, _frozen_model, _partition_transfer, evaluate_transfer,
)
from multimodal import PreparedMatcher
from geometry import RATIO_COLUMNS
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE


def descriptor(index, variant=0):
    values = np.zeros(PALM_FEATURE_SIZE, dtype=float)
    values[index % PALM_FEATURE_SIZE] = 1.0
    values[(index + 1) % PALM_FEATURE_SIZE] = variant * 0.001
    return (values / np.linalg.norm(values)).tolist()


def make_model(templates, *, learned=False):
    model = {
        "schema_version": 2,
        "descriptor_version": DESCRIPTOR_VERSION,
        "geometry_feature_names": list(RATIO_COLUMNS),
        "palm_feature_size": PALM_FEATURE_SIZE,
        "model_kind": "palm_geometry",
        "quality": dict(DEFAULT_QUALITY),
        "thresholds": {"geometry": 0.31, "palm": 0.42, "fusion": 0.27},
        "fusion": {"geometry_scale": 0.2, "palm_scale": 0.3,
                   "weights": ({"geometry": 0.4, "palm": 0.6} if learned else
                               {"geometry": 0.5, "palm": 0.5})},
        "templates": templates,
        "enrollment_counts": {subject: 1 for subject in templates},
        "threshold_source": "synthetic fixture",
    }
    if learned:
        projection = np.zeros((PALM_FEATURE_SIZE, 1))
        projection[0, 0] = 1.0
        model["palm_metric"] = {
            "kind": "regularized_lda_v1", "input_size": PALM_FEATURE_SIZE,
            "output_size": 1, "mean": [value * 0.5 for value in descriptor(0)],
            "projection": projection.tolist(),
            "training_samples": 4, "training_classes": 2, "shrinkage": 0.1,
        }
    return model


def create_bound_fixture(root):
    records, assignments = [], []
    serial = 0

    def add(subject, role, vector_index):
        nonlocal serial
        name = f"{subject}_{role}_{serial}.jpg"
        record = {
            "image_name": name, "subject_id": subject,
            "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "geometry_status": "ok", "palm_status": "ok",
            "geometry": [0.5 + vector_index * 0.01 + j * 0.001 for j in range(5)],
            "palm": descriptor(vector_index, serial % 3),
        }
        records.append(record)
        assignments.append({**{key: record[key] for key in
                               ("image_name", "subject_id", "image_sha256", "geometry_status", "palm_status")},
                            "split": role})
        serial += 1
        return record

    training_samples = []
    for user, vector_index in (("trainA", 0), ("trainB", 1)):
        for role in ("enrollment", "calibration", "test"):
            training_samples.append(add(user, role, vector_index))
    add("trainA", "insufficient_samples", 0)
    for user, vector_index, count in (("newA", 2, 3), ("newB", 3, 3), ("few", 4, 2)):
        for _ in range(count):
            add(user, "insufficient_samples", vector_index)

    training_templates = {
        user: {"geometry": [0.5 + index * 0.01 + j * 0.001 for j in range(5)],
               "palm": descriptor(index)}
        for user, index in (("trainA", 0), ("trainB", 1))
    }
    baseline = make_model(training_templates)
    incumbent = make_model(training_templates, learned=True)
    tuned = make_model(training_templates, learned=True)

    features_path = root / "features.jsonl"
    features_bytes = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records).encode()
    features_path.write_bytes(features_bytes)
    splits_path = root / "splits.csv"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("image_name", "subject_id", "image_sha256",
                                                "geometry_status", "palm_status", "split"))
    writer.writeheader()
    writer.writerows(assignments)
    split_bytes = output.getvalue().encode()
    splits_path.write_bytes(split_bytes)
    for value in (baseline, incumbent, tuned):
        value["source_features_sha256"] = hashlib.sha256(features_bytes).hexdigest()
        value["split_manifest_sha256"] = hashlib.sha256(split_bytes).hexdigest()

    baseline_path, incumbent_path, tuned_path = (root / name for name in
                                                 ("baseline.json", "incumbent.json", "tuned.json"))
    for path, value in ((baseline_path, baseline), (incumbent_path, incumbent), (tuned_path, tuned)):
        path.write_text(json.dumps(value), encoding="utf-8")
    return tuned_path, incumbent_path, features_path, splits_path, baseline_path


class PalmTransferTests(unittest.TestCase):
    def test_partition_is_fixed_seed_and_disjoint(self):
        rows = [{"image_name": f"U{user}_{index}.jpg", "subject_id": f"U{user}",
                 "image_sha256": f"{user}{index}"}
                for user in range(3) for index in range(5)]
        first = _partition_transfer(rows)
        second = _partition_transfer(list(reversed(rows)))
        self.assertEqual(first, second)
        enrollment, probes, manifest = first
        self.assertEqual(len(enrollment), 6)
        self.assertEqual(len(probes), 9)
        self.assertFalse({row["image_name"] for row in enrollment} &
                         {row["image_name"] for row in probes})
        self.assertEqual(sum(row["role"] == "enrollment" for row in manifest), 6)

    def test_no_eligible_identity_fails_clearly(self):
        rows = [{"image_name": "one.jpg", "subject_id": "one"},
                {"image_name": "two.jpg", "subject_id": "two"}]
        with self.assertRaisesRegex(ValueError, "No unseen identity"):
            _partition_transfer(rows)

    def test_transfer_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "a" * 64
            split_hash = "b" * 64
            model = make_model({"trainA": {"geometry": [1.0] * 5,
                                             "palm": descriptor(0)}})
            model.update(source_features_sha256=source, split_manifest_sha256=split_hash)
            incumbent = {**model, "split_manifest_sha256": "c" * 64}
            for name, value in (("model.json", model), ("incumbent.json", incumbent),
                                ("baseline.json", model)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "share feature and split source hashes"):
                evaluate_transfer(root / "model.json", root / "incumbent.json",
                                  root / "features.jsonl", root / "splits.csv",
                                  root / "baseline.json", root / "result")

    def test_bound_transfer_publishes_only_eligible_images_and_frozen_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tuned, incumbent, features, splits, baseline = create_bound_fixture(root)
            original_tuned = tuned.read_bytes()
            original_incumbent = incumbent.read_bytes()
            summary = evaluate_transfer(tuned, incumbent, features, splits, baseline, root / "transfer")
            output = root / "transfer"
            self.assertEqual((summary["training_identity_count"], summary["eligible_identity_count"]), (2, 2))
            self.assertEqual(summary["enrollment_images"], 4)
            self.assertEqual(summary["probe_images"], 2)
            self.assertEqual(summary["identity_count_excluded_as_training_overlap"], 1)
            self.assertEqual(summary["skipped_insufficient_identities"], [
                {"subject_id": "trainA", "common_valid_images": 1,
                 "reason": "identity overlaps the metric-training cohort"},
                {"subject_id": "few", "common_valid_images": 2,
                 "reason": "fewer than 3 common-valid images"},
            ])
            self.assertEqual({path.name for path in output.iterdir()}, {"summary.json", "transfer_split.csv"})
            self.assertEqual(tuned.read_bytes(), original_tuned)
            self.assertEqual(incumbent.read_bytes(), original_incumbent)
            with (output / "transfer_split.csv").open(newline="", encoding="utf-8") as handle:
                split_rows = list(csv.DictReader(handle))
            self.assertEqual(len(split_rows), 6)
            self.assertNotIn("trainA", {row["subject_id"] for row in split_rows})
            self.assertTrue(all(row["image_sha256"] for row in split_rows))

    def test_batch_distances_match_live_matcher_scores(self):
        templates = {"newA": {"geometry": [1.0 + j * 0.01 for j in range(5)],
                               "palm": descriptor(0)},
                     "newB": {"geometry": [1.1 + j * 0.01 for j in range(5)],
                               "palm": descriptor(1)}}
        enrollment = [{"subject_id": "newA"}, {"subject_id": "newA"},
                      {"subject_id": "newB"}, {"subject_id": "newB"}]
        rows = [{"subject_id": "newB", "geometry": [1.12 + j * 0.01 for j in range(5)],
                 "palm": descriptor(1, 1)}]
        subjects = sorted(templates)
        for learned in (False, True):
            model = _frozen_model(make_model({"old": templates["newA"]}, learned=learned), templates, enrollment)
            actual = _distances(rows, subjects, model)
            expected = PreparedMatcher(model).score_all(rows[0]["geometry"], rows[0]["palm"])
            for mode in ("geometry", "palm", "fusion"):
                np.testing.assert_allclose(actual[mode][0],
                                           [expected[subject]["modes"][mode]["distance"] for subject in subjects],
                                           rtol=0, atol=1e-15)

    def test_frozen_replacement_preserves_operating_parameters(self):
        templates = {"newA": {"geometry": [1.0] * 5, "palm": descriptor(0)},
                     "newB": {"geometry": [1.0] * 5, "palm": descriptor(1)}}
        enrollment = [{"subject_id": "newA"}, {"subject_id": "newA"}]
        for learned in (False, True):
            original = make_model({"oldA": templates["newA"]}, learned=learned)
            before = {key: json.dumps(original.get(key), sort_keys=True)
                      for key in ("palm_metric", "thresholds", "fusion")}
            replaced = _frozen_model(original, templates, enrollment)
            self.assertEqual(replaced["templates"], templates)
            self.assertEqual(replaced["enrollment_counts"], {"newA": 2})
            for key, value in before.items():
                self.assertEqual(json.dumps(replaced.get(key), sort_keys=True), value)


if __name__ == "__main__":
    unittest.main()
