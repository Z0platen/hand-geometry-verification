import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from palm_learning import fit_metric, project_features, validate_metric
from palmprint import PALM_FEATURE_SIZE
from prepare_palm import evaluate_records
from train_palm import PALM_WEIGHTS, train


def descriptor(subject, nuisance):
    values = np.zeros(PALM_FEATURE_SIZE)
    values[subject] = 1
    values[3] = nuisance
    return (values / np.linalg.norm(values)).tolist()


def records():
    rows = []
    for subject in range(3):
        for index in range(10):
            name = f"person-{subject}-{index}.jpg"
            rows.append({
                "image_name": name,
                "subject_id": str(subject),
                "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "geometry_status": "ok",
                "palm_status": "ok",
                "geometry": [1 + 0.02 * subject + 0.01 * index] * 5,
                "palm": descriptor(subject, 0.15 * index),
            })
    return rows


class PalmMetricTests(unittest.TestCase):
    def test_fit_projects_signed_unit_embeddings_and_round_trips(self):
        rows = records()
        metric = fit_metric([row["palm"] for row in rows], [row["subject_id"] for row in rows])
        restored = validate_metric(json.loads(json.dumps(metric)))
        self.assertEqual(restored["training_samples"], 30)
        self.assertEqual(restored["training_classes"], 3)
        self.assertEqual(restored["output_size"], 2)
        raw = [descriptor(subject, 0.65) for subject in range(3)]
        projected = project_features(raw, restored)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1)
        np.testing.assert_allclose(project_features(raw[0], restored), projected[0])
        self.assertTrue(np.any(projected < 0))
        self.assertGreater(np.linalg.norm(projected[0] - projected[1]), 1)

    def test_rejects_bad_metric_and_zero_projection(self):
        unit = descriptor(0, 0)
        metric = fit_metric([unit, descriptor(1, 0)], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "require.*identities"):
            fit_metric([unit, unit], ["a", "a"])
        with self.assertRaisesRegex(ValueError, "discriminative directions|no variation"):
            fit_metric([unit, unit], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "projection"):
            validate_metric({**metric, "projection": [[0]] * PALM_FEATURE_SIZE})
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            project_features(unit, {**metric, "mean": unit})
        with self.assertRaisesRegex(ValueError, "unit length"):
            project_features([0] * PALM_FEATURE_SIZE, metric)

    def test_test_split_mutation_cannot_change_training_or_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "baseline"
            source_rows = records()
            evaluate_records(source_rows, base, make_plot=False)
            feature_path = base / "features.jsonl"
            split_path = base / "splits.csv"
            baseline_path = base / "model.json"
            original_model = baseline_path.read_bytes()
            first_output = Path(directory) / "original"
            original = train(feature_path, split_path, base, first_output, make_plot=False)

            with split_path.open(encoding="utf-8", newline="") as handle:
                test_names = {row["image_name"] for row in csv.DictReader(handle) if row["split"] == "test"}
            for index, row in enumerate(source_rows):
                if row["image_name"] in test_names:
                    row["palm"] = descriptor((int(row["subject_id"]) + 1) % 3, 0.15 * index)
            changed_features = Path(directory) / "changed.jsonl"
            changed = "".join(json.dumps(row) + "\n" for row in source_rows).encode()
            changed_features.write_bytes(changed)
            second_output = Path(directory) / "mutated"
            with self.assertRaisesRegex(ValueError, "source_features_sha256"):
                train(changed_features, split_path, base, second_output, make_plot=False)

            # Rebind only the modified cache digest to test the model's split isolation.
            rebound_baseline = Path(directory) / "rebound.json"
            baseline = json.loads(original_model)
            baseline["source_features_sha256"] = hashlib.sha256(changed).hexdigest()
            rebound_baseline.write_text(json.dumps(baseline), encoding="utf-8")
            mutated = train(changed_features, split_path, rebound_baseline, second_output, make_plot=False)
            first = json.loads((first_output / "model.json").read_text(encoding="utf-8"))
            second = json.loads((second_output / "model.json").read_text(encoding="utf-8"))
            for key in ("templates", "palm_metric", "thresholds", "fusion"):
                self.assertEqual(first[key], second[key])
            self.assertEqual(first["templates"], baseline["templates"])
            self.assertEqual(baseline_path.read_bytes(), original_model)
            reference = json.loads((first_output / "reference.json").read_text(encoding="utf-8"))
            self.assertNotIn("templates", reference)
            self.assertNotIn("enrollment_counts", reference)
            self.assertEqual(reference["palm_metric"], first["palm_metric"])
            self.assertIn(first["fusion"]["weights"]["palm"], PALM_WEIGHTS)
            self.assertEqual(original["fusion_calibration_candidates"], mutated["fusion_calibration_candidates"])
            self.assertEqual(original["methods"]["palm"]["learned"]["calibration"],
                             mutated["methods"]["palm"]["learned"]["calibration"])
            self.assertNotEqual(original["methods"]["palm"]["learned"]["test"],
                                mutated["methods"]["palm"]["learned"]["test"])
            self.assertEqual(original["methods"]["geometry"]["baseline"],
                             original["methods"]["geometry"]["learned"])


if __name__ == "__main__":
    unittest.main()
