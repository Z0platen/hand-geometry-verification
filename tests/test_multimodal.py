import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate import split_samples
from geometry import RATIO_COLUMNS
from multimodal import PreparedMatcher, build_personal_model, load_model, load_reference, score_probe, validate_reference
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE
from palm_learning import fit_metric, validate_metric
from prepare_palm import _calibration_scale, evaluate_records, pairwise_distances


def palm(x=1.0, y=0.0):
    values = np.zeros(PALM_FEATURE_SIZE)
    values[:2] = (x, y)
    return (values / np.linalg.norm(values)).tolist()


def reference():
    return {"schema_version": 2, "reference_kind": "palm_geometry_calibration",
            "descriptor_version": DESCRIPTOR_VERSION, "geometry_feature_names": list(RATIO_COLUMNS),
            "palm_feature_size": PALM_FEATURE_SIZE, "quality": dict(DEFAULT_QUALITY),
            "thresholds": {"geometry": 0.5, "palm": 0.75, "fusion": 0.9},
            "fusion": {"geometry_scale": 0.2, "palm_scale": 0.4,
                       "weights": {"geometry": 0.5, "palm": 0.5}}}


def records():
    rows = []
    for subject, center in (("alice", 1), ("bob", 2)):
        for number in range(10):
            name = f"{subject}_{number}.jpg"
            descriptor = palm(1, 0.01 * number) if subject == "alice" else palm(0.01 * number, 1)
            rows.append({"image_name": name, "subject_id": subject,
                         "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                         "geometry_status": "ok", "palm_status": "ok",
                         "geometry": [center + 0.005 * number] * 5, "palm": descriptor, "quality": {}})
    rows.append({"image_name": "failed.jpg", "subject_id": "excluded", "image_sha256": "failed-hash",
                 "geometry_status": "no_hand", "palm_status": "palm_no_hand",
                 "geometry": None, "palm": None, "quality": {}})
    return rows


class MultimodalTests(unittest.TestCase):
    def model(self):
        return build_personal_model([{"geometry": [1.0] * 5, "palm": palm()}], "alice", reference())

    def test_personal_medians_and_threshold_transfer(self):
        ref = reference()
        before = copy.deepcopy(ref)
        model = build_personal_model([
            {"geometry": [value] * 5, "palm": descriptor}
            for value, descriptor in ((1, palm(1, 0)), (2, palm(0.8, 0.6)), (99, palm(0.6, 0.8)))
        ], "alice", ref)
        self.assertEqual(model["templates"]["alice"]["geometry"], [2.0] * 5)
        np.testing.assert_allclose(model["templates"]["alice"]["palm"], palm(0.8, 0.6))
        self.assertAlmostEqual(math.hypot(*model["templates"]["alice"]["palm"]), 1)
        self.assertEqual(model["thresholds"], before["thresholds"])
        self.assertEqual(model["fusion"], before["fusion"])
        self.assertEqual(ref, before)
        self.assertEqual(model["enrollment_counts"], {"alice": 3})
        self.assertFalse(model["personal_calibration"])

    def test_score_polarity_equal_weight_normalization_and_missing_components(self):
        model = self.model()
        geometry = [1.1] * 5
        descriptor = palm(0.8, 0.6)
        result = score_probe(geometry, descriptor, "alice", model)["modes"]
        expected_geometry = math.dist(geometry, [1.0] * 5)
        expected_palm = math.dist(descriptor, palm())
        self.assertAlmostEqual(result["geometry"]["distance"], expected_geometry)
        self.assertAlmostEqual(result["palm"]["distance"], expected_palm)
        self.assertAlmostEqual(result["fusion"]["distance"], 0.5 * expected_geometry / 0.2 + 0.5 * expected_palm / 0.4)
        self.assertTrue(result["geometry"]["accepted"])
        self.assertFalse(result["fusion"]["accepted"])
        failed = score_probe(dict.fromkeys(RATIO_COLUMNS, 1), None, "alice", model,
                             palm_status="palm_too_small")["modes"]
        self.assertTrue(failed["geometry"]["accepted"])
        self.assertEqual(failed["palm"]["status"], "palm_too_small")
        self.assertIsNone(failed["palm"]["accepted"])
        self.assertIsNone(failed["fusion"]["accepted"])
        self.assertEqual(failed["fusion"]["status"], "component_unavailable")

    def test_fusion_boundary_and_unknown_claim(self):
        model = self.model()
        model["thresholds"] = dict.fromkeys(("geometry", "palm", "fusion"), 0)
        result = score_probe([1] * 5, palm(), "alice", model)
        self.assertTrue(all(decision["accepted"] for decision in result["modes"].values()))
        with self.assertRaises(ValueError):
            score_probe([1] * 5, palm(), "missing", model)

    def test_malformed_vectors_and_calibration_fail_cleanly(self):
        for values in ([True] * 5, ["1"] * 5, [float("nan")] * 5, [1], [[1]] * 5):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score_probe(values, palm(), "alice", self.model())
        for descriptor in ([0] * PALM_FEATURE_SIZE, [1] * PALM_FEATURE_SIZE, [True] * PALM_FEATURE_SIZE):
            with self.assertRaises(ValueError):
                score_probe([1] * 5, descriptor, "alice", self.model())
        for replacement in (True, 0, float("nan"), "0.2"):
            ref = reference()
            ref["fusion"]["geometry_scale"] = replacement
            with self.subTest(scale=replacement), self.assertRaises(ValueError):
                validate_reference(ref)
        bad = reference()
        bad["fusion"]["weights"] = {"geometry": 0.7, "palm": 0.3}
        with self.assertRaises(ValueError):
            validate_reference(bad)

    def test_reference_is_aggregate_and_models_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference()), encoding="utf-8")
            self.assertNotIn("templates", load_reference(path))
            model = self.model()
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertEqual(load_model(path), model)
            with self.assertRaises(ValueError):
                load_reference(path)
            for bad in ([], {**reference(), "schema_version": True}, {**reference(), "templates": {}}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_reference(path)

    def test_pairwise_computation_matches_small_euclidean_example(self):
        probes = np.asarray([[1, 2, 3], [2, 4, 6]])
        references = np.asarray([[1, 2, 3], [5, 1, 3]])
        expected = np.asarray([[math.dist(a, b) for b in references] for a in probes])
        np.testing.assert_allclose(pairwise_distances(probes, references), expected)
        scale, details = _calibration_scale(np.zeros((2, 2)), np.eye(2, dtype=bool))
        self.assertEqual(scale, 1)
        self.assertTrue(details["fallback_used"])

    def test_common_cohort_split_and_manifests_are_disjoint(self):
        rows = records()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = evaluate_records(rows, output, make_plot=False)
            model = load_model(output / "model.json")
            ref = load_reference(output / "reference.json")
            self.assertEqual(summary["enrolled_users"], 2)
            self.assertEqual(summary["excluded_users"], {"excluded": 0})
            self.assertEqual(summary["acquisition"]["common_failure_rate"], 1 / 21)
            self.assertEqual(ref["split_manifest_sha256"], hashlib.sha256((output / "splits.csv").read_bytes()).hexdigest())
            self.assertEqual(model["split_manifest_sha256"], ref["split_manifest_sha256"])
            self.assertNotIn("templates", ref)
            with (output / "splits.csv").open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len({row["image_name"] for row in manifest}), len(rows))
            self.assertEqual(next(row["split"] for row in manifest if row["image_name"] == "failed.jpg"), "acquisition_failed")
            enrollment = {row["image_name"] for row in manifest if row["split"] == "enrollment"}
            calibration = {row["image_name"] for row in manifest if row["split"] == "calibration"}
            test = {row["image_name"] for row in manifest if row["split"] == "test"}
            self.assertFalse(enrollment & calibration or enrollment & test or calibration & test)
            for subject in ("alice", "bob"):
                expected = np.median([row["geometry"] for row in rows if row["subject_id"] == subject and row["image_name"] in enrollment], axis=0)
                np.testing.assert_array_equal(model["templates"][subject]["geometry"], expected)

    def test_test_feature_changes_cannot_modify_thresholds_or_templates(self):
        rows = records()
        splits, _ = split_samples(rows[:-1])
        test_names = {row["image_name"] for row in splits["test"]}
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["image_name"] in test_names:
                row["geometry"] = [9.0] * 5
                row["palm"] = palm(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            evaluate_records(rows, first, make_plot=False)
            evaluate_records(changed, second, make_plot=False)
            before, after = load_model(first / "model.json"), load_model(second / "model.json")
            self.assertEqual(before["templates"], after["templates"])
            self.assertEqual(before["thresholds"], after["thresholds"])
            self.assertEqual(before["fusion"], after["fusion"])

    def test_duplicate_content_is_rejected_before_splitting(self):
        rows = records()
        rows[1]["image_sha256"] = rows[0]["image_sha256"]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "Duplicate image content"):
            evaluate_records(rows, directory, make_plot=False)

    def test_prepared_scoring_exactly_matches_single_claims_and_freezes_model(self):
        model = self.model()
        dense = np.random.default_rng(42).uniform(0.01, 1, PALM_FEATURE_SIZE)
        dense = (dense / math.hypot(*dense)).tolist()
        model["templates"]["bob"] = {"geometry": [1.5] * 5, "palm": dense}
        matcher = PreparedMatcher(model)
        for geometry, descriptor, geometry_status, palm_status in (
            ([1.1] * 5, dense, "ok", "ok"),
            (dict.fromkeys(RATIO_COLUMNS, 1.3), palm(0.8, 0.6), "ok", "ok"),
            (None, dense, "finger_out_of_frame", "ok"),
            ([1] * 5, None, "ok", "palm_too_small"),
            (None, None, "no_hand", "palm_no_hand"),
        ):
            expected = {claim: score_probe(geometry, descriptor, claim, model, geometry_status, palm_status)
                        for claim in model["templates"]}
            self.assertEqual(matcher.score_all(geometry, descriptor, geometry_status, palm_status), expected)
            for claim in model["templates"]:
                self.assertEqual(matcher.score_probe(geometry, descriptor, claim, geometry_status, palm_status), expected[claim])
        before = matcher.score_all([1.1] * 5, dense)
        model["templates"]["alice"]["geometry"][0] = 99
        model["thresholds"]["fusion"] = 99
        self.assertEqual(matcher.score_all([1.1] * 5, dense), before)
        with self.assertRaises(ValueError):
            matcher.score_probe([1] * 5, dense, "unknown")

    def test_prepared_learned_metric_matches_standalone_without_revalidation_per_probe(self):
        model = self.model()
        descriptors = [palm(1, 0), palm(0.8, 0.6), palm(0, 1), palm(0.6, 0.8)]
        model["palm_metric"] = fit_metric(descriptors, ["alice", "alice", "bob", "bob"])
        model["templates"]["bob"] = {"geometry": [1.5] * 5, "palm": palm(0, 1)}
        probes = (
            ([1.1] * 5, palm(0.8, 0.6), "ok", "ok"),
            (None, palm(1, 0), "finger_out_of_frame", "ok"),
            ([1] * 5, None, "ok", "palm_too_small"),
        )
        expected = [
            {claim: score_probe(geometry, descriptor, claim, model, geometry_status, palm_status)
             for claim in model["templates"]}
            for geometry, descriptor, geometry_status, palm_status in probes
        ]
        with patch("multimodal.validate_metric", wraps=validate_metric) as validate:
            matcher = PreparedMatcher(model)
            self.assertEqual(validate.call_count, 1)
        with patch("palm_learning._metric_arrays", side_effect=AssertionError("metric arrays were revalidated per probe")):
            for (geometry, descriptor, geometry_status, palm_status), expected_row in zip(probes, expected, strict=True):
                self.assertEqual(matcher.score_all(geometry, descriptor, geometry_status, palm_status), expected_row)
                for claim in model["templates"]:
                    self.assertEqual(matcher.score_probe(geometry, descriptor, claim, geometry_status, palm_status), expected_row[claim])


if __name__ == "__main__":
    unittest.main()
