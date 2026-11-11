import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import calibrate_threshold, error_curve, read_features, split_samples, summarize_scores
from matching import FEATURE_SETS, build_templates, feature_vector, load_model, verify_features


def sample(subject, number, value):
    row = {"subject_id": subject, "image_name": f"{subject}_{number}.jpg", "status": "ok"}
    for columns in FEATURE_SETS.values():
        row.update(dict.fromkeys(columns, value))
    return row


class MatchingTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_eligible(self):
        rows = [sample(subject, i, i + 1) for subject in ("a", "b") for i in range(10)]
        rows += [sample("excluded", i, 1) for i in range(5)]
        splits, excluded = split_samples(rows)
        again, _ = split_samples(list(reversed(rows)))
        self.assertEqual(splits, again)
        self.assertEqual(excluded, {"excluded": 5})
        seen = set()
        for phase, samples in splits.items():
            names = {row["image_name"] for row in samples}
            self.assertFalse(seen & names)
            seen |= names
            self.assertEqual({row["subject_id"] for row in samples}, {"a", "b"})
        self.assertEqual(len(seen), 20)

    def test_templates_use_median(self):
        templates = build_templates([sample("a", i, value) for i, value in enumerate((1, 2, 99))], FEATURE_SETS["ratios"])
        self.assertEqual(templates["a"], [2.0] * 5)

    def test_threshold_polarity_and_ties(self):
        distances = np.array([0.1, 0.2, 1.0, 2.0])
        genuine = np.array([True, True, False, False])
        threshold = calibrate_threshold(distances, genuine)
        self.assertEqual(threshold, 0.2)
        summary = summarize_scores(distances, genuine, threshold)
        self.assertEqual(summary["false_accepts"], 0)
        self.assertEqual(summary["false_rejects"], 0)
        self.assertEqual(summary["roc_auc"], 1)

    def test_curve_treats_tied_distances_together(self):
        thresholds, fmr, fnmr = error_curve([0.2, 0.2], [True, False])
        self.assertEqual(len(thresholds), 2)
        np.testing.assert_array_equal(fmr, [0, 1])
        np.testing.assert_array_equal(fnmr, [1, 0])

    def test_duplicate_hash_rejected(self):
        rows = [sample("a", 0, 1), sample("b", 0, 2)]
        for row in rows:
            row["image_sha256"] = "same-content"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate image content"):
                read_features(path)

    def test_nonfinite_features_rejected(self):
        with self.assertRaises(ValueError):
            build_templates([sample("a", 0, float("nan"))], FEATURE_SETS["ratios"])

    def test_model_roundtrip_and_claim_validation(self):
        model = {"schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
                 "threshold": 0.1, "templates": {"a": [1.0] * 5}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            restored = load_model(path)
            self.assertTrue(verify_features(sample("a", 0, 1), "a", restored)["accepted"])
            self.assertFalse(verify_features(sample("a", 0, 2), "a", restored)["accepted"])
            with self.assertRaisesRegex(ValueError, "Unknown"):
                verify_features(sample("a", 0, 1), "missing", restored)

    def test_malformed_models_have_clear_value_errors(self):
        examples = [[], {"schema_version": 1, "method": []}, {
            "schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
            "threshold": 1, "templates": {"a": [None] * 5},
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for content in examples:
                path.write_text(json.dumps(content), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_model(path)

    def test_csv_numeric_strings_remain_supported_but_boolean_and_nested_features_fail(self):
        columns = FEATURE_SETS["ratios"]
        np.testing.assert_array_equal(feature_vector(dict.fromkeys(columns, "1.25"), columns), [1.25] * 5)
        for value in (True, np.bool_(True), [1.0], np.array([1.0]), 10 ** 1000):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                feature_vector(dict.fromkeys(columns, value), columns)
        with self.assertRaises(ValueError):
            feature_vector({}, [])

    def test_model_json_requires_integer_schema_and_numeric_scalars(self):
        valid = {"schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
                 "threshold": 0.1, "templates": {"a": [1.0] * 5}}
        invalid_models = [
            {**valid, "schema_version": value} for value in (True, 1.0, "1")
        ]
        invalid_models.extend({**valid, "templates": {"a": [value] * 5}} for value in (True, "1.0", [1.0], {}))
        invalid_models.append({**valid, "threshold": 10 ** 1000})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for index, model in enumerate(invalid_models):
                with self.subTest(case=index):
                    path.write_text(json.dumps(model), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_model(path)

    def test_verification_validates_in_memory_model_before_broadcasting(self):
        model = {"schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
                 "threshold": 0.1, "templates": {"a": [1.0]}}
        with self.assertRaises(ValueError):
            verify_features(sample("a", 0, 1), "a", model)

    def test_distance_handles_finite_overflow_and_underflow_cases(self):
        columns = FEATURE_SETS["ratios"]
        model = {"schema_version": 1, "method": "ratios", "feature_names": columns,
                 "threshold": 1e201, "templates": {"a": [1.0] * 5}}
        result = verify_features(dict.fromkeys(columns, 1e200), "a", model)
        self.assertTrue(np.isfinite(result["distance"]))
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["distance"] / 1e200, np.sqrt(5))
        model = {**model, "threshold": 0, "templates": {"a": [1e-200] * 5}}
        result = verify_features(dict.fromkeys(columns, 2e-200), "a", model)
        self.assertGreater(result["distance"], 0)
        self.assertFalse(result["accepted"])
        with self.assertRaises(ValueError):
            verify_features(dict.fromkeys(columns, 1.7e308), "a", model)

    def test_subnormal_squared_distance_does_not_false_accept(self):
        columns = FEATURE_SETS["ratios"]
        reference = [1.5e-160, 1e-160, 1e-160, 1e-160, 1e-160]
        model = {"schema_version": 1, "method": "ratios", "feature_names": columns,
                 "threshold": 2.49999e-160, "templates": {"a": reference}}
        probe = dict(zip(columns, [2 * value for value in reference], strict=True))
        result = verify_features(probe, "a", model)
        self.assertEqual(result["distance"], math.hypot(*reference))
        self.assertFalse(result["accepted"])

    def test_finite_extreme_enrollment_median_stays_finite(self):
        columns = FEATURE_SETS["ratios"]
        templates = build_templates([sample("a", i, 1e308) for i in range(2)], columns)
        np.testing.assert_array_equal(templates["a"], [1e308] * 5)

    def test_normal_distance_keeps_original_numpy_result(self):
        columns = FEATURE_SETS["ratios"]
        reference = [1.11, 1.51, 1.62, 1.49, 1.22]
        probe = dict(zip(columns, [1.15, 1.48, 1.65, 1.44, 1.21], strict=True))
        model = {"schema_version": 1, "method": "ratios", "feature_names": columns,
                 "threshold": 0.2, "templates": {"a": reference}}
        expected = float(np.linalg.norm(np.asarray(list(probe.values())) - np.asarray(reference)))
        self.assertEqual(verify_features(probe, "a", model)["distance"], expected)


if __name__ == "__main__":
    unittest.main()
