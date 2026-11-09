import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import calibrate_threshold, error_curve, read_features, split_samples, summarize_scores
from matching import FEATURE_SETS, build_templates, load_model, verify_features


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


if __name__ == "__main__":
    unittest.main()
