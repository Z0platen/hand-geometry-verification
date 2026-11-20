import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from multimodal import score_probe
import optimize_palm
from optimize_palm import _exact_distances, optimize
from palm_learning import fit_metric
from palmprint import PALM_FEATURE_SIZE
from prepare_palm import evaluate_records
from train_palm import train


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
                "image_name": name, "subject_id": str(subject),
                "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "geometry_status": "ok", "palm_status": "ok",
                "geometry": [1 + 0.02 * subject + 0.01 * index] * 5,
                "palm": descriptor(subject, 0.15 * index),
            })
    return rows


class PalmOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = records()
        self.baseline = self.root / "baseline"
        evaluate_records(self.rows, self.baseline, make_plot=False)
        self.incumbent_dir = self.root / "incumbent"
        train(self.baseline / "features.jsonl", self.baseline / "splits.csv",
              self.baseline, self.incumbent_dir, make_plot=False)

    def tearDown(self):
        self.temp.cleanup()

    def run_optimizer(self, out, *, features=None, baseline=None, incumbent=None, **kwargs):
        return optimize(features or self.baseline / "features.jsonl",
                        self.baseline / "splits.csv", baseline or self.baseline,
                        incumbent or self.incumbent_dir, out, make_plot=False,
                        shrinkages=(0.1,), dimensions=(1, 2, 64), palm_weights=(0.5, 0.75), **kwargs)

    def test_grid_outputs_public_reference_and_exact_live_distance(self):
        output = self.root / "optimized"
        summary = self.run_optimizer(output)
        self.assertEqual(len(summary["search_candidates"]), 1 + 1 * 2 * 2)
        self.assertIn(summary["selection"]["candidate"], {row["candidate"] for row in summary["search_candidates"]})
        model = json.loads((output / "model.json").read_text(encoding="utf-8"))
        reference = json.loads((output / "reference.json").read_text(encoding="utf-8"))
        self.assertNotIn("templates", reference)
        self.assertNotIn("enrollment_counts", reference)
        self.assertTrue((output / "search.csv").exists())
        with (self.baseline / "splits.csv").open(newline="", encoding="utf-8") as f:
            first_test_name = next(row["image_name"] for row in csv.DictReader(f) if row["split"] == "test")
        row = next(row for row in self.rows if row["image_name"] == first_test_name)
        subjects = list(model["templates"])
        scores = _exact_distances([row], subjects, model["templates"], model["palm_metric"])
        for subject_index, subject in enumerate(subjects):
            live = score_probe(row["geometry"], row["palm"], subject, model)
            self.assertEqual(scores["geometry"][0, subject_index], live["modes"]["geometry"]["distance"])
            self.assertEqual(scores["palm"][0, subject_index], live["modes"]["palm"]["distance"])

    def test_hash_binding_rejects_mutated_features_and_output_is_not_overwritten(self):
        original = (self.baseline / "features.jsonl").read_bytes()
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["palm"] = descriptor(1, 0.2)
        changed_path = self.root / "changed.jsonl"
        changed_path.write_text("".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_features_sha256"):
            self.run_optimizer(self.root / "rejected", features=changed_path)
        self.assertEqual((self.baseline / "features.jsonl").read_bytes(), original)
        existing = self.root / "occupied"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty output directory"):
            self.run_optimizer(existing)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_test_split_mutation_cannot_change_selection_or_frozen_model(self):
        output = self.root / "original"
        first = self.run_optimizer(output)
        with (self.baseline / "splits.csv").open(newline="", encoding="utf-8") as f:
            split_rows = list(csv.DictReader(f))
        test_names = {row["image_name"] for row in split_rows if row["split"] == "test"}
        changed_rows = [dict(row) for row in self.rows]
        for row in changed_rows:
            if row["image_name"] in test_names:
                row["palm"] = descriptor((int(row["subject_id"]) + 1) % 3, 0.4)
        changed_bytes = "".join(json.dumps(row) + "\n" for row in changed_rows).encode()
        changed_path = self.root / "changed.jsonl"
        changed_path.write_bytes(changed_bytes)
        baseline_data = json.loads((self.baseline / "model.json").read_text(encoding="utf-8"))
        baseline_data["source_features_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
        rebound_baseline = self.root / "rebound-baseline.json"
        rebound_bytes = (json.dumps(baseline_data) + "\n").encode()
        rebound_baseline.write_bytes(rebound_bytes)
        incumbent_data = json.loads((self.incumbent_dir / "model.json").read_text(encoding="utf-8"))
        incumbent_data["source_features_sha256"] = baseline_data["source_features_sha256"]
        incumbent_data["baseline_model_sha256"] = hashlib.sha256(rebound_bytes).hexdigest()
        rebound_incumbent = self.root / "rebound-incumbent.json"
        rebound_incumbent.write_text(json.dumps(incumbent_data), encoding="utf-8")
        second = self.run_optimizer(self.root / "mutated", features=changed_path,
                                    baseline=rebound_baseline, incumbent=rebound_incumbent)
        first_model = json.loads((output / "model.json").read_text(encoding="utf-8"))
        second_model = json.loads((self.root / "mutated" / "model.json").read_text(encoding="utf-8"))
        for key in ("templates", "palm_metric", "thresholds", "fusion"):
            self.assertEqual(first_model[key], second_model[key])
        self.assertEqual(first["selection"], second["selection"])
        self.assertNotEqual(first["methods"]["palm"]["tuned"]["test"],
                            second["methods"]["palm"]["tuned"]["test"])

    def test_incumbent_wins_numerical_calibration_ties(self):
        starting = self.run_optimizer(self.root / "starting")
        incumbent_record = starting["search_candidates"][0]
        incumbent = json.loads((self.incumbent_dir / "model.json").read_text(encoding="utf-8"))
        metric = incumbent["palm_metric"]
        incumbent_weight = incumbent["fusion"]["weights"]["palm"]
        original_operating_point = optimize_palm._operating_point

        def tied_operating_point(*args, **kwargs):
            thresholds, fusion, metrics = original_operating_point(*args, **kwargs)
            metrics["fusion"]["balanced_error"] = incumbent_record["fusion_ber"]
            metrics["fusion"]["fmr"] = incumbent_record["fusion_fmr"]
            metrics["palm"]["balanced_error"] = incumbent_record["palm_ber"]
            return thresholds, fusion, metrics

        # Reuse the incumbent projection and make the synthetic calibration objective an exact tie.
        with patch("optimize_palm.fit_metric", return_value=metric), \
                patch("optimize_palm._operating_point", side_effect=tied_operating_point):
            summary = optimize(self.baseline / "features.jsonl", self.baseline / "splits.csv", self.baseline,
                               self.incumbent_dir, self.root / "tie", make_plot=False,
                               shrinkages=(metric["shrinkage"],), dimensions=(metric["output_size"],),
                               palm_weights=(incumbent_weight,))
        self.assertFalse(summary["selection"]["calibration_strictly_improved"])
        self.assertEqual(summary["selection"]["candidate"], "incumbent")


if __name__ == "__main__":
    unittest.main()
