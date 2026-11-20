import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from palm_trials import MODES, PalmTrialLog
from geometry import RATIO_COLUMNS
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE


def model():
    return {
        "schema_version": 2, "model_kind": "palm_geometry",
        "descriptor_version": DESCRIPTOR_VERSION, "geometry_feature_names": list(RATIO_COLUMNS),
        "palm_feature_size": PALM_FEATURE_SIZE, "quality": dict(DEFAULT_QUALITY),
        "fusion": {"geometry_scale": 1, "palm_scale": 1, "weights": {"geometry": 0.5, "palm": 0.5}},
        "templates": {"a": {"geometry": [1] * 5, "palm": [1] + [0] * (PALM_FEATURE_SIZE - 1)}},
        "thresholds": dict.fromkeys(MODES, 0.5),
    }


def scores(distance=0.2):
    return {mode: {"status": "ok", "accepted": distance <= 0.5, "distance": distance, "threshold": 0.5}
            for mode in MODES}


class PalmTrialTests(unittest.TestCase):
    def test_one_attempt_exports_three_methods_with_declared_distance(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "b", 35, "day2-far")
            result = log.record({"claimed_id": "a", "modes": scores()}, {"native_roi_pixels": 80, "contrast": 9})
            self.assertEqual(result, {"status": "saved", "count": 1})
            summary = log.summary()
            self.assertEqual(summary["distance_cm"], 35)
            self.assertEqual(summary["distance_source"], "operator_declared")
            for method in summary["methods"].values():
                self.assertEqual(method["impostor_comparisons"], 1)
                self.assertEqual(method["false_accepts"], 1)
                self.assertEqual(method["fmr"], 1)
                self.assertIsNone(method["fnmr"])
            with (log.run_dir / "trials.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["method"] for row in rows}, set(MODES))
            self.assertEqual({row["distance_cm"] for row in rows}, {"35.0"})
            saved = json.loads(next((log.run_dir / "trials").glob("*.json")).read_text())
            self.assertEqual(set(saved["quality"]), {"native_roi_pixels", "contrast"})
            self.assertNotIn("templates", (log.run_dir / "summary.json").read_text())
            self.assertFalse(list(log.run_dir.rglob("*.jpg")))
            self.assertFalse(list(log.run_dir.rglob("*.png")))

    def test_quality_failures_are_excluded_from_conditional_identity_rates(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            decision = scores(0.8)
            decision["palm"] = {"status": "palm_too_small", "accepted": None, "distance": None, "threshold": 0.5}
            decision["fusion"] = {"status": "component_unavailable", "accepted": None, "distance": None, "threshold": 0.5}
            log.record(decision, {})
            methods = log.summary()["methods"]
            self.assertEqual(methods["geometry"]["fnmr"], 1)
            self.assertEqual(methods["palm"]["acquisition_failures"], 1)
            self.assertIsNone(methods["palm"]["fnmr"])
            self.assertIsNone(methods["fusion"]["fmr"])

    def test_empty_run_rates_are_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            for method in log.summary()["methods"].values():
                self.assertIsNone(method["fmr"])
                self.assertIsNone(method["fnmr"])
                self.assertIsNone(method["acquisition_failure_rate"])

    def test_malformed_decisions_and_accidental_feature_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            invalid = scores()
            invalid["geometry"]["accepted"] = False
            changed = scores()
            changed["palm"]["threshold"] = 0.7
            for decision, quality in ((invalid, {}), (changed, {}), (scores(), {"features": [1, 2]}),
                                      ({"claimed_id": "b", "modes": scores()}, {})):
                with self.subTest(decision=decision, quality=quality):
                    with self.assertRaises(ValueError):
                        log.record(decision, quality)
            self.assertEqual(log.count, 0)
            self.assertEqual(list((log.run_dir / "trials").iterdir()), [])

    def test_model_mutation_cannot_change_the_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            source = model()
            log = PalmTrialLog(folder, source, "a", "a")
            source["thresholds"]["geometry"] = 0.6
            with self.assertRaisesRegex(ValueError, "model has changed"):
                log.record(scores(), {})
            self.assertEqual(log.count, 0)

    def test_summary_cannot_mutate_frozen_operating_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            log.summary()["thresholds"]["geometry"] = 0.8
            changed = scores()
            changed["geometry"]["threshold"] = 0.8
            with self.assertRaisesRegex(ValueError, "frozen model"):
                log.record(changed, {})
            self.assertEqual(log.summary()["thresholds"]["geometry"], 0.5)

    def test_failed_publication_does_not_count_an_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            with patch("palm_trials.os.link", side_effect=OSError("publication failed")):
                with self.assertRaises(OSError):
                    log.record(scores(), {})
            self.assertEqual(log.count, 0)
            self.assertEqual(list((log.run_dir / "trials").iterdir()), [])

    def test_export_retry_preserves_one_committed_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            log = PalmTrialLog(folder, model(), "a", "a")
            with patch.object(log, "export", side_effect=OSError("report unavailable")):
                result = log.record(scores(), {})
            self.assertEqual(result["status"], "saved")
            self.assertIn("export_error", result)
            self.assertEqual(log.count, 1)
            self.assertEqual(len(list((log.run_dir / "trials").glob("*.json"))), 1)
            log.export()
            self.assertEqual(json.loads((log.run_dir / "summary.json").read_text())["attempts"], 1)

    def test_invalid_declared_distance_and_identifiers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for distance in (0, -1, True, float("nan")):
                with self.subTest(distance=distance), self.assertRaises(ValueError):
                    PalmTrialLog(folder, model(), "a", "a", distance)
            with self.assertRaises(ValueError):
                PalmTrialLog(folder, model(), "a", "../other")

    def test_digital_input_scale_is_validated_and_persisted_with_declared_distance(self):
        with tempfile.TemporaryDirectory() as folder:
            for scale in (0.09, 1.1, True, float("nan")):
                with self.subTest(scale=scale), self.assertRaises(ValueError):
                    PalmTrialLog(folder, model(), "a", "a", input_scale=scale)
            log = PalmTrialLog(folder, model(), "a", "a", 40, input_scale=0.5)
            log.record(scores(), {})
            self.assertEqual(log.summary()["input_scale"], 0.5)
            self.assertEqual(log.summary()["distance_cm"], 40)
            run = json.loads((log.run_dir / "run.json").read_text())
            trial = json.loads(next((log.run_dir / "trials").glob("*.json")).read_text())
            self.assertEqual(run["input_scale"], 0.5)
            self.assertEqual(trial["input_scale"], 0.5)
            with (log.run_dir / "trials.csv").open(newline="") as file:
                self.assertEqual({row["input_scale"] for row in csv.DictReader(file)}, {"0.5"})


if __name__ == "__main__":
    unittest.main()
