import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matching import FEATURE_SETS
from trial_log import QUALITY_STATUSES, TrialLog


class TrialLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "reference.json"
        self.model_path.write_text(json.dumps({
            "schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
            "threshold": 0.2, "templates": {"alice": [1.0] * 5},
        }), encoding="utf-8")

    def log(self, actual="alice", **kwargs):
        return TrialLog(self.root / "experiments", self.model_path, "alice", actual, **kwargs)

    def decision(self, distance=0.1, **kwargs):
        accepted = isinstance(distance, (int, float)) and distance <= 0.2
        return {"status": "ok", "claimed_id": "alice", "accepted": accepted,
                "distance": distance, "threshold": 0.2, **kwargs}

    def test_valid_trials_and_acquisition_failures_have_separate_denominators(self):
        log = self.log()
        self.assertIsNone(log.summary()["fmr"])
        self.assertIsNone(log.summary()["fnmr"])
        self.assertIsNone(log.summary()["acquisition_failure_rate"])
        self.assertEqual(log.record(self.decision(distance=0.2)), {"status": "saved", "count": 1})
        log.record(self.decision(distance=0.3))
        for status in sorted(QUALITY_STATUSES - {"ok"}):
            log.record(self.decision(status=status, accepted=None, distance=None))
        summary = log.summary()
        self.assertEqual(summary["attempts"], 8)
        self.assertEqual(summary["valid_acquisitions"], 2)
        self.assertEqual(summary["acquisition_failures"], 6)
        self.assertEqual(summary["genuine_comparisons"], 2)
        self.assertEqual(summary["false_rejects"], 1)
        self.assertEqual(summary["fnmr"], 0.5)
        self.assertIsNone(summary["fmr"])
        self.assertEqual(summary["acquisition_failure_rate"], 0.75)

    def test_impostor_false_accepts_and_boundary_use_fixed_threshold(self):
        log = self.log(actual="bob")
        log.record(self.decision(distance=0.2))
        log.record(self.decision(distance=0.3))
        summary = log.summary()
        self.assertEqual(summary["impostor_comparisons"], 2)
        self.assertEqual(summary["false_accepts"], 1)
        self.assertEqual(summary["fmr"], 0.5)
        self.assertIsNone(summary["fnmr"])

    def test_snapshot_provenance_unique_run_and_exports_contain_no_image_data(self):
        original = self.model_path.read_bytes()
        log = self.log(condition=" different distance ")
        another = self.log()
        self.assertNotEqual(log.run_dir, another.run_dir)
        self.model_path.write_text("changed after run started", encoding="utf-8")
        log.record(self.decision(measurements={"private": "not stored"}, frame="not stored"))
        self.assertEqual((log.run_dir / "model.json").read_bytes(), original)
        trial = json.loads((log.run_dir / "trials" / "000001.json").read_text(encoding="utf-8"))
        self.assertEqual(trial["model_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(trial["condition"], "different distance")
        self.assertTrue(trial["timestamp_utc"].endswith("+00:00"))
        self.assertNotIn("measurements", trial)
        self.assertNotIn("frame", trial)
        self.assertFalse(list(log.run_dir.rglob("*.jpg")))
        paths = log.export()
        with paths["trials_csv"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trial_index"], "1")
        summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
        self.assertEqual(summary["attempts"], 1)
        self.assertIn("not independent", summary["protocol"])

    def test_inconsistent_or_malformed_decisions_never_create_trials(self):
        log = self.log()
        examples = [None, {}, self.decision(claimed_id="bob"), self.decision(status="unknown"),
                    self.decision(threshold=0.3), self.decision(threshold=True),
                    self.decision(distance=-0.1), self.decision(distance=float("nan")),
                    self.decision(distance=float("inf")), self.decision(distance=True),
                    self.decision(distance="0.1"), self.decision(accepted=False),
                    self.decision(accepted=1), self.decision(status="no_hand"),
                    self.decision(status="no_hand", accepted=None, distance=0.1)]
        for index, decision in enumerate(examples):
            with self.subTest(case=index), self.assertRaises(ValueError):
                log.record(decision)
        self.assertEqual(log.count, 0)
        self.assertEqual(list((log.run_dir / "trials").iterdir()), [])

    def test_invalid_inputs_do_not_publish_a_run(self):
        for actual, condition in (("../bob", "standard"), ("", "standard"), ("bob", ""),
                                  ("bob", "two\nlines"), ("bob", "x" * 81)):
            with self.subTest(actual=actual, condition=condition), self.assertRaises(ValueError):
                self.log(actual=actual, condition=condition)
        with self.assertRaises(ValueError):
            TrialLog(self.root / "experiments", self.model_path, "missing", "bob")
        self.model_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.log()
        self.assertEqual(list((self.root / "experiments").iterdir()), [])

    def test_changed_in_memory_model_is_rejected(self):
        log = self.log()
        log.model["templates"]["alice"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "fixed model"):
            log.record(self.decision())
        self.assertEqual(log.count, 0)

    def test_failed_trial_write_leaves_count_and_canonical_files_unchanged(self):
        log = self.log()
        with patch("trial_log.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                log.record(self.decision())
        self.assertEqual(log.count, 0)
        self.assertEqual(list((log.run_dir / "trials").iterdir()), [])
        log.record(self.decision())
        self.assertEqual(log.count, 1)

    def test_export_failure_preserves_saved_trial_and_retry_does_not_duplicate(self):
        log = self.log()
        with patch("trial_log._replace_bytes", side_effect=OSError("CSV open elsewhere")):
            result = log.record(self.decision())
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["count"], 1)
        self.assertIn("export_error", result)
        self.assertTrue((log.run_dir / "trials" / "000001.json").is_file())
        log.export()
        with (log.run_dir / "trials.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        self.assertEqual(log.count, 1)

    def test_existing_canonical_attempt_is_not_overwritten(self):
        log = self.log()
        target = log.run_dir / "trials" / "000001.json"
        target.write_text("existing trial", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            log.record(self.decision())
        self.assertEqual(log.count, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "existing trial")

    def test_cleanup_failure_after_commit_keeps_saved_count(self):
        log = self.log()
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.name.startswith(".trial-"):
                raise OSError("temporary file busy")
            return original_unlink(path, *args, **kwargs)

        with patch("trial_log.Path.unlink", new=unlink):
            result = log.record(self.decision())
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["count"], 1)
        self.assertIn("cleanup_error", result)
        self.assertEqual(log.count, 1)


if __name__ == "__main__":
    unittest.main()
