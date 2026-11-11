import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from snapshots import save_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.preview = np.zeros((240, 400, 3), dtype=np.uint8)
        self.decision = {"status": "ok", "claimed_id": "demo", "distance": 0.1,
                         "threshold": 0.2, "accepted": True}
        self.model = {"method": "ratios", "templates": {"demo": [1.0] * 5}, "threshold": 0.2}

    def test_saved_pair_matches_preview_and_decision_without_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            target = save_snapshot(self.preview, self.decision, self.model, Path(directory))
            image = target / "preview.png"
            np.testing.assert_array_equal(cv2.imread(str(image)), self.preview)
            report = json.loads((target / "decision.json").read_text())
            self.assertEqual(report["decision"], self.decision)
            self.assertEqual(report["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertNotIn("templates", report)
            self.assertNotEqual(target, save_snapshot(self.preview, self.decision, self.model, Path(directory)))

    def test_publication_failure_leaves_previous_results_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_snapshot(self.preview, self.decision, self.model, root)
            with patch("snapshots.Path.rename", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    save_snapshot(self.preview, self.decision, self.model, root)
            self.assertEqual(list(root.iterdir()), [first])
            self.assertTrue((first / "decision.json").is_file())

    def test_no_hand_snapshot_preserves_null_decision(self):
        decision = {**self.decision, "status": "no_hand", "distance": None, "accepted": None}
        with tempfile.TemporaryDirectory() as directory:
            target = save_snapshot(self.preview, decision, self.model, Path(directory))
            report = json.loads((target / "decision.json").read_text())
            self.assertIsNone(report["decision"]["accepted"])

    def test_nonfinite_report_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_snapshot(self.preview, {**self.decision, "distance": float("nan")}, self.model, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
