import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from enrollment import EnrollmentSession, validate_user_id
from matching import FEATURE_SETS, load_model


def reference():
    return {"schema_version": 1, "method": "ratios", "feature_names": FEATURE_SETS["ratios"],
            "threshold": 0.123, "templates": {"dataset_user": [1.0] * 5},
            "evaluation_protocol": "Exploratory image split."}


def measurement(value=1.0):
    return {"status": "ok", **dict.fromkeys(FEATURE_SETS["ratios"], value)}


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = np.full((20, 30, 3), 80, dtype=np.uint8)

    def test_user_id_validation_prevents_path_escape(self):
        self.assertEqual(validate_user_id("P01_demo-2"), "P01_demo-2")
        for invalid in ("", "../user", "a/b", "a\\b", "two words", "üser", "x" * 33, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_user_id(invalid)

    def test_manual_captures_save_pairs_and_median_model(self):
        session = EnrollmentSession("P01", self.root, reference(), sample_count=3)
        for index, value in enumerate((1, 99, 2)):
            result = session.add_sample(self.frame, measurement(value), index * 1.0)
            self.assertEqual(result, {"status": "saved", "count": index + 1, "target": 3})
            capture = session.session_dir / f"sample_{index + 1:03d}"
            image = cv2.imread(str(capture / "hand.jpg"))
            self.assertEqual(image.shape, self.frame.shape)
            metadata = json.loads((capture / "metadata.json").read_text())
            self.assertEqual(metadata["image_sha256"], hashlib.sha256((capture / "hand.jpg").read_bytes()).hexdigest())
        model_path = session.finalize()
        model = load_model(model_path)
        self.assertEqual(model["templates"], {"P01": [2.0] * 5})
        self.assertEqual(model["threshold"], 0.123)
        self.assertFalse(model["personal_calibration"])
        self.assertEqual(model["model_kind"], "personal_demo")
        self.assertEqual(model_path, session.finalize())
        with self.assertRaises(ValueError):
            session.add_sample(self.frame, measurement(), 99)

    def test_invalid_or_too_soon_samples_do_not_increase_count(self):
        session = EnrollmentSession("P01", self.root, reference(), sample_count=2)
        self.assertEqual(session.add_sample(self.frame, {"status": "no_hand"}, 0)["status"], "no_hand")
        self.assertEqual(session.add_sample(self.frame, measurement(float("nan")), 0)["status"], "invalid_measurements")
        self.assertEqual(session.add_sample(np.zeros((10, 10)), measurement(), 0)["status"], "invalid_frame")
        self.assertEqual(session.add_sample(self.frame, measurement(), float("nan"))["status"], "invalid_timestamp")
        self.assertEqual(session.count, 0)
        self.assertEqual(session.add_sample(self.frame, measurement(), 2)["status"], "saved")
        self.assertEqual(session.add_sample(self.frame, measurement(), 2.1)["status"], "capture_too_soon")
        self.assertEqual(session.add_sample(self.frame, measurement(), 1)["status"], "capture_too_soon")
        self.assertEqual(session.count, 1)
        with self.assertRaises(ValueError):
            session.finalize()
        self.assertFalse((session.session_dir / "model.json").exists())

    def test_unique_sessions_and_reference_validation(self):
        first = EnrollmentSession("P01", self.root, reference())
        second = EnrollmentSession("P01", self.root, reference())
        self.assertNotEqual(first.session_dir, second.session_dir)
        for changes in ({"method": "pixels"}, {"feature_names": ["wrong"]}, {"threshold": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EnrollmentSession("P02", self.root, {**reference(), **changes})

    def test_failed_metadata_write_preserves_previous_samples(self):
        session = EnrollmentSession("P01", self.root, reference(), sample_count=2)
        session.add_sample(self.frame, measurement(), 0)
        with patch("enrollment.write_json", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                session.add_sample(self.frame, measurement(2), 1)
        self.assertEqual(session.count, 1)
        self.assertTrue((session.session_dir / "sample_001" / "hand.jpg").exists())
        self.assertFalse((session.session_dir / "sample_002").exists())
        self.assertEqual(session.add_sample(self.frame, measurement(2), 1)["status"], "saved")

    def test_reference_file_provenance_and_capture_limit(self):
        path = self.root / "reference.json"
        path.write_text(json.dumps(reference()), encoding="utf-8")
        session = EnrollmentSession("P01", self.root / "captures", reference(), sample_count=1, reference_model_path=path)
        session.add_sample(self.frame, measurement(), 0)
        self.assertEqual(session.add_sample(self.frame, measurement(), 2)["status"], "enrollment_complete")
        model = json.loads(session.finalize().read_text())
        self.assertEqual(model["reference_model_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(model["reference_model_hash_kind"], "file_bytes")
        self.assertEqual(model["enrollment_counts"], {"P01": 1})


if __name__ == "__main__":
    unittest.main()
