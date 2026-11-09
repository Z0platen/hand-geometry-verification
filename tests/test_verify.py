import unittest
from unittest.mock import Mock, patch

from geometry import RATIO_COLUMNS
from verify import decision_from_measurements, verify_camera


class VerificationDecisionTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "schema_version": 1,
            "method": "ratios",
            "feature_names": list(RATIO_COLUMNS),
            "threshold": 0.2,
            "templates": {"P01": [1.0] * 5},
        }

    def test_valid_measurements_use_calibrated_distance(self):
        measurements = {"status": "ok", **dict.fromkeys(RATIO_COLUMNS, 1.0)}
        decision = decision_from_measurements(measurements, "P01", self.model)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["distance"], 0.0)
        self.assertEqual(decision["threshold"], 0.2)

    def test_quality_failure_produces_no_match_decision(self):
        decision = decision_from_measurements({"status": "no_hand"}, "P01", self.model)
        self.assertIsNone(decision["accepted"])
        self.assertIsNone(decision["distance"])
        self.assertEqual(decision["status"], "no_hand")

    def test_unknown_claim_fails_even_when_capture_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown claimed identity"):
            decision_from_measurements({"status": "no_hand"}, "P99", self.model)

    def test_camera_read_failure_releases_resources(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with patch("verify.cv2.VideoCapture", return_value=camera), \
                patch("verify.cv2.namedWindow"), \
                patch("verify.cv2.destroyAllWindows") as close_windows:
            with self.assertRaisesRegex(RuntimeError, "did not return a frame"):
                verify_camera(0, Mock(), "P01", self.model)
        camera.release.assert_called_once()
        close_windows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
