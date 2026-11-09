from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from enroll import run_enrollment
from geometry import RATIO_COLUMNS


class EnrollmentFlowTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.measurements = {"status": "ok", **dict.fromkeys(RATIO_COLUMNS, 1.0)}
        self.model = {
            "schema_version": 1,
            "method": "ratios",
            "feature_names": list(RATIO_COLUMNS),
            "threshold": 0.2,
            "templates": {"demo": [1.0] * 5},
        }
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, self.frame)
        self.session = Mock(count=0, target=5)

    def test_opening_and_quitting_does_not_capture(self):
        with patch("enroll.cv2.VideoCapture", return_value=self.camera), \
                patch("enroll.EnrollmentSession", return_value=self.session), \
                patch("enroll.cv2.namedWindow"), \
                patch("enroll.cv2.imshow"), \
                patch("enroll.cv2.waitKey", return_value=ord("q")), \
                patch("enroll.cv2.destroyAllWindows") as close_windows, \
                patch("enroll.analyze_frame", return_value=(self.measurements, [])), \
                patch("enroll.enrollment_preview", return_value=self.frame):
            result = run_enrollment(0, Mock(), "demo", Path("captures"), self.model, 5)
        self.assertIsNone(result)
        self.session.add_sample.assert_not_called()
        self.session.finalize.assert_not_called()
        self.camera.release.assert_called_once()
        close_windows.assert_called_once()

    def test_completed_enrollment_waits_for_hand_removal_before_scoring(self):
        def save_sample(*args):
            self.session.count += 1
            return {"status": "saved", "count": self.session.count, "target": 5}

        self.session.add_sample.side_effect = save_sample
        model_path = Path("captures/demo/session/model.json")
        self.session.finalize.return_value = model_path
        observations = [(self.measurements, [])] * 5
        observations += [({"status": "no_hand"}, []), (self.measurements, [])]
        with patch("enroll.cv2.VideoCapture", return_value=self.camera), \
                patch("enroll.EnrollmentSession", return_value=self.session), \
                patch("enroll.cv2.namedWindow"), \
                patch("enroll.cv2.imshow"), \
                patch("enroll.cv2.waitKey", side_effect=[32] * 5 + [-1, ord("q")]), \
                patch("enroll.cv2.getWindowProperty", return_value=1), \
                patch("enroll.cv2.destroyAllWindows"), \
                patch("enroll.analyze_frame", side_effect=observations), \
                patch("enroll.enrollment_preview", return_value=self.frame), \
                patch("enroll.verification_preview", return_value=self.frame) as verify_preview, \
                patch("enroll.load_model", return_value=self.model):
            result = run_enrollment(0, Mock(), "demo", Path("captures"), self.model, 5)
        self.assertEqual(result, model_path)
        self.assertEqual(self.session.add_sample.call_count, 5)
        self.session.finalize.assert_called_once()
        verify_preview.assert_called_once()
        self.assertEqual(self.camera.read.call_count, 7)
        self.camera.release.assert_called_once()

    def test_camera_read_failure_releases_resources(self):
        self.camera.read.return_value = (False, None)
        with patch("enroll.cv2.VideoCapture", return_value=self.camera), \
                patch("enroll.EnrollmentSession", return_value=self.session), \
                patch("enroll.cv2.namedWindow"), \
                patch("enroll.cv2.destroyAllWindows") as close_windows:
            with self.assertRaisesRegex(RuntimeError, "did not return a frame"):
                run_enrollment(0, Mock(), "demo", Path("captures"), self.model, 5)
        self.session.add_sample.assert_not_called()
        self.camera.release.assert_called_once()
        close_windows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
