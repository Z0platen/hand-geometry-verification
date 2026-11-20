import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from geometry import RATIO_COLUMNS
from multimodal import build_personal_model, load_model
from palm_demo import run_camera
from palm_enrollment import PalmEnrollment, latest_palm_model
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE


def reference():
    return {"schema_version": 2, "reference_kind": "palm_geometry_calibration",
            "descriptor_version": DESCRIPTOR_VERSION, "geometry_feature_names": list(RATIO_COLUMNS),
            "palm_feature_size": PALM_FEATURE_SIZE, "quality": dict(DEFAULT_QUALITY),
            "thresholds": {"geometry": 0.5, "palm": 0.75, "fusion": 0.9},
            "fusion": {"geometry_scale": 0.2, "palm_scale": 0.4,
                       "weights": {"geometry": 0.5, "palm": 0.5}}}


class PalmWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.geometry = {"status": "ok", "features": [1.0] * 5}
        self.palm = {"status": "ok", "features": (np.ones(PALM_FEATURE_SIZE) / np.sqrt(PALM_FEATURE_SIZE)).tolist(),
                     "quality": {"native_roi_pixels": 100.0, "contrast": 5.0, "sharpness": 15.0}}

    def test_lossless_palm_profile_preserves_existing_geometry_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "demo" / "old"
            old.mkdir(parents=True)
            (old / "model.json").write_text("old geometry profile")
            session = PalmEnrollment("demo", reference(), root, 5)
            self.assertIsNone(latest_palm_model("demo", root))
            self.assertEqual(session.add_sample(self.frame, self.geometry, {**self.palm, "status": "palm_too_small"}), "palm_too_small")
            self.assertEqual(session.count, 0)
            for index in range(5):
                frame = np.full_like(self.frame, index)
                self.assertEqual(session.add_sample(frame, self.geometry, self.palm), "saved")
            path = session.finalize()
            model = load_model(path)
            self.assertEqual(model["enrollment_counts"], {"demo": 5})
            self.assertEqual(model["thresholds"], reference()["thresholds"])
            self.assertEqual(latest_palm_model("demo", root), path)
            self.assertEqual(session.finalize(), path)
            self.assertEqual((old / "model.json").read_text(), "old geometry profile")
            self.assertTrue((session.session_dir / "sample_001" / "hand.png").is_file())

    def test_duplicate_and_failed_writes_leave_capture_count_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            session = PalmEnrollment("demo", reference(), Path(directory), 5)
            self.assertEqual(session.add_sample(self.frame, self.geometry, self.palm), "saved")
            self.assertEqual(session.add_sample(self.frame, self.geometry, self.palm), "duplicate_capture")
            with patch("palm_enrollment.write_json", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    session.add_sample(np.ones_like(self.frame), self.geometry, self.palm)
            self.assertEqual(session.count, 1)
            self.assertFalse((session.session_dir / "sample_002").exists())
            self.assertFalse((session.session_dir / "palm_model.json").exists())

    def test_live_preview_does_not_automatically_enroll_or_log(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", return_value=ord("q")), \
                patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.analyze", return_value=(self.geometry, self.palm, [])), \
                patch("palm_demo.palm_preview", return_value=self.frame), \
                patch("palm_demo.PalmEnrollment") as session, patch("palm_demo.PalmTrialLog") as log:
            run_camera(0, Mock(), "demo", reference(), None)
        session.return_value.add_sample.assert_not_called()
        log.assert_not_called()
        camera.release.assert_called_once()

    def test_completed_enrollment_waits_for_removal_before_manual_trial(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        model = build_personal_model([{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference())
        session = Mock(count=5)
        session.add_sample.return_value = "saved"
        session.finalize.return_value = Path("synthetic-palm-model.json")
        no_hand = ({"status": "no_hand", "features": None},
                   {**self.palm, "status": "palm_no_hand", "features": None}, [])
        valid = (self.geometry, self.palm, [])
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), \
                patch("palm_demo.cv2.waitKey", side_effect=[32, ord("t"), -1, ord("t"), ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), \
                patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.monotonic_ns", side_effect=[0, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000]), \
                patch("palm_demo.analyze", side_effect=[valid, valid, no_hand, valid, valid]), \
                patch("palm_demo.palm_preview", return_value=self.frame), \
                patch("palm_demo.PalmEnrollment", return_value=session), \
                patch("palm_demo.load_model", return_value=model), patch("palm_demo.PalmTrialLog") as log:
            log.return_value.record.return_value = {"status": "saved"}
            run_camera(0, Mock(), "demo", reference(), None, actual="demo", samples=5)
        self.assertEqual(log.return_value.record.call_count, 1)
        self.assertTrue(log.return_value.record.call_args.args[0]["modes"]["fusion"]["accepted"])
        log.return_value.export.assert_called_once()
        camera.release.assert_called_once()

    def test_quality_warning_clears_as_soon_as_current_frame_is_usable(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        bad_palm = {**self.palm, "status": "palm_low_contrast", "features": None}
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", side_effect=[-1, ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), \
                patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.analyze", side_effect=[(self.geometry, bad_palm, []), (self.geometry, self.palm, [])]), \
                patch("palm_demo.palm_preview", return_value=self.frame) as preview, \
                patch("palm_demo.PalmEnrollment"):
            run_camera(0, Mock(), "demo", reference(), None)
        self.assertIn("Improve palm lighting", preview.call_args_list[0].args[9])
        self.assertIn("Ready. Press SPACE", preview.call_args_list[1].args[9])
        self.assertNotIn("Improve palm lighting", preview.call_args_list[1].args[9])

    def test_save_button_click_saves_one_sample_and_waits_for_hand_removal(self):
        import cv2
        from palm_view import SAVE_BUTTON_BOUNDS
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        session = Mock(count=0)

        def save(*_args):
            session.count += 1
            return "saved"

        session.add_sample.side_effect = save
        callback = None

        def bind(_name, function):
            nonlocal callback
            callback = function

        def click_save(_delay):
            left, top, right, bottom = SAVE_BUTTON_BOUNDS
            callback(cv2.EVENT_LBUTTONUP, (left + right) // 2, (top + bottom) // 2, 0, None)
            return -1

        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), \
                patch("palm_demo.cv2.setMouseCallback", side_effect=bind), \
                patch("palm_demo.cv2.imshow"), \
                patch("palm_demo.cv2.waitKey", side_effect=[-1, -1, ord("q")]) as wait, \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), \
                patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.analyze", return_value=(self.geometry, self.palm, [])), \
                patch("palm_demo.palm_preview", return_value=self.frame) as preview, \
                patch("palm_demo.PalmEnrollment", return_value=session):
            calls = 0
            def events(delay):
                nonlocal calls
                calls += 1
                return click_save(delay) if calls < 3 else ord("q")
            wait.side_effect = events
            run_camera(0, Mock(), "demo", reference(), None)
        self.assertEqual(session.add_sample.call_count, 1)
        self.assertEqual(preview.call_args_list[1].args[6], "remove_hand")
        self.assertEqual(preview.call_args_list[1].args[7], 1)
        session.finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
