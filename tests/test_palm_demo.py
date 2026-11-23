import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from geometry import RATIO_COLUMNS
from multimodal import PreparedMatcher, build_personal_model, load_model, validate_model
from palm_demo import compare, live_guidance, run_camera
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

    @staticmethod
    def focus_hand():
        return [SimpleNamespace(x=0.40 + (index % 5) * 0.05, y=0.30 + (index // 5) * 0.05)
                for index in range(21)]

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

    def test_mirror_toggle_is_display_only_and_enrollment_uses_original_frame(self):
        frame = self.frame.copy()
        frame[:, :40] = (10, 20, 210)
        frame[:, 240:] = (190, 35, 5)
        original = frame.copy()
        hand = object()
        hands = [hand]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        session = Mock(count=0)
        session.add_sample.return_value = "saved"
        preview = Mock(side_effect=lambda *args, **kwargs: args[0])
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", side_effect=[-1, ord("m"), 32, ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.monotonic_ns", side_effect=[0, 2_000_000_000, 4_000_000_000, 6_000_000_000]), \
                patch("palm_demo.analyze", return_value=(self.geometry, self.palm, hands)) as analyze, \
                patch("palm_demo.palm_preview", preview), patch("palm_demo.PalmEnrollment", return_value=session):
            run_camera(0, Mock(), "demo", reference(), None)
        self.assertEqual([call.kwargs["mirror"] for call in preview.call_args_list], [True, True, False, False])
        self.assertTrue(all(call.args[0] is frame for call in analyze.call_args_list))
        self.assertIs(session.add_sample.call_args.args[0], frame)
        np.testing.assert_array_equal(frame, original)
        self.assertIs(session.add_sample.call_args.args[2], self.palm)

    def test_focus_defaults_on_f_toggles_view_and_keeps_sample_source_original(self):
        frame = np.full((720, 1280, 3), (12, 70, 210), dtype=np.uint8)
        original = frame.copy()
        hands = [self.focus_hand()]
        palm = {**self.palm, "corners": np.array([[540, 260], [760, 260], [760, 460], [540, 460]], dtype=float)}
        geometry = dict(self.geometry)
        geometry_before = dict(geometry)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        session = Mock(count=0)
        session.add_sample.return_value = "saved"
        preview = Mock(side_effect=lambda *args, **kwargs: args[0])
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", side_effect=[-1, ord("f"), -1, ord("f"), 32, ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.monotonic_ns", side_effect=[0, 2_000_000_000, 4_000_000_000, 6_000_000_000, 8_000_000_000, 10_000_000_000]), \
                patch("palm_demo.analyze", return_value=(geometry, palm, hands)) as analyze, \
                patch("palm_demo.palm_preview", preview), patch("palm_demo.PalmEnrollment", return_value=session):
            run_camera(0, Mock(), "demo", reference(), None)
        viewports = [call.kwargs["viewport"] for call in preview.call_args_list]
        self.assertIsNotNone(viewports[0])
        self.assertIsNotNone(viewports[1])
        self.assertIsNone(viewports[2])
        self.assertIsNone(viewports[3])
        self.assertIsNotNone(viewports[4])
        self.assertIsNotNone(viewports[5])
        self.assertTrue(all(call.args[0] is frame for call in analyze.call_args_list))
        self.assertIs(session.add_sample.call_args.args[0], frame)
        self.assertIs(session.add_sample.call_args.args[1], geometry)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(geometry, geometry_before)

    def test_focus_toggle_leaves_screenshot_and_trial_keys_active(self):
        frame = np.full((720, 1280, 3), (22, 35, 80), dtype=np.uint8)
        hands = [self.focus_hand()]
        palm = {**self.palm, "corners": np.array([[540, 260], [760, 260], [760, 460], [540, 460]], dtype=float)}
        model = build_personal_model(
            [{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference(),
        )
        scores = {"modes": {"fusion": {"accepted": True, "distance": 0.2, "threshold": 0.9}}}
        matcher = Mock()
        matcher.score_probe.return_value = scores
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        preview = Mock(return_value=frame.copy())
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", side_effect=[ord("f"), ord("s"), ord("t"), ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.monotonic_ns", side_effect=[0, 2_000_000_000, 4_000_000_000, 6_000_000_000]), \
                patch("palm_demo.analyze", return_value=(self.geometry, palm, hands)) as analyze, \
                patch("palm_demo.palm_preview", preview), patch("palm_demo.PreparedMatcher", return_value=matcher), \
                patch("palm_demo.snapshot_feedback", return_value="Screenshot saved") as screenshot, \
                patch("palm_demo.PalmTrialLog") as trial_log:
            trial_log.return_value.record.return_value = {"status": "saved"}
            run_camera(0, Mock(), "demo", reference(), model, actual="demo")
        self.assertTrue(preview.call_args_list[0].kwargs["viewport"])
        self.assertIsNone(preview.call_args_list[1].kwargs["viewport"])
        self.assertEqual(screenshot.call_count, 1)
        self.assertEqual(trial_log.return_value.record.call_count, 1)
        self.assertIs(trial_log.return_value.record.call_args.args[0], scores)
        self.assertTrue(all(call.args[0] is frame for call in analyze.call_args_list))

    def test_live_guidance_distinguishes_no_hand_tracking_and_usability(self):
        palm_small = {**self.palm, "status": "palm_too_small", "minimum_roi_pixels": 64,
                      "quality": {**self.palm["quality"], "native_roi_pixels": 28}}
        no_hand = live_guidance({"status": "invalid_landmarks"}, palm_small, "verifying", False, hands=[])
        tracking = live_guidance({"status": "invalid_landmarks"}, palm_small, "verifying", False, hands=[object()])
        out_of_frame = live_guidance({"status": "finger_out_of_frame"}, palm_small, "verifying", False, hands=[object()])
        too_small = live_guidance({"status": "ok"}, palm_small, "verifying", False, hands=[object()])
        low_contrast = live_guidance(
            {"status": "ok"}, {**self.palm, "status": "palm_low_contrast"}, "verifying", False, hands=[object()],
        )
        self.assertIn("Show your whole right hand", no_hand)
        self.assertNotIn("Hand detected", no_hand)
        self.assertIn("Hand detected", tracking)
        self.assertIn("unclear", tracking)
        self.assertIn("whole hand", out_of_frame)
        self.assertIn("28px", too_small)
        self.assertIn("64px", too_small)
        self.assertIn("whole hand in frame", too_small)
        self.assertIn("even lighting", low_contrast)

    def test_invalid_current_geometry_cannot_become_accept_or_saved_sample(self):
        model = build_personal_model(
            [{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference(),
        )
        invalid = {**self.geometry, "status": "finger_out_of_frame"}
        result = compare(invalid, self.palm, "demo", model)
        self.assertIsNone(result["modes"]["fusion"]["accepted"])
        with tempfile.TemporaryDirectory() as directory:
            session = PalmEnrollment("demo", reference(), Path(directory), 5)
            status = session.add_sample(self.frame, invalid, self.palm)
            self.assertEqual(status, "finger_out_of_frame")
            self.assertEqual(session.count, 0)
            self.assertEqual(list(session.session_dir.glob("sample_*")), [])

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
                patch("palm_demo.load_model", return_value=model), \
                patch("palm_demo.PreparedMatcher", wraps=PreparedMatcher) as prepared_matcher, \
                patch("palm_demo.PalmTrialLog") as log:
            log.return_value.record.return_value = {"status": "saved"}
            run_camera(0, Mock(), "demo", reference(), None, actual="demo", samples=5)
        self.assertEqual(log.return_value.record.call_count, 1)
        self.assertTrue(log.return_value.record.call_args.args[0]["modes"]["fusion"]["accepted"])
        log.return_value.export.assert_called_once()
        prepared_matcher.assert_called_once_with(model)
        camera.release.assert_called_once()

    def test_live_frames_validate_and_prepare_model_only_once(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        model = build_personal_model([{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference())
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", side_effect=[-1, -1, -1, ord("q")]), \
                patch("palm_demo.cv2.getWindowProperty", return_value=1), patch("palm_demo.cv2.destroyAllWindows"), \
                patch("palm_demo.analyze", return_value=(self.geometry, self.palm, [])), \
                patch("palm_demo.palm_preview", return_value=self.frame), \
                patch("multimodal.validate_model", wraps=validate_model) as validate:
            run_camera(0, Mock(), "demo", reference(), model)
        validate.assert_called_once_with(model)
        camera.release.assert_called_once()

    def test_camera_resources_are_released_when_matcher_preparation_fails(self):
        camera = Mock()
        camera.isOpened.return_value = True
        model = build_personal_model([{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference())
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.destroyAllWindows") as destroy, \
                patch("palm_demo.PreparedMatcher", side_effect=ValueError("invalid model")), \
                self.assertRaisesRegex(ValueError, "invalid model"):
            run_camera(0, Mock(), "demo", reference(), model)
        camera.release.assert_called_once()
        destroy.assert_called_once()

    def test_camera_resources_are_released_when_prepared_scoring_fails(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, self.frame)
        model = build_personal_model([{"geometry": self.geometry["features"], "palm": self.palm["features"]}], "demo", reference())
        matcher = Mock()
        matcher.score_probe.side_effect = ValueError("invalid probe")
        with patch("palm_demo.cv2.VideoCapture", return_value=camera), \
                patch("palm_demo.cv2.namedWindow"), patch("palm_demo.cv2.resizeWindow"), patch("palm_demo.cv2.setMouseCallback"), \
                patch("palm_demo.cv2.destroyAllWindows") as destroy, \
                patch("palm_demo.cv2.imshow"), patch("palm_demo.cv2.waitKey", return_value=-1), \
                patch("palm_demo.analyze", return_value=(self.geometry, self.palm, [])), \
                patch("palm_demo.palm_preview", return_value=self.frame), \
                patch("palm_demo.PreparedMatcher", return_value=matcher), \
                self.assertRaisesRegex(ValueError, "invalid probe"):
            run_camera(0, Mock(), "demo", reference(), model)
        camera.release.assert_called_once()
        destroy.assert_called_once()

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
                patch("palm_demo.analyze", side_effect=[(self.geometry, bad_palm, [object()]), (self.geometry, self.palm, [object()])]), \
                patch("palm_demo.palm_preview", return_value=self.frame) as preview, \
                patch("palm_demo.PalmEnrollment"):
            run_camera(0, Mock(), "demo", reference(), None)
        self.assertIn("even lighting", preview.call_args_list[0].args[9])
        self.assertIn("Ready. Press SPACE", preview.call_args_list[1].args[9])
        self.assertNotIn("even lighting", preview.call_args_list[1].args[9])

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
