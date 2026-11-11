import unittest
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from geometry import RATIO_COLUMNS
from verify import analyze_image, decision_from_measurements, latest_personal_model, main, verify_camera


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

    def test_empty_image_has_a_clear_decode_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jpg"
            path.touch()
            with self.assertRaisesRegex(ValueError, "Could not decode image"):
                analyze_image(path, Mock(), "P01", self.model)

    def test_uppercase_q_closes_camera_and_releases_resources(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        measurements = {"status": "ok", **dict.fromkeys(RATIO_COLUMNS, 1.0)}
        with patch("verify.cv2.VideoCapture", return_value=camera), \
                patch("verify.cv2.namedWindow"), patch("verify.cv2.imshow"), \
                patch("verify.cv2.waitKey", return_value=ord("Q")), \
                patch("verify.analyze_frame", return_value=(measurements, [])), \
                patch("verify.verification_preview", return_value=frame), \
                patch("verify.cv2.destroyAllWindows"):
            verify_camera(0, Mock(), "P01", self.model)
        camera.read.assert_called_once()
        camera.release.assert_called_once()

    def test_latest_profile_skips_newer_incomplete_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "P01" / name for name in ("20260101-old", "20260102-new", "20260103-partial")]
            for path in paths:
                path.mkdir(parents=True)
            for path in paths[:2]:
                (path / "model.json").write_text(json.dumps(self.model), encoding="utf-8")
            (paths[2] / "model.json").mkdir()
            self.assertEqual(latest_personal_model("P01", root), paths[1] / "model.json")

    def test_missing_profile_and_invalid_user_id_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No completed enrollment"):
                latest_personal_model("P01", Path(directory))
            with self.assertRaises(ValueError):
                latest_personal_model("../P01", Path(directory))

    def test_user_shortcut_opens_selected_personal_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "P01" / "20260101-session"
            session.mkdir(parents=True)
            model = {**self.model, "model_kind": "personal_demo"}
            (session / "model.json").write_text(json.dumps(model), encoding="utf-8")
            landmarker = root / "landmarker.task"
            landmarker.touch()
            with patch("sys.argv", ["verify.py", "--user", "P01", "--captures", directory, "--camera", "0"]), \
                    patch("verify.DEFAULT_LANDMARKER", landmarker), \
                    patch("verify.mp.tasks.vision.HandLandmarker.create_from_options") as factory, \
                    patch("verify.verify_camera") as camera:
                self.assertEqual(main(), 0)
            camera.assert_called_once_with(0, factory.return_value.__enter__.return_value, "P01", model)

    def test_conflicting_identity_flags_fail_before_camera(self):
        combinations = [
            ["--model", "unused.json"],
            ["--user", "P01", "--claim", "P02"],
            ["--model", "unused.json", "--claim", "P01", "--captures", "unused"],
        ]
        for flags in combinations:
            with self.subTest(flags=flags), redirect_stderr(StringIO()), \
                    patch("sys.argv", ["verify.py", *flags, "--camera", "0"]), \
                    patch("verify.cv2.VideoCapture") as camera:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                camera.assert_not_called()

    def test_saves_only_manually_selected_current_frame_and_decision(self):
        frames = [np.full((240, 320, 3), value, dtype=np.uint8) for value in range(3)]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames]
        measurements = {"status": "ok", **dict.fromkeys(RATIO_COLUMNS, 1.0)}
        with patch("verify.cv2.VideoCapture", return_value=camera), \
                patch("verify.cv2.namedWindow"), patch("verify.cv2.imshow"), \
                patch("verify.cv2.waitKey", side_effect=[-1, ord("s"), ord("q")]), \
                patch("verify.cv2.getWindowProperty", return_value=1), \
                patch("verify.analyze_frame", side_effect=[(measurements, []), ({"status": "no_hand"}, []), (measurements, [])]), \
                patch("verify.verification_preview", side_effect=frames), \
                patch("verify.snapshot_feedback", return_value="Saved") as save, \
                patch("verify.cv2.destroyAllWindows"):
            verify_camera(0, Mock(), "P01", self.model)
        self.assertEqual(save.call_count, 1)
        self.assertIs(save.call_args.args[0], frames[1])
        self.assertEqual(save.call_args.args[1]["status"], "no_hand")
        self.assertIsNone(save.call_args.args[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
