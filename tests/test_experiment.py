from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiment import main, rate_label, run_experiment
from geometry import RATIO_COLUMNS
from trial_log import TrialLog


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.measurements = {"status": "ok", **dict.fromkeys(RATIO_COLUMNS, 1.0)}
        self.model = {
            "schema_version": 1, "model_kind": "personal_demo", "method": "ratios",
            "feature_names": list(RATIO_COLUMNS), "threshold": 0.2,
            "templates": {"P01": [1.0] * 5},
        }
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, self.frame)
        self.log = Mock(model=self.model, count=0, run_dir=Path("unused-trial-run"))
        self.log.summary.return_value = {"attempts": 0}

        def record(decision):
            self.log.count += 1
            return {"status": "saved", "count": self.log.count}

        self.log.record.side_effect = record

    def run_camera(self, keys, observations=None, times=None, visible=1, actual="P01"):
        with ExitStack() as stack:
            patches = {
                "cv2.VideoCapture": {"return_value": self.camera},
                "cv2.namedWindow": {}, "cv2.imshow": {}, "cv2.destroyAllWindows": {},
                "cv2.waitKey": {"side_effect": keys},
                "cv2.getWindowProperty": {"return_value": visible},
                "experiment_preview": {"return_value": self.frame},
                "analyze_frame": ({"side_effect": observations} if observations is not None
                                  else {"return_value": (self.measurements, [])}),
                "monotonic_ns": ({"side_effect": times} if times is not None
                                 else {"return_value": 0}),
            }
            self.mocks = {
                name: stack.enter_context(patch(f"experiment.{name}", **options))
                for name, options in patches.items()
            }
            return run_experiment(0, Mock(), self.log, "P01", actual)

    def test_live_frames_and_quitting_do_not_record_attempts(self):
        self.run_camera([-1, -1, ord("q")])
        self.log.record.assert_not_called()
        self.log.export.assert_called_once()
        self.camera.release.assert_called_once()
        self.mocks["cv2.destroyAllWindows"].assert_called_once()

    def test_successful_attempts_require_hand_removal_and_one_second_spacing(self):
        observations = [(self.measurements, []), (self.measurements, []), ({"status": "no_hand"}, [])]
        observations += [(self.measurements, []), ({"status": "no_hand"}, [])]
        observations += [(self.measurements, [])] * 3
        self.run_camera(
            [32, 32, -1, 32, -1, 32, 32, ord("q")], observations,
            [0, 2_000_000_000, 2_100_000_000, 2_200_000_000,
             2_300_000_000, 2_400_000_000, 3_300_000_000, 4_500_000_000],
        )
        self.assertEqual(self.log.record.call_count, 3)
        self.assertTrue(all(call.args[0]["status"] == "ok" for call in self.log.record.call_args_list))
        self.log.export.assert_called_once()

    def test_manual_acquisition_failures_are_recorded_without_identity_decisions(self):
        self.run_camera(
            [32, 32, 32, ord("q")], [({"status": "no_hand"}, [])] * 4,
            [0, 500_000_000, 1_000_000_000, 1_100_000_000],
        )
        self.assertEqual(self.log.record.call_count, 2)
        for call in self.log.record.call_args_list:
            decision = call.args[0]
            self.assertEqual(decision["status"], "no_hand")
            self.assertIsNone(decision["accepted"])
            self.assertIsNone(decision["distance"])

    def test_window_close_and_uppercase_q_export_without_recording(self):
        for keys, visible in (([ord("Q")], 1), ([-1], 0)):
            with self.subTest(keys=keys):
                self.log.reset_mock()
                self.camera.reset_mock()
                self.run_camera(keys, visible=visible)
                self.log.record.assert_not_called()
                self.log.export.assert_called_once()
                self.camera.release.assert_called_once()

    def test_camera_failure_still_exports_and_releases_camera(self):
        self.camera.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "did not return a frame"):
            self.run_camera([])
        self.log.record.assert_not_called()
        self.log.export.assert_called_once()
        self.camera.release.assert_called_once()
        self.mocks["cv2.destroyAllWindows"].assert_called_once()

    def test_camera_open_failure_still_exports_and_releases_camera(self):
        self.camera.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
            self.run_camera([])
        self.log.export.assert_called_once()
        self.camera.release.assert_called_once()

    def test_nonpersonal_session_fails_before_opening_camera(self):
        self.model.pop("model_kind")
        with self.assertRaisesRegex(ValueError, "not a personal enrollment"):
            self.run_camera([])
        self.mocks["cv2.VideoCapture"].assert_not_called()
        self.log.export.assert_called_once()

    def test_export_failure_does_not_prevent_camera_cleanup(self):
        self.log.export.side_effect = OSError("export unavailable")
        with self.assertRaisesRegex(OSError, "export unavailable"):
            self.run_camera([27])
        self.camera.release.assert_called_once()
        self.mocks["cv2.destroyAllWindows"].assert_called_once()

    def test_video_timestamps_are_strictly_increasing(self):
        self.run_camera([-1, -1, ord("q")], times=[2_000_000, 2_000_000, 1_000_000])
        self.assertEqual([call.args[2] for call in self.mocks["analyze_frame"].call_args_list], [2, 3, 4])

    def test_rates_with_zero_denominators_are_shown_as_unavailable(self):
        self.assertEqual(rate_label(None), "n/a")
        self.assertEqual(rate_label(0.125), "12.50%")

    def test_real_logger_exports_manual_impostor_and_acquisition_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_path = root / "model.json"
            model_path.write_text(json.dumps(self.model), encoding="utf-8")
            self.log = TrialLog(root / "trials", model_path, "P01", "P02")
            summary = self.run_camera(
                [32, 32, ord("q")],
                [(self.measurements, []), ({"status": "no_hand"}, []), ({"status": "no_hand"}, [])],
                [0, 1_100_000_000, 2_200_000_000], actual="P02",
            )
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["valid_acquisitions"], 1)
            self.assertEqual(summary["acquisition_failures"], 1)
            self.assertEqual(summary["false_accepts"], 1)
            self.assertEqual(summary["fmr"], 1.0)
            self.assertIsNone(summary["fnmr"])
            with (self.log.run_dir / "trials.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["quality_status"] for row in rows], ["ok", "no_hand"])
            self.assertEqual(rows[1]["accepted"], "")
            self.assertFalse(list(self.log.run_dir.rglob("*.jpg")))
            self.assertFalse(list(self.log.run_dir.rglob("*.png")))

    def test_missing_profile_fails_before_opening_camera(self):
        with patch("sys.argv", ["experiment.py", "--user", "P01", "--actual", "P02"]), \
                patch("experiment.latest_personal_model", side_effect=ValueError("No completed enrollment")), \
                patch("experiment.cv2.VideoCapture") as camera, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 1)
        camera.assert_not_called()

    def test_cli_passes_declared_ids_and_condition_to_trial_log(self):
        landmarker = Mock()
        landmarker.is_file.return_value = True
        with patch("sys.argv", ["experiment.py", "--user", "P01", "--actual", "P02", "--condition", "far"]), \
                patch("experiment.DEFAULT_LANDMARKER", landmarker), \
                patch("experiment.latest_personal_model", return_value=Path("fixture-model.json")), \
                patch("experiment.mp.tasks.vision.HandLandmarker.create_from_options"), \
                patch("experiment.TrialLog", return_value=self.log) as factory, \
                patch("experiment.run_experiment", return_value={"attempts": 0}) as run, \
                redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual(factory.call_args.args[1:], (Path("fixture-model.json"), "P01", "P02", "far"))
        self.assertEqual(run.call_args.args[2:], (self.log, "P01", "P02", "far"))


if __name__ == "__main__":
    unittest.main()
