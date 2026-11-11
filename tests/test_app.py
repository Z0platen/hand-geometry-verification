import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from app import ChildRunner, build_command, camera_index, condition_label, list_profiles, process_error_detail, validate_profile_id


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_session(self, user_id, session_id, count=10):
        directory = self.root / user_id / session_id
        directory.mkdir(parents=True)
        model = directory / "model.json"
        model.write_text(json.dumps({"enrollment_counts": {user_id: count}}), encoding="utf-8")
        return model

    def test_profiles_use_latest_completed_session_and_ignore_incomplete(self):
        self.make_session("P01", "20260101", 5)
        latest = self.make_session("P01", "20260201", 10)
        (self.root / "P01" / "20260301").mkdir()
        self.make_session("P02", "20260101", 7)
        self.make_session("invalid user", "20260101")
        profiles = list_profiles(self.root)
        self.assertEqual([profile.user_id for profile in profiles], ["P01", "P02"])
        self.assertEqual(profiles[0].model_path, latest)
        self.assertEqual(profiles[0].sample_count, 10)
        self.assertEqual(profiles[0].session_count, 2)
        self.assertEqual(list_profiles(self.root / "absent"), [])

    def test_unreadable_summary_does_not_hide_latest_profile(self):
        self.make_session("P01", "20260101")
        latest = self.make_session("P01", "20260201")
        latest.write_text("broken json", encoding="utf-8")
        profile, = list_profiles(self.root)
        self.assertEqual(profile.model_path, latest)
        self.assertIsNone(profile.sample_count)

    def test_validation_rejects_path_ids_and_invalid_cameras(self):
        self.assertEqual(validate_profile_id("P01_demo-2"), "P01_demo-2")
        for value in ("", "../P01", "a/b", "a\\b", "two words", "üser", "x" * 33, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_profile_id(value)
        self.assertEqual(camera_index(" 01 "), 1)
        for value in ("", "-1", "0.5", "camera0", True, "١", "1;echo bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera_index(value)
        for value in ("", "a\nb", "a\x00b", "a\x7fb", "x" * 81):
            with self.subTest(value=value), self.assertRaises(ValueError):
                condition_label(value)

    def test_commands_preserve_spaces_and_pass_explicit_identity(self):
        project = self.root / "project with spaces"
        image = self.root / "my photo.png"
        image.write_bytes(b"synthetic test file")
        command = build_command("photo", "P01", image_path=image, root=project)
        self.assertEqual(command, [str(Path(sys.executable).resolve()), str(project / "verify.py"),
                                  "--user=P01", "--image", str(image), "--show"])
        command = build_command("experiment", "P01", actual_id="P02", condition="low light", camera="1", root=project)
        self.assertEqual(command[2:], ["--user=P01", "--actual=P02", "--condition=low light", "--camera", "1"])
        self.assertEqual(build_command("enroll", "P01", root=project)[1], str(project / "enroll.py"))
        self.assertEqual(build_command("live", "P01", root=project)[1], str(project / "verify.py"))
        with self.assertRaises(ValueError):
            build_command("photo", "P01", image_path=self.root / "missing.png")

    def test_leading_hyphen_values_reach_real_child_profile_lookup(self):
        image = self.root / "synthetic.png"
        image.write_bytes(b"synthetic; not decoded because no profile exists")
        commands = [
            build_command("photo", "-claimed", image_path=image),
            build_command("experiment", "-claimed", actual_id="-actual", condition="-far"),
        ]
        for command in commands:
            # An isolated empty capture root forces exit before image decoding,
            # detector creation, or opening a camera. Exit 1 from profile lookup
            # proves the real child parser accepted every leading-hyphen value.
            with self.subTest(script=command[1]):
                result = subprocess.run(
                    command + ["--captures", str(self.root / "empty")],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("No completed enrollment for '-claimed'", result.stderr)

    def test_blank_identity_errors_name_the_required_field(self):
        with self.assertRaisesRegex(ValueError, "Enter the Actual person ID"):
            build_command("experiment", "musa", actual_id=" ")
        with self.assertRaisesRegex(ValueError, "Enter the New user ID"):
            build_command("enroll", "")

    def test_runner_allows_one_child_and_reports_logged_failure(self):
        runner = ChildRunner(self.root)
        child = Mock()
        child.poll.return_value = None
        with patch("app.subprocess.Popen", return_value=child) as popen:
            log_path = runner.start([sys.executable, "synthetic.py"], "Synthetic test")
            self.assertIsNone(runner.poll())
            with self.assertRaises(ValueError):
                runner.start([sys.executable, "other.py"], "Other")
            self.assertEqual(popen.call_count, 1)
            options = popen.call_args.kwargs
            self.assertFalse(options["shell"])
            self.assertEqual(options["cwd"], self.root)
            self.assertEqual(options["stderr"], subprocess.STDOUT)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Camera unavailable. Try camera 1.\n")
        child.poll.return_value = 1
        result = runner.poll()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Camera unavailable", result.detail)
        self.assertEqual(result.log_path, log_path)
        self.assertIsNone(runner.process)
        self.assertIsNone(runner.poll())

    def test_spawn_failure_does_not_leave_runner_busy(self):
        runner = ChildRunner(self.root)
        with patch("app.subprocess.Popen", side_effect=OSError("cannot launch")):
            with self.assertRaisesRegex(OSError, "cannot launch"):
                runner.start([sys.executable, "synthetic.py"], "Synthetic test")
        self.assertIsNone(runner.process)
        log_path, = (self.root / "outputs" / "launcher").glob("*.log")
        self.assertIn("cannot launch", log_path.read_text(encoding="utf-8"))

    def test_error_popup_prefers_action_error_over_detector_warnings(self):
        detail = process_error_detail(
            "W0000 noisy detector warning\nVerification error: Could not open camera 0. Try --camera 1.\n"
        )
        self.assertEqual(detail, "Verification error: Could not open camera 0. Try --camera 1.")


if __name__ == "__main__":
    unittest.main()
