import json
from pathlib import Path
import tempfile
import unittest

from geometry import RATIO_COLUMNS
from multimodal import MODES, build_personal_model
from palm_reports import compile_reports, render_html, write_html_report
from palm_trials import PalmTrialLog
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE


def reference():
    return {"schema_version": 2, "reference_kind": "palm_geometry_calibration",
            "descriptor_version": DESCRIPTOR_VERSION, "geometry_feature_names": list(RATIO_COLUMNS),
            "palm_feature_size": PALM_FEATURE_SIZE, "quality": dict(DEFAULT_QUALITY),
            "thresholds": {"geometry": 0.5, "palm": 0.5, "fusion": 0.5},
            "fusion": {"geometry_scale": 1.0, "palm_scale": 1.0,
                       "weights": {"geometry": 0.5, "palm": 0.5}}}


def model(fusion_threshold=0.5):
    calibration = reference()
    calibration["thresholds"]["fusion"] = fusion_threshold
    return build_personal_model([{"geometry": [1] * 5,
                                  "palm": [1] + [0] * (PALM_FEATURE_SIZE - 1)}],
                                "alice", calibration)


def decisions(model_value, fusion_distance=0.2, *, acquisition_failure=False):
    result = {}
    for mode in MODES:
        threshold = model_value["thresholds"][mode]
        if acquisition_failure and mode in ("palm", "fusion"):
            status, distance, accepted = ("palm_too_small", None, None) if mode == "palm" else ("component_unavailable", None, None)
        else:
            distance = fusion_distance if mode == "fusion" else 0.2
            status, accepted = "ok", distance <= threshold
        result[mode] = {"status": status, "distance": distance, "accepted": accepted, "threshold": threshold}
    return result


def add_run(root, model_value=None, *, claim="alice", actual="alice", condition="indoor",
            distance=20, scale=1.0, attempts=()):
    model_value = model_value or model()
    log = PalmTrialLog(root, model_value, claim, actual, distance, condition, scale)
    for attempt in attempts:
        log.record(decisions(model_value, attempt[0], acquisition_failure=attempt[1]),
                   {"native_roi_pixels": 80, "sharpness": 10, "contrast": 5})
    return log


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class PalmReportsTests(unittest.TestCase):
    def test_compiles_fusion_counts_with_quality_failures_and_separate_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = model()
            add_run(root, first, claim="alice", actual="alice", condition="indoor", distance=20,
                    scale=1.0, attempts=((0.2, False), (0.8, False), (0.2, True)))
            add_run(root, first, claim="alice", actual="bob", condition="indoor", distance=20,
                    scale=1.0, attempts=((0.2, False), (0.8, False), (0.2, True)))
            add_run(root, model(0.4), claim="alice", actual="bob", condition="night", distance=35,
                    scale=0.5, attempts=((0.2, False),))

            report = compile_reports(root)
            self.assertEqual((report["runs_seen"], report["valid_runs"], report["skipped_runs"]), (3, 3, 0))
            self.assertEqual(len(report["groups"]), 2)
            group = next(row for row in report["groups"] if row["condition"] == "indoor")
            self.assertEqual(group["runs"], 2)
            self.assertEqual(group["attempts"], 6)
            self.assertEqual(group["genuine_attempts"], 3)
            self.assertEqual(group["impostor_attempts"], 3)
            self.assertEqual(group["valid_acquisitions"], 4)
            self.assertEqual(group["acquisition_failures"], 2)
            self.assertEqual(group["genuine_comparisons"], 2)
            self.assertEqual(group["impostor_comparisons"], 2)
            self.assertEqual(group["false_accepts"], 1)
            self.assertEqual(group["false_rejects"], 1)
            self.assertEqual(group["fmr"], 0.5)
            self.assertEqual(group["fnmr"], 0.5)
            self.assertEqual(group["acquisition_failure_rate"], 1 / 3)
            self.assertEqual(group["coverage"], 2 / 3)
            other = next(row for row in report["groups"] if row["condition"] == "night")
            self.assertEqual((other["distance_cm"], other["input_scale"], other["thresholds"]["fusion"]), (35, 0.5, 0.4))

    def test_same_model_hash_with_different_thresholds_warns_and_stays_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_model = model()
            first = add_run(root, frozen_model, attempts=((0.2, False),))
            second = add_run(root, frozen_model, condition="night", attempts=((0.2, False),))
            run_path = second.run_dir / "run.json"
            run_data = read_json(run_path)
            run_data["thresholds"]["fusion"] = 0.4
            write_json(run_path, run_data)
            trial_path = next((second.run_dir / "trials").glob("*.json"))
            trial = read_json(trial_path)
            trial["modes"]["fusion"]["threshold"] = 0.4
            trial["modes"]["fusion"]["accepted"] = trial["modes"]["fusion"]["distance"] <= 0.4
            write_json(trial_path, trial)
            report = compile_reports(root)
            self.assertEqual(len(report["groups"]), 2)
            self.assertTrue(any("different thresholds" in warning for warning in report["warnings"]))
            self.assertEqual({group["model_sha256"] for group in report["groups"]}, {first.model_sha256})

    def test_malformed_run_is_skipped_atomically_for_binding_types_and_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = add_run(root, attempts=((0.2, False), (0.2, False)))
            trial_paths = sorted((malformed.run_dir / "trials").glob("*.json"))
            second = read_json(trial_paths[1])
            second["actual_id"] = "elsewhere"
            write_json(trial_paths[1], second)
            extra = add_run(root, condition="bool", attempts=((0.2, False),))
            run_path = extra.run_dir / "run.json"
            run_data = read_json(run_path)
            run_data["input_scale"] = True
            write_json(run_path, run_data)
            mismatch = add_run(root, condition="index", attempts=((0.2, False),))
            path = next((mismatch.run_dir / "trials").glob("*.json"))
            trial = read_json(path)
            trial["trial_index"] = 7
            write_json(path, trial)

            report = compile_reports(root)
            self.assertEqual((report["runs_seen"], report["valid_runs"], report["skipped_runs"]), (3, 0, 3))
            self.assertEqual(report["groups"], [])
            self.assertEqual(len(report["warnings"]), 3)

    def test_duplicate_json_keys_and_duplicate_indices_reject_whole_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_key = add_run(root, condition="dupe-key", attempts=())
            path = duplicate_key.run_dir / "run.json"
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1), encoding="utf-8")
            duplicate_index = add_run(root, condition="dupe-index", attempts=((0.2, False),))
            original = next((duplicate_index.run_dir / "trials").glob("*.json"))
            (duplicate_index.run_dir / "trials" / "000001-copy.json").write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
            report = compile_reports(root)
            self.assertEqual((report["valid_runs"], report["skipped_runs"]), (0, 2))
            self.assertEqual(report["groups"], [])

    def test_empty_valid_run_has_unavailable_rates_and_html_shows_na(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_run(root)
            report = compile_reports(root)
            group = report["groups"][0]
            self.assertEqual(group["attempts"], 0)
            self.assertIsNone(group["fmr"])
            self.assertIsNone(group["fnmr"])
            self.assertIsNone(group["coverage"])
            page = render_html(report)
            self.assertIn("n/a", page)
            self.assertIn("not independent observations", page)
            output = write_html_report(report, root / "reports" / "report.html")
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_text(encoding="utf-8"), page)

    def test_html_escapes_labels_and_malformed_run_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_run(root, condition='<img src=x onerror="alert(1)">', attempts=((0.2, False),))
            unsafe = add_run(root, condition="bad-run")
            unsafe_run = root / "run&broken"
            unsafe.run_dir.rename(unsafe_run)
            run_data = read_json(unsafe_run / "run.json")
            run_data["schema_version"] = True
            write_json(unsafe_run / "run.json", run_data)
            report = compile_reports(root)
            page = render_html(report)
            self.assertIn("run&amp;broken", page)
            self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", page)
            self.assertNotIn("run&broken", page)
            self.assertNotIn('<img src=x onerror="alert(1)">', page)

    def test_missing_root_is_read_only_and_returns_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-created"
            report = compile_reports(missing)
            self.assertFalse(missing.exists())
            self.assertEqual((report["runs_seen"], report["valid_runs"], report["skipped_runs"]), (0, 0, 0))
            self.assertTrue(report["warnings"])
            self.assertEqual(report["groups"], [])


if __name__ == "__main__":
    unittest.main()
