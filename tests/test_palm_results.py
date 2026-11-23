import unittest

from palm_results import group_details, group_row_values, report_intro


def sample_group(**changes):
    group = {
        "model_sha256": "a" * 64,
        "condition": "low_light",
        "distance_cm": 40.0,
        "input_scale": 0.5,
        "thresholds": {"geometry": 0.2, "palm": 1.1, "fusion": 1.9},
        "run_ids": ["run-a", "run-b"],
        "claimed_ids": ["claimed1", "claimed2"],
        "actual_ids": ["claimed1", "other"],
        "runs": 2,
        "attempts": 20,
        "genuine_attempts": 10,
        "impostor_attempts": 10,
        "valid_acquisitions": 18,
        "acquisition_failures": 2,
        "genuine_comparisons": 9,
        "impostor_comparisons": 9,
        "false_accepts": 1,
        "false_rejects": 2,
        "fmr": 1 / 9,
        "fnmr": 2 / 9,
        "acquisition_failure_rate": 0.1,
        "coverage": 0.9,
    }
    group.update(changes)
    return group


class PalmResultsTests(unittest.TestCase):
    def test_group_row_displays_conditional_rates_and_attempt_denominators(self):
        values = group_row_values(sample_group())
        self.assertEqual(values[0], "claimed1, claimed2")
        self.assertEqual(values[4], "20")
        self.assertIn("1/9", values[5])
        self.assertIn("2/9", values[6])
        self.assertEqual(values[7], "10.0% (2/20)")
        self.assertIn("Coverage: 90.0% (18/20)", group_details(sample_group()))

    def test_missing_rate_uses_na_and_keeps_zero_denominator(self):
        values = group_row_values(sample_group(
            fmr=None, fnmr=None, coverage=None, acquisition_failure_rate=None,
            false_accepts=0, false_rejects=0, impostor_comparisons=0, genuine_comparisons=0,
        ))
        self.assertEqual(values[7], "n/a (2/20)")
        self.assertEqual(values[5], "n/a (0/0)")
        self.assertEqual(values[6], "n/a (0/0)")

    def test_group_details_keep_full_hash_and_profile_ids(self):
        details = group_details(sample_group())
        self.assertIn("Model SHA-256: " + "a" * 64, details)
        self.assertIn("Claimed profile IDs: claimed1, claimed2", details)
        self.assertIn("Actual person IDs: claimed1, other", details)
        self.assertIn("Thresholds:", details)
        self.assertIn("Genuine comparisons: 9; false rejects: 2", details)

    def test_empty_and_all_skipped_reports_give_actionable_guidance(self):
        empty = report_intro({"runs_seen": 0, "valid_runs": 0, "skipped_runs": 0, "groups": []})
        self.assertIn("select a profile and Actual person ID", empty)
        self.assertIn("press T to save attempts", empty)
        skipped = report_intro({"runs_seen": 3, "valid_runs": 0, "skipped_runs": 3, "groups": []})
        self.assertIn("No complete valid runs", skipped)
        self.assertIn("Skipped runs: 3", skipped)
        self.assertNotIn("accuracy", empty.casefold())


if __name__ == "__main__":
    unittest.main()
