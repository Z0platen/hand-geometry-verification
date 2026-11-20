from copy import deepcopy
from math import nan
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from palmprint import DEFAULT_QUALITY, PALM_FEATURE_SIZE, analyze_palm


def sample_hand(width=512, height=512):
    positions = {0: (256, 410), 5: (156, 210), 9: (220, 185),
                 13: (290, 188), 17: (356, 210)}
    return [SimpleNamespace(x=positions.get(index, (256, 256))[0] / width,
                            y=positions.get(index, (256, 256))[1] / height)
            for index in range(21)]


def textured_frame():
    frame = np.full((512, 512, 3), 175, dtype=np.uint8)
    for start, end in (((195, 230), (320, 310)), ((200, 290), (310, 250)),
                       ((232, 215), (252, 330))):
        cv2.line(frame, start, end, (65, 65, 65), 3, cv2.LINE_AA)
    cv2.ellipse(frame, (257, 262), (35, 46), 30, 20, 175, (110, 110, 110), 2, cv2.LINE_AA)
    return frame


class PalmprintTests(unittest.TestCase):
    def test_descriptor_has_fixed_finite_normalized_shape_and_stable_repeats(self):
        first = analyze_palm(textured_frame(), [sample_hand()])
        second = analyze_palm(textured_frame(), [sample_hand()])
        self.assertEqual(first["status"], "ok")
        features = np.asarray(first["features"])
        self.assertEqual(features.shape, (PALM_FEATURE_SIZE,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features >= 0))
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0)
        np.testing.assert_array_equal(features, second["features"])
        for key in ("roi", "lines"):
            self.assertEqual(first[key].shape, (128, 128))
            self.assertEqual(first[key].dtype, np.uint8)
        self.assertEqual(first["corners"].shape, (4, 2))
        self.assertGreater(first["quality"]["sharpness"], 0)

    def test_alignment_follows_ninety_degree_rotation(self):
        frame = textured_frame()
        hand = sample_hand()
        original = analyze_palm(frame, [hand])
        rotated_hand = [SimpleNamespace(x=(511 - point.y * 512) / 512, y=point.x) for point in hand]
        rotated = analyze_palm(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), [rotated_hand])
        self.assertEqual(rotated["status"], "ok")
        np.testing.assert_allclose(original["roi"], rotated["roi"], atol=1)
        self.assertLess(float(np.linalg.norm(np.asarray(original["features"]) - rotated["features"])), 0.01)

    def test_native_extent_is_measured_before_upsampling_and_scales_with_source(self):
        frame, hand = textured_frame(), sample_hand()
        normal = analyze_palm(frame, [hand])
        large = analyze_palm(cv2.resize(frame, (1024, 1024)), [hand])
        small = analyze_palm(cv2.resize(frame, (256, 256)), [hand])
        self.assertEqual(normal["quality"]["native_roi_pixels"], 100)
        self.assertEqual(large["quality"]["native_roi_pixels"], 200)
        self.assertEqual(small["quality"]["native_roi_pixels"], 50)
        self.assertEqual(small["status"], "palm_too_small")
        self.assertIsNone(small["features"])
        self.assertEqual(normal["roi"].shape, large["roi"].shape)
        self.assertLess(float(np.linalg.norm(np.asarray(normal["features"]) - large["features"])), 0.15)

    def test_uniform_palm_is_quality_failure_without_match_features(self):
        result = analyze_palm(np.full((512, 512, 3), 150, dtype=np.uint8), [sample_hand()])
        self.assertEqual(result["status"], "palm_low_contrast")
        self.assertEqual(result["quality"]["contrast"], 0)
        self.assertIsNone(result["features"])
        self.assertIsNone(result["lines"])

    def test_blur_metric_is_reported_without_an_unvalidated_blur_rejection(self):
        frame, hand = textured_frame(), sample_hand()
        original = analyze_palm(frame, [hand])
        blurred = analyze_palm(cv2.GaussianBlur(frame, (25, 25), 6), [hand])
        self.assertGreater(blurred["quality"]["contrast"], DEFAULT_QUALITY["min_contrast"])
        self.assertLess(blurred["quality"]["sharpness"], original["quality"]["sharpness"])
        self.assertEqual(blurred["status"], "ok")

    def test_cropped_or_missing_wrist_can_use_visible_central_palm(self):
        hand = sample_hand()
        hand[0].y = 1.4
        self.assertEqual(analyze_palm(textured_frame(), [hand])["status"], "ok")
        hand[0].x = nan
        fallback = analyze_palm(textured_frame(), [hand])
        self.assertEqual(fallback["status"], "ok")
        np.testing.assert_allclose(fallback["corners"], analyze_palm(textured_frame(), [sample_hand()])["corners"])

    def test_corner_bounds_and_degenerate_or_nonfinite_landmarks_fail_cleanly(self):
        frame, hand = textured_frame(), sample_hand()
        offscreen = deepcopy(hand)
        for point in offscreen:
            point.y -= 0.5
        result = analyze_palm(frame, [offscreen])
        self.assertEqual(result["status"], "palm_out_of_frame")
        self.assertIsNone(result["features"])
        hand[17] = hand[5]
        self.assertEqual(analyze_palm(frame, [hand])["status"], "palm_degenerate_geometry")
        hand = sample_hand()
        hand[9].x = nan
        self.assertEqual(analyze_palm(frame, [hand])["status"], "palm_invalid_landmarks")
        hand = sample_hand()
        hand[0].x = nan
        hand[9].y = hand[13].y = hand[5].y
        self.assertEqual(analyze_palm(frame, [hand])["status"], "palm_degenerate_geometry")

    def test_invalid_frame_and_hand_counts_and_quality_configuration(self):
        frame, hand = textured_frame(), sample_hand()
        self.assertEqual(analyze_palm(None, [hand])["status"], "palm_invalid_frame")
        self.assertEqual(analyze_palm(frame, [])["status"], "palm_no_hand")
        self.assertEqual(analyze_palm(frame, [hand, hand])["status"], "palm_multiple_hands")
        self.assertEqual(analyze_palm(frame, [[hand[0]]])["status"], "palm_invalid_landmarks")
        for quality in ({"min_roi_pixels": nan}, {"min_roi_pixels": True},
                        {"min_contrast": -1}, {"blur_threshold": 3}):
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                analyze_palm(frame, [hand], quality)
        before = DEFAULT_QUALITY.copy()
        self.assertEqual(analyze_palm(frame, [hand], {"min_roi_pixels": 101})["status"], "palm_too_small")
        self.assertEqual(DEFAULT_QUALITY, before)


if __name__ == "__main__":
    unittest.main()
