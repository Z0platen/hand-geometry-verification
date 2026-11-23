from math import nan
from types import SimpleNamespace
import unittest

import numpy as np

from geometry import FINGER_PATHS, RATIO_COLUMNS, analyze_hand
from pilot import make_preview


def sample_hand():
    points = [SimpleNamespace(x=0.4, y=0.6) for _ in range(21)]
    for finger, (_, path) in enumerate(FINGER_PATHS):
        for joint, index in enumerate(path):
            points[index] = SimpleNamespace(x=0.2 + finger * 0.1, y=0.5 - joint * 0.1)
    return points


class GeometryTests(unittest.TestCase):
    def test_pixel_aspect_ratio_and_scale(self):
        hand = sample_hand()
        normal = analyze_hand([hand], 1000, 500)
        larger = analyze_hand([hand], 2000, 1000)
        self.assertEqual(normal['status'], 'ok')
        self.assertAlmostEqual(normal['index_px'], 150)
        self.assertAlmostEqual(normal['palm_width_px'], 300)
        self.assertAlmostEqual(normal['index_ratio'], 0.5)
        for name in RATIO_COLUMNS:
            self.assertAlmostEqual(normal[name], larger[name])

    def test_unused_wrist_can_be_cropped_or_nonfinite(self):
        hand = sample_hand()
        hand[0].y = -0.2
        hand[1].x = nan
        self.assertEqual(analyze_hand([hand], 640, 480)['status'], 'ok')
        preview = make_preview(np.zeros((480, 640, 3), dtype=np.uint8), [hand])
        self.assertEqual(preview.shape, (480, 900, 3))

    def test_required_tip_must_be_visible(self):
        hand = sample_hand()
        hand[12].y = 1.01
        self.assertEqual(analyze_hand([hand], 640, 480)['status'], 'finger_out_of_frame')
        hand[12].y = nan
        self.assertEqual(analyze_hand([hand], 640, 480)['status'], 'invalid_landmarks')

    def test_cropped_tip_keeps_current_visible_skeleton_and_status(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        hand = sample_hand()
        hand[12].y = 1.01
        status = analyze_hand([hand], 640, 480)
        status_before = dict(status)
        preview = make_preview(frame, [hand], measurements=status)
        self.assertEqual(status['status'], 'finger_out_of_frame')
        self.assertTrue(np.array_equal(preview[144, 256], [70, 240, 125]))
        self.assertTrue(np.array_equal(frame, original))
        self.assertEqual(hand[12].y, 1.01)
        self.assertEqual(status, status_before)

    def test_nonfinite_current_point_is_skipped_without_hiding_other_landmarks(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = sample_hand()
        hand[10].x = nan
        status = analyze_hand([hand], 640, 480)
        preview = make_preview(frame, [hand], measurements=status)
        self.assertEqual(status['status'], 'invalid_landmarks')
        self.assertTrue(np.array_equal(preview[144, 256], [70, 240, 125]))

    def test_malformed_current_point_does_not_abort_preview(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = sample_hand()
        hand[10] = object()
        with self.assertRaises(AttributeError):
            analyze_hand([hand], 640, 480)
        preview = make_preview(frame, [hand])
        self.assertTrue(np.array_equal(preview[144, 256], [70, 240, 125]))

    def test_partial_hand_draws_safe_points_without_changing_invalid_status(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = sample_hand()
        partial = hand[:8]
        before = [(point.x, point.y) for point in partial]
        status = analyze_hand([partial], 640, 480)
        preview = make_preview(frame, [partial], measurements=status)
        self.assertEqual(status['status'], 'invalid_landmarks')
        self.assertTrue(np.array_equal(preview[144, 192], [70, 240, 125]))
        self.assertEqual([(point.x, point.y) for point in partial], before)

    def test_unknown_status_uses_fallback_and_still_draws_hands(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = sample_hand()
        status = {'status': 'new_status', 'payload': [1, 2, 3]}
        before = dict(status)
        preview = make_preview(frame, [hand], measurements=status)
        self.assertTrue(np.array_equal(preview[144, 256], [70, 240, 125]))
        self.assertEqual(status, before)

    def test_multiple_hands_status_still_draws_each_current_hand(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = sample_hand()
        second = sample_hand()
        for point in second:
            point.x += 0.3
        status = analyze_hand([first, second], 640, 480)
        preview = make_preview(frame, [first, second], measurements=status)
        self.assertEqual(status['status'], 'multiple_hands')
        self.assertTrue(np.array_equal(preview[144, 256], [70, 240, 125]))
        self.assertTrue(np.array_equal(preview[144, 448], [70, 240, 125]))

    def test_no_hand_frame_has_no_overlay_from_previous_detection(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = sample_hand()
        detected = make_preview(frame, [hand])
        empty = make_preview(frame, [])
        self.assertTrue(np.any(np.all(detected[:480, :640] == [70, 240, 125], axis=2)))
        self.assertFalse(np.any(np.all(empty[:480, :640] == [70, 240, 125], axis=2)))
        self.assertTrue(np.array_equal(empty[:480, :640], frame))

    def test_acquisition_failures(self):
        hand = sample_hand()
        self.assertEqual(analyze_hand([], 640, 480)['status'], 'no_hand')
        self.assertEqual(analyze_hand([hand, hand], 640, 480)['status'], 'multiple_hands')
        self.assertEqual(analyze_hand([[hand[0]]], 640, 480)['status'], 'invalid_landmarks')
        hand[17] = hand[5]
        self.assertEqual(analyze_hand([hand], 640, 480)['status'], 'degenerate_palm')


if __name__ == '__main__':
    unittest.main()
