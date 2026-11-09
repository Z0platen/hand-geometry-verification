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

    def test_acquisition_failures(self):
        hand = sample_hand()
        self.assertEqual(analyze_hand([], 640, 480)['status'], 'no_hand')
        self.assertEqual(analyze_hand([hand, hand], 640, 480)['status'], 'multiple_hands')
        self.assertEqual(analyze_hand([[hand[0]]], 640, 480)['status'], 'invalid_landmarks')
        hand[17] = hand[5]
        self.assertEqual(analyze_hand([hand], 640, 480)['status'], 'degenerate_palm')


if __name__ == '__main__':
    unittest.main()
