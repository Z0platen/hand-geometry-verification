import unittest

import numpy as np

from palm_view import render_preview


class PalmPreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((10, 12, 3), (18, 74, 130), dtype=np.uint8)
        self.palm = {
            "status": "palm_too_small",
            "quality": {"native_roi_pixels": 7.0, "contrast": 0.0, "sharpness": 0.0},
            "corners": np.array([[2, 1], [9, 1], [9, 8], [2, 8]], dtype=float),
        }

    def render(self, scores=None, show_details=False):
        return render_preview(
            self.frame, [], {"status": "no_hand"}, self.palm, scores,
            "sample-user", "verifying", 0, 0, "Show your right palm.",
            show_details=show_details,
        )

    def test_small_frame_and_unavailable_palm(self):
        original = self.frame.copy()
        preview = self.render()
        self.assertEqual(preview.shape, (730, 1000, 3))
        np.testing.assert_array_equal(preview[275, 325], original[0, 0])
        np.testing.assert_array_equal(self.frame, original)

    def test_toggling_details_preserves_images_and_camera_preview(self):
        self.palm["roi"] = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.palm["lines"] = np.flipud(self.palm["roi"]).copy()
        before = {key: self.palm[key].copy() for key in ("roi", "lines", "corners")}
        simple = self.render()
        technical = self.render(show_details=True)
        np.testing.assert_array_equal(simple[70:476, 20:630], technical[70:476, 20:630])
        self.assertFalse(np.array_equal(simple[70:515, 650:980], technical[70:515, 650:980]))
        for key, value in before.items():
            np.testing.assert_array_equal(self.palm[key], value)

    def test_verdict_depends_on_fusion_only(self):
        accepted = {"accepted": True, "distance": 0.10, "threshold": 0.20}
        rejected = {"accepted": False, "distance": 0.30, "threshold": 0.20}
        scores = {"modes": {"geometry": accepted, "palm": rejected, "fusion": accepted}}
        first = self.render(scores)
        scores["modes"].update(geometry=rejected, palm=accepted)
        np.testing.assert_array_equal(first, self.render(scores))
        scores["modes"]["fusion"] = rejected
        self.assertFalse(np.array_equal(first[540:600], self.render(scores)[540:600]))
