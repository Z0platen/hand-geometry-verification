from math import inf, nan
from types import SimpleNamespace
import unittest

from focus_view import FocusViewport, validate_viewport


def hand(cx=0.5, cy=0.5, sx=0.12, sy=0.24, *, z=None):
    points = []
    for index in range(21):
        column = index % 5
        row = index // 5
        x = cx + (column / 4 - 0.5) * sx
        y = cy + (row / 4 - 0.5) * sy
        point = SimpleNamespace(x=x, y=y)
        if z is not None:
            point.z = z
        points.append(point)
    return points


def inside(crop, width, height, points):
    left, top, right, bottom = crop
    return all(left <= point.x * width < right and top <= point.y * height < bottom for point in points)


class FocusViewTests(unittest.TestCase):
    def test_crop_includes_current_hand_and_roi_and_clamps_to_source(self):
        frame_shape = (720, 1280, 3)
        points = hand(0.06, 0.92, 0.12, 0.12)
        corners = [(5, 620), (95, 620), (95, 690), (5, 690)]
        original_points = [(point.x, point.y) for point in points]
        original_corners = list(corners)
        view = FocusViewport()
        crop = view.update(frame_shape, [points], corners)
        self.assertIsNotNone(crop)
        left, top, right, bottom = crop
        self.assertEqual(left, 0)
        self.assertTrue(inside(crop, 1280, 720, points))
        self.assertTrue(all(left <= x < right and top <= y < bottom for x, y in corners))
        self.assertEqual(validate_viewport(crop, frame_shape, [points], corners), crop)
        self.assertEqual([(point.x, point.y) for point in points], original_points)
        self.assertEqual(corners, original_corners)

    def test_crop_has_padding_when_not_limited_by_frame_edges(self):
        frame_shape = (1080, 1920, 3)
        points = hand(0.5, 0.5, 0.10, 0.12)
        crop = FocusViewport().update(frame_shape, [points])
        self.assertIsNotNone(crop)
        left, top, right, bottom = crop
        x_values = [point.x * 1920 for point in points]
        y_values = [point.y * 1080 for point in points]
        self.assertLess(left, min(x_values))
        self.assertGreater(right, max(x_values))
        self.assertLess(top, min(y_values))
        self.assertGreater(bottom, max(y_values))

    def test_only_one_complete_finite_inframe_hand_is_eligible_and_invalid_resets(self):
        shape = (720, 1280, 3)
        view = FocusViewport()
        valid = hand()
        self.assertIsNotNone(view.update(shape, [valid]))
        for invalid_hands in ([], [valid, hand(0.7)], [valid[:20]]):
            with self.subTest(count=len(invalid_hands)):
                self.assertIsNone(view.update(shape, invalid_hands))
                self.assertIsNone(view._crop)
                self.assertIsNotNone(view.update(shape, [valid]))

        invalid = hand()
        invalid[7].x = nan
        self.assertIsNone(view.update(shape, [invalid]))
        invalid = hand()
        invalid[0].y = -0.01  # The geometry measurement can ignore a cropped wrist; focus cannot.
        self.assertIsNone(view.update(shape, [invalid]))

    def test_optional_depth_must_be_finite_when_present(self):
        shape = (720, 1280, 3)
        view = FocusViewport()
        self.assertIsNotNone(view.update(shape, [hand(z=-0.2)]))
        self.assertIsNone(view.update(shape, [hand(z=inf)]))

    def test_roi_requires_four_finite_inframe_pixel_corners(self):
        shape = (720, 1280, 3)
        view = FocusViewport()
        points = hand()
        for corners in (
            [(10, 10), (20, 10), (20, 20)],
            [(10, 10), (20, 10), (20, 20), (10, nan)],
            [(10, 10), (20, 10), (1300, 20), (10, 20)],
        ):
            with self.subTest(corners=corners):
                self.assertIsNone(view.update(shape, [points], corners))
                self.assertIsNone(view._crop)

    def test_full_small_frame_uses_none_and_disabled_update_resets(self):
        view = FocusViewport()
        points = hand(0.5, 0.5, 0.20, 0.35)
        self.assertIsNotNone(view.update((720, 1280), [points]))
        self.assertIsNone(view.update((240, 320), [points]))
        self.assertIsNone(validate_viewport((0, 0, 320, 240), (240, 320), [points]))
        self.assertIsNone(view._crop)
        self.assertIsNotNone(view.update((720, 1280), [points]))
        self.assertIsNone(view.update((720, 1280), [points], enabled=False))
        self.assertIsNone(view._crop)

    def test_crop_respects_native_pixel_minimum_and_two_times_zoom_limit(self):
        for shape, min_width, min_height in (((1080, 1920), 960, 540), ((480, 640), 610, 406)):
            points = hand(0.5, 0.5, 0.01, 0.01)
            crop = FocusViewport().update(shape, [points])
            self.assertIsNotNone(crop)
            self.assertGreaterEqual(crop[2] - crop[0], min_width)
            self.assertGreaterEqual(crop[3] - crop[1], min_height)
            self.assertTrue(inside(crop, shape[1], shape[0], points))

    def test_crop_scales_with_resolution_when_limits_do_not_bind(self):
        points_small = hand(0.5, 0.5, 0.28, 0.55)
        points_large = hand(0.5, 0.5, 0.28, 0.55)
        small = FocusViewport().update((1200, 2000), [points_small])
        large = FocusViewport().update((2400, 4000), [points_large])
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        self.assertLessEqual(abs((large[2] - large[0]) - 2 * (small[2] - small[0])), 2)
        self.assertLessEqual(abs((large[3] - large[1]) - 2 * (small[3] - small[1])), 2)

    def test_hysteresis_holds_inside_margin_then_reframes_and_resets_on_resolution(self):
        view = FocusViewport()
        shape = (720, 1280, 3)
        initial = view.update(shape, [hand(0.35, 0.5)])
        held = view.update(shape, [hand(0.36, 0.5)])
        self.assertEqual(held, initial)
        moved = hand(0.63, 0.5)
        reframed = view.update(shape, [moved])
        self.assertNotEqual(reframed, initial)
        self.assertTrue(inside(reframed, 1280, 720, moved))
        changed_resolution = view.update((1440, 2560, 3), [hand(0.63, 0.5)])
        self.assertIsNotNone(changed_resolution)
        self.assertNotEqual(changed_resolution, reframed)
        view.reset()
        self.assertIsNone(view._crop)

    def test_validate_viewport_rejects_stale_outside_or_malformed_crop(self):
        shape = (720, 1280)
        points = hand()
        crop = FocusViewport().update(shape, [points])
        self.assertEqual(validate_viewport(crop, shape, [points]), crop)
        self.assertIsNone(validate_viewport((0, 0, 10, 10), shape, [points]))
        self.assertIsNone(validate_viewport((-1, 0, 20, 20), shape, [points]))
        self.assertIsNone(validate_viewport((False, 0, 20, 20), shape, [points]))
        self.assertIsNone(validate_viewport(crop, shape, [points[:19]]))
        self.assertIsNone(validate_viewport(crop, shape, [points, hand(0.8)]))
        self.assertIsNone(validate_viewport(crop, shape, [hand(0.5, 0.5, 0.12, 0.24)],
                                            [(10, 10), (20, 10), (20, 20)]))


if __name__ == "__main__":
    unittest.main()
