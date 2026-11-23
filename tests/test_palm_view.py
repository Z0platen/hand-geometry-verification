import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from focus_view import FocusViewport
from palm_view import render_preview


class PalmPreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((10, 12, 3), (18, 74, 130), dtype=np.uint8)
        self.palm = {
            "status": "palm_too_small",
            "quality": {"native_roi_pixels": 7.0, "contrast": 0.0, "sharpness": 0.0},
            "corners": np.array([[2, 1], [9, 1], [9, 8], [2, 8]], dtype=float),
        }

    def render(self, scores=None, show_details=False, *, mirror=False, image_input=False, hands=None, geometry=None):
        return render_preview(
            self.frame, [] if hands is None else hands,
            {"status": "no_hand"} if geometry is None else geometry, self.palm, scores,
            "sample-user", "verifying", 0, 0, "Show your right palm.",
            show_details=show_details, mirror=mirror, image_input=image_input,
        )

    @staticmethod
    def focused_hand():
        return [SimpleNamespace(x=0.40 + (index % 5) * 0.05, y=0.30 + (index // 5) * 0.05)
                for index in range(21)]

    @staticmethod
    def focused_geometry():
        names = ("thumb", "index", "middle", "ring", "pinky")
        return {"status": "ok", **{f"{name}_px": 20.0 for name in names},
                **{f"{name}_ratio": 0.5 for name in names}}

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

    def test_mirror_flips_only_composed_camera_image_and_roi_guides(self):
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        frame[:, :4] = (15, 60, 210)
        frame[:, 8:] = (190, 30, 5)
        frame[2:4, 2:5] = (20, 240, 80)
        corners = np.array([[1, 1], [8, 2], [9, 7], [2, 8]], dtype=float)
        frame_before = frame.copy()
        corners_before = corners.copy()
        hand = object()
        hands = [hand]
        with patch("palm_view.make_preview", side_effect=lambda image, *_args, **_kwargs: image.copy()):
            ordinary = render_preview(
                frame, hands, {"status": "ok"}, {**self.palm, "corners": corners}, None,
                "sample-user", "verifying", 0, 0, "Ready", mirror=False,
            )
            mirrored = render_preview(
                frame, hands, {"status": "ok"}, {**self.palm, "corners": corners}, None,
                "sample-user", "verifying", 0, 0, "Ready", mirror=True,
            )
        factor = min(610 / frame.shape[1], 406 / frame.shape[0])
        width, height = round(frame.shape[1] * factor), round(frame.shape[0] * factor)
        x, y = 20 + (610 - width) // 2, 70 + (406 - height) // 2
        ordinary_view = ordinary[y:y + height, x:x + width]
        mirrored_view = mirrored[y:y + height, x:x + width]
        np.testing.assert_array_equal(mirrored_view, cv2.flip(ordinary_view, 1))
        np.testing.assert_array_equal(frame, frame_before)
        np.testing.assert_array_equal(corners, corners_before)
        self.assertIs(hands[0], hand)

    def test_photo_preview_stays_unmirrored_and_has_no_mirror_control(self):
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        frame[:, :4] = (15, 60, 210)
        frame[:, 8:] = (190, 30, 5)
        with patch("palm_view.make_preview", side_effect=lambda image, *_args, **_kwargs: image.copy()):
            normal = render_preview(
                frame, [], {"status": "no_hand"}, self.palm, None,
                "sample-user", "verifying", 0, 0, "Photo", image_input=True,
            )
            requested_mirror = render_preview(
                frame, [], {"status": "no_hand"}, self.palm, None,
                "sample-user", "verifying", 0, 0, "Photo", image_input=True, mirror=True,
            )
        np.testing.assert_array_equal(normal, requested_mirror)
        with patch("palm_view._text", wraps=__import__("palm_view")._text) as draw_text:
            self.render()
            camera_labels = [call.args[1] for call in draw_text.call_args_list]
        with patch("palm_view._text", wraps=__import__("palm_view")._text) as draw_text:
            self.render(image_input=True)
            photo_labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("M  Mirror" in label for label in camera_labels))
        self.assertFalse(any("M  Mirror" in label for label in photo_labels))

    def test_empty_current_hand_list_is_labeled_as_no_hand(self):
        with patch("palm_view._text", wraps=__import__("palm_view")._text) as draw_text:
            self.render(geometry={"status": "invalid_landmarks"}, hands=[])
        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("No hand detected", labels)
        self.assertNotIn("Hand detected", labels)

    def test_proportional_frames_keep_display_skeleton_readable_and_aligned(self):
        hand = [SimpleNamespace(x=0.25 + (index % 5) * 0.08, y=0.12 + (index // 5) * 0.16)
                for index in range(21)]
        hands = [hand]
        geometry = {"status": "ok", **{f"{name}_px": 20.0 for name in ("thumb", "index", "middle", "ring", "pinky")},
                    **{f"{name}_ratio": 0.5 for name in ("thumb", "index", "middle", "ring", "pinky")}}
        geometry_before = dict(geometry)
        normalized_corners = np.array([[0.3, 0.25], [0.7, 0.25], [0.7, 0.75], [0.3, 0.75]])
        frames = [np.full((640, 800, 3), (20, 30, 40), dtype=np.uint8),
                  np.full((1280, 1600, 3), (20, 30, 40), dtype=np.uint8)]
        corners = [normalized_corners * (frame.shape[1], frame.shape[0]) for frame in frames]
        frames_before = [frame.copy() for frame in frames]
        corners_before = [points.copy() for points in corners]
        previews = []
        for frame, roi_corners in zip(frames, corners):
            previews.append(render_preview(
                frame, hands, geometry,
                {**self.palm, "status": "ok", "corners": roi_corners}, None,
                "sample-user", "verifying", 0, 0, "Ready",
            ))
        np.testing.assert_array_equal(previews[0], previews[1])
        factor = min(610 / frames[0].shape[1], 406 / frames[0].shape[0])
        view_width, view_height = round(800 * factor), round(640 * factor)
        x, y = 20 + (610 - view_width) // 2, 70 + (406 - view_height) // 2
        view = previews[0][y:y + view_height, x:x + view_width]
        skeleton_pixels = np.all(view == (42, 200, 255), axis=2)
        self.assertGreater(int(skeleton_pixels.sum()), 100)
        for frame, original in zip(frames, frames_before):
            np.testing.assert_array_equal(frame, original)
        for points, original in zip(corners, corners_before):
            np.testing.assert_array_equal(points, original)
        self.assertEqual(geometry, geometry_before)
        self.assertIs(hands[0], hand)

    def test_full_and_focused_views_map_raw_pixel_roi_corners_correctly(self):
        frame = np.full((720, 1280, 3), (20, 30, 40), dtype=np.uint8)
        hand = self.focused_hand()
        hands = [hand]
        corners = np.array([[540, 260], [760, 260], [760, 460], [540, 460]], dtype=float)
        crop = FocusViewport().update(frame.shape, hands, corners)
        self.assertIsNotNone(crop)
        geometry = self.focused_geometry()
        before_frame, before_corners = frame.copy(), corners.copy()

        def first_corner(viewport):
            with patch("palm_view.make_preview", side_effect=lambda image, *_args, **_kwargs: image.copy()) as make, \
                    patch("palm_view.cv2.line", wraps=cv2.line) as draw_line:
                render_preview(
                    frame, hands, geometry, {**self.palm, "status": "ok", "corners": corners}, None,
                    "sample-user", "verifying", 0, 0, "Ready", viewport=viewport,
                )
            self.assertIs(make.call_args.kwargs["measurements"], geometry)
            left, top, right, bottom = viewport if viewport is not None else (0, 0, frame.shape[1], frame.shape[0])
            view_width, view_height = right - left, bottom - top
            factor = min(610 / view_width, 406 / view_height)
            size = (round(view_width * factor), round(view_height * factor))
            scaled = (corners[0] - (left, top)) * (size[0] / view_width, size[1] / view_height)
            expected = tuple(np.rint(scaled).astype(int))
            self.assertEqual(draw_line.call_args_list[0].args[1], expected)

        first_corner(None)
        first_corner(crop)
        np.testing.assert_array_equal(frame, before_frame)
        np.testing.assert_array_equal(corners, before_corners)
        self.assertIs(hands[0], hand)

    def test_unsafe_current_hand_or_roi_rejects_stale_crop(self):
        frame = np.full((720, 1280, 3), (20, 30, 40), dtype=np.uint8)
        hand = self.focused_hand()
        hands = [hand]
        geometry = self.focused_geometry()
        corners = np.array([[540, 260], [760, 260], [760, 460], [540, 460]], dtype=float)
        crop = FocusViewport().update(frame.shape, hands, corners)
        self.assertIsNotNone(crop)
        partial_hand = [hand[:-1]]
        second_hand = [hand, hand]
        changed_roi = np.array([[1000, 500], [1120, 500], [1120, 620], [1000, 620]], dtype=float)
        unsafe_cases = (([], corners), (partial_hand, corners), (second_hand, corners), (hands, changed_roi))
        with patch("palm_view.make_preview", side_effect=lambda image, *_args, **_kwargs: image.copy()):
            for current_hands, current_corners in unsafe_cases:
                with self.subTest(current_hands=len(current_hands)):
                    full = render_preview(
                        frame, current_hands, geometry,
                        {**self.palm, "corners": current_corners}, None,
                        "sample-user", "verifying", 0, 0, "Ready",
                    )
                    preview = render_preview(
                        frame, current_hands, geometry,
                        {**self.palm, "corners": current_corners}, None,
                        "sample-user", "verifying", 0, 0, "Ready", viewport=crop,
                    )
                    np.testing.assert_array_equal(preview[70:476, 20:630], full[70:476, 20:630])

    def test_photos_ignore_focus_viewport_and_have_no_focus_control(self):
        frame = self.frame.copy()
        viewport = (2, 1, 9, 8)
        with patch("palm_view.make_preview", side_effect=lambda image, *_args, **_kwargs: image.copy()):
            ordinary = render_preview(
                frame, [], {"status": "no_hand"}, self.palm, None,
                "sample-user", "verifying", 0, 0, "Photo", image_input=True,
            )
            requested_focus = render_preview(
                frame, [], {"status": "no_hand"}, self.palm, None,
                "sample-user", "verifying", 0, 0, "Photo", image_input=True,
                mirror=True, viewport=viewport,
            )
        np.testing.assert_array_equal(ordinary, requested_focus)
        with patch("palm_view._text", wraps=__import__("palm_view")._text) as draw_text:
            render_preview(frame, [], {"status": "no_hand"}, self.palm, None,
                           "sample-user", "verifying", 0, 0, "Photo", image_input=True)
        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertFalse(any("F  Focus" in label for label in labels))
