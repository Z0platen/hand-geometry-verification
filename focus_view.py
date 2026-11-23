"""Pure geometry and short-term state for a hand-focused camera viewport."""

from math import ceil, floor, isfinite
from numbers import Integral, Real


VIEW_WIDTH = 610
VIEW_HEIGHT = 406
VIEW_ASPECT = VIEW_WIDTH / VIEW_HEIGHT
PADDING_FRACTION = 0.20
MAX_RELATIVE_ZOOM = 2.0
HYSTERESIS_MARGIN = 0.05


def _frame_size(frame_shape):
    try:
        if len(frame_shape) < 2:
            return None
        height, width = frame_shape[0], frame_shape[1]
    except (IndexError, TypeError):
        return None
    if (isinstance(height, bool) or not isinstance(height, Integral)
            or isinstance(width, bool) or not isinstance(width, Integral)):
        return None
    height, width = int(height), int(width)
    return (height, width) if height > 0 and width > 0 else None


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _point_xy(point):
    try:
        x = _finite_number(point.x)
        y = _finite_number(point.y)
        try:
            z_value = point.z
        except AttributeError:
            z_value = None
            has_z = False
        else:
            has_z = True
        z = _finite_number(z_value) if has_z else None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if x is None or y is None or (has_z and z is None):
        return None
    return x, y


def _current_bounds(frame_size, hands, corners=None):
    """Return pixel bounds for a fully valid hand and optional four ROI corners."""
    if frame_size is None:
        return None
    height, width = frame_size
    try:
        if hands is None or len(hands) != 1:
            return None
        hand = hands[0]
        if len(hand) != 21:
            return None
    except (IndexError, KeyError, TypeError):
        return None

    points = []
    try:
        for point in hand:
            xy = _point_xy(point)
            if xy is None:
                return None
            x, y = xy
            if not (0 <= x < 1 and 0 <= y < 1):
                return None
            points.append((x * width, y * height))
    except (TypeError, ValueError):
        return None

    if corners is not None:
        try:
            if len(corners) != 4:
                return None
            for corner in corners:
                if len(corner) != 2:
                    return None
                x = _finite_number(corner[0])
                y = _finite_number(corner[1])
                if x is None or y is None or not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                    return None
                points.append((x, y))
        except (IndexError, KeyError, TypeError):
            return None

    return min(point[0] for point in points), min(point[1] for point in points), \
        max(point[0] for point in points), max(point[1] for point in points)


def _crop_for_bounds(bounds, frame_size):
    height, width = frame_size
    left_bound, top_bound, right_bound, bottom_bound = bounds
    box_width = max(1.0, right_bound - left_bound)
    box_height = max(1.0, bottom_bound - top_bound)
    padded_width = box_width * (1 + 2 * PADDING_FRACTION)
    padded_height = box_height * (1 + 2 * PADDING_FRACTION)

    minimum_width = max(width / MAX_RELATIVE_ZOOM, min(VIEW_WIDTH, width))
    minimum_height = max(height / MAX_RELATIVE_ZOOM, min(VIEW_HEIGHT, height))
    scale = max(
        padded_width / VIEW_WIDTH,
        padded_height / VIEW_HEIGHT,
        minimum_width / VIEW_WIDTH,
        minimum_height / VIEW_HEIGHT,
    )

    # Keep the preferred viewport aspect while it fits. At source boundaries,
    # clamp dimensions independently, allowing the closest bounded aspect.
    crop_width = min(width, max(1, int(ceil(VIEW_WIDTH * scale))))
    crop_height = min(height, max(1, int(ceil(VIEW_HEIGHT * scale))))
    if crop_width < right_bound - left_bound or crop_height < bottom_bound - top_bound:
        return None
    if crop_width == width and crop_height == height:
        return None

    centered_left = int(floor((left_bound + right_bound - crop_width) / 2))
    centered_top = int(floor((top_bound + bottom_bound - crop_height) / 2))
    min_left = max(0, int(floor(right_bound - crop_width)) + 1)
    max_left = min(width - crop_width, int(floor(left_bound)))
    min_top = max(0, int(floor(bottom_bound - crop_height)) + 1)
    max_top = min(height - crop_height, int(floor(top_bound)))
    if min_left > max_left or min_top > max_top:
        return None
    crop_left = min(max(centered_left, min_left), max_left)
    crop_top = min(max(centered_top, min_top), max_top)
    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    if not (crop_left <= left_bound and crop[2] > right_bound
            and crop_top <= top_bound and crop[3] > bottom_bound):
        return None
    if crop == (0, 0, width, height):
        return None
    return crop


def validate_viewport(crop, frame_shape, hands, corners=None):
    """Return a safe crop containing this frame's current hand and ROI, else None."""
    frame_size = _frame_size(frame_shape)
    bounds = _current_bounds(frame_size, hands, corners)
    if crop is None or frame_size is None or bounds is None:
        return None
    height, width = frame_size
    if not isinstance(crop, (tuple, list)) or len(crop) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in crop):
        return None
    left, top, right, bottom = (int(value) for value in crop)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    x0, y0, x1, y1 = bounds
    if left <= x0 and right > x1 and top <= y0 and bottom > y1:
        if (left, top, right, bottom) == (0, 0, width, height):
            return None
        return left, top, right, bottom
    return None


class FocusViewport:
    """Keep a stable viewport while validating current bounds on every frame."""

    def __init__(self):
        self._frame_size = None
        self._crop = None

    def reset(self):
        self._frame_size = None
        self._crop = None

    def update(self, frame_shape, hands, corners=None, enabled=True):
        """Return a half-open source crop, or None to display the full frame."""
        if not enabled:
            self.reset()
            return None
        frame_size = _frame_size(frame_shape)
        bounds = _current_bounds(frame_size, hands, corners)
        if frame_size is None or bounds is None:
            self.reset()
            return None
        if self._frame_size != frame_size:
            self.reset()
        if self._crop is not None:
            left, top, right, bottom = self._crop
            margin_x = (right - left) * HYSTERESIS_MARGIN
            margin_y = (bottom - top) * HYSTERESIS_MARGIN
            x0, y0, x1, y1 = bounds
            if (left + margin_x <= x0 and x1 <= right - margin_x
                    and top + margin_y <= y0 and y1 <= bottom - margin_y):
                return self._crop
        crop = _crop_for_bounds(bounds, frame_size)
        if crop is None:
            self.reset()
            return None
        self._frame_size = frame_size
        self._crop = crop
        return crop
