"""Five finger measurements shared by dataset extraction and live verification."""

from itertools import pairwise
from math import dist, isfinite


FINGER_PATHS = (
    ("thumb", (2, 3, 4)),
    ("index", (5, 6, 7, 8)),
    ("middle", (9, 10, 11, 12)),
    ("ring", (13, 14, 15, 16)),
    ("pinky", (17, 18, 19, 20)),
)
FINGER_NAMES = tuple(name for name, _ in FINGER_PATHS)
PIXEL_COLUMNS = tuple(f"{name}_px" for name in FINGER_NAMES)
RATIO_COLUMNS = tuple(f"{name}_ratio" for name in FINGER_NAMES)
REQUIRED_LANDMARKS = frozenset(index for _, path in FINGER_PATHS for index in path)


def analyze_hand(hands, image_width, image_height):
    """Return quality status and measurements from normalized MediaPipe points.

    Palm width is the projected distance between landmarks 5 and 17. The wrist
    and thumb CMC are unused, so their being cropped does not invalidate fingers.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not hands:
        return {"status": "no_hand"}
    if len(hands) != 1:
        return {"status": "multiple_hands"}
    if len(hands[0]) != 21:
        return {"status": "invalid_landmarks"}

    points = [(point.x * image_width, point.y * image_height) for point in hands[0]]
    for index in REQUIRED_LANDMARKS:
        x, y = points[index]
        if not (isfinite(x) and isfinite(y)):
            return {"status": "invalid_landmarks"}
        if not (0 <= x < image_width and 0 <= y < image_height):
            return {"status": "finger_out_of_frame"}

    palm_width = dist(points[5], points[17])
    if palm_width < 1:
        return {"status": "degenerate_palm"}
    record = {"status": "ok", "palm_width_px": palm_width}
    for name, path in FINGER_PATHS:
        length = sum(dist(points[start], points[end]) for start, end in pairwise(path))
        if length <= 0:
            return {"status": "degenerate_finger"}
        record[f"{name}_px"] = length
        record[f"{name}_ratio"] = length / palm_width
    return record
