"""Draw the combined hand verification preview without changing its inputs."""

from types import SimpleNamespace

import cv2
import numpy as np

from focus_view import validate_viewport
from pilot import make_preview


SAVE_BUTTON_BOUNDS = (20, 540, 340, 600)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = (237, 240, 243)
MUTED = (167, 175, 185)
GREEN = (148, 227, 129)
AMBER = (110, 205, 250)
RED = (128, 132, 255)
PANEL = (44, 40, 37)
TRACKING_GUIDANCE = {
    "no_hand": "Show whole hand",
    "multiple_hands": "Show one hand",
    "invalid_landmarks": "Hold steady",
    "finger_out_of_frame": "Fingertips out of frame",
    "degenerate_palm": "Move closer",
    "degenerate_finger": "Open fingers",
}


def _text(canvas, value, x, y, size=0.52, color=TEXT, weight=1):
    cv2.putText(canvas, value, (x, y), FONT, size, color, weight, cv2.LINE_AA)


def _photo(canvas, image, x, y, size):
    if image is None:
        cv2.rectangle(canvas, (x, y), (x + size - 1, y + size - 1), (27, 25, 23), 1)
        return
    square = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    canvas[y:y + size, x:x + size] = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)


def _camera(canvas, frame, hands, geometry, corners, mirror=False, viewport=None):
    height, width = frame.shape[:2]
    crop = validate_viewport(viewport, frame.shape, hands, corners) if viewport is not None else None
    if crop is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        left, top, right, bottom = crop
    view_width, view_height = right - left, bottom - top
    factor = min(610 / view_width, 406 / view_height)
    size = (max(1, round(view_width * factor)), max(1, round(view_height * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    display_frame = cv2.resize(frame[top:bottom, left:right].copy(), size, interpolation=interpolation)
    display_hands = hands
    if crop is not None:
        display_hands = [
            [
                SimpleNamespace(x=(point.x * width - left) / view_width,
                                y=(point.y * height - top) / view_height)
                for point in hand
            ]
            for hand in hands
        ]
    annotated = make_preview(display_frame, display_hands, measurements=geometry)
    view = annotated[:size[1], :size[0]]
    if corners is not None:
        points = (np.asarray(corners) - (left, top)) * (size[0] / view_width, size[1] / view_height)
        for index, point in enumerate(points):
            for neighbor in (points[index - 1], points[(index + 1) % len(points)]):
                start = tuple(np.rint(point).astype(int))
                end = tuple(np.rint(point + (neighbor - point) * 0.22).astype(int))
                cv2.line(view, start, end, (38, 38, 38), 3, cv2.LINE_AA)
                cv2.line(view, start, end, (209, 216, 219), 1, cv2.LINE_AA)
    if mirror:
        view = cv2.flip(view, 1)
    x = 20 + (610 - size[0]) // 2
    y = 70 + (406 - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = view
    return crop


def _readiness(geometry, palm, phase, hands=None):
    if phase == "remove_hand":
        return "Remove your hand", AMBER
    if hands is not None and not hands:
        return "Show whole hand", AMBER
    if hands is not None and len(hands) > 1:
        return "Show one hand", AMBER
    if geometry["status"] != "ok":
        if hands:
            return {
                "finger_out_of_frame": "Fingertips out of frame",
                "invalid_landmarks": "Hold steady",
                "degenerate_palm": "Move closer",
                "degenerate_finger": "Open fingers",
            }.get(geometry["status"], "Reposition hand"), AMBER
        return TRACKING_GUIDANCE.get(geometry["status"], "Show whole hand"), AMBER
    status = palm["status"]
    if status == "ok":
        return "Ready", GREEN
    if status == "palm_too_small":
        minimum = palm.get("minimum_roi_pixels", 64)
        native = palm.get("quality", {}).get("native_roi_pixels", 0)
        return f"Palm {native:.0f}/{minimum:g}px: move closer", AMBER
    return {
        "palm_low_contrast": "Use even lighting",
        "palm_empty_texture": "Use even lighting",
        "palm_out_of_frame": "Palm out of frame",
        "palm_degenerate_geometry": "Face palm to camera",
        "palm_invalid_landmarks": "Hold steady",
    }.get(status, "Reposition palm"), AMBER


def _tracking_label(hands):
    count = len(hands) if hands is not None else 0
    return "No hand detected" if count == 0 else ("Hand detected" if count == 1 else f"{count} hands detected")


def _palm_panel(canvas, frame, hands, geometry, palm, phase, actual, distance_cm, scale,
                show_details):
    cv2.rectangle(canvas, (650, 70), (979, 514), PANEL, -1)
    roi = palm.get("roi")
    quality = palm["quality"]
    native_pixels = quality["native_roi_pixels"]
    minimum = palm.get("minimum_roi_pixels", 64)
    if show_details:
        _text(canvas, "Raw palm crop", 673, 109, 0.47)
        _text(canvas, "Gabor response", 827, 109, 0.47)
        _photo(canvas, roi, 673, 121, 128)
        _photo(canvas, palm.get("lines"), 827, 121, 128)
        _text(canvas, f"Frame {frame.shape[1]} x {frame.shape[0]} px", 673, 281)
        _text(canvas, f"Palm {native_pixels:.0f} px  /  minimum {minimum:g} px", 673, 306)
        _text(canvas, f"Sharpness {quality['sharpness']:.1f}", 673, 331)
        _text(canvas, f"Contrast {quality['contrast']:.1f}", 673, 356)
        _text(canvas, f"Input scale {scale:g}", 673, 381)
        status_y = 414
    else:
        _text(canvas, "Palm detail", 673, 109, 0.65, weight=2)
        enhanced = None
        if roi is not None:
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        _photo(canvas, enhanced, 705, 126, 220)
        if roi is None:
            _text(canvas, "Palm detail unavailable", 730, 240, 0.45, MUTED)
        else:
            _text(canvas, "Contrast enhanced view", 719, 371, 0.46, MUTED)
        status_y = 414
        _text(canvas, f"Palm {native_pixels:.0f} px  /  minimum {minimum:g} px",
              673, 447, 0.52, MUTED)
    status, color = _readiness(geometry, palm, phase, hands)
    _text(canvas, status, 673, status_y, 0.55, color, 1)
    if actual is not None:
        _text(canvas, f"Actual ID  {actual}", 673, 475, 0.47, MUTED)
    if distance_cm is not None:
        _text(canvas, f"Measured distance  {distance_cm:g} cm", 673, 499, 0.47, MUTED)


def _enrollment(canvas, geometry, palm, phase, count, target):
    if phase == "remove_hand":
        _text(canvas, "Remove your hand", 20, 581, 1.0, AMBER, 2)
    else:
        ready = geometry["status"] == palm["status"] == "ok"
        left, top, right, bottom = SAVE_BUTTON_BOUNDS
        cv2.rectangle(canvas, (left, top), (right, bottom),
                      (86, 157, 64) if ready else PANEL, -1)
        _text(canvas, "SAVE SAMPLE [SPACE]", 40, 578, 0.72, TEXT if ready else MUTED, 2)
    if target > 0:
        boundaries = np.linspace(20, 620, target + 1, dtype=int)
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            color = GREEN if index < count else (67, 65, 61)
            cv2.rectangle(canvas, (int(left), 615), (int(right) - 5, 625), color, -1)


def _verification(canvas, scores, phase):
    decision = scores["modes"]["fusion"] if scores is not None else None
    if phase == "remove_hand":
        _text(canvas, "Remove your hand", 20, 581, 1.0, AMBER, 2)
    elif decision is None or decision["accepted"] is None:
        _text(canvas, "Waiting for a usable hand image", 20, 581, 0.77, AMBER, 2)
    else:
        accepted = decision["accepted"]
        _text(canvas, "ACCEPT" if accepted else "REJECT", 20, 585, 1.45,
              GREEN if accepted else RED, 2)
        _text(canvas, f"Distance {decision['distance']:.5f}  /  Threshold {decision['threshold']:.5f}",
              257, 577, 0.55)


def _feedback(canvas, feedback):
    line = ""
    y = 655
    for word in feedback[:125].split():
        candidate = f"{line} {word}".strip()
        if cv2.getTextSize(candidate, FONT, 0.52, 1)[0][0] > 950:
            _text(canvas, line, 20, y, color=AMBER)
            y += 23
            line = word
        else:
            line = candidate
    _text(canvas, line, 20, y, color=AMBER)


def render_preview(frame, hands, geometry, palm, scores, user_id, phase, count, target, feedback,
                   actual=None, distance_cm=None, scale=1.0, image_input=False,
                   show_details=False, mirror=False, viewport=None):
    canvas = np.full((730, 1000, 3), (29, 26, 24), dtype=np.uint8)
    enrolling = phase == "enrolling" or (phase == "remove_hand" and target > 0)
    title = "Register your hand" if enrolling else "Combined hand verification"
    _text(canvas, title, 20, 38, 0.87, weight=2)
    _text(canvas, f"User  {user_id}", 650, 30, 0.49, MUTED)
    if enrolling:
        _text(canvas, f"{count} of {target} samples saved", 650, 54, 0.55)
    cv2.rectangle(canvas, (20, 70), (629, 475), (18, 16, 14), -1)
    display_mirror = bool(mirror and not image_input)
    display_viewport = _camera(
        canvas, frame, hands, geometry, palm.get("corners"), mirror=display_mirror,
        viewport=None if image_input else viewport,
    )
    _palm_panel(canvas, frame, hands, geometry, palm, phase, actual, distance_cm, scale, show_details)
    if not image_input:
        _text(canvas, _tracking_label(hands), 20, 505, 0.48, MUTED)
        _text(canvas, "Hand focus" if display_viewport is not None else "Full view", 270, 505, 0.48, MUTED)
        _text(canvas, "Mirrored view" if display_mirror else "Original view", 465, 505, 0.48, MUTED)
    controls = ["D  Details", "Q / Esc  Close"]
    if enrolling:
        _enrollment(canvas, geometry, palm, phase, count, target)
        controls.insert(0, "Space / Click  Save sample")
    else:
        _verification(canvas, scores, phase)
        controls.insert(0, "S  Screenshot")
        if actual is not None and not image_input:
            controls.insert(0, "T  Record trial")
    if not image_input:
        controls.insert(0, "M  Mirror")
        controls.insert(1, "F  Focus/full")
    _feedback(canvas, feedback)
    cv2.line(canvas, (20, 695), (979, 695), PANEL, 1)
    _text(canvas, "    |    ".join(controls), 20, 717, 0.48, MUTED)
    return canvas
