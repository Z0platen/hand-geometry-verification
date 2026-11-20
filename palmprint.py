"""Aligned central-palm texture with a handcrafted Gabor baseline.

This descriptor is not an implementation of a specific published palmprint
recognizer. Native resolution and contrast gates describe acquisition quality;
sharpness is recorded without imposing an unvalidated blur threshold.
"""

from functools import lru_cache
from math import isfinite, pi

import cv2
import numpy as np


DESCRIPTOR_VERSION = "gabor-palm-v1"
PALM_FEATURE_SIZE = 384
DEFAULT_QUALITY = {"min_roi_pixels": 64, "min_contrast": 3}
ROI_SIZE = 128
ROI_SIDE_FRACTION = 0.50
ROI_CENTER_OFFSET = 0.30


def _result(status, *, features=None, roi=None, lines=None, quality=None, corners=None):
    return {
        "status": status,
        "features": features,
        "roi": roi,
        "lines": lines,
        "quality": quality if quality is not None else {
            "native_roi_pixels": 0.0, "sharpness": 0.0, "contrast": 0.0,
        },
        "corners": corners,
    }


def _quality_limits(quality):
    if quality is not None and (not isinstance(quality, dict) or set(quality) - set(DEFAULT_QUALITY)):
        raise ValueError("Palm quality settings must contain min_roi_pixels and/or min_contrast.")
    limits = {**DEFAULT_QUALITY, **(quality or {})}
    for name, minimum in (("min_roi_pixels", 2), ("min_contrast", 0)):
        value = limits[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number at least {minimum}.")
        try:
            valid = isfinite(value) and value >= minimum
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite number at least {minimum}.")
    return limits


def _palm_corners(hand, width, height):
    """Orient the ROI anatomically, so rotating the image rotates the ROI too.

    The first axis runs from index MCP (5) to pinky MCP (17). The second points
    toward the wrist, using only the sign of its perpendicular projection.
    A cropped wrist is allowed; a missing/nonfinite wrist falls back to the
    opposite side of the middle/ring MCP arc (9,13).
    """
    try:
        anchors = np.asarray([
            (float(hand[index].x) * width, float(hand[index].y) * height)
            for index in (5, 9, 13, 17)
        ], dtype=np.float64)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None, 0.0, "palm_invalid_landmarks"
    if not np.all(np.isfinite(anchors)):
        return None, 0.0, "palm_invalid_landmarks"
    index_mcp, middle_mcp, ring_mcp, pinky_mcp = anchors
    lateral = pinky_mcp - index_mcp
    palm_width = float(np.linalg.norm(lateral))
    if not isfinite(palm_width) or palm_width < 1:
        return None, 0.0, "palm_degenerate_geometry"
    lateral /= palm_width
    perpendicular = np.asarray([-lateral[1], lateral[0]])
    midpoint = (index_mcp + pinky_mcp) / 2
    wrist_projection = 0.0
    try:
        wrist = np.asarray((float(hand[0].x) * width, float(hand[0].y) * height))
        if np.all(np.isfinite(wrist)):
            wrist_projection = float(np.dot(wrist - midpoint, perpendicular))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        pass
    if abs(wrist_projection) >= 0.15 * palm_width:
        direction = wrist_projection
    else:
        direction = float(np.dot(midpoint - (middle_mcp + ring_mcp) / 2, perpendicular))
        if abs(direction) < 0.015 * palm_width:
            return None, 0.0, "palm_degenerate_geometry"
    toward_wrist = perpendicular if direction > 0 else -perpendicular
    side = ROI_SIDE_FRACTION * palm_width
    center = midpoint + ROI_CENTER_OFFSET * palm_width * toward_wrist
    half_x = lateral * side / 2
    half_y = toward_wrist * side / 2
    corners = np.asarray([
        center - half_x - half_y, center + half_x - half_y,
        center + half_x + half_y, center - half_x + half_y,
    ], dtype=np.float64)
    if not np.all(np.isfinite(corners)):
        return None, side, "palm_invalid_landmarks"
    if (np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > width - 1)
            or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > height - 1)):
        return corners, side, "palm_out_of_frame"
    return corners, side, "ok"


@lru_cache(maxsize=1)
def _gabor_kernels():
    kernels = []
    for orientation in range(6):
        kernel = cv2.getGaborKernel(
            (15, 15), 3.0, orientation * pi / 6, 7.0, 0.5, 0,
            ktype=cv2.CV_32F,
        )
        kernel -= kernel.mean()
        kernel /= np.abs(kernel).sum()
        kernels.append(kernel)
    return tuple(kernels)


def _texture_features(roi):
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    enhanced = enhanced.astype(np.float32) / 255.0
    responses = [
        np.abs(cv2.filter2D(enhanced, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101))
        for kernel in _gabor_kernels()
    ]
    # Six oriented response maps, each pooled into 8x8 spatial cells.
    vector = np.concatenate([
        response.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel()
        for response in responses
    ]).astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not isfinite(norm) or norm <= 1e-10:
        return None, None
    vector /= norm
    line_response = np.maximum.reduce(responses)
    upper = float(np.percentile(line_response, 99))
    lines = np.clip(line_response * (255.0 / max(upper, 1e-10)), 0, 255).astype(np.uint8)
    return vector.tolist(), lines


def analyze_palm(frame, hands, quality=None):
    """Return aligned texture, acquisition metrics, and source-image ROI corners.

    ``frame`` is the original uint8 BGR image and ``hands`` is MediaPipe's list
    of normalized 21-landmark hands. No camera, files, or model downloads are
    opened here. All failures return ``features=None``; they are not non-matches.
    ``native_roi_pixels`` is the square's source-image side before resizing.
    """
    limits = _quality_limits(quality)
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2):
        return _result("palm_invalid_frame")
    try:
        count = len(hands) if hands is not None else 0
    except TypeError:
        return _result("palm_invalid_landmarks")
    if count == 0:
        return _result("palm_no_hand")
    if count != 1:
        return _result("palm_multiple_hands")
    try:
        if len(hands[0]) != 21:
            return _result("palm_invalid_landmarks")
    except (TypeError, IndexError, KeyError):
        return _result("palm_invalid_landmarks")
    height, width = frame.shape[:2]
    corners, native_side, status = _palm_corners(hands[0], width, height)
    metrics = {"native_roi_pixels": native_side, "sharpness": 0.0, "contrast": 0.0}
    if status != "ok":
        return _result(status, quality=metrics, corners=corners)
    if native_side < limits["min_roi_pixels"]:
        return _result("palm_too_small", quality=metrics, corners=corners)
    native_size = max(2, int(np.floor(native_side)))
    destination = np.asarray([
        (0, 0), (native_size - 1, 0),
        (native_size - 1, native_size - 1), (0, native_size - 1),
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    native_roi = cv2.warpPerspective(
        gray, transform, (native_size, native_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    metrics["contrast"] = float(native_roi.std())
    metrics["sharpness"] = float(cv2.Laplacian(native_roi, cv2.CV_64F).var())
    interpolation = cv2.INTER_AREA if native_size >= ROI_SIZE else cv2.INTER_LINEAR
    roi = cv2.resize(native_roi, (ROI_SIZE, ROI_SIZE), interpolation=interpolation)
    if metrics["contrast"] < limits["min_contrast"]:
        return _result("palm_low_contrast", roi=roi, quality=metrics, corners=corners)
    features, lines = _texture_features(roi)
    if features is None:
        return _result("palm_empty_texture", roi=roi, quality=metrics, corners=corners)
    return _result("ok", features=features, roi=roi, lines=lines, quality=metrics, corners=corners)
