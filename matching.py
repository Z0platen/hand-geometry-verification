"""Median enrollment templates and distance-based hand verification."""

import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FEATURE_SETS = {
    "pixels": [f"{finger}_px" for finger in FINGERS],
    "ratios": [f"{finger}_ratio" for finger in FINGERS],
}


def _finite_number(value, label, *, allow_string=False):
    if isinstance(value, (bool, np.bool_)) or not (
        isinstance(value, Real) or (allow_string and isinstance(value, str))
    ):
        raise ValueError(f"{label} must be numeric scalars.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite numbers.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numbers.")
    return number


def feature_vector(measurements, feature_names):
    if not isinstance(feature_names, (list, tuple)) or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("Feature names must describe the five fingers in their defined order.")
    if list(feature_names) not in FEATURE_SETS.values():
        raise ValueError("Feature names must describe the five fingers in their defined order.")
    try:
        # CSV measurements are strings; saved model templates have stricter validation.
        vector = np.asarray([
            _finite_number(measurements[name], "Finger measurements", allow_string=True)
            for name in feature_names
        ], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("All five finger measurements must be numeric and present.") from error
    if np.any(vector <= 0):
        raise ValueError("Finger measurements must be finite, positive values.")
    return vector


def build_templates(enrollment_rows, feature_names):
    grouped = {}
    for row in enrollment_rows:
        grouped.setdefault(str(row["subject_id"]), []).append(feature_vector(row, feature_names))
    if not grouped:
        raise ValueError("At least one enrollment sample is required.")
    templates = {}
    for subject, samples in sorted(grouped.items()):
        with np.errstate(over="ignore", invalid="ignore"):
            median = np.median(samples, axis=0)
        if not np.all(np.isfinite(median)):
            # NumPy averages the two middle values by adding first; avoid overflow
            # only in that exceptional case, retaining the original normal scores.
            ordered = np.sort(np.asarray(samples), axis=0)
            upper = ordered[len(ordered) // 2]
            lower = ordered[(len(ordered) - 1) // 2]
            median = lower + (upper - lower) / 2
        templates[subject] = median.tolist()
    return templates


def validate_model(model):
    """Validate a schema-1 model without changing its values or metadata."""
    if not isinstance(model, dict) or type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("Unsupported hand verification model schema or method.")
    if not isinstance(model.get("method"), str) or model["method"] not in FEATURE_SETS:
        raise ValueError("Unsupported hand verification model schema or method.")
    feature_names = model.get("feature_names")
    if (not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names)
            or feature_names != FEATURE_SETS[model["method"]]):
        raise ValueError("Model feature names do not match the five-finger feature definition.")
    _finite_number(model.get("threshold"), "The model threshold")
    templates = model.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("The model must contain enrollment templates.")
    for subject, values in templates.items():
        if not isinstance(subject, str) or not subject or not isinstance(values, list) or len(values) != len(feature_names):
            raise ValueError("Each template must identify a subject and contain five measurements.")
        if any(_finite_number(value, "Template measurements") <= 0 for value in values):
            raise ValueError("Template measurements must be finite, positive values.")
    return model


def load_model(path):
    return validate_model(json.loads(Path(path).read_text(encoding="utf-8")))


def verify_features(measurements, claimed_id, model):
    validate_model(model)
    claimed_id = str(claimed_id)
    if claimed_id not in model["templates"]:
        raise ValueError(f"Unknown claimed user {claimed_id!r}.")
    vector = feature_vector(measurements, model["feature_names"])
    reference = np.asarray(model["templates"][claimed_id], dtype=float)
    difference = vector - reference
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance = float(np.linalg.norm(difference))
    magnitude = np.abs(difference)
    squared_underflow = np.any((magnitude > 0) & (magnitude < math.sqrt(np.finfo(float).tiny)))
    if not math.isfinite(distance) or squared_underflow:
        distance = math.hypot(*difference)
    if not math.isfinite(distance):
        raise ValueError("The verification distance exceeds the finite numeric range.")
    return {
        "claimed_id": claimed_id,
        "distance": distance,
        "threshold": float(model["threshold"]),
        "accepted": bool(distance <= model["threshold"]),
    }
