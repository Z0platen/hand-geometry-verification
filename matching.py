"""Median enrollment templates and distance-based hand verification."""

import json
from pathlib import Path

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FEATURE_SETS = {
    "pixels": [f"{finger}_px" for finger in FINGERS],
    "ratios": [f"{finger}_ratio" for finger in FINGERS],
}


def feature_vector(measurements, feature_names):
    try:
        vector = np.asarray([float(measurements[name]) for name in feature_names], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("All five finger measurements must be numeric and present.") from error
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
        raise ValueError("Finger measurements must be finite, positive values.")
    return vector


def build_templates(enrollment_rows, feature_names):
    grouped = {}
    for row in enrollment_rows:
        grouped.setdefault(str(row["subject_id"]), []).append(feature_vector(row, feature_names))
    if not grouped:
        raise ValueError("At least one enrollment sample is required.")
    return {subject: np.median(samples, axis=0).tolist() for subject, samples in sorted(grouped.items())}


def load_model(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise ValueError("Unsupported hand verification model schema or method.")
    if not isinstance(model.get("method"), str) or model["method"] not in FEATURE_SETS:
        raise ValueError("Unsupported hand verification model schema or method.")
    if model.get("feature_names") != FEATURE_SETS[model["method"]]:
        raise ValueError("Model feature names do not match the five-finger feature definition.")
    threshold = model.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("The model threshold must be a finite number.")
    templates = model.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("The model must contain enrollment templates.")
    for subject, values in templates.items():
        if not subject or not isinstance(values, list) or len(values) != len(model["feature_names"]):
            raise ValueError("Each template must identify a subject and contain five measurements.")
        feature_vector(dict(zip(model["feature_names"], values, strict=True)), model["feature_names"])
    return model


def verify_features(measurements, claimed_id, model):
    claimed_id = str(claimed_id)
    if claimed_id not in model["templates"]:
        raise ValueError(f"Unknown claimed user {claimed_id!r}.")
    vector = feature_vector(measurements, model["feature_names"])
    reference = np.asarray(model["templates"][claimed_id], dtype=float)
    distance = float(np.linalg.norm(vector - reference))
    return {
        "claimed_id": claimed_id,
        "distance": distance,
        "threshold": float(model["threshold"]),
        "accepted": bool(distance <= model["threshold"]),
    }
