"""Fixed-calibration geometry, palm texture, and equal-weight score fusion."""

from collections.abc import Mapping
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np

from geometry import RATIO_COLUMNS
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE
from palm_learning import project_features, validate_metric


MODES = ("geometry", "palm", "fusion")
def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric scalar.") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}.")
    return number


def _vector(values, size, name, *, unit=False):
    if isinstance(values, Mapping) and name == "geometry":
        try:
            values = [values[column] for column in RATIO_COLUMNS]
        except KeyError as error:
            raise ValueError("Geometry requires all five finger ratios.") from error
    if not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1 or len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} numeric measurements.")
    vector = np.asarray([_number(value, name) for value in values], dtype=float)
    if np.any(vector < 0) or (not unit and np.any(vector == 0)):
        raise ValueError(f"{name} measurements must be {'non-negative' if unit else 'positive'}.")
    if unit:
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("Palm features must be a non-negative unit-length descriptor.")
        vector = vector / norm
    return vector


def _validate_common(value):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 2:
        raise ValueError("Unsupported multimodal schema; version 2 is required.")
    if value.get("descriptor_version") != DESCRIPTOR_VERSION:
        raise ValueError("The palm descriptor version differs from this implementation.")
    if value.get("geometry_feature_names") != list(RATIO_COLUMNS):
        raise ValueError("The model requires the five ordered finger ratios.")
    if type(value.get("palm_feature_size")) is not int or value["palm_feature_size"] != PALM_FEATURE_SIZE:
        raise ValueError("The palm descriptor size differs from this implementation.")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(MODES):
        raise ValueError("Exactly three operating thresholds are required.")
    for mode in MODES:
        _number(thresholds[mode], f"{mode} threshold")
    fusion = value.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError("Fusion calibration is required.")
    for component in ("geometry", "palm"):
        _number(fusion.get(f"{component}_scale"), f"{component} normalization scale", positive=True)
    weights = fusion.get("weights")
    if not isinstance(weights, dict) or set(weights) != {"geometry", "palm"}:
        raise ValueError("Fusion requires fixed geometry and palm weights.")
    numeric_weights = [_number(weights[component], "Fusion weight", positive=True) for component in ("geometry", "palm")]
    if not math.isclose(sum(numeric_weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Fusion weights must be positive and sum to one.")
    if "palm_metric" in value:
        validate_metric(value["palm_metric"])
    elif numeric_weights != [0.5, 0.5]:
        raise ValueError("The handcrafted baseline uses equal fusion weights of 0.5.")
    quality = value.get("quality")
    if not isinstance(quality, dict) or set(quality) != set(DEFAULT_QUALITY):
        raise ValueError("The palm quality configuration is missing or incompatible.")
    for key, setting in quality.items():
        number = _number(setting, f"Quality setting {key}")
        if number < 0 or (key == "min_roi_pixels" and number < 2):
            raise ValueError("Palm quality limits must be non-negative, with an ROI side of at least two pixels.")
    return value


def validate_reference(reference):
    _validate_common(reference)
    if reference.get("reference_kind") != "palm_geometry_calibration" or "templates" in reference:
        raise ValueError("An aggregate palm/geometry calibration reference without templates is required.")
    return reference


def load_reference(path):
    return validate_reference(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_model(model):
    _validate_common(model)
    if model.get("model_kind") != "palm_geometry":
        raise ValueError("A palm_geometry model is required.")
    templates = model.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("The model must contain enrollment templates.")
    for subject, template in templates.items():
        if not isinstance(subject, str) or not subject or not isinstance(template, dict):
            raise ValueError("Each enrollment template must identify a user and two components.")
        _vector(template.get("geometry"), 5, "geometry")
        _vector(template.get("palm"), PALM_FEATURE_SIZE, "palm", unit=True)
    return model


def load_model(path):
    return validate_model(json.loads(Path(path).read_text(encoding="utf-8")))


def _median(values):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.median(values, axis=0)
    if not np.all(np.isfinite(result)):
        ordered = np.sort(values, axis=0)
        lower, upper = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
        result = lower + (upper - lower) / 2
    return result


def build_templates(samples):
    """Build component medians using enrollment rows only; normalize palm medians."""
    grouped = {}
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("subject_id"), str) or not sample["subject_id"]:
            raise ValueError("Enrollment rows require a nonempty subject_id.")
        grouped.setdefault(sample["subject_id"], []).append((
            _vector(sample.get("geometry"), 5, "geometry"),
            _vector(sample.get("palm"), PALM_FEATURE_SIZE, "palm", unit=True),
        ))
    if not grouped:
        raise ValueError("At least one common-valid enrollment sample is required.")
    templates = {}
    for subject, vectors in sorted(grouped.items()):
        geometry = _median(np.asarray([pair[0] for pair in vectors]))
        palm = _median(np.asarray([pair[1] for pair in vectors]))
        norm = math.hypot(*palm)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("The palm enrollment median is degenerate; collect more consistent samples.")
        templates[subject] = {"geometry": geometry.tolist(), "palm": (palm / norm).tolist()}
    return templates


def build_personal_model(samples, user_id, reference):
    validate_reference(reference)
    if not isinstance(user_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", user_id) is None:
        raise ValueError("User ID requires 1–32 ASCII letters, digits, underscores, or hyphens.")
    try:
        samples = list(samples)
    except TypeError as error:
        raise ValueError("Personal enrollment requires common-valid sample dictionaries.") from error
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("Personal enrollment requires common-valid sample dictionaries.")
    template = build_templates([{**sample, "subject_id": user_id} for sample in samples])
    # Copy the aggregate calibration unchanged; no personal capture changes its threshold.
    model = json.loads(json.dumps(reference, allow_nan=False))
    model.pop("reference_kind")
    model.update({
        "model_kind": "palm_geometry", "personal_demo": True,
        "templates": template, "enrollment_counts": {user_id: len(samples)},
        "threshold_source": "transferred_from_dataset", "personal_calibration": False,
        "reference_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    })
    return validate_model(model)


def score_probe(geometry, palm, claim, model, geometry_status="ok", palm_status="ok"):
    validate_model(model)
    if not isinstance(claim, str) or claim not in model["templates"]:
        raise ValueError(f"Unknown claimed identity {claim!r}.")
    template = model["templates"][claim]
    modes = {}
    for name, values, status, size in (
        ("geometry", geometry, geometry_status, 5), ("palm", palm, palm_status, PALM_FEATURE_SIZE),
    ):
        if not isinstance(status, str) or not status:
            raise ValueError("Acquisition statuses must be nonempty strings.")
        threshold = float(model["thresholds"][name])
        if status != "ok" or values is None:
            modes[name] = {"status": status if status != "ok" else f"missing_{name}",
                           "accepted": None, "distance": None, "threshold": threshold}
            continue
        probe = _vector(values, size, name, unit=name == "palm")
        reference = _vector(template[name], size, name, unit=name == "palm")
        if name == "palm" and "palm_metric" in model:
            probe, reference = project_features(np.asarray([values, template[name]], dtype=float), model["palm_metric"])
        distance = math.dist(probe, reference)
        if not math.isfinite(distance):
            raise ValueError("The component distance exceeds the finite numeric range.")
        modes[name] = {"status": "ok", "distance": distance, "threshold": threshold,
                       "accepted": distance <= threshold}
    threshold = float(model["thresholds"]["fusion"])
    if any(modes[name]["status"] != "ok" for name in ("geometry", "palm")):
        modes["fusion"] = {"status": "component_unavailable", "accepted": None,
                           "distance": None, "threshold": threshold}
    else:
        calibration = model["fusion"]
        distance = sum(calibration["weights"][name] * modes[name]["distance"] / calibration[f"{name}_scale"] for name in ("geometry", "palm"))
        if not math.isfinite(distance):
            raise ValueError("The fusion score exceeds the finite numeric range.")
        modes["fusion"] = {"status": "ok", "distance": distance, "threshold": threshold,
                           "accepted": distance <= threshold}
    return {"claimed_id": claim, "modes": modes}


class PreparedMatcher:
    """Immutable scoring snapshot for repeated probes or all enrolled claims.

    Validation and template normalization happen once. score_all validates each
    probe once and retains exactly the same math.dist calculations as score_probe.
    Subsequent mutations of the supplied model cannot change this snapshot.
    """

    def __init__(self, model):
        model = copy.deepcopy(model)
        validate_model(model)
        self._metric = model.get("palm_metric")
        if self._metric is not None:
            self._metric = {**self._metric, "mean": np.asarray(self._metric["mean"], dtype=float),
                            "projection": np.asarray(self._metric["projection"], dtype=float)}
        self.subjects = tuple(model["templates"])
        self._templates = {
            subject: {
                "geometry": _vector(template["geometry"], 5, "geometry"),
                "palm": _vector(template["palm"], PALM_FEATURE_SIZE, "palm", unit=True),
            }
            for subject, template in model["templates"].items()
        }
        self._thresholds = {mode: float(model["thresholds"][mode]) for mode in MODES}
        self._scales = {name: model["fusion"][f"{name}_scale"] for name in ("geometry", "palm")}
        self._weights = dict(model["fusion"]["weights"])
        if self._metric is not None:
            projected = project_features(np.asarray([model["templates"][subject]["palm"] for subject in self.subjects]), self._metric)
            for subject, vector in zip(self.subjects, projected, strict=True):
                self._templates[subject]["palm"] = vector

    def _prepare_probe(self, geometry, palm, geometry_status, palm_status):
        vectors, statuses = {}, {}
        for name, values, status, size in (
            ("geometry", geometry, geometry_status, 5), ("palm", palm, palm_status, PALM_FEATURE_SIZE),
        ):
            if not isinstance(status, str) or not status:
                raise ValueError("Acquisition statuses must be nonempty strings.")
            if status != "ok" or values is None:
                vectors[name] = None
                statuses[name] = status if status != "ok" else f"missing_{name}"
            else:
                vectors[name] = _vector(values, size, name, unit=name == "palm")
                statuses[name] = "ok"
        if self._metric is not None and vectors["palm"] is not None:
            vectors["palm"] = project_features(palm, self._metric)
        return vectors, statuses

    def _score(self, vectors, statuses, claim):
        modes = {}
        for name in ("geometry", "palm"):
            threshold = self._thresholds[name]
            if statuses[name] != "ok":
                modes[name] = {"status": statuses[name], "accepted": None,
                               "distance": None, "threshold": threshold}
                continue
            distance = math.dist(vectors[name], self._templates[claim][name])
            if not math.isfinite(distance):
                raise ValueError("The component distance exceeds the finite numeric range.")
            modes[name] = {"status": "ok", "distance": distance, "threshold": threshold,
                           "accepted": distance <= threshold}
        threshold = self._thresholds["fusion"]
        if any(statuses[name] != "ok" for name in ("geometry", "palm")):
            modes["fusion"] = {"status": "component_unavailable", "accepted": None,
                               "distance": None, "threshold": threshold}
        else:
            distance = sum(self._weights[name] * modes[name]["distance"] / self._scales[name] for name in ("geometry", "palm"))
            if not math.isfinite(distance):
                raise ValueError("The fusion score exceeds the finite numeric range.")
            modes["fusion"] = {"status": "ok", "distance": distance, "threshold": threshold,
                               "accepted": distance <= threshold}
        return {"claimed_id": claim, "modes": modes}

    def score_probe(self, geometry, palm, claim, geometry_status="ok", palm_status="ok"):
        if not isinstance(claim, str) or claim not in self._templates:
            raise ValueError(f"Unknown claimed identity {claim!r}.")
        vectors, statuses = self._prepare_probe(geometry, palm, geometry_status, palm_status)
        return self._score(vectors, statuses, claim)

    def score_all(self, geometry, palm, geometry_status="ok", palm_status="ok"):
        vectors, statuses = self._prepare_probe(geometry, palm, geometry_status, palm_status)
        return {claim: self._score(vectors, statuses, claim) for claim in self.subjects}
