"""Durable manual comparisons for geometry, palm texture, and their fusion."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
from time import monotonic

from multimodal import MODES, validate_model
from trial_log import _identity, _json_bytes, _number, _replace_bytes, _utc_now, _write_bytes


QUALITY_FIELDS = ("native_roi_pixels", "sharpness", "contrast")
TRIAL_COLUMNS = (
    "trial_index", "timestamp_utc", "actual_id", "claimed_id", "genuine", "distance_cm", "input_scale",
    "condition", "method", "quality_status", "accepted", "distance", "threshold",
    "model_sha256", *QUALITY_FIELDS,
)
PROTOCOL = (
    "Manually triggered comparisons with operator-declared actual identity and optional "
    "measured camera distance in centimetres. The model and thresholds stay fixed. "
    "Acquisition failures are counted separately from conditional FMR and FNMR. "
    "Repeated attempts may be correlated; independent sessions are required for evaluation."
)


def model_fingerprint(model):
    return hashlib.sha256(json.dumps(
        model, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class PalmTrialLog:
    """One fixed model, claimed identity, and declared actual person per run."""

    def __init__(self, root, model, claim, actual, distance_cm=None, condition="standard", input_scale=1.0):
        self.claim = _identity(claim, "Claimed ID")
        self.actual = _identity(actual, "Actual ID")
        validate_model(model)
        if self.claim not in model["templates"]:
            raise ValueError(f"Unknown claimed identity {self.claim!r}.")
        thresholds = model.get("thresholds")
        if not isinstance(thresholds, dict) or not set(MODES).issubset(thresholds):
            raise ValueError("The model must define geometry, palm, and fusion thresholds.")
        self._thresholds = {mode: _number(thresholds[mode], "Model threshold") for mode in MODES}
        self.distance_cm = None if distance_cm is None else _number(distance_cm, "Camera distance")
        if self.distance_cm is not None and self.distance_cm <= 0:
            raise ValueError("Camera distance must be positive when supplied.")
        self.input_scale = _number(input_scale, "Input resolution scale")
        if not 0.1 <= self.input_scale <= 1:
            raise ValueError("Input resolution scale must be between 0.1 and 1.")
        if not isinstance(condition, str) or not condition.strip() or len(condition.strip()) > 80:
            raise ValueError("Condition must be a nonempty printable label of at most 80 characters.")
        self.condition = condition.strip()
        if not self.condition.isprintable():
            raise ValueError("Condition must be a nonempty printable label of at most 80 characters.")
        self.model = model
        self.model_sha256 = model_fingerprint(model)
        self._records = []
        self._lock = threading.RLock()
        self._started = monotonic()
        self._created_at = _utc_now()
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = root / f"{stamp}-{secrets.token_hex(8)}"
        with tempfile.TemporaryDirectory(prefix=".palm-run-", dir=root) as folder:
            staging = Path(folder)
            (staging / "trials").mkdir()
            _write_bytes(staging / "run.json", _json_bytes(self._metadata()))
            staging.rename(self.run_dir)
        self.export()

    @property
    def count(self):
        return len(self._records)

    def _metadata(self):
        return {
            "schema_version": 1, "model_schema_version": 2,
            "created_at_utc": self._created_at, "claimed_id": self.claim, "actual_id": self.actual,
            "distance_cm": self.distance_cm, "distance_source": "operator_declared",
            "input_scale": self.input_scale, "input_scale_source": "digital_full_frame_resize",
            "condition": self.condition, "model_sha256": self.model_sha256,
            "thresholds": dict(self._thresholds), "protocol": PROTOCOL,
        }

    def _validate(self, scores, quality):
        if model_fingerprint(self.model) != self.model_sha256:
            raise ValueError("The run's fixed model has changed.")
        if not isinstance(scores, dict):
            raise ValueError("A trial requires scores for all three methods.")
        if "modes" in scores:
            if scores.get("claimed_id") != self.claim:
                raise ValueError("The scored identity differs from the run's claimed identity.")
            scores = scores["modes"]
        if not isinstance(scores, dict) or set(scores) != set(MODES):
            raise ValueError("A trial requires exactly geometry, palm, and fusion scores.")
        validated = {}
        for mode in MODES:
            result = scores[mode]
            if not isinstance(result, dict) or not {"status", "accepted", "distance", "threshold"}.issubset(result):
                raise ValueError("Each method requires status, acceptance, distance, and threshold.")
            status = result["status"]
            if not isinstance(status, str) or not status or not status.isprintable() or len(status) > 80:
                raise ValueError("Each method needs a printable acquisition status.")
            threshold = _number(result["threshold"], "Decision threshold")
            if threshold != self._thresholds[mode]:
                raise ValueError("A decision threshold differs from the run's frozen model.")
            distance, accepted = result["distance"], result["accepted"]
            if status == "ok":
                distance = _number(distance, "Comparison distance")
                if distance < 0 or type(accepted) is not bool or accepted != (distance <= threshold):
                    raise ValueError("Acceptance must agree with the nonnegative distance and frozen threshold.")
            elif distance is not None or accepted is not None:
                raise ValueError("Acquisition failures require null distance and acceptance.")
            validated[mode] = {"status": status, "distance": distance, "accepted": accepted, "threshold": threshold}
        if not isinstance(quality, dict) or set(quality) - set(QUALITY_FIELDS):
            raise ValueError("Quality must contain only ROI pixel size, sharpness, and contrast measurements.")
        validated_quality = {}
        for name, value in quality.items():
            value = None if value is None else _number(value, name)
            if value is not None and value < 0:
                raise ValueError("Quality measurements must be nonnegative.")
            validated_quality[name] = value
        return validated, validated_quality

    def record(self, scores, quality):
        """Commit one explicit attempt, then refresh reports which can be retried."""
        with self._lock:
            modes, quality = self._validate(scores, quality)
            trial = {
                "trial_index": self.count + 1, "timestamp_utc": _utc_now(),
                "actual_id": self.actual, "claimed_id": self.claim, "genuine": self.actual == self.claim,
                "distance_cm": self.distance_cm, "input_scale": self.input_scale, "condition": self.condition,
                "model_sha256": self.model_sha256, "modes": modes, "quality": quality,
            }
            target = self.run_dir / "trials" / f"{trial['trial_index']:06d}.json"
            temporary = target.with_name(f".trial-{secrets.token_hex(8)}.tmp")
            committed = False
            result = {"status": "saved", "count": trial["trial_index"]}
            try:
                _write_bytes(temporary, _json_bytes(trial))
                os.link(temporary, target)
                self._records.append(trial)
                committed = True
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    if not committed:
                        raise
                    result["cleanup_error"] = str(error)
            try:
                self.export()
            except OSError as error:
                result["export_error"] = str(error)
            return result

    def summary(self):
        with self._lock:
            methods = {}
            for mode in MODES:
                valid = [trial for trial in self._records if trial["modes"][mode]["status"] == "ok"]
                genuine = [trial for trial in valid if trial["genuine"]]
                impostor = [trial for trial in valid if not trial["genuine"]]
                false_accepts = sum(trial["modes"][mode]["accepted"] for trial in impostor)
                false_rejects = sum(not trial["modes"][mode]["accepted"] for trial in genuine)
                failures = self.count - len(valid)
                methods[mode] = {
                    "attempts": self.count, "valid_acquisitions": len(valid), "acquisition_failures": failures,
                    "genuine_comparisons": len(genuine), "impostor_comparisons": len(impostor),
                    "false_accepts": false_accepts, "false_rejects": false_rejects,
                    "fmr": false_accepts / len(impostor) if impostor else None,
                    "fnmr": false_rejects / len(genuine) if genuine else None,
                    "acquisition_failure_rate": failures / self.count if self.count else None,
                }
            return {
                **self._metadata(), "attempts": self.count, "methods": methods,
                "elapsed_seconds": max(0.0, monotonic() - self._started),
            }

    def export(self):
        with self._lock:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=TRIAL_COLUMNS)
            writer.writeheader()
            for trial in self._records:
                base = {key: trial[key] for key in TRIAL_COLUMNS if key in trial}
                for mode, result in trial["modes"].items():
                    writer.writerow({
                        **base, **trial["quality"], "method": mode,
                        "quality_status": result["status"], "accepted": result["accepted"],
                        "distance": result["distance"], "threshold": result["threshold"],
                    })
            csv_path, summary_path = self.run_dir / "trials.csv", self.run_dir / "summary.json"
            _replace_bytes(csv_path, stream.getvalue().encode("utf-8"))
            _replace_bytes(summary_path, _json_bytes(self.summary()))
            return {"trials_csv": csv_path, "summary_json": summary_path}
