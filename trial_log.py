"""Durable records of manually requested verification attempts, without images."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from time import monotonic

from matching import load_model


QUALITY_STATUSES = frozenset({
    "ok", "no_hand", "multiple_hands", "invalid_landmarks", "finger_out_of_frame",
    "degenerate_palm", "degenerate_finger",
})
TRIAL_COLUMNS = (
    "trial_index", "timestamp_utc", "actual_id", "claimed_id", "genuine", "quality_status",
    "accepted", "distance", "threshold", "model_sha256", "method", "condition",
)
PROTOCOL = (
    "Exploratory, manually triggered webcam attempts with declared actual identity. "
    "Repeated attempts are not independent observations. The model and threshold stay fixed; "
    "these trials do not train or calibrate a threshold. Acquisition failures are reported "
    "separately and excluded from genuine/impostor comparison error rates."
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _identity(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value) is None:
        raise ValueError(f"{name} must contain 1–32 ASCII letters, digits, underscores, or hyphens.")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric scalar.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    return number


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_bytes(path, contents):
    with Path(path).open("xb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_bytes(path, contents):
    path = Path(path)
    temporary = path.with_name(f".{path.name}-{secrets.token_hex(8)}.tmp")
    try:
        _write_bytes(temporary, contents)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class TrialLog:
    """One fixed actual/claimed identity pair and model snapshot per experiment run.

    Each call to record represents one explicit manual attempt. Canonical trial JSON
    is committed before derived exports. If an export fails, record still returns
    status='saved' with export_error; retry export(), not record(), for that attempt.
    """

    def __init__(self, root, model_path, claimed_id, actual_id, condition="standard"):
        self.claim = _identity(claimed_id, "Claimed ID")
        self.actual = _identity(actual_id, "Actual ID")
        if not isinstance(condition, str):
            raise ValueError("Condition must be a nonempty, printable label of at most 80 characters.")
        self.condition = condition.strip()
        if not self.condition or len(self.condition) > 80 or not self.condition.isprintable():
            raise ValueError("Condition must be a nonempty, printable label of at most 80 characters.")

        self._source_path = Path(model_path).resolve()
        source_bytes = self._source_path.read_bytes()
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = root / f"{stamp}-{secrets.token_hex(8)}"
        self._records = []
        self._lock = threading.RLock()
        self._started = monotonic()
        self._created_at = _utc_now()
        self._model_sha256 = hashlib.sha256(source_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix=".run-", dir=root) as staging:
            staging = Path(staging)
            _write_bytes(staging / "model.json", source_bytes)
            self.model = load_model(staging / "model.json")
            if self.claim not in self.model["templates"]:
                raise ValueError(f"Unknown claimed identity {self.claim!r}.")
            self._model_signature = json.dumps(self.model, sort_keys=True, allow_nan=False)
            self._threshold = float(self.model["threshold"])
            self._method = self.model["method"]
            (staging / "trials").mkdir()
            _write_bytes(staging / "run.json", _json_bytes({
                "schema_version": 1,
                "created_at_utc": self._created_at,
                "claimed_id": self.claim,
                "actual_id": self.actual,
                "condition": self.condition,
                "model_source_path": str(self._source_path),
                "model_snapshot": "model.json",
                "model_sha256": self._model_sha256,
                "method": self._method,
                "threshold": self._threshold,
                "protocol": PROTOCOL,
            }))
            staging.rename(self.run_dir)
        self.export()

    @property
    def count(self):
        return len(self._records)

    def _validated_decision(self, decision):
        if not isinstance(decision, dict) or not {
            "status", "claimed_id", "accepted", "distance", "threshold",
        }.issubset(decision):
            raise ValueError("A trial requires a complete verification decision.")
        try:
            signature = json.dumps(self.model, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("The run's fixed model has been changed.") from error
        if signature != self._model_signature:
            raise ValueError("The run's fixed model has been changed.")
        status = decision["status"]
        if not isinstance(status, str) or status not in QUALITY_STATUSES:
            raise ValueError("Unknown acquisition quality status.")
        if decision["claimed_id"] != self.claim:
            raise ValueError("The decision's claimed identity differs from this run.")
        threshold = _number(decision["threshold"], "Decision threshold")
        if threshold != self._threshold:
            raise ValueError("The decision threshold differs from this run's fixed model.")
        if status != "ok":
            if decision["accepted"] is not None or decision["distance"] is not None:
                raise ValueError("Acquisition failures must have no distance or acceptance decision.")
            return status, None, None
        distance = _number(decision["distance"], "Decision distance")
        if distance < 0:
            raise ValueError("Decision distance must be non-negative.")
        accepted = distance <= self._threshold
        if type(decision["accepted"]) is not bool or decision["accepted"] != accepted:
            raise ValueError("The decision's acceptance is inconsistent with its distance and threshold.")
        return status, accepted, distance

    def record(self, decision):
        with self._lock:
            status, accepted, distance = self._validated_decision(decision)
            trial = dict(zip(TRIAL_COLUMNS, (
                self.count + 1, _utc_now(), self.actual, self.claim, self.actual == self.claim,
                status, accepted, distance, self._threshold, self._model_sha256,
                self._method, self.condition,
            ), strict=True))
            target = self.run_dir / "trials" / f"{trial['trial_index']:06d}.json"
            temporary = target.with_name(f".trial-{secrets.token_hex(8)}.tmp")
            committed = False
            cleanup_error = None
            try:
                _write_bytes(temporary, _json_bytes(trial))
                # Exclusive hard-link publication cannot overwrite another attempt.
                os.link(temporary, target)
                self._records.append(trial)
                committed = True
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    if not committed:
                        raise
                    cleanup_error = str(error)
            result = {"status": "saved", "count": self.count}
            if cleanup_error:
                result["cleanup_error"] = cleanup_error
            try:
                self.export()
            except OSError as error:
                result["export_error"] = str(error)
            return result

    def summary(self):
        with self._lock:
            valid = [trial for trial in self._records if trial["quality_status"] == "ok"]
            genuine = [trial for trial in valid if trial["genuine"]]
            impostor = [trial for trial in valid if not trial["genuine"]]
            false_accepts = sum(trial["accepted"] for trial in impostor)
            false_rejects = sum(not trial["accepted"] for trial in genuine)
            failures = self.count - len(valid)
            return {
                "schema_version": 1,
                "created_at_utc": self._created_at,
                "claimed_id": self.claim,
                "actual_id": self.actual,
                "condition": self.condition,
                "model_sha256": self._model_sha256,
                "method": self._method,
                "threshold": self._threshold,
                "attempts": self.count,
                "valid_acquisitions": len(valid),
                "acquisition_failures": failures,
                "genuine_comparisons": len(genuine),
                "impostor_comparisons": len(impostor),
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "fmr": false_accepts / len(impostor) if impostor else None,
                "fnmr": false_rejects / len(genuine) if genuine else None,
                "acquisition_failure_rate": failures / self.count if self.count else None,
                "elapsed_seconds": max(0.0, monotonic() - self._started),
                "protocol": PROTOCOL,
            }

    def export(self):
        """Regenerate reports from committed attempts; safe to retry after export errors."""
        with self._lock:
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=TRIAL_COLUMNS)
            writer.writeheader()
            writer.writerows(self._records)
            csv_path = self.run_dir / "trials.csv"
            summary_path = self.run_dir / "summary.json"
            _replace_bytes(csv_path, output.getvalue().encode("utf-8"))
            _replace_bytes(summary_path, _json_bytes(self.summary()))
            return {"trials_csv": csv_path, "summary_json": summary_path}
