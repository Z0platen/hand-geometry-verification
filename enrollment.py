"""Explicit webcam captures and a personal median hand template."""

from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading

import cv2
import numpy as np

from matching import FEATURE_SETS, feature_vector


def validate_user_id(user_id):
    if not isinstance(user_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", user_id) is None:
        raise ValueError("User ID must contain 1–32 ASCII letters, digits, underscores, or hyphens.")
    return user_id


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_json(path, contents):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(contents, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class EnrollmentSession:
    def __init__(
        self, user_id, root, reference_model, sample_count=10,
        min_interval_seconds=0.8, reference_model_path=None,
    ):
        self.user_id = validate_user_id(user_id)
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("The capture target must be a positive integer.")
        if isinstance(min_interval_seconds, bool):
            raise ValueError("The minimum capture interval must be finite and non-negative.")
        try:
            interval = float(min_interval_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError("The minimum capture interval must be finite and non-negative.") from error
        if not isfinite(interval) or interval < 0:
            raise ValueError("The minimum capture interval must be finite and non-negative.")
        if not isinstance(reference_model, dict) or reference_model.get("schema_version") != 1:
            raise ValueError("A valid version 1 ratio reference model is required.")
        if reference_model.get("method") != "ratios" or reference_model.get("feature_names") != FEATURE_SETS["ratios"]:
            raise ValueError("Personal enrollment requires the five-finger ratio reference model.")
        threshold = reference_model.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not isfinite(threshold):
            raise ValueError("The reference threshold must be finite.")

        try:
            encoded_reference = json.dumps(reference_model, sort_keys=True, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("The reference model must contain valid JSON values.") from error
        source_kind = "canonical_json"
        source_path = None
        if reference_model_path is not None:
            source_path = Path(reference_model_path).resolve()
            source_bytes = source_path.read_bytes()
            if json.loads(source_bytes) != reference_model:
                raise ValueError("The source model file differs from the supplied reference model.")
            encoded_reference = source_bytes
            source_kind = "file_bytes"

        self.target = sample_count
        self.min_interval_seconds = interval
        self._threshold = float(threshold)
        self._samples = []
        self._last_timestamp = None
        self._finalized_model = None
        self._lock = threading.Lock()
        self._provenance = {
            "reference_model_sha256": hashlib.sha256(encoded_reference).hexdigest(),
            "reference_model_hash_kind": source_kind,
            "reference_model_path": str(source_path) if source_path else None,
            "reference_evaluation_protocol": reference_model.get("evaluation_protocol"),
        }

        root = Path(root).resolve()
        user_directory = root / self.user_id
        if not user_directory.resolve().is_relative_to(root):
            raise ValueError("The user capture directory resolves outside the selected root.")
        user_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.session_dir = user_directory / f"{stamp}-{secrets.token_hex(4)}"
        self.session_dir.mkdir()
        write_json(self.session_dir / "session.json", {
            "schema_version": 1,
            "user_id": self.user_id,
            "created_at_utc": utc_now(),
            "capture_target": self.target,
            "min_interval_seconds": interval,
            "feature_names": FEATURE_SETS["ratios"],
            "threshold": self._threshold,
            "threshold_source": "transferred_from_dataset",
            "personal_calibration": False,
            **self._provenance,
        })

    @property
    def count(self):
        return len(self._samples)

    def _result(self, status):
        return {"status": status, "count": self.count, "target": self.target}

    def add_sample(self, frame, measurements, timestamp):
        """Persist one explicitly requested raw frame when its measurements are usable."""
        with self._lock:
            if self._finalized_model is not None:
                raise ValueError("This enrollment session is already finalized.")
            if self.count >= self.target:
                return self._result("enrollment_complete")
            if not isinstance(measurements, dict):
                return self._result("invalid_measurements")
            status = measurements.get("status")
            if status != "ok":
                return self._result(status if isinstance(status, str) and status else "invalid_measurements")
            try:
                vector = feature_vector(measurements, FEATURE_SETS["ratios"])
            except ValueError:
                return self._result("invalid_measurements")
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3:
                return self._result("invalid_frame")
            if frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                return self._result("invalid_frame")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not isfinite(timestamp) or timestamp < 0:
                return self._result("invalid_timestamp")
            if self._last_timestamp is not None and timestamp - self._last_timestamp < self.min_interval_seconds:
                return self._result("capture_too_soon")

            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise RuntimeError("Could not encode the camera capture.")
            image_bytes = encoded.tobytes()
            capture_index = self.count + 1
            capture_name = f"sample_{capture_index:03d}"
            metadata = {
                "schema_version": 1,
                "user_id": self.user_id,
                "capture_index": capture_index,
                "captured_at_utc": utc_now(),
                "timestamp_seconds": float(timestamp),
                "image_file": "hand.jpg",
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "measurements": dict(zip(FEATURE_SETS["ratios"], vector.tolist(), strict=True)),
            }
            with tempfile.TemporaryDirectory(prefix=".capture-", dir=self.session_dir) as staging:
                staging = Path(staging)
                with (staging / "hand.jpg").open("xb") as handle:
                    handle.write(image_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                write_json(staging / "metadata.json", metadata)
                staging.rename(self.session_dir / capture_name)
            self._samples.append(vector)
            self._last_timestamp = float(timestamp)
            return self._result("saved")

    def finalize(self):
        """Publish a model only after the requested number of captures were saved."""
        with self._lock:
            if self._finalized_model is not None:
                return self._finalized_model
            if self.count < self.target:
                raise ValueError(f"Capture {self.target - self.count} more usable samples before finalizing.")
            model = {
                "schema_version": 1,
                "model_kind": "personal_demo",
                "method": "ratios",
                "feature_names": FEATURE_SETS["ratios"],
                "threshold": self._threshold,
                "threshold_source": "transferred_from_dataset",
                "personal_calibration": False,
                "templates": {self.user_id: np.median(self._samples, axis=0).tolist()},
                "enrollment_counts": {self.user_id: self.count},
                "session_id": self.session_dir.name,
                "created_at_utc": utc_now(),
                "evaluation_protocol": (
                    "Personal webcam enrollment demonstration. The threshold is transferred "
                    "from the reference dataset model. Personal calibration and independent "
                    "personal verification testing remain future measurements."
                ),
                **self._provenance,
            }
            model_path = self.session_dir / "model.json"
            temporary_path = self.session_dir / f".model-{secrets.token_hex(4)}.json"
            write_json(temporary_path, model)
            try:
                temporary_path.rename(model_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            self._finalized_model = model_path
            return model_path
