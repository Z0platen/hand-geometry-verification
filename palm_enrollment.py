"""Explicit lossless webcam samples for a personal palm-and-geometry profile."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile

import cv2

from enrollment import validate_user_id, write_json
from multimodal import build_personal_model, validate_reference


DEFAULT_CAPTURES = Path(__file__).with_name("captures")


def latest_palm_model(user_id, root=DEFAULT_CAPTURES):
    validate_user_id(user_id)
    models = [path for path in (Path(root) / user_id).glob("*/palm_model.json") if path.is_file()]
    return max(models, key=lambda path: path.parent.name) if models else None


class PalmEnrollment:
    def __init__(self, user_id, reference, root=DEFAULT_CAPTURES, target=10):
        self.user_id = validate_user_id(user_id)
        if type(target) is not int or not 5 <= target <= 30:
            raise ValueError("Collect between 5 and 30 palm samples.")
        self.reference = json.loads(json.dumps(reference, allow_nan=False))
        validate_reference(self.reference)
        self.target = target
        self.samples = []
        self.hashes = set()
        self.model_path = None
        root = Path(root).resolve()
        user_directory = root / user_id
        if not user_directory.resolve().is_relative_to(root):
            raise ValueError("The capture directory must remain inside the chosen capture root.")
        user_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.session_dir = user_directory / f"{stamp}-palm-{secrets.token_hex(4)}"
        self.session_dir.mkdir()

    @property
    def count(self):
        return len(self.samples)

    def add_sample(self, frame, geometry, palm):
        if self.model_path is not None or self.count >= self.target:
            raise ValueError("This palm enrollment is complete.")
        if geometry["status"] != "ok":
            return geometry["status"]
        if palm["status"] != "ok":
            return palm["status"]
        sample = {"geometry": geometry["features"], "palm": palm["features"]}
        # Use the same descriptor validation as the final model before saving data.
        build_personal_model([sample], self.user_id, self.reference)
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise ValueError("Could not encode the palm sample.")
        image_bytes = encoded.tobytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in self.hashes:
            return "duplicate_capture"
        destination = self.session_dir / f"sample_{self.count + 1:03d}"
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=self.session_dir) as directory:
            staging = Path(directory)
            with (staging / "hand.png").open("xb") as handle:
                handle.write(image_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            write_json(staging / "measurements.json", {
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "image_sha256": digest, "quality": palm["quality"], **sample,
            })
            staging.rename(destination)
        self.samples.append(sample)
        self.hashes.add(digest)
        return "saved"

    def finalize(self):
        if self.model_path is not None:
            return self.model_path
        if self.count < self.target:
            raise ValueError(f"Collect {self.target - self.count} more usable palm samples.")
        model = build_personal_model(self.samples, self.user_id, self.reference)
        model.update({
            "model_kind": "palm_geometry", "personal_calibration": False,
            "enrollment_counts": {self.user_id: self.count},
            "session_id": self.session_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        destination = self.session_dir / "palm_model.json"
        temporary = self.session_dir / f".model-{secrets.token_hex(4)}.json"
        try:
            write_json(temporary, model)
            os.link(temporary, destination)
            self.model_path = destination
        finally:
            temporary.unlink(missing_ok=True)
        return destination
