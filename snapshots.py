"""Save a manually requested verification screenshot and its matching decision."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile

import cv2


DEFAULT_SNAPSHOTS = Path(__file__).with_name("outputs") / "snapshots"


def save_snapshot(preview, decision, model, root=DEFAULT_SNAPSHOTS):
    """Publish an image/result pair together; never include enrollment templates."""
    success, encoded = cv2.imencode(".png", preview)
    if not success:
        raise ValueError("Could not encode the screenshot.")
    image_bytes = encoded.tobytes()
    now = datetime.now(timezone.utc)
    metadata = {
        "schema_version": 1,
        "saved_at_utc": now.isoformat(),
        "image_file": "preview.png",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "decision": decision,
        "method": model["method"],
        "model_kind": model.get("model_kind", "dataset_experiment"),
        "model_sha256": hashlib.sha256(
            json.dumps(model, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "model_hash_kind": "canonical_json",
        "note": "One manually saved comparison; this screenshot is not an accuracy estimate.",
    }
    result_bytes = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=root) as directory:
        staging = Path(directory)
        for name, contents in (("preview.png", image_bytes), ("decision.json", result_bytes)):
            with (staging / name).open("xb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
        staging.rename(destination)
    return destination
