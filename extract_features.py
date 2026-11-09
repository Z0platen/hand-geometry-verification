"""Extract right-palmar finger measurements directly from the 11K Hands ZIP."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from time import perf_counter
from zipfile import BadZipFile, ZipFile

import cv2
import mediapipe as mp
import numpy as np

from geometry import PIXEL_COLUMNS, RATIO_COLUMNS, analyze_hand
from pilot import DEFAULT_MODEL


COLUMNS = ("image_name", "subject_id", "image_sha256", "status", "palm_width_px", *PIXEL_COLUMNS, *RATIO_COLUMNS)


def read_metadata(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "imageName", "aspectOfHand"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Metadata requires id, imageName, and aspectOfHand columns.")
        selected = []
        seen = set()
        for row in reader:
            if not all(isinstance(row.get(name), str) and row[name].strip() for name in required):
                raise ValueError("Metadata has a row with missing id, imageName, or aspectOfHand.")
            if row["aspectOfHand"].strip().lower() != "palmar right":
                continue
            image_name, subject_id = row["imageName"].strip(), row["id"].strip()
            if not image_name or not subject_id or image_name in seen:
                raise ValueError("Right-palmar metadata has an empty identifier or duplicate filename.")
            selected.append((image_name, subject_id))
            seen.add(image_name)
    if not selected:
        raise ValueError("Metadata contains no right-palmar images.")
    return sorted(selected)


def archive_index(archive):
    index = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or path.name.startswith("._"):
            continue
        if path.name in index:
            raise ValueError(f"Ambiguous image basename in ZIP: {path.name}")
        index[path.name] = name
    return index


def extract_record(archive, index, image_name, subject_id, detector):
    record = {"image_name": image_name, "subject_id": subject_id}
    if image_name not in index:
        return {**record, "status": "missing_image"}
    encoded = archive.read(index[image_name])
    record["image_sha256"] = hashlib.sha256(encoded).hexdigest()
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {**record, "status": "unreadable_image"}
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
    return {**record, **analyze_hand(result.hand_landmarks, width, height)}


def extract(archive_path, metadata_path, model_path, output_path, limit=None):
    output_path = output_path.resolve()
    metadata_output = output_path.with_suffix(".meta.json")
    sources = {path.resolve() for path in (archive_path, metadata_path, model_path)}
    if output_path in sources or metadata_output in sources:
        raise ValueError("Output files must have different paths from the input files.")
    selected = read_metadata(metadata_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("The image limit must be positive.")
        selected = selected[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
    )
    counts = Counter()
    started = perf_counter()
    temporary_path = None
    try:
        with ZipFile(archive_path) as archive:
            index = archive_index(archive)
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent,
                                                 suffix=".tmp", delete=False) as file:
                    temporary_path = Path(file.name)
                    writer = csv.DictWriter(file, fieldnames=COLUMNS)
                    writer.writeheader()
                    for number, (image_name, subject_id) in enumerate(selected, 1):
                        record = extract_record(archive, index, image_name, subject_id, detector)
                        writer.writerow(record)
                        counts[record["status"]] += 1
                        if number % 100 == 0:
                            print(f"Processed {number}/{len(selected)} images", flush=True)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    summary = {
        "dataset": "11K Hands", "view": "palmar right", "selected_images": len(selected),
        "statuses": dict(sorted(counts.items())), "feature_columns": list(RATIO_COLUMNS),
        "mediapipe_version": mp.__version__, "opencv_version": cv2.__version__,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    metadata_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/features.csv"))
    parser.add_argument("--limit", type=int, help="Process the first N filenames for a smoke check")
    args = parser.parse_args()
    try:
        summary = extract(args.archive, args.metadata, args.landmarker, args.output, args.limit)
    except (OSError, ValueError, RuntimeError, BadZipFile, cv2.error) as error:
        parser.exit(1, f"Extraction error: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
