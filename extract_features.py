"""Extract right-palmar finger measurements directly from the 11K Hands ZIP."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
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
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "imageName", "aspectOfHand"}
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or any(not name for name in headers):
            raise ValueError("Metadata column names must be nonempty and unique.")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Metadata requires id, imageName, and aspectOfHand columns.")
        selected = []
        seen = set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Metadata has an incorrect number of fields on CSV line {line}.")
            if not all(isinstance(row.get(name), str) and row[name].strip() for name in required):
                raise ValueError("Metadata has a row with missing id, imageName, or aspectOfHand.")
            image_name, subject_id = row["imageName"].strip(), row["id"].strip()
            if not image_name or not subject_id or image_name in seen:
                raise ValueError("Metadata has an empty identifier or duplicate filename.")
            seen.add(image_name)
            aspect = row["aspectOfHand"].strip().lower()
            if aspect not in {"palmar right", "palmar left", "dorsal right", "dorsal left"}:
                raise ValueError(f"Unknown hand aspect on CSV line {line}: {aspect!r}.")
            if aspect == "palmar right":
                selected.append((image_name, subject_id))
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
    try:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR) if encoded else None
    except cv2.error:
        frame = None
    if frame is None:
        return {**record, "status": "unreadable_image"}
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
    return {**record, **analyze_hand(result.hand_landmarks, width, height)}


def publish_outputs(staged_files):
    """Replace the CSV and provenance together, restoring prior files on failure."""
    backups = {}
    for staged, target in staged_files:
        if target.exists():
            backup = staged.with_name(staged.name + ".previous")
            shutil.copy2(target, backup)
            backups[target] = backup
    published = []
    try:
        for staged, target in staged_files:
            os.replace(staged, target)
            published.append(target)
    except OSError:
        for target in reversed(published):
            if target in backups:
                os.replace(backups[target], target)
            else:
                target.unlink(missing_ok=True)
        raise


def extract(archive_path, metadata_path, model_path, output_path, limit=None):
    archive_path, metadata_path, model_path = map(Path, (archive_path, metadata_path, model_path))
    output_path = Path(output_path).resolve()
    metadata_output = output_path.with_suffix(".meta.json").resolve()
    sources = {path.resolve() for path in (archive_path, metadata_path, model_path)}
    if output_path in sources or metadata_output in sources or output_path == metadata_output:
        raise ValueError("Output files must have different paths from the input files.")
    selected = read_metadata(metadata_path)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
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
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}-build-", dir=output_path.parent) as folder:
        build = Path(folder)
        temporary_path = build / "features.csv"
        with ZipFile(archive_path) as archive:
            index = archive_index(archive)
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                with temporary_path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=COLUMNS)
                    writer.writeheader()
                    for number, (image_name, subject_id) in enumerate(selected, 1):
                        record = extract_record(archive, index, image_name, subject_id, detector)
                        writer.writerow(record)
                        counts[record["status"]] += 1
                        if number % 100 == 0:
                            print(f"Processed {number}/{len(selected)} images", flush=True)
                    file.flush()
                    os.fsync(file.fileno())

        summary = {
            "dataset": "11K Hands", "view": "palmar right", "selected_images": len(selected),
            "statuses": dict(sorted(counts.items())), "feature_columns": list(RATIO_COLUMNS),
            "mediapipe_version": mp.__version__, "opencv_version": cv2.__version__,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(temporary_path.read_bytes()).hexdigest(),
            "elapsed_seconds": round(perf_counter() - started, 2),
        }
        staged_metadata = build / "features.meta.json"
        with staged_metadata.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        publish_outputs(((temporary_path, output_path), (staged_metadata, metadata_output)))
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
