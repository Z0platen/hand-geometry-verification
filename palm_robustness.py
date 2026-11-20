"""Evaluate frozen hand verification under simulated full-image downsampling."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import BadZipFile, ZipFile

import cv2
import mediapipe as mp
import numpy as np

from extract_features import archive_index
from geometry import analyze_hand
from multimodal import PreparedMatcher, load_model
from palm_trials import MODES, model_fingerprint
from palmprint import analyze_palm
from verify import DEFAULT_LANDMARKER


PROTOCOL = (
    "Held-out test photographs from the bound split manifest. Each complete image is "
    "downsampled before hand landmarks, geometry, and palm texture are extracted again. "
    "Templates and thresholds remain frozen at their full-resolution calibration values. "
    "Image scale simulates resolution loss and does not measure physical camera distance. "
    "Conditional-valid and common-valid comparisons are reported separately. Comparison "
    "pairs share probes, subjects, and templates and may be correlated."
)
SUMMARY_COLUMNS = (
    "image_scale", "method", "subset", "threshold", "attempted_probes", "valid_probes",
    "excluded_probes", "coverage", "acquisition_failures", "acquisition_failure_rate",
    "genuine_comparisons", "impostor_comparisons", "false_accepts", "false_rejects", "fmr", "fnmr",
)
PROBE_COLUMNS = (
    "image_scale", "image_name", "subject_id", "original_width", "original_height",
    "resized_width", "resized_height", "geometry_status", "palm_status", "fusion_status",
    "common_valid", "native_roi_pixels", "sharpness", "contrast",
)


def read_test_probes(path, model):
    path = Path(path)
    expected = model.get("split_manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Use a dataset model with the prepare_palm.py split manifest fingerprint.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("The split manifest does not match the frozen model's fingerprint.")
    selected, seen_names, seen_hashes = [], set(), set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        required = {"image_name", "subject_id", "image_sha256", "split"}
        if len(headers) != len(set(headers)) or not required.issubset(headers):
            raise ValueError("Split manifest requires unique image_name, subject_id, image_sha256, and split columns.")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed split manifest row on CSV line {line}.")
            name, subject, phase = (row[field].strip() for field in ("image_name", "subject_id", "split"))
            if not name or PurePosixPath(name).name != name or "\\" in name or not subject or not phase:
                raise ValueError(f"Missing or invalid split identifiers on CSV line {line}.")
            if name in seen_names:
                raise ValueError("The split manifest repeats an image name.")
            seen_names.add(name)
            digest = row["image_sha256"].strip().lower()
            if phase in {"enrollment", "calibration", "test"}:
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError("Eligible split images require their SHA-256 content hashes.")
                if digest in seen_hashes:
                    raise ValueError("Duplicate eligible image content would leak across split roles.")
                seen_hashes.add(digest)
            if phase == "test":
                if subject not in model["templates"]:
                    raise ValueError(f"Held-out subject {subject!r} has no frozen enrollment template.")
                selected.append({"image_name": name, "subject_id": subject, "image_sha256": digest})
    if not selected:
        raise ValueError("The split manifest contains no held-out test images.")
    return sorted(selected, key=lambda row: row["image_name"])


def validate_scales(scales):
    values = []
    for value in scales:
        if isinstance(value, bool):
            raise ValueError("Resolution scales must be numbers greater than zero and at most one.")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Resolution scales must be numbers greater than zero and at most one.") from error
        if not math.isfinite(value) or not 0 < value <= 1 or value in values:
            raise ValueError("Resolution scales must be unique numbers greater than zero and at most one.")
        values.append(value)
    if not values:
        raise ValueError("Supply at least one resolution scale.")
    return values


def analyze_probe(frame, detector, model):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hands = detector.detect(image).hand_landmarks
    height, width = frame.shape[:2]
    return analyze_hand(hands, width, height), analyze_palm(frame, hands, model["quality"])


def empty_counts():
    return dict.fromkeys((
        "valid_probes", "genuine_comparisons", "impostor_comparisons", "false_accepts", "false_rejects",
    ), 0)


def add_comparison(counts, result, genuine):
    if genuine:
        counts["genuine_comparisons"] += 1
        counts["false_rejects"] += int(not result["accepted"])
    else:
        counts["impostor_comparisons"] += 1
        counts["false_accepts"] += int(result["accepted"])


def finish_counts(counts, attempts, conditional):
    result = {
        **counts, "attempted_probes": attempts, "excluded_probes": attempts - counts["valid_probes"],
        "coverage": counts["valid_probes"] / attempts,
        "fmr": counts["false_accepts"] / counts["impostor_comparisons"] if counts["impostor_comparisons"] else None,
        "fnmr": counts["false_rejects"] / counts["genuine_comparisons"] if counts["genuine_comparisons"] else None,
    }
    if conditional:
        result["acquisition_failures"] = result["excluded_probes"]
        result["acquisition_failure_rate"] = result["excluded_probes"] / attempts
    return result


def evaluate_scale(archive, index, probes, scale, detector, model, matcher, writer):
    methods = {mode: {subset: empty_counts() for subset in ("conditional_valid", "common_valid")} for mode in MODES}
    statuses = {mode: Counter() for mode in MODES}
    subjects = sorted(model["templates"])
    for number, probe in enumerate(probes, 1):
        name, actual = probe["image_name"], probe["subject_id"]
        frame, image_status = None, "missing_image"
        if name in index:
            encoded = archive.read(index[name])
            if hashlib.sha256(encoded).hexdigest() != probe["image_sha256"]:
                raise ValueError(f"Held-out image content differs from its split manifest: {name}.")
            try:
                frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR) if encoded else None
            except cv2.error:
                frame = None
            image_status = "unreadable_image" if frame is None else "ok"
        dimensions = {}
        if frame is None:
            geometry, palm = {"status": image_status}, {"status": image_status, "features": None, "quality": {}}
        else:
            height, width = frame.shape[:2]
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = frame if scale == 1 else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            geometry, palm = analyze_probe(resized, detector, model)
            dimensions = dict(zip(("original_width", "original_height", "resized_width", "resized_height"), (width, height, *size)))
        geometry_features = geometry if geometry["status"] == "ok" else None
        palm_features = palm["features"] if palm["status"] == "ok" else None

        all_scores = matcher.score_all(
            geometry_features, palm_features,
            geometry_status=geometry["status"], palm_status=palm["status"],
        )
        first = all_scores[actual]["modes"]
        common = all(first[mode]["status"] == "ok" for mode in MODES)
        for mode in MODES:
            statuses[mode][first[mode]["status"]] += 1
            methods[mode]["conditional_valid"]["valid_probes"] += int(first[mode]["status"] == "ok")
            methods[mode]["common_valid"]["valid_probes"] += int(common)
        for claim in subjects:
            scores = all_scores[claim]["modes"]
            for mode in MODES:
                if scores[mode]["status"] == "ok":
                    add_comparison(methods[mode]["conditional_valid"], scores[mode], actual == claim)
                    if common:
                        add_comparison(methods[mode]["common_valid"], scores[mode], actual == claim)
        writer.writerow({
            "image_scale": scale, "image_name": name, "subject_id": actual, **dimensions,
            **{f"{mode}_status": first[mode]["status"] for mode in MODES},
            "common_valid": int(common),
            **{key: value for key, value in palm.get("quality", {}).items() if key in {"native_roi_pixels", "sharpness", "contrast"}},
        })
        if number % 100 == 0:
            print(f"Resolution scale {scale:g}: processed {number}/{len(probes)} test photographs", flush=True)
    return {
        "image_scale": scale,
        "methods": {
            mode: {
                "threshold": model["thresholds"][mode], "status_counts": dict(sorted(statuses[mode].items())),
                **{subset: finish_counts(counts, len(probes), subset == "conditional_valid") for subset, counts in subsets.items()},
            }
            for mode, subsets in methods.items()
        },
    }


def plot_results(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    colors = {"geometry": "#2775b6", "palm": "#ce681f", "fusion": "#289166"}
    try:
        for row, subset in enumerate(("conditional_valid", "common_valid")):
            for column, (metric, label) in enumerate((("coverage", "Usable probes (%)"), ("fmr", "False match rate (%)"), ("fnmr", "False non-match rate (%)"))):
                axis = axes[row, column]
                for mode in MODES:
                    values = [result["methods"][mode][subset][metric] for result in results]
                    axis.plot(range(len(results)), [np.nan if value is None else value * 100 for value in values], marker="o", label=mode, color=colors[mode])
                axis.set_xticks(range(len(results)), [f"{result['image_scale']:g}" for result in results])
                axis.set(xlabel="Full-image resolution scale", ylabel=label, title=subset.replace("_", " ").title())
                axis.set_ylim(-2, 102)
                axis.grid(alpha=0.2)
                axis.legend()
        figure.suptitle("Simulated downsampling with frozen full-resolution calibration")
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def run(archive_path, model_path, splits_path, output, scales=(1, 0.5, 0.25), landmarker=DEFAULT_LANDMARKER, limit=None):
    scales = validate_scales(scales)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("The held-out probe limit must be a positive integer.")
    archive_path, model_path, splits_path, output, landmarker = map(Path, (archive_path, model_path, splits_path, output, landmarker))
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty output directory to preserve existing experiments.")
    model = load_model(model_path)
    matcher = PreparedMatcher(model)
    probes = read_test_probes(splits_path, model)
    available = len(probes)
    probes = probes if limit is None else probes[:limit]
    if not landmarker.is_file():
        raise FileNotFoundError(f"Hand landmarker model missing at {landmarker}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{output.name}-build-", dir=output.parent) as folder:
        build = Path(folder)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        )
        with ZipFile(archive_path) as archive, \
                mp.tasks.vision.HandLandmarker.create_from_options(options) as detector, \
                (build / "probes.csv").open("w", encoding="utf-8", newline="") as file:
            index = archive_index(archive)
            writer = csv.DictWriter(file, fieldnames=PROBE_COLUMNS)
            writer.writeheader()
            results = [evaluate_scale(archive, index, probes, scale, detector, model, matcher, writer) for scale in scales]
        summary = {
            "schema_version": 1, "protocol": PROTOCOL,
            "model_sha256": model_fingerprint(model),
            "split_manifest_sha256": model["split_manifest_sha256"], "split_manifest_binding_verified": True,
            "landmarker_sha256": hashlib.sha256(landmarker.read_bytes()).hexdigest(),
            "scales": scales, "available_test_images": available, "selected_test_images": len(probes),
            "probe_limit": limit, "enrolled_users": len(model["templates"]), "thresholds": model["thresholds"],
            "resolution_simulation": "Entire photographs resized before landmark redetection; no physical distance inference.",
            "results": results,
        }
        (build / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (build / "summary.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for result in results:
                for mode, method in result["methods"].items():
                    for subset in ("conditional_valid", "common_valid"):
                        writer.writerow({
                            "image_scale": result["image_scale"], "method": mode, "subset": subset,
                            "threshold": method["threshold"], **method[subset],
                        })
        plot_results(build / "resolution_comparison.png", results)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.5, 0.25])
    parser.add_argument("--output", type=Path, default=Path("outputs/palm_robustness"))
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--limit", type=int, help="Use the first N sorted held-out probes for a smoke run")
    args = parser.parse_args()
    try:
        summary = run(args.archive, args.model, args.splits, args.output, args.scales, args.landmarker, args.limit)
    except (OSError, ValueError, RuntimeError, BadZipFile, cv2.error) as error:
        parser.exit(1, f"Resolution experiment error: {error}\n")
    print(f"Evaluated {summary['selected_test_images']} held-out photographs at {len(summary['scales'])} resolution scales.")
    print(f"Reports saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
