"""Compare geometry, handcrafted palm texture, and fixed equal-weight fusion."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from zipfile import BadZipFile, ZipFile

import cv2
import mediapipe as mp
import numpy as np

from evaluate import calibrate_threshold, error_curve, score_rows, split_samples, summarize_scores, write_csv
from extract_features import archive_index, read_metadata
from geometry import RATIO_COLUMNS, analyze_hand
from multimodal import MODES, build_templates, validate_model, validate_reference
from palmprint import DEFAULT_QUALITY, DESCRIPTOR_VERSION, PALM_FEATURE_SIZE, analyze_palm
from pilot import DEFAULT_MODEL


PROTOCOL = (
    "Exploratory enrolled-user image split on 11K Hands right palms. All three modes use "
    "the same common-valid cohort, enrollment images, calibration images, and test probes. "
    "Templates use enrollment images only. Thresholds and fusion normalization use calibration "
    "only; fusion weights are fixed at 0.5 each. Capture-session labels are unavailable, and "
    "nearby frames may remain correlated. This does not establish cross-session or webcam "
    "generalization, and comparison pairs are not independent observations. The handcrafted "
    "Gabor descriptor is a baseline, not a reproduction of a published palmprint system."
)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_record(archive, index, image_name, subject_id, detector):
    record = {"image_name": image_name, "subject_id": subject_id, "image_sha256": "",
              "geometry": None, "palm": None, "quality": {}}
    if image_name not in index:
        return {**record, "geometry_status": "missing_image", "palm_status": "missing_image"}
    encoded = archive.read(index[image_name])
    record["image_sha256"] = hashlib.sha256(encoded).hexdigest()
    try:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR) if encoded else None
    except cv2.error:
        frame = None
    if frame is None:
        return {**record, "geometry_status": "unreadable_image", "palm_status": "unreadable_image"}
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    height, width = frame.shape[:2]
    geometry = analyze_hand(result.hand_landmarks, width, height)
    palm = analyze_palm(frame, result.hand_landmarks, quality=DEFAULT_QUALITY)
    record.update({"geometry_status": geometry["status"], "palm_status": palm["status"],
                   "quality": palm.get("quality", {})})
    if geometry["status"] == "ok":
        record["geometry"] = [float(geometry[column]) for column in RATIO_COLUMNS]
    if palm["status"] == "ok":
        record["palm"] = np.asarray(palm["features"], dtype=float).tolist()
    return record


def pairwise_distances(probes, references):
    """Euclidean distances using O(probes*users) memory, without a 3-D tensor."""
    probes, references = np.asarray(probes, dtype=float), np.asarray(references, dtype=float)
    if probes.ndim != 2 or references.ndim != 2 or probes.shape[1] != references.shape[1]:
        raise ValueError("Probe and reference matrices must have matching feature dimensions.")
    if not np.all(np.isfinite(probes)) or not np.all(np.isfinite(references)):
        raise ValueError("Comparison vectors must be finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.sum(probes * probes, axis=1)[:, None] + np.sum(references * references, axis=1)[None, :]
        squared -= 2 * (probes @ references.T)
    if not np.all(np.isfinite(squared)):
        raise ValueError("Pairwise distances exceed the finite numeric range.")
    return np.sqrt(np.maximum(squared, 0))


def _calibration_scale(distances, genuine):
    median = float(np.median(distances[genuine]))
    # Fixed fallback for degenerate calibration, never selected using test scores.
    return (median if median > 0 else 1.0), {"genuine_median": median, "fallback_used": median <= 0}


def _macro_statistics(distances, genuine, rows, threshold):
    per_user = {}
    actual = np.asarray([row["subject_id"] for row in rows])
    accepted = distances <= threshold
    for subject in sorted(set(actual)):
        mask = actual == subject
        labels, decisions = genuine[mask], accepted[mask]
        per_user[str(subject)] = {
            "probes": int(np.sum(mask)),
            "fmr": float(np.mean(decisions[~labels])),
            "fnmr": float(np.mean(~decisions[labels])),
        }
    return {"mean_user_fmr": float(np.mean([entry["fmr"] for entry in per_user.values()])),
            "mean_user_fnmr": float(np.mean([entry["fnmr"] for entry in per_user.values()])),
            "per_user": per_user}


def plot_comparison(path, results, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    try:
        for row, mode in enumerate(MODES):
            result = results[mode]
            distances = result["test_distances"]
            axes[row, 0].hist(distances[labels], bins=30, density=True, alpha=0.65, label="Genuine")
            axes[row, 0].hist(distances[~labels], bins=30, density=True, alpha=0.55, label="Impostor")
            axes[row, 0].axvline(result["threshold"], color="black", linestyle="--", label="Calibration threshold")
            axes[row, 0].set(title=f"{mode.capitalize()} test scores", xlabel="Distance / fused score", ylabel="Density")
            axes[row, 0].legend()
            _, fmr, fnmr = error_curve(distances, labels)
            axes[row, 1].plot(fmr, 1 - fnmr, label="Descriptive test ROC")
            axes[row, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            metrics = result["summary"]["test"]
            axes[row, 1].scatter([metrics["fmr"]], [1 - metrics["fnmr"]], label="Fixed calibration threshold")
            axes[row, 1].set(title=f"{mode.capitalize()} ROC", xlabel="False match rate", ylabel="True match rate",
                             xlim=(0, 1), ylim=(0, 1))
            axes[row, 1].legend()
        figure.suptitle("Exploratory common-valid image split; sessions unknown")
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def evaluate_records(records, output, seed=42, provenance=None, *, make_plot=True):
    """Evaluate retained extraction records; output is an internal empty build directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = list(records)
    names, hashes = set(), set()
    common = []
    subject_counts = Counter()
    geometry_counts, palm_counts = Counter(), Counter()
    for record in records:
        name, subject = record.get("image_name"), record.get("subject_id")
        if not isinstance(name, str) or not name or not isinstance(subject, str) or not subject:
            raise ValueError("Every extraction record requires an image_name and subject_id.")
        if name in names:
            raise ValueError("Duplicate image names cannot be split independently.")
        names.add(name)
        digest = record.get("image_sha256", "")
        if digest and digest in hashes:
            raise ValueError("Duplicate image content cannot be split independently.")
        if digest:
            hashes.add(digest)
        geometry_status, palm_status = record.get("geometry_status"), record.get("palm_status")
        if not isinstance(geometry_status, str) or not geometry_status or not isinstance(palm_status, str) or not palm_status:
            raise ValueError("Every extraction record requires both acquisition statuses.")
        subject_counts[subject] += 1
        geometry_counts[geometry_status] += 1
        palm_counts[palm_status] += 1
        if geometry_status == palm_status == "ok":
            # Validate each successful descriptor before it can influence a split.
            build_templates([record])
            common.append(record)
    splits, excluded = split_samples(common, seed=seed, min_samples=6)
    common_counts = Counter(record["subject_id"] for record in common)
    excluded.update({subject: common_counts[subject] for subject in subject_counts if subject not in common_counts})
    templates = build_templates(splits["enrollment"])
    subjects = list(templates)
    component_scores = {}
    labels = {}
    for phase in ("calibration", "test"):
        labels[phase] = np.asarray([[row["subject_id"] == subject for subject in subjects] for row in splits[phase]])
        component_scores[phase] = {
            component: pairwise_distances([row[component] for row in splits[phase]],
                                          [templates[subject][component] for subject in subjects])
            for component in ("geometry", "palm")
        }
    scales, scale_details = {}, {}
    for component in ("geometry", "palm"):
        scales[component], scale_details[component] = _calibration_scale(component_scores["calibration"][component], labels["calibration"])
    for phase in ("calibration", "test"):
        component_scores[phase]["fusion"] = sum(
            0.5 * component_scores[phase][component] / scales[component] for component in ("geometry", "palm")
        )
    thresholds = {mode: calibrate_threshold(component_scores["calibration"][mode], labels["calibration"]) for mode in MODES}
    assignment = {row["image_name"]: phase for phase, rows in splits.items() for row in rows}
    write_csv(output / "splits.csv", ["image_name", "subject_id", "image_sha256", "split", "geometry_status", "palm_status"], (
        {"image_name": row["image_name"], "subject_id": row["subject_id"], "image_sha256": row.get("image_sha256", ""),
         "geometry_status": row["geometry_status"], "palm_status": row["palm_status"],
         "split": assignment.get(row["image_name"], "insufficient_samples" if row["geometry_status"] == row["palm_status"] == "ok" else "acquisition_failed")}
        for row in records
    ))
    with (output / "features.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    source = dict(provenance or {})
    source.update({"split_manifest_sha256": hash_file(output / "splits.csv"),
                   "source_features_sha256": hash_file(output / "features.jsonl")})
    reference = {
        "schema_version": 2, "reference_kind": "palm_geometry_calibration",
        "descriptor_version": DESCRIPTOR_VERSION, "geometry_feature_names": list(RATIO_COLUMNS),
        "palm_feature_size": PALM_FEATURE_SIZE, "quality": dict(DEFAULT_QUALITY), "thresholds": thresholds,
        "fusion": {"geometry_scale": scales["geometry"], "palm_scale": scales["palm"],
                   "weights": {"geometry": 0.5, "palm": 0.5},
                   "normalization_source": "Calibration genuine-distance medians; non-positive median uses fixed scale 1.",
                   "calibration_details": scale_details},
        "calibration": {"users": len(subjects), "probes": len(splits["calibration"]), "seed": seed,
                        "selection": "Minimum calibration balanced error; ties prefer lower FMR, then lower threshold."},
        "evaluation_protocol": PROTOCOL, **source,
    }
    validate_reference(reference)
    model = {key: value for key, value in reference.items() if key != "reference_kind"}
    model.update({"model_kind": "palm_geometry", "templates": templates,
                  "enrollment_counts": dict(Counter(row["subject_id"] for row in splits["enrollment"]))})
    validate_model(model)
    results, method_summaries = {}, {}
    for mode in MODES:
        metrics = {phase: summarize_scores(component_scores[phase][mode], labels[phase], thresholds[mode])
                   for phase in ("calibration", "test")}
        metrics["test_macro_users"] = _macro_statistics(component_scores["test"][mode], labels["test"], splits["test"], thresholds[mode])
        method_summaries[mode] = metrics
        results[mode] = {"threshold": thresholds[mode], "summary": metrics,
                         "test_distances": component_scores["test"][mode]}
    count = len(records)
    summary = {
        "schema_version": 2, "protocol": PROTOCOL, "seed": seed, "minimum_samples": 6,
        "source_images": count, "source_users": len(subject_counts), "common_valid_images": len(common),
        "enrolled_users": len(subjects), "excluded_users": dict(sorted(excluded.items())),
        "source_subject_image_counts": dict(sorted(subject_counts.items())),
        "split_counts": {phase: len(rows) for phase, rows in splits.items()},
        "acquisition": {"geometry_status_counts": dict(sorted(geometry_counts.items())),
                        "palm_status_counts": dict(sorted(palm_counts.items())),
                        "geometry_failure_rate": (count - geometry_counts["ok"]) / count if count else None,
                        "palm_failure_rate": (count - palm_counts["ok"]) / count if count else None,
                        "common_failure_rate": (count - len(common)) / count if count else None},
        "quality": dict(DEFAULT_QUALITY), "descriptor_version": DESCRIPTOR_VERSION,
        "fusion_normalization": reference["fusion"], "methods": method_summaries, **source,
    }
    columns = ["method", "phase", "image_name", "actual_subject", "claimed_subject", "genuine", "distance", "accepted"]
    write_csv(output / "scores.csv", columns, (
        row for mode in MODES for phase in ("calibration", "test")
        for row in score_rows(mode, phase, splits[phase], subjects, component_scores[phase][mode], thresholds[mode])
    ))
    _write_json(output / "reference.json", reference)
    _write_json(output / "model.json", model)
    _write_json(output / "summary.json", summary)
    if make_plot:
        plot_comparison(output / "comparison.png", results, labels["test"])
    return summary


def prepare(archive_path, metadata_path, output, landmarker=DEFAULT_MODEL, limit=None, seed=42):
    archive_path, metadata_path, landmarker, output = map(Path, (archive_path, metadata_path, landmarker, output))
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty output directory; existing experiments will not be overwritten.")
    if any(path.resolve().is_relative_to(output) for path in (archive_path, metadata_path, landmarker)):
        raise ValueError("Input sources must be outside the output directory.")
    selected = read_metadata(metadata_path)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("The image limit must be a positive integer.")
        selected = selected[:limit]
    started = perf_counter()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-build-", dir=output.parent) as build_path:
        build = Path(build_path)
        records = []
        with ZipFile(archive_path) as archive:
            index = archive_index(archive)
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                for number, (image_name, subject_id) in enumerate(selected, 1):
                    records.append(extract_record(archive, index, image_name, subject_id, detector))
                    if number % 100 == 0:
                        print(f"Processed {number}/{len(selected)} right palms", flush=True)
        provenance = {"dataset": "11K Hands", "view": "palmar right", "limited_run": limit is not None,
                      "archive_sha256": hash_file(archive_path), "metadata_sha256": hash_file(metadata_path),
                      "landmarker_sha256": hash_file(landmarker), "mediapipe_version": mp.__version__,
                      "opencv_version": cv2.__version__, "numpy_version": np.__version__}
        summary = evaluate_records(records, build, seed=seed, provenance=provenance)
        summary["elapsed_seconds"] = round(perf_counter() - started, 2)
        _write_json(build / "summary.json", summary)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/palm_evaluation"))
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="First N filenames only; this may leave too few eligible users.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = prepare(args.archive, args.metadata, args.output, args.landmarker, args.limit, args.seed)
    except (OSError, ValueError, RuntimeError, BadZipFile, cv2.error) as error:
        parser.exit(1, f"Palm preparation error: {error}\n")
    print(f"Common-valid images: {summary['common_valid_images']}/{summary['source_images']}; enrolled users: {summary['enrolled_users']}")
    for mode, result in summary["methods"].items():
        metrics = result["test"]
        print(f"{mode}: fixed-threshold test FMR {metrics['fmr']:.2%}, FNMR {metrics['fnmr']:.2%}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
