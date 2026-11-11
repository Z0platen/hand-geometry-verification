"""Exploratory enrolled-user verification on disjoint image subsets."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from matching import FEATURE_SETS, build_templates, feature_vector


PROTOCOL = (
    "Exploratory enrolled-user image split. Enrollment, calibration, and test use "
    "different image files from the same enrolled users. 11K Hands supplies no "
    "capture-session labels. Nearby video frames may remain correlated. Results "
    "do not establish cross-session or webcam generalization."
)
ACQUISITION_STATUSES = {
    "ok", "no_hand", "multiple_hands", "invalid_landmarks", "finger_out_of_frame",
    "degenerate_palm", "degenerate_finger", "missing_image", "unreadable_image",
}


def read_features(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or any(not name for name in headers):
            raise ValueError("Feature CSV column names must be nonempty and unique.")
        required = {"image_name", "subject_id", "status", *sum(FEATURE_SETS.values(), [])}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Feature CSV is missing columns: {', '.join(sorted(missing))}")
        rows = []
        status_counts = Counter()
        subject_counts = Counter()
        seen_images = set()
        seen_hashes = set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incorrect number of fields on CSV line {line}.")
            name = (row["image_name"] or "").strip()
            subject = (row["subject_id"] or "").strip()
            status = (row["status"] or "").strip()
            if not name or not subject or not status:
                raise ValueError(f"Missing image, subject, or status on CSV line {line}.")
            if status not in ACQUISITION_STATUSES:
                raise ValueError(f"Unknown acquisition status on CSV line {line}: {status!r}.")
            if name in seen_images:
                raise ValueError(f"Duplicate image name {name!r}; each image must appear once.")
            seen_images.add(name)
            digest = (row.get("image_sha256") or "").strip().casefold()
            if digest and digest in seen_hashes:
                raise ValueError("Duplicate image content hash; remove duplicate images before splitting.")
            if digest:
                seen_hashes.add(digest)
            status_counts[status] += 1
            subject_counts[subject] += 1
            if status != "ok":
                continue
            row["image_name"], row["subject_id"], row["status"] = name, subject, status
            for columns in FEATURE_SETS.values():
                try:
                    feature_vector(row, columns)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid successful measurements on CSV line {line}: {error}") from error
            rows.append(row)
    return rows, dict(status_counts), dict(subject_counts)


def split_samples(rows, seed=42, min_samples=6):
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 4:
        raise ValueError("At least four samples are required for two enrollment and separate validation/test images.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)
    rng = np.random.default_rng(seed)
    splits = {"enrollment": [], "calibration": [], "test": []}
    excluded = {}
    for subject, samples in sorted(grouped.items()):
        if len(samples) < min_samples:
            excluded[subject] = len(samples)
            continue
        samples = sorted(samples, key=lambda item: item["image_name"])
        shuffled = [samples[index] for index in rng.permutation(len(samples))]
        enrollment_count = max(2, int(len(samples) * 0.4))
        calibration_count = max(1, int(len(samples) * 0.3))
        splits["enrollment"].extend(shuffled[:enrollment_count])
        splits["calibration"].extend(shuffled[enrollment_count:enrollment_count + calibration_count])
        splits["test"].extend(shuffled[enrollment_count + calibration_count:])
    if len({row["subject_id"] for row in splits["enrollment"]}) < 2:
        raise ValueError("Evaluation requires at least two eligible users.")
    return splits, excluded


def comparisons(rows, templates, columns):
    subjects = list(templates)
    probes = np.asarray([feature_vector(row, columns) for row in rows])
    reference = np.asarray([templates[subject] for subject in subjects])
    distances = np.linalg.norm(probes[:, None, :] - reference[None, :, :], axis=2)
    genuine = np.asarray([[row["subject_id"] == subject for subject in subjects] for row in rows])
    return distances, genuine, subjects


def comparison_arrays(distances, genuine):
    scores = np.asarray(distances, dtype=float)
    labels = np.asarray(genuine)
    if scores.shape != labels.shape or labels.dtype != np.dtype(bool):
        raise ValueError("Each distance requires a corresponding Boolean genuine/impostor label.")
    scores, labels = scores.ravel(), labels.ravel()
    if not scores.size or not np.all(np.isfinite(scores)) or np.any(scores < 0) or not labels.any() or labels.all():
        raise ValueError("Error curves require finite genuine and impostor distances.")
    return scores, labels


def error_curve(distances, genuine):
    scores, labels = comparison_arrays(distances, genuine)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    accepted_genuine = np.r_[0, np.cumsum(sorted_labels)[ends]]
    accepted_impostor = np.r_[0, np.cumsum(~sorted_labels)[ends]]
    thresholds = np.r_[np.nextafter(sorted_scores[0], -np.inf), sorted_scores[ends]]
    fmr = accepted_impostor / np.sum(~labels)
    fnmr = 1 - accepted_genuine / np.sum(labels)
    return thresholds, fmr, fnmr


def calibrate_threshold(distances, genuine):
    thresholds, fmr, fnmr = error_curve(distances, genuine)
    # Primary objective is balanced error. Ties prefer fewer false accepts, then a smaller threshold.
    best = np.lexsort((thresholds, fmr, np.round((fmr + fnmr) / 2, 14)))[0]
    return float(thresholds[best])


def summarize_scores(distances, genuine, threshold):
    distances, genuine = comparison_arrays(distances, genuine)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("The operating threshold must be a finite number.")
    accepted = distances <= threshold
    genuine_count = int(np.sum(genuine))
    impostor_count = int(np.sum(~genuine))
    false_accepts = int(np.sum(accepted & ~genuine))
    false_rejects = int(np.sum(~accepted & genuine))
    _, fmr_curve, fnmr_curve = error_curve(distances, genuine)
    closest = int(np.argmin(np.abs(fmr_curve - fnmr_curve)))
    tpr = 1 - fnmr_curve
    auc = float(np.sum(np.diff(fmr_curve) * (tpr[:-1] + tpr[1:]) / 2))
    return {
        "threshold": threshold,
        "genuine_comparisons": genuine_count,
        "impostor_comparisons": impostor_count,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "fmr": false_accepts / impostor_count,
        "fnmr": false_rejects / genuine_count,
        "balanced_error": (false_accepts / impostor_count + false_rejects / genuine_count) / 2,
        "eer_descriptive": float((fmr_curve[closest] + fnmr_curve[closest]) / 2),
        "roc_auc": auc,
    }


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(method, phase, samples, subjects, distances, threshold):
    for row_index, sample in enumerate(samples):
        for column_index, subject in enumerate(subjects):
            distance = float(distances[row_index, column_index])
            yield {
                "method": method, "phase": phase, "image_name": sample["image_name"],
                "actual_subject": sample["subject_id"], "claimed_subject": subject,
                "genuine": int(subject == sample["subject_id"]), "distance": distance,
                "accepted": int(distance <= threshold),
            }


def plot_results(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    try:
        for row, (method, result) in enumerate(results.items()):
            distances, genuine = result["test_distances"], result["test_genuine"]
            axes[row, 0].hist(distances[genuine], bins=35, density=True, alpha=0.65, label="Genuine")
            axes[row, 0].hist(distances[~genuine], bins=35, density=True, alpha=0.55, label="Impostor")
            axes[row, 0].axvline(result["threshold"], color="black", linestyle="--", label="Calibration threshold")
            axes[row, 0].set(title=f"{method.capitalize()} — test distances", xlabel="Euclidean distance", ylabel="Density")
            axes[row, 0].legend()
            _, fmr, fnmr = error_curve(distances, genuine)
            axes[row, 1].plot(fmr, 1 - fnmr, label="Test ROC")
            axes[row, 1].plot([0, 1], [0, 1], color="gray", linestyle="--")
            metrics = result["summary"]["test"]
            axes[row, 1].scatter([metrics["fmr"]], [1 - metrics["fnmr"]], label="Fixed calibration threshold", zorder=3)
            axes[row, 1].set(title=f"{method.capitalize()} — test ROC", xlabel="False match rate", ylabel="True match rate", xlim=(0, 1), ylim=(0, 1))
            axes[row, 1].legend()
        figure.suptitle("Exploratory image-split hand verification")
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def evaluate(features_path, output, seed=42, min_samples=6):
    features_path = Path(features_path).resolve()
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty output directory so existing experiments remain intact.")
    rows, statuses, subject_counts = read_features(features_path)
    splits, excluded = split_samples(rows, seed, min_samples)
    successful_subjects = {row["subject_id"] for row in rows}
    excluded.update({subject: 0 for subject in subject_counts if subject not in successful_subjects})
    subjects = sorted({row["subject_id"] for row in splits["enrollment"]})
    summary = {
        "schema_version": 1, "protocol": PROTOCOL, "seed": seed, "minimum_samples": min_samples,
        "source_csv_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "input_status_counts": statuses, "source_users": len(subject_counts),
        "enrolled_users": len(subjects), "excluded_users": excluded,
        "split_counts": {name: len(samples) for name, samples in splits.items()},
        "threshold_selection": "Minimum calibration balanced error; ties prefer smaller FMR, then smaller threshold.",
        "uncertainty_note": "Comparison pairs share users, templates, and images and are not independent trials.",
        "methods": {},
    }
    results = {}
    for method, columns in FEATURE_SETS.items():
        templates = build_templates(splits["enrollment"], columns)
        calibration, calibration_genuine, order = comparisons(splits["calibration"], templates, columns)
        threshold = calibrate_threshold(calibration, calibration_genuine)
        test, test_genuine, _ = comparisons(splits["test"], templates, columns)
        method_summary = {
            "calibration": summarize_scores(calibration, calibration_genuine, threshold),
            "test": summarize_scores(test, test_genuine, threshold),
        }
        summary["methods"][method] = method_summary
        model = {
            "schema_version": 1, "method": method, "feature_names": columns,
            "threshold": threshold, "templates": templates, "evaluation_protocol": PROTOCOL,
            "enrollment_counts": dict(Counter(row["subject_id"] for row in splits["enrollment"])),
        }
        results[method] = {
            "model": model, "threshold": threshold, "subjects": order, "summary": method_summary,
            "calibration_distances": calibration, "test_distances": test, "test_genuine": test_genuine,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{output.name}-build-", dir=output.parent) as build_path:
        build = Path(build_path)
        write_csv(build / "splits.csv", ["image_name", "subject_id", "split"], (
            {"image_name": row["image_name"], "subject_id": row["subject_id"], "split": phase}
            for phase, samples in splits.items() for row in samples
        ))
        comparison_columns = ["method", "phase", "image_name", "actual_subject", "claimed_subject", "genuine", "distance", "accepted"]
        write_csv(build / "scores.csv", comparison_columns, (
            row for method, result in results.items() for phase in ("calibration", "test")
            for row in score_rows(method, phase, splits[phase], result["subjects"], result[f"{phase}_distances"], result["threshold"])
        ))
        for method, result in results.items():
            (build / f"model_{method}.json").write_text(json.dumps(result["model"], indent=2, allow_nan=False), encoding="utf-8")
        (build / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        plot_results(build / "verification_curves.png", results)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=6)
    args = parser.parse_args()
    try:
        summary = evaluate(args.features, args.output, args.seed, args.min_samples)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Evaluation error: {error}\n")
    print(f"Enrolled users: {summary['enrolled_users']}")
    print(summary["protocol"])
    for method, result in summary["methods"].items():
        test = result["test"]
        print(f"{method}: test FMR {test['fmr']:.2%}, FNMR {test['fnmr']:.2%}, descriptive EER {test['eer_descriptive']:.2%}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
