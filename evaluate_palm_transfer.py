"""Evaluate frozen palm models on identities excluded from metric training."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from evaluate import summarize_scores
from geometry import RATIO_COLUMNS
from multimodal import MODES, build_templates, validate_model
from palm_learning import project_features
from prepare_palm import _write_json
from train_palm import _read_bound_splits


SEED = 20261120
MIN_IMAGES = 3
ENROLLMENT_IMAGES = 2
PROTOCOL = (
    "Small-cohort transfer check on common-valid images from identities marked insufficient_samples "
    "in the source split manifest, excluding every identity in the metric-training cohort. "
    "For each eligible identity, two images selected by fixed-seed SHA-256 ordering form a raw "
    "median enrollment template and remaining images are probes. The same raw templates replace "
    "the templates in each frozen model; learned palm metrics, thresholds, fusion scales, and "
    "fusion weights are unchanged. No fitting or tuning uses this cohort. This measures unseen "
    "identities within the same dataset and common-valid acquisition condition; capture sessions "
    "are unknown, and webcam performance and cross-session generalization are not established. "
    "Pairwise comparisons are correlated and the cohort is small."
)


def _read_transfer_rows(features_path, splits_path, binding_model):
    """Validate source bindings with the training reader, then return eligible rows."""
    _read_bound_splits(features_path, splits_path, binding_model)
    rows = []
    features = {json.loads(line)["image_name"]: json.loads(line)
                for line in Path(features_path).read_text(encoding="utf-8").splitlines()}
    reader = csv.DictReader(io.StringIO(Path(splits_path).read_text(encoding="utf-8"), newline=""))
    for manifest_row in reader:
        if manifest_row["split"] != "insufficient_samples":
            continue
        record = features[manifest_row["image_name"]]
        if record["geometry_status"] == record["palm_status"] == "ok" and record["image_sha256"]:
            rows.append(record)
    return rows


def _partition_transfer(rows, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)
    eligible = {subject: members for subject, members in grouped.items() if len(members) >= MIN_IMAGES}
    if not eligible:
        raise ValueError("No unseen identity has at least three common-valid insufficient_samples images.")
    enrollment, probes, manifest = [], [], []
    for subject in sorted(eligible):
        ordered = sorted(eligible[subject], key=lambda row: (
            hashlib.sha256(f"{seed}:{row['image_name']}".encode("utf-8")).hexdigest(), row["image_name"],
        ))
        for index, row in enumerate(ordered):
            role = "enrollment" if index < ENROLLMENT_IMAGES else "probe"
            manifest.append({"image_name": row["image_name"], "subject_id": subject,
                             "image_sha256": row["image_sha256"], "role": role})
            (enrollment if role == "enrollment" else probes).append(row)
    return enrollment, probes, manifest


def _frozen_model(model, templates, enrollment):
    clone = json.loads(json.dumps(model, allow_nan=False))
    clone["templates"] = templates
    clone["enrollment_counts"] = dict(sorted(Counter(row["subject_id"] for row in enrollment).items()))
    return validate_model(clone)


def _distances(rows, subjects, model):
    templates = model["templates"]
    result = {}
    geometry_references = [templates[subject]["geometry"] for subject in subjects]
    raw_palm_references = [templates[subject]["palm"] for subject in subjects]
    metric = model.get("palm_metric")
    if metric is None:
        palm_references = raw_palm_references
    else:
        palm_references = project_features(raw_palm_references, metric).tolist()
    for mode in MODES:
        matrix = []
        for row in rows:
            if mode == "geometry":
                probe, references = row["geometry"], geometry_references
                values = [math.dist(probe, reference) for reference in references]
            elif mode == "palm":
                probe = row["palm"] if metric is None else project_features(row["palm"], metric).tolist()
                values = [math.dist(probe, reference) for reference in palm_references]
            else:
                geometry = [math.dist(row["geometry"], reference) for reference in geometry_references]
                probe = row["palm"] if metric is None else project_features(row["palm"], metric).tolist()
                palm = [math.dist(probe, reference) for reference in palm_references]
                fusion = model["fusion"]
                values = [fusion["weights"]["geometry"] * g / fusion["geometry_scale"]
                          + fusion["weights"]["palm"] * p / fusion["palm_scale"]
                          for g, p in zip(geometry, palm)]
            matrix.append(values)
        result[mode] = np.asarray(matrix, dtype=float)
    return result


def _macro_scores(distances, rows, subjects, threshold):
    genuine = np.asarray([row["subject_id"] for row in rows])[:, None] == np.asarray(subjects)[None, :]
    overall = summarize_scores(distances, genuine, threshold)
    user_rates = {}
    for subject in subjects:
        rows_for_user = np.asarray([row["subject_id"] == subject for row in rows])
        subject_genuine = genuine[rows_for_user]
        accepted = distances[rows_for_user] <= threshold
        user_rates[subject] = {
            "fmr": float(np.sum(accepted & ~subject_genuine) / np.sum(~subject_genuine)),
            "fnmr": float(np.sum(~accepted & subject_genuine) / np.sum(subject_genuine)),
        }
    overall["per_user_macro_fmr"] = float(np.mean([rates["fmr"] for rates in user_rates.values()]))
    overall["per_user_macro_fnmr"] = float(np.mean([rates["fnmr"] for rates in user_rates.values()]))
    overall["per_user"] = user_rates
    return overall


def evaluate_transfer(model_path, incumbent_path, features_path, splits_path, baseline_path, output):
    output = Path(output).resolve()
    paths = [Path(path) for path in (model_path, incumbent_path, features_path, splits_path, baseline_path)]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory for the transfer experiment.")
    if any(path.resolve().is_relative_to(output) for path in paths):
        raise ValueError("Transfer inputs must be outside the output directory.")
    if Path(baseline_path).is_dir():
        baseline_path = Path(baseline_path) / "model.json"
    model_bytes = Path(model_path).read_bytes()
    incumbent_bytes = Path(incumbent_path).read_bytes()
    baseline_bytes = Path(baseline_path).read_bytes()
    model, incumbent, baseline = (validate_model(json.loads(data)) for data in
                                  (model_bytes, incumbent_bytes, baseline_bytes))
    source_keys = ("source_features_sha256", "split_manifest_sha256")
    if any(model.get(key) != incumbent.get(key) for key in source_keys) or any(
            model.get(key) != baseline.get(key) for key in source_keys):
        raise ValueError("The compared models and baseline must share feature and split source hashes.")
    training_ids = set(model["templates"]) | set(incumbent["templates"]) | set(baseline["templates"])
    rows = _read_transfer_rows(features_path, splits_path, baseline)
    candidate_counts = Counter(row["subject_id"] for row in rows)
    overlap_ids = {row["subject_id"] for row in rows} & training_ids
    rows = [row for row in rows if row["subject_id"] not in training_ids]
    counts = Counter(row["subject_id"] for row in rows)
    skipped_ids = sorted(subject for subject, count in counts.items() if count < MIN_IMAGES)
    skipped = [
        {"subject_id": subject, "common_valid_images": candidate_counts[subject],
         "reason": "identity overlaps the metric-training cohort"}
        for subject in sorted(overlap_ids)
    ] + [
        {"subject_id": subject, "common_valid_images": counts[subject],
         "reason": f"fewer than {MIN_IMAGES} common-valid images"}
        for subject in skipped_ids
    ]
    enrollment, probes, split_manifest = _partition_transfer(rows)
    templates = build_templates(enrollment)
    if len(templates) < 2:
        raise ValueError("At least two eligible unseen identities are required for impostor comparisons.")

    frozen = {"tuned": _frozen_model(model, templates, enrollment),
              "incumbent": _frozen_model(incumbent, templates, enrollment)}
    subjects = sorted(templates)
    score_data = {}
    for name, candidate in frozen.items():
        distances = _distances(probes, subjects, candidate)
        score_data[name] = {
            mode: _macro_scores(distances[mode], probes, subjects, candidate["thresholds"][mode])
            for mode in MODES
        }
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "eligible_identity_count": len(subjects),
        "training_identity_count": len(training_ids),
        "identity_count_excluded_as_training_overlap": len(overlap_ids),
        "insufficient_manifest_common_valid_identity_count": len(candidate_counts),
        "insufficient_manifest_common_valid_image_count": sum(candidate_counts.values()),
        "common_valid_insufficient_identity_count": len(counts),
        "common_valid_insufficient_image_count": sum(counts.values()),
        "skipped_insufficient_identities": skipped,
        "enrollment_images": len(enrollment),
        "probe_images": len(probes),
        "source_features_sha256": model["source_features_sha256"],
        "split_manifest_sha256": model["split_manifest_sha256"],
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "incumbent_sha256": hashlib.sha256(incumbent_bytes).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "methods": score_data,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-build-", dir=output.parent) as temporary:
        build = Path(temporary)
        with (build / "transfer_split.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_name", "subject_id", "image_sha256", "role"))
            writer.writeheader()
            writer.writerows(split_manifest)
        _write_json(build / "summary.json", summary)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/palm_tuned/model.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("outputs/palm_trained/model.json"))
    parser.add_argument("--features", type=Path, default=Path("outputs/palm_evaluation/features.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/palm_evaluation/splits.csv"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/palm_evaluation/model.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/palm_transfer"))
    args = parser.parse_args()
    try:
        summary = evaluate_transfer(args.model, args.incumbent, args.features, args.splits,
                                    args.baseline, args.output)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Palm transfer evaluation error: {error}\n")
    print(f"Evaluated {summary['probe_images']} probe images from {summary['eligible_identity_count']} unseen identities.")
    print(f"Transfer outputs saved to {args.output}")


if __name__ == "__main__":
    main()
