"""Train a supervised palm metric on a saved enrollment/calibration/test split."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from evaluate import calibrate_threshold, error_curve, summarize_scores
from multimodal import MODES, build_templates, validate_model, validate_reference
from palm_learning import fit_metric, project_features
from prepare_palm import _calibration_scale, _write_json, pairwise_distances


BASELINE_DIR = Path("outputs/palm_evaluation")
PHASES = ("enrollment", "calibration", "test")
EXCLUDED_ROLES = ("insufficient_samples", "acquisition_failed")
MANIFEST_FIELDS = ("image_name", "subject_id", "image_sha256", "geometry_status", "palm_status")
PALM_WEIGHTS = (0.25, 0.5, 0.75)
FUSION_SELECTION = "Minimum calibration balanced error, then lower FMR, then closest to equal weights, then lower palm weight."
TRAINING_PROTOCOL = (
    "Regularized LDA uses enrollment labels and palm descriptors only. Templates are the same raw "
    "enrollment medians as the handcrafted baseline. Fusion scales, fusion weight selection and "
    "operating thresholds use calibration only. The test split evaluates the frozen models. "
    "This is an enrolled-user image split with unknown capture sessions and potentially correlated "
    "frames and comparison pairs. Cross-session and webcam performance remain unverified."
)


def _read_bound_splits(features_path, splits_path, baseline):
    features_bytes = Path(features_path).read_bytes()
    splits_bytes = Path(splits_path).read_bytes()
    for key, content in (("source_features_sha256", features_bytes), ("split_manifest_sha256", splits_bytes)):
        if hashlib.sha256(content).hexdigest() != baseline.get(key):
            raise ValueError(f"Input does not match the baseline {key}.")

    features, hashes = {}, set()
    for line in features_bytes.splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Each feature record must be an object.")
        name, subject, digest, geometry_status, palm_status = (record.get(key) for key in MANIFEST_FIELDS)
        if not all(isinstance(field, str) and field for field in (name, subject, geometry_status, palm_status)):
            raise ValueError("Feature rows require an image name, subject ID, and both acquisition statuses.")
        if not isinstance(digest, str) or name in features or (digest and digest in hashes):
            raise ValueError("Feature records must have unique image names and image hashes.")
        if digest:
            hashes.add(digest)
        features[name] = record

    splits = {phase: [] for phase in PHASES}
    seen = set()
    reader = csv.DictReader(io.StringIO(splits_bytes.decode("utf-8"), newline=""))
    if not reader.fieldnames or set(reader.fieldnames) != {*MANIFEST_FIELDS, "split"}:
        raise ValueError("The split manifest has incompatible columns.")
    for row in reader:
        name, phase = row["image_name"], row["split"]
        if name in seen or name not in features:
            raise ValueError("Each split-manifest image must have one feature record and one role.")
        seen.add(name)
        record = features[name]
        if any(row[key] != record[key] for key in MANIFEST_FIELDS):
            raise ValueError("The split manifest disagrees with its feature record.")
        common_valid = record["geometry_status"] == record["palm_status"] == "ok"
        if phase in splits and common_valid and row["image_sha256"]:
            splits[phase].append(record)
        elif phase not in EXCLUDED_ROLES or (phase == "insufficient_samples") != common_valid:
            raise ValueError("The split manifest contains an incompatible acquisition status or role.")
    if seen != features.keys():
        raise ValueError("The split manifest does not cover every feature record.")
    for phase, rows in splits.items():
        if {row["subject_id"] for row in rows} != baseline["templates"].keys():
            raise ValueError(f"The {phase} identities differ from the baseline enrollment cohort.")
    return splits


def _distances(rows, subjects, templates, metric=None, *, geometry=None):
    probes = [row["palm"] for row in rows]
    references = [templates[subject]["palm"] for subject in subjects]
    if metric is None:
        palm = pairwise_distances(probes, references)
    else:
        # Use the same Euclidean calculation as the live matcher at the operating threshold.
        projected_probes = project_features(probes, metric).tolist()
        projected_references = project_features(references, metric).tolist()
        palm = np.asarray([[math.dist(probe, reference) for reference in projected_references]
                           for probe in projected_probes])
    if geometry is None:
        geometry = pairwise_distances([row["geometry"] for row in rows],
                                      [templates[subject]["geometry"] for subject in subjects])
    return {"geometry": geometry, "palm": palm}


def _genuine(rows, subjects):
    identities = np.asarray([row["subject_id"] for row in rows])
    return identities[:, None] == np.asarray(subjects)[None, :]


def _fusion_scores(components, fusion):
    return sum(fusion["weights"][name] * components[name] / fusion[f"{name}_scale"]
               for name in ("geometry", "palm"))


def _fit_operating_point(components, genuine, geometry_threshold):
    scales, details = {}, {}
    for name, distances in components.items():
        scales[f"{name}_scale"], details[name] = _calibration_scale(distances, genuine)
    fusion = {
        **scales,
        "normalization_source": "Calibration genuine-distance medians with fixed scale 1 if the median is zero.",
        "calibration_details": details,
    }
    candidates = []
    for palm_weight in PALM_WEIGHTS:
        weights = {"geometry": 1 - palm_weight, "palm": palm_weight}
        scores = _fusion_scores(components, {**fusion, "weights": weights})
        threshold = calibrate_threshold(scores, genuine)
        metrics = summarize_scores(scores, genuine, threshold)
        candidates.append({"weights": weights, **metrics})

    def candidate_key(candidate):
        weight = candidate["weights"]["palm"]
        return round(candidate["balanced_error"], 14), candidate["fmr"], abs(weight - 0.5), weight

    selected = min(candidates, key=candidate_key)
    fusion.update({"weights": selected["weights"], "weight_selection": FUSION_SELECTION,
                   "candidate_palm_weights": list(PALM_WEIGHTS)})
    # Geometry retains the baseline threshold calibrated on these same validation probes.
    thresholds = {"geometry": geometry_threshold,
                  "palm": calibrate_threshold(components["palm"], genuine),
                  "fusion": selected["threshold"]}
    return thresholds, fusion, candidates


def fit_and_calibrate(enrollment, calibration, baseline):
    """Return the frozen learned model and calibration-only evidence."""
    templates = baseline["templates"]
    subjects = list(templates)
    rebuilt = build_templates(enrollment)
    counts = dict(Counter(row["subject_id"] for row in enrollment))
    if (rebuilt.keys() != templates.keys() or counts != baseline.get("enrollment_counts")
            or any(not np.allclose(rebuilt[subject][name], templates[subject][name], rtol=0, atol=1e-12)
                   for subject in subjects for name in ("geometry", "palm"))):
        raise ValueError("Baseline templates must match the split's raw enrollment medians.")

    metric = fit_metric([row["palm"] for row in enrollment], [row["subject_id"] for row in enrollment])
    metric.update({"training_split": "enrollment", "numpy_version": np.__version__})
    baseline_components = _distances(calibration, subjects, templates)
    learned_components = _distances(calibration, subjects, templates, metric, geometry=baseline_components["geometry"])
    genuine = _genuine(calibration, subjects)
    thresholds, fusion, candidates = _fit_operating_point(learned_components, genuine, baseline["thresholds"]["geometry"])
    learned = {**baseline, "palm_metric": metric, "thresholds": thresholds, "fusion": fusion,
               "evaluation_protocol": TRAINING_PROTOCOL}
    baseline_components["fusion"] = _fusion_scores(baseline_components, baseline["fusion"])
    learned_components["fusion"] = _fusion_scores(learned_components, fusion)
    return learned, genuine, {"baseline": baseline_components, "learned": learned_components}, candidates


def _plot_comparison(path, metrics, distances, genuine):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    try:
        for row, mode in enumerate(("palm", "fusion")):
            roc, bars = axes[row]
            for index, (variant, color, label) in enumerate((
                ("baseline", "#566476", "Handcrafted"),
                ("learned", "#007A71", "Trained LDA"),
            )):
                _, fmr, fnmr = error_curve(distances[variant][mode], genuine)
                actual = metrics[mode][variant]["test"]
                roc.plot(fmr, 1 - fnmr, color=color, label=label)
                roc.scatter(actual["fmr"], 1 - actual["fnmr"], color=color, zorder=3)
                bars.bar(np.arange(2) + 0.32 * (index - 0.5), [actual["fmr"], actual["fnmr"]],
                         width=0.32, color=color, label=label)
            roc.set(title=f"{mode.capitalize()} ROC", xlabel="False match rate", ylabel="True match rate",
                    xlim=(0, 1), ylim=(0, 1.01))
            roc.legend(loc="lower right")
            bars.set(title=f"{mode.capitalize()} at calibration thresholds", ylabel="Test error rate",
                     xticks=[0, 1], xticklabels=["False match", "False nonmatch"])
            bars.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            bars.grid(axis="y", alpha=0.2)
            bars.legend()
        figure.suptitle("11K Hands enrolled-user image split")
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def train(features, splits, baseline_path, output, *, make_plot=True):
    output = Path(output).resolve()
    baseline_path = Path(baseline_path)
    if baseline_path.is_dir():
        baseline_path /= "model.json"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty output directory for the trained experiment.")
    if any(Path(path).resolve().is_relative_to(output) for path in (features, splits, baseline_path)):
        raise ValueError("Training inputs must be outside the output directory.")
    source = baseline_path.read_bytes()
    baseline = validate_model(json.loads(source))
    if "palm_metric" in baseline:
        raise ValueError("The comparison baseline must use the handcrafted palm descriptor.")
    baseline_hash = hashlib.sha256(source).hexdigest()
    partitions = _read_bound_splits(features, splits, baseline)
    learned, calibration_genuine, calibration_distances, candidates = fit_and_calibrate(
        partitions["enrollment"], partitions["calibration"], baseline,
    )
    learned["baseline_model_sha256"] = baseline_hash
    validate_model(learned)
    reference = {key: value for key, value in learned.items()
                 if key not in ("model_kind", "templates", "enrollment_counts")}
    reference["reference_kind"] = "palm_geometry_calibration"
    validate_reference(reference)

    subjects, templates = list(baseline["templates"]), baseline["templates"]
    test_rows = partitions["test"]
    test_genuine = _genuine(test_rows, subjects)
    test_baseline = _distances(test_rows, subjects, templates)
    test_learned = _distances(test_rows, subjects, templates, learned["palm_metric"], geometry=test_baseline["geometry"])
    test_distances = {"baseline": test_baseline, "learned": test_learned}
    models = {"baseline": baseline, "learned": learned}
    for variant, distances in test_distances.items():
        distances["fusion"] = _fusion_scores(distances, models[variant]["fusion"])
    metrics = {
        mode: {variant: {
            "calibration": summarize_scores(calibration_distances[variant][mode], calibration_genuine, model["thresholds"][mode]),
            "test": summarize_scores(test_distances[variant][mode], test_genuine, model["thresholds"][mode]),
        } for variant, model in models.items()}
        for mode in MODES
    }
    summary = {
        "schema_version": 2,
        "protocol": TRAINING_PROTOCOL,
        "baseline_protocol": baseline.get("evaluation_protocol"),
        "baseline_model_sha256": baseline_hash,
        "source_features_sha256": baseline["source_features_sha256"],
        "split_manifest_sha256": baseline["split_manifest_sha256"],
        "split_counts": {phase: len(rows) for phase, rows in partitions.items()},
        "palm_metric": {key: value for key, value in learned["palm_metric"].items()
                        if key not in ("mean", "projection")},
        "fusion": learned["fusion"],
        "fusion_calibration_candidates": candidates,
        "methods": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-build-", dir=output.parent) as build_path:
        build = Path(build_path)
        _write_json(build / "model.json", learned)
        _write_json(build / "reference.json", reference)
        _write_json(build / "summary.json", summary)
        if make_plot:
            _plot_comparison(build / "comparison.png", metrics, test_distances, test_genuine)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASELINE_DIR / "features.jsonl")
    parser.add_argument("--splits", type=Path, default=BASELINE_DIR / "splits.csv")
    parser.add_argument("--baseline", type=Path, default=BASELINE_DIR, help="Handcrafted model JSON or its experiment directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/palm_trained"))
    args = parser.parse_args()
    try:
        summary = train(args.features, args.splits, args.baseline, args.output)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Palm training error: {error}\n")
    metric = summary["palm_metric"]
    print(f"Trained {metric['output_size']} dimensions on {metric['training_samples']} enrollment images "
          f"from {metric['training_classes']} users.")
    print(f"Calibration-selected fusion palm weight: {summary['fusion']['weights']['palm']:.2f}")
    for mode, variants in summary["methods"].items():
        for variant, results in variants.items():
            test = results["test"]
            print(f"{mode} ({variant}): test FMR {test['fmr']:.2%}, FNMR {test['fnmr']:.2%}")
    print(f"Training outputs saved to {args.output}")


if __name__ == "__main__":
    main()
