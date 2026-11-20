"""Search calibration-only palm metric settings on a hash-bound saved split."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from evaluate import calibrate_threshold, error_curve, summarize_scores
from multimodal import MODES, build_templates, validate_model, validate_reference
from palm_learning import fit_metric, project_features
from prepare_palm import _calibration_scale, _write_json
from train_palm import _distances, _fusion_scores, _genuine, _read_bound_splits


BASELINE_DIR = Path("outputs/palm_evaluation")
INCUMBENT_PATH = Path("outputs/palm_trained/model.json")
SHRINKAGES = (0.01, 0.1, 0.5)
DIMENSIONS = (16, 32, 64, 96, 128)
PALM_WEIGHTS = (0.5, 0.75, 0.9, 0.95)
PROTOCOL = (
    "Regularized LDA settings and fusion weights were selected using calibration rows only. "
    "Metric fitting uses enrollment descriptors only; raw enrollment median templates are preserved. "
    "Thresholds and normalization scales are frozen before the historical exploratory test split is scored. "
    "That enrolled-user image split has unknown capture sessions and potentially correlated frames and pairs; "
    "it is not a fresh untouched holdout, and cross-session or webcam performance remains unverified."
)


def _exact_distances(rows, subjects, templates, metric=None, *, geometry=None):
    """Use per-pair math.dist, matching the live matcher even for near-identical vectors."""
    probes = [row["palm"] for row in rows]
    references = [templates[subject]["palm"] for subject in subjects]
    if metric is not None:
        probes = project_features(probes, metric).tolist()
        references = project_features(references, metric).tolist()
    palm = np.asarray([[math.dist(probe, reference) for reference in references] for probe in probes])
    if geometry is None:
        geometry = np.asarray([[math.dist(row["geometry"], templates[subject]["geometry"])
                                for subject in subjects] for row in rows])
    return {"geometry": geometry, "palm": palm}


def _operating_point(components, genuine, geometry_threshold, palm_weight):
    scales, details = {}, {}
    for component, distances in components.items():
        scales[f"{component}_scale"], details[component] = _calibration_scale(distances, genuine)
    fusion = {
        **scales,
        "weights": {"geometry": 1 - palm_weight, "palm": palm_weight},
        "normalization_source": "Calibration genuine-distance medians with fixed scale 1 if the median is zero.",
        "calibration_details": details,
    }
    fusion_scores = _fusion_scores(components, fusion)
    thresholds = {
        "geometry": geometry_threshold,
        "palm": calibrate_threshold(components["palm"], genuine),
        "fusion": calibrate_threshold(fusion_scores, genuine),
    }
    metrics = {mode: summarize_scores(components[mode] if mode != "fusion" else fusion_scores,
                                      genuine, thresholds[mode]) for mode in MODES}
    return thresholds, fusion, metrics


def _metric_variant(full_metric, dimension):
    actual = min(dimension, full_metric["output_size"])
    return {**full_metric, "output_size": actual,
            "projection": [row[:actual] for row in full_metric["projection"]]}


def _key(candidate):
    return (round(candidate["fusion_ber"], 14), candidate["fusion_fmr"],
            candidate["palm_ber"], candidate["dimension"], candidate["shrinkage"],
            abs(candidate["palm_weight"] - 0.5), candidate["palm_weight"])


def _candidate_record(name, shrinkage, dimension, palm_weight, metrics, *, is_incumbent=False):
    return {"candidate": name, "shrinkage": shrinkage, "dimension": dimension,
            "palm_weight": palm_weight, "fusion_ber": metrics["fusion"]["balanced_error"],
            "fusion_fmr": metrics["fusion"]["fmr"], "palm_ber": metrics["palm"]["balanced_error"],
            "incumbent": is_incumbent}


def _verify_incumbent(incumbent, baseline, baseline_hash, partitions):
    validate_model(incumbent)
    if "palm_metric" not in incumbent:
        raise ValueError("The incumbent must contain a learned palm metric.")
    if (incumbent.get("baseline_model_sha256") != baseline_hash
            or incumbent.get("source_features_sha256") != baseline.get("source_features_sha256")
            or incumbent.get("split_manifest_sha256") != baseline.get("split_manifest_sha256")):
        raise ValueError("The incumbent is not bound to this baseline and split.")
    templates = baseline["templates"]
    rebuilt = build_templates(partitions["enrollment"])
    counts = dict(Counter(row["subject_id"] for row in partitions["enrollment"]))
    if (incumbent["templates"] != baseline["templates"]
            or rebuilt.keys() != templates.keys()
            or counts != baseline.get("enrollment_counts")
            or incumbent.get("enrollment_counts") != baseline.get("enrollment_counts")
            or any(not np.allclose(rebuilt[s][k], templates[s][k], rtol=0, atol=1e-12)
                   for s in templates for k in ("geometry", "palm"))):
        raise ValueError("Baseline and incumbent templates must match raw enrollment medians.")


def optimize(features, splits, baseline_path, incumbent_path, output, *, shrinkages=SHRINKAGES,
             dimensions=DIMENSIONS, palm_weights=PALM_WEIGHTS, make_plot=True):
    output = Path(output).resolve()
    baseline_path, incumbent_path = Path(baseline_path), Path(incumbent_path)
    if baseline_path.is_dir():
        baseline_path /= "model.json"
    if incumbent_path.is_dir():
        incumbent_path /= "model.json"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty output directory for the tuned experiment.")
    if any(Path(p).resolve().is_relative_to(output) for p in (features, splits, baseline_path, incumbent_path)):
        raise ValueError("Training inputs must be outside the output directory.")
    for values, name, upper in ((shrinkages, "shrinkages", 1), (dimensions, "dimensions", 128),
                                (palm_weights, "palm weights", 1)):
        if not values or any(type(v) not in (int, float) or not 0 < v <= upper for v in values):
            raise ValueError(f"Invalid {name} grid.")
    if any(type(v) is not int for v in dimensions):
        raise ValueError("Invalid dimensions grid.")
    if any(weight >= 1 for weight in palm_weights):
        raise ValueError("Palm weights must leave a positive geometry weight.")
    baseline_bytes = baseline_path.read_bytes()
    baseline = validate_model(json.loads(baseline_bytes))
    if "palm_metric" in baseline:
        raise ValueError("The baseline must use the handcrafted palm descriptor.")
    baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
    partitions = _read_bound_splits(features, splits, baseline)
    incumbent = json.loads(incumbent_path.read_text(encoding="utf-8"))
    _verify_incumbent(incumbent, baseline, baseline_hash, partitions)

    templates, subjects = baseline["templates"], list(baseline["templates"])
    enrollment = partitions["enrollment"]
    calibration, test_rows = partitions["calibration"], partitions["test"]
    geometry_cal = np.asarray([[math.dist(row["geometry"], templates[s]["geometry"]) for s in subjects]
                               for row in calibration])
    genuine_cal = _genuine(calibration, subjects)
    incumbent_cal = _exact_distances(calibration, subjects, templates, incumbent["palm_metric"], geometry=geometry_cal)
    incumbent_cal["fusion"] = _fusion_scores(incumbent_cal, incumbent["fusion"])
    incumbent_cal_metrics = {mode: summarize_scores(incumbent_cal[mode], genuine_cal,
                                                    incumbent["thresholds"][mode]) for mode in MODES}
    records = [_candidate_record("incumbent", incumbent["palm_metric"]["shrinkage"],
                                 incumbent["palm_metric"]["output_size"], incumbent["fusion"]["weights"]["palm"],
                                 incumbent_cal_metrics, is_incumbent=True)]
    candidate_models = {"incumbent": incumbent}
    candidate_cal_metrics = {"incumbent": incumbent_cal_metrics}

    for alpha in shrinkages:
        print(f"Fitting enrollment metric for shrinkage {alpha:g}", flush=True)
        full = fit_metric([row["palm"] for row in enrollment], [row["subject_id"] for row in enrollment],
                          max_components=max(dimensions), shrinkage=alpha)
        seen_dims = set()
        for requested in dimensions:
            metric = _metric_variant(full, requested)
            effective = metric["output_size"]
            if effective in seen_dims:
                continue
            seen_dims.add(effective)
            calibration_components = _exact_distances(calibration, subjects, templates, metric, geometry=geometry_cal)
            for weight in palm_weights:
                thresholds, fusion, metrics = _operating_point(calibration_components, genuine_cal,
                                                                baseline["thresholds"]["geometry"], weight)
                name = f"a{alpha:g}_d{effective}_w{weight:g}"
                model = {**baseline, "palm_metric": {**metric, "training_split": "enrollment",
                                                      "numpy_version": np.__version__},
                         "thresholds": thresholds, "fusion": fusion, "evaluation_protocol": PROTOCOL,
                         "baseline_model_sha256": baseline_hash}
                records.append(_candidate_record(name, float(alpha), effective, float(weight), metrics))
                candidate_models[name] = model
                candidate_cal_metrics[name] = metrics

    incumbent_key = (round(records[0]["fusion_ber"], 14), records[0]["fusion_fmr"])
    grid = records[1:]
    selected_record = min(grid, key=_key)
    selected_key = (round(selected_record["fusion_ber"], 14), selected_record["fusion_fmr"])
    # The incumbent remains selected on ties, so the tuned model cannot regress this calibration objective.
    selected_name = selected_record["candidate"] if selected_key < incumbent_key else "incumbent"
    selected_model = candidate_models[selected_name]
    incumbent_sha256 = hashlib.sha256(incumbent_path.read_bytes()).hexdigest()
    selected_model["optimization"] = {
        "selection_phase": "calibration", "candidate": selected_name,
        "incumbent_model_sha256": incumbent_sha256,
        "bounded_grid": {"shrinkages": list(shrinkages), "dimensions": list(dimensions),
                         "palm_weights": list(palm_weights)},
        "calibration_strictly_improved": selected_key < incumbent_key,
    }
    selected_components = _exact_distances(calibration, subjects, templates, selected_model["palm_metric"], geometry=geometry_cal)
    selected_components["fusion"] = _fusion_scores(selected_components, selected_model["fusion"])

    # Thresholds and fusion are frozen above; only now score the historical exploratory test split.
    genuine_test = _genuine(test_rows, subjects)
    geometry_test = np.asarray([[math.dist(row["geometry"], templates[s]["geometry"]) for s in subjects]
                                for row in test_rows])
    test_components = {}
    for label, model in (("incumbent", incumbent), ("tuned", selected_model)):
        comp = _exact_distances(test_rows, subjects, templates, model["palm_metric"], geometry=geometry_test)
        comp["fusion"] = _fusion_scores(comp, model["fusion"])
        test_components[label] = comp
    methods = {mode: {label: {
        "calibration": summarize_scores(
            (incumbent_cal if label == "incumbent" else selected_components)[mode], genuine_cal,
            (incumbent if label == "incumbent" else selected_model)["thresholds"][mode]),
        "test": summarize_scores(test_components[label][mode], genuine_test, model["thresholds"][mode]),
    } for label, model in (("incumbent", incumbent), ("tuned", selected_model))} for mode in MODES}
    selected_public = {k: v for k, v in selected_model.items() if k not in ("templates", "enrollment_counts", "model_kind")}
    selected_reference = {**selected_public, "reference_kind": "palm_geometry_calibration"}
    validate_reference(selected_reference)
    summary = {
        "schema_version": 1, "protocol": PROTOCOL, "selection_phase": "calibration",
        "test_split_label": "Previously reported exploratory test split; not a fresh untouched holdout.",
        "source_features_sha256": baseline["source_features_sha256"],
        "split_manifest_sha256": baseline["split_manifest_sha256"], "baseline_model_sha256": baseline_hash,
        "incumbent_model_sha256": incumbent_sha256,
        "split_counts": {phase: len(rows) for phase, rows in partitions.items()},
        "grid": {"shrinkages": list(shrinkages), "dimensions": list(dimensions), "palm_weights": list(palm_weights)},
        "selection": {"candidate": selected_name,
                      "calibration_strictly_improved": selected_key < incumbent_key,
                      "incumbent_calibration_key": list(incumbent_key),
                      "best_grid_calibration_key": list(selected_key),
                      "tieguard": "Incumbent wins unless rounded calibration fusion BER then calibration FMR strictly improves.",
                      "key": ["rounded calibration fusion BER (14 decimals)", "calibration fusion FMR",
                              "calibration palm BER", "lower dimension", "lower shrinkage",
                              "closest palm weight to 0.5", "lower palm weight"]},
        "selected_model": {"shrinkage": selected_model["palm_metric"]["shrinkage"],
                           "dimension": selected_model["palm_metric"]["output_size"],
                           "weights": selected_model["fusion"]["weights"], "thresholds": selected_model["thresholds"]},
        "search_candidates": records, "methods": methods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-build-", dir=output.parent) as temp:
        build = Path(temp)
        _write_json(build / "model.json", selected_model)
        _write_json(build / "reference.json", selected_reference)
        _write_json(build / "summary.json", summary)
        with (build / "search.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        if make_plot:
            _plot_comparison(build / "comparison.png", methods, test_components, genuine_test, selected_name)
        if output.exists():
            output.rmdir()
        os.replace(build, output)
    return summary


def _plot_comparison(path, methods, distances, genuine, selected_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    try:
        for i, mode in enumerate(("palm", "fusion")):
            roc, bars = axes[i]
            for j, (variant, color, label) in enumerate((
                ("incumbent", "#566476", "Incumbent"), ("tuned", "#007A71", "Calibration-selected"),
            )):
                _, fmr, fnmr = error_curve(distances[variant][mode], genuine)
                point = methods[mode][variant]["test"]
                roc.plot(fmr, 1 - fnmr, color=color, label=label)
                roc.scatter([point["fmr"]], [1 - point["fnmr"]], color=color, zorder=3)
                bars.bar(np.arange(2) + 0.32 * (j - 0.5), [point["fmr"], point["fnmr"]],
                         width=0.32, color=color, label=label)
            roc.set(title=f"{mode.capitalize()} ROC", xlabel="False match rate", ylabel="True match rate", xlim=(0, 1), ylim=(0, 1.01))
            roc.legend(loc="lower right")
            bars.set(title=f"{mode.capitalize()} at frozen calibration thresholds", ylabel="Test error rate",
                     xticks=[0, 1], xticklabels=["False match", "False nonmatch"])
            bars.yaxis.set_major_formatter(PercentFormatter(xmax=1)); bars.grid(axis="y", alpha=0.2); bars.legend()
        fig.suptitle(f"Incumbent vs tuned palm model ({selected_name})")
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASELINE_DIR / "features.jsonl")
    parser.add_argument("--splits", type=Path, default=BASELINE_DIR / "splits.csv")
    parser.add_argument("--baseline", type=Path, default=BASELINE_DIR)
    parser.add_argument("--incumbent", type=Path, default=INCUMBENT_PATH)
    parser.add_argument("--output", type=Path, default=Path("outputs/palm_tuned"))
    args = parser.parse_args()
    try:
        summary = optimize(args.features, args.splits, args.baseline, args.incumbent, args.output)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Palm optimization error: {error}\n")
    print(f"Selected {summary['selection']['candidate']} using calibration-only evidence.", flush=True)
    print(f"Historical exploratory test comparison saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
