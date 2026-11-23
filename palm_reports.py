"""Compile durable manual palm-trial JSON logs into scoped fusion reports."""

import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import secrets
from time import strftime, gmtime


MODES = ("geometry", "palm", "fusion")
QUALITY_FIELDS = frozenset(("native_roi_pixels", "sharpness", "contrast"))
IDENTITY_RE = re.compile(r"[A-Za-z0-9_-]{1,32}\Z", re.ASCII)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
RUN_FIELDS = frozenset((
    "schema_version", "model_schema_version", "created_at_utc", "claimed_id", "actual_id",
    "distance_cm", "distance_source", "input_scale", "input_scale_source", "condition",
    "model_sha256", "thresholds", "protocol",
))
TRIAL_FIELDS = frozenset((
    "trial_index", "timestamp_utc", "actual_id", "claimed_id", "genuine", "distance_cm",
    "input_scale", "condition", "model_sha256", "modes", "quality",
))
DECISION_FIELDS = frozenset(("status", "accepted", "distance", "threshold"))


class _InvalidRecord(ValueError):
    pass


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidRecord(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise _InvalidRecord(f"non-finite JSON number {value}")


def _read_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_pairs_without_duplicates,
                             parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _InvalidRecord(f"cannot read {path.name}: {error}") from error


def _is_link_or_junction(path):
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (is_junction is not None and is_junction())


def _within_root(path, root):
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidRecord(f"{label} must be a finite numeric value")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise _InvalidRecord(f"{label} must be a finite numeric value") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise _InvalidRecord(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _identity(value, label):
    if not isinstance(value, str) or IDENTITY_RE.fullmatch(value) is None:
        raise _InvalidRecord(f"{label} must be 1–32 ASCII letters, digits, underscores, or hyphens")
    return value


def _timestamp(value, label):
    if not isinstance(value, str) or not value:
        raise _InvalidRecord(f"{label} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _InvalidRecord(f"{label} must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _InvalidRecord(f"{label} must be a timezone-aware ISO timestamp")


def _condition(value):
    if (not isinstance(value, str) or not value or len(value) > 80
            or value != value.strip() or not value.isprintable()):
        raise _InvalidRecord("condition must be a trimmed, printable label of 1–80 characters")
    return value


def _thresholds(value):
    if not isinstance(value, dict) or set(value) != set(MODES):
        raise _InvalidRecord("thresholds must define geometry, palm, and fusion exactly")
    return {mode: _number(value[mode], f"{mode} threshold") for mode in MODES}


def _validate_run(value):
    if not isinstance(value, dict) or set(value) != RUN_FIELDS:
        raise _InvalidRecord("run metadata has missing or unexpected fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise _InvalidRecord("unsupported run schema version")
    if type(value["model_schema_version"]) is not int or value["model_schema_version"] != 2:
        raise _InvalidRecord("unsupported model schema version")
    _timestamp(value["created_at_utc"], "created_at_utc")
    claim = _identity(value["claimed_id"], "claimed_id")
    actual = _identity(value["actual_id"], "actual_id")
    distance = value["distance_cm"]
    if distance is not None:
        distance = _number(distance, "distance_cm", positive=True)
    scale = _number(value["input_scale"], "input_scale")
    if not 0.1 <= scale <= 1:
        raise _InvalidRecord("input_scale must be between 0.1 and 1")
    condition = _condition(value["condition"])
    if value["distance_source"] != "operator_declared" or value["input_scale_source"] != "digital_full_frame_resize":
        raise _InvalidRecord("run acquisition source markers are incompatible")
    model_hash = value["model_sha256"]
    if not isinstance(model_hash, str) or SHA256_RE.fullmatch(model_hash) is None:
        raise _InvalidRecord("model_sha256 must be a lowercase 64-character SHA-256")
    if not isinstance(value["protocol"], str) or not value["protocol"].strip():
        raise _InvalidRecord("protocol must be a nonempty string")
    return {
        "claimed_id": claim, "actual_id": actual, "distance_cm": distance,
        "input_scale": scale, "condition": condition, "model_sha256": model_hash,
        "thresholds": _thresholds(value["thresholds"]),
    }


def _validate_trial(value, run, filename):
    if not isinstance(value, dict) or set(value) != TRIAL_FIELDS:
        raise _InvalidRecord(f"{filename} has missing or unexpected fields")
    index = value["trial_index"]
    if type(index) is not int or index < 1:
        raise _InvalidRecord(f"{filename} trial_index must be a positive integer")
    if filename != f"{index:06d}.json":
        raise _InvalidRecord(f"{filename} does not match trial_index {index}")
    _timestamp(value["timestamp_utc"], f"{filename} timestamp_utc")
    for key in ("claimed_id", "actual_id"):
        _identity(value[key], f"{filename} {key}")
        if value[key] != run[key]:
            raise _InvalidRecord(f"{filename} {key} differs from its run")
    if type(value["genuine"]) is not bool or value["genuine"] != (run["actual_id"] == run["claimed_id"]):
        raise _InvalidRecord(f"{filename} genuine flag does not match its declared identities")
    distance = value["distance_cm"]
    if distance is not None:
        distance = _number(distance, f"{filename} distance_cm", positive=True)
    scale = _number(value["input_scale"], f"{filename} input_scale")
    if distance != run["distance_cm"] or scale != run["input_scale"]:
        raise _InvalidRecord(f"{filename} acquisition settings differ from its run")
    if _condition(value["condition"]) != run["condition"]:
        raise _InvalidRecord(f"{filename} condition differs from its run")
    if value["model_sha256"] != run["model_sha256"]:
        raise _InvalidRecord(f"{filename} model_sha256 differs from its run")
    modes = value["modes"]
    if not isinstance(modes, dict) or set(modes) != set(MODES):
        raise _InvalidRecord(f"{filename} must contain all three mode decisions")
    for mode in MODES:
        result = modes[mode]
        if not isinstance(result, dict) or set(result) != DECISION_FIELDS:
            raise _InvalidRecord(f"{filename} {mode} decision has missing or unexpected fields")
        status = result["status"]
        if not isinstance(status, str) or not status or len(status) > 80 or not status.isprintable():
            raise _InvalidRecord(f"{filename} {mode} status must be a printable nonempty string")
        threshold = _number(result["threshold"], f"{filename} {mode} threshold")
        if threshold != run["thresholds"][mode]:
            raise _InvalidRecord(f"{filename} {mode} threshold differs from its run")
        if status == "ok":
            distance_value = _number(result["distance"], f"{filename} {mode} distance")
            accepted = result["accepted"]
            if distance_value < 0 or type(accepted) is not bool or accepted != (distance_value <= threshold):
                raise _InvalidRecord(f"{filename} {mode} acceptance disagrees with its distance and threshold")
        elif result["distance"] is not None or result["accepted"] is not None:
            raise _InvalidRecord(f"{filename} {mode} acquisition failure requires null decision values")
    quality = value["quality"]
    if not isinstance(quality, dict) or set(quality) - QUALITY_FIELDS:
        raise _InvalidRecord(f"{filename} quality contains unknown fields")
    for name, measurement in quality.items():
        if measurement is not None and _number(measurement, f"{filename} quality {name}") < 0:
            raise _InvalidRecord(f"{filename} quality {name} must be nonnegative")
    if all(modes[name]["status"] == "ok" for name in ("geometry", "palm")) != (modes["fusion"]["status"] == "ok"):
        raise _InvalidRecord(f"{filename} fusion availability disagrees with component availability")
    return {"trial_index": index, "genuine": value["genuine"], "fusion": modes["fusion"]}


def _load_run(run_dir):
    metadata_path = run_dir / "run.json"
    if _is_link_or_junction(metadata_path) or not metadata_path.is_file():
        raise _InvalidRecord("run.json is missing or is a symlink")
    run = _validate_run(_read_json(metadata_path))
    trials_dir = run_dir / "trials"
    if _is_link_or_junction(trials_dir) or not trials_dir.is_dir():
        raise _InvalidRecord("trials directory is missing or is a symlink")
    trials = []
    indices = set()
    for path in sorted(trials_dir.iterdir(), key=lambda item: item.name):
        if path.suffix != ".json":
            continue
        if _is_link_or_junction(path) or not path.is_file():
            raise _InvalidRecord(f"{path.name} is not a regular JSON file")
        record = _validate_trial(_read_json(path), run, path.name)
        if record["trial_index"] in indices:
            raise _InvalidRecord(f"duplicate trial_index {record['trial_index']}")
        indices.add(record["trial_index"])
        trials.append(record)
    if indices != set(range(1, len(indices) + 1)):
        raise _InvalidRecord("trial indices must be contiguous starting at 1")
    return run, trials


def _group_key(run):
    return (run["model_sha256"], tuple((mode, run["thresholds"][mode]) for mode in MODES),
            run["condition"], run["distance_cm"], run["input_scale"])


def _new_group(run):
    return {
        "model_sha256": run["model_sha256"], "condition": run["condition"],
        "distance_cm": run["distance_cm"], "input_scale": run["input_scale"],
        "thresholds": dict(run["thresholds"]), "run_ids": [], "claimed_ids": [], "actual_ids": [],
        "runs": 0, "attempts": 0, "genuine_attempts": 0, "impostor_attempts": 0,
        "valid_acquisitions": 0, "acquisition_failures": 0,
        "genuine_comparisons": 0, "impostor_comparisons": 0,
        "false_accepts": 0, "false_rejects": 0,
        "fmr": None, "fnmr": None, "acquisition_failure_rate": None, "coverage": None,
    }


def _finish_group(group):
    genuine, impostor = group["genuine_comparisons"], group["impostor_comparisons"]
    attempts = group["attempts"]
    group["fmr"] = group["false_accepts"] / impostor if impostor else None
    group["fnmr"] = group["false_rejects"] / genuine if genuine else None
    group["acquisition_failure_rate"] = group["acquisition_failures"] / attempts if attempts else None
    group["coverage"] = group["valid_acquisitions"] / attempts if attempts else None
    for key in ("run_ids", "claimed_ids", "actual_ids"):
        group[key] = sorted(set(group[key]))
    return group


def compile_reports(root):
    """Read and validate durable run/trial JSON, grouping comparable fusion attempts."""
    root_path = Path(root).expanduser()
    source_root = str(root_path.resolve())
    report = {"schema_version": 1, "source_root": source_root, "runs_seen": 0,
              "valid_runs": 0, "skipped_runs": 0, "warnings": [], "groups": []}
    if _is_link_or_junction(root_path) or not root_path.exists() or not root_path.is_dir():
        report["warnings"].append(f"Source root is missing or is not a directory: {source_root}")
        return report
    groups = {}
    hash_thresholds = {}
    try:
        candidates = sorted(root_path.iterdir(), key=lambda item: item.name)
    except OSError as error:
        report["warnings"].append(f"Could not read source root {source_root}: {error}")
        return report
    for run_dir in candidates:
        if run_dir.name.startswith(".palm-run-"):
            continue
        if _is_link_or_junction(run_dir):
            report["runs_seen"] += 1
            report["skipped_runs"] += 1
            report["warnings"].append(f"Skipped run {run_dir.name}: run directory is a symlink or junction")
            continue
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run.json"
        if not metadata_path.exists() and not _is_link_or_junction(metadata_path):
            continue
        report["runs_seen"] += 1
        try:
            if not _within_root(run_dir, root_path) or not _within_root(metadata_path, root_path):
                raise _InvalidRecord("run path resolves outside the source root")
            trials_path = run_dir / "trials"
            if not _within_root(trials_path, root_path):
                raise _InvalidRecord("trials path resolves outside the source root")
            run, trials = _load_run(run_dir)
        except (OSError, _InvalidRecord, ValueError) as error:
            report["skipped_runs"] += 1
            report["warnings"].append(f"Skipped run {run_dir.name}: {error}")
            continue
        report["valid_runs"] += 1
        hash_prior = hash_thresholds.setdefault(run["model_sha256"], set())
        threshold_key = tuple(run["thresholds"][mode] for mode in MODES)
        if hash_prior and threshold_key not in hash_prior:
            report["warnings"].append(
                f"Model {run['model_sha256']} appears with different thresholds; runs are reported separately."
            )
        hash_prior.add(threshold_key)
        key = _group_key(run)
        group = groups.setdefault(key, _new_group(run))
        group["runs"] += 1
        group["run_ids"].append(run_dir.name)
        group["claimed_ids"].append(run["claimed_id"])
        group["actual_ids"].append(run["actual_id"])
        for trial in trials:
            group["attempts"] += 1
            if trial["genuine"]:
                group["genuine_attempts"] += 1
            else:
                group["impostor_attempts"] += 1
            fusion = trial["fusion"]
            if fusion["status"] != "ok":
                group["acquisition_failures"] += 1
                continue
            group["valid_acquisitions"] += 1
            if trial["genuine"]:
                group["genuine_comparisons"] += 1
                if not fusion["accepted"]:
                    group["false_rejects"] += 1
            else:
                group["impostor_comparisons"] += 1
                if fusion["accepted"]:
                    group["false_accepts"] += 1
    report["groups"] = [_finish_group(group) for _, group in sorted(groups.items(), key=lambda item: repr(item[0]))]
    return report


def _display(value):
    return "n/a" if value is None else f"{value:.3f}"


def _percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_html(report):
    """Render a self-contained, escaped fusion-only summary page."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Expected a schema_version 1 palm trial report.")
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for group in report.get("groups", []):
        threshold_text = ", ".join(f"{name}: {value:g}" for name, value in group["thresholds"].items())
        details = (f"<details><summary>Model and thresholds</summary><code>{esc(group['model_sha256'])}</code>"
                   f"<br>{esc(threshold_text)}</details><details><summary>Run IDs ({len(group['run_ids'])})</summary>"
                   f"{esc(', '.join(group['run_ids']))}</details>")
        rows.append("<tr>" + "".join((
            f"<td>{details}</td>",
            f"<td>{esc(', '.join(group['claimed_ids']))}</td>",
            f"<td>{esc(', '.join(group['actual_ids']))}</td>",
            f"<td>{esc(group['condition'])}</td>",
            f"<td>{esc(_display(group['distance_cm']))}</td>",
            f"<td>{esc(_display(group['input_scale']))}</td>",
            f"<td>{group['runs']}</td>",
            f"<td>{group['attempts']}<br>G {group['genuine_attempts']} / I {group['impostor_attempts']}</td>",
            f"<td>{group['valid_acquisitions']} valid<br>{group['acquisition_failures']} failed<br>{esc(_percent(group['coverage']))} coverage</td>",
            f"<td>G {group['genuine_comparisons']} / I {group['impostor_comparisons']}</td>",
            f"<td>FA {group['false_accepts']} / FR {group['false_rejects']}</td>",
            f"<td>{esc(_percent(group['fmr']))}</td><td>{esc(_percent(group['fnmr']))}</td>",
            f"<td>{esc(_percent(group['acquisition_failure_rate']))}</td>",
        )) + "</tr>")
    headers = (
        "Model, thresholds, and run IDs", "Claimed IDs", "Actual IDs", "Condition", "Distance cm",
        "Input scale", "Runs", "Attempts (genuine / impostor)", "Acquisitions", "Valid comparisons (genuine / impostor)",
        "False accepts / rejects", "Fusion FMR", "Fusion FNMR", "Acquisition failure rate",
    )
    header_html = "".join(f"<th scope=\"col\">{esc(title)}</th>" for title in headers)
    warning_html = "".join(f"<li>{esc(warning)}</li>" for warning in report.get("warnings", []))
    if not rows:
        rows.append(f"<tr><td colspan=\"{len(headers)}\">No valid trial runs found. Start a manual camera trial to create a run; rates remain n/a until there are valid genuine or impostor comparisons.</td></tr>")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Palm trial report</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:2rem;color:#17212b}h1{margin-bottom:.25rem}
.meta,.caveat{color:#465767} .table-wrap{overflow-x:auto;margin:1rem 0}table{border-collapse:collapse;font-size:.86rem;min-width:1500px}
th,td{border:1px solid #aab7c4;padding:.45rem .55rem;text-align:left;vertical-align:top}th{background:#edf2f6;position:sticky;top:0}
td:first-child{font-family:ui-monospace,monospace;overflow-wrap:anywhere;max-width:16rem}ul{padding-left:1.4rem}
</style></head><body>
<h1>Palm trial report</h1>
<p class="meta">Source: <code>""" + esc(report.get("source_root", "")) + """</code><br>
Runs seen: """ + str(report.get("runs_seen", 0)) + """ · Valid: """ + str(report.get("valid_runs", 0)) + """ · Skipped: """ + str(report.get("skipped_runs", 0)) + """</p>
<h2>Fusion results by run configuration</h2><div class="table-wrap"><table><thead><tr>""" + header_html + """</tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
<h2>Warnings</h2><ul>""" + (warning_html or "<li>None</li>") + """</ul>
<p class="caveat">These are descriptive, manually triggered attempts with operator-declared identities and conditions. Repeated attempts may be correlated and are not independent observations. Acquisition failures are shown separately and excluded from conditional FMR and FNMR denominators. Fusion FMR = false accepts / valid impostor comparisons. Fusion FNMR = false rejects / valid genuine comparisons. Each rate is n/a when its denominator is zero. These rates do not represent overall accuracy.</p>
</body></html>
"""


def write_html_report(report, path):
    """Write rendered HTML and return its destination path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(report), encoding="utf-8")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=project_root / "outputs" / "palm_trials",
                        help="Root containing durable PalmTrialLog run directories")
    parser.add_argument("--output", type=Path, help="HTML output path (default: unique file under outputs/reports)")
    args = parser.parse_args(argv)
    report = compile_reports(args.root)
    output = args.output
    if output is None:
        output = project_root / "outputs" / "reports" / f"palm-trials-{strftime('%Y%m%dT%H%M%SZ', gmtime())}-{secrets.token_hex(4)}.html"
    try:
        written = write_html_report(report, output)
    except OSError as error:
        parser.exit(1, f"Palm report error: {error}\n")
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
