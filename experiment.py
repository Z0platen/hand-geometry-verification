"""Record explicitly requested webcam verification attempts at a fixed threshold."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic_ns

import cv2
import mediapipe as mp

from trial_log import TrialLog
from verify import (
    DEFAULT_CAPTURES,
    DEFAULT_LANDMARKER,
    analyze_frame,
    decision_from_measurements,
    latest_personal_model,
    verification_preview,
)


DEFAULT_OUTPUT = Path(__file__).with_name("outputs") / "experiments"
WINDOW_NAME = "Hand Geometry Manual Trials"
MIN_INTERVAL_NS = 1_000_000_000


def rate_label(value):
    return "n/a" if value is None else f"{value:.2%}"


def experiment_preview(frame, hands, decision, model, summary, claim, actual, condition, feedback):
    preview = verification_preview(
        frame, hands, decision, model["method"], model=model, snapshot_enabled=False,
        max_frame_height=360,
    )
    height = preview.shape[0]
    preview = cv2.copyMakeBorder(
        preview, 0, 192, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    lines = [
        f"Declared actual: {actual}  |  Claimed: {claim}",
        f"Condition: {condition[:60]}  |  Fixed threshold: {model['threshold']:.5f}",
        f"Attempts: {summary['attempts']}  |  Valid acquisition: {summary['valid_acquisitions']}"
        f"  |  Acquisition failures: {summary['acquisition_failures']}",
        f"FMR: {rate_label(summary['fmr'])} ({summary['impostor_comparisons']} impostor)"
        f"  |  FNMR: {rate_label(summary['fnmr'])} ({summary['genuine_comparisons']} genuine)",
        "SPACE records one attempt. Live preview alone records nothing. Q / Esc exports and closes.",
        feedback,
        "Declared IDs define labels. No tuning; later independent sessions are still needed.",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            preview, text, (12, height + 24 + row * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return preview


def run_experiment(camera_index, detector, log, claim, actual, condition="standard"):
    """Count manual attempts, keeping acquisition failures outside match rates."""
    camera = None
    try:
        if log.model.get("model_kind") != "personal_demo":
            raise ValueError("The selected session is not a personal enrollment.")
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        last_timestamp = -1
        last_attempt_ns = None
        awaiting_removal = False
        feedback = "Press SPACE to record. Wait at least 1 second between attempts."
        summary = log.summary()
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera did not return a frame.")
            now_ns = monotonic_ns()
            last_timestamp = max(last_timestamp + 1, now_ns // 1_000_000)
            measurements, hands = analyze_frame(frame, detector, last_timestamp)
            decision = decision_from_measurements(measurements, claim, log.model)
            if awaiting_removal and measurements["status"] == "no_hand":
                awaiting_removal = False
                feedback = "Hand removed. Reposition, then press SPACE when ready."
            preview = experiment_preview(
                frame, hands, decision, log.model, summary, claim, actual, condition, feedback,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key != 32:
                continue
            if awaiting_removal:
                feedback = "Remove your hand completely before recording another attempt."
                continue
            if last_attempt_ns is not None and now_ns - last_attempt_ns < MIN_INTERVAL_NS:
                feedback = "Wait at least 1 second after the last recorded attempt."
                continue
            update = log.record(decision)
            if update.get("status") != "saved":
                raise RuntimeError("The trial logger did not save the requested attempt.")
            last_attempt_ns = now_ns
            awaiting_removal = decision["status"] == "ok"
            summary = log.summary()
            if awaiting_removal:
                feedback = f"Attempt {log.count} recorded. Remove your hand before the next attempt."
            else:
                feedback = f"Attempt {log.count}: acquisition failure recorded ({decision['status']})."
            if update.get("export_error"):
                feedback = f"Attempt {log.count} saved. CSV/summary export pending; close to retry."
                print(f"Trial saved; export will be retried: {update['export_error']}", file=sys.stderr, flush=True)
        return log.summary()
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                log.export()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Claimed ID from the latest completed local enrollment")
    parser.add_argument("--actual", required=True, help="Declared actual person ID; equality with --user means genuine")
    parser.add_argument("--condition", default="standard", help="Capture condition label for this run")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        model_path = latest_personal_model(args.user, args.captures)
        if not args.landmarker.is_file():
            raise FileNotFoundError(f"Hand landmarker model missing at {args.landmarker}.")
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.landmarker)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            log = TrialLog(args.output, model_path, args.user, args.actual, args.condition)
            print(f"Trial results directory: {log.run_dir}", flush=True)
            summary = run_experiment(args.camera, detector, log, args.user, args.actual, args.condition)
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Experiment error: {error}\n")
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
