"""Verify one claimed identity using a calibrated hand geometry model."""

import argparse
import json
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
import numpy as np

from geometry import analyze_hand
from matching import load_model, verify_features
from pilot import make_preview


DEFAULT_LANDMARKER = Path(__file__).with_name("hand_landmarker.task")
WINDOW_NAME = "Hand Geometry Verification"
CAPTURE_MESSAGES = {
    "no_hand": "Show one right palm facing the camera.",
    "multiple_hands": "Show only one hand.",
    "invalid_landmarks": "Hold your hand steady and try again.",
    "finger_out_of_frame": "Keep all fingertips inside the frame.",
    "degenerate_palm": "Move closer and face your palm toward the camera.",
    "degenerate_finger": "Open your fingers and hold still.",
}


def decision_from_measurements(measurements, claim, model):
    """Keep acquisition failures separate from valid match or non-match decisions."""
    if claim not in model["templates"]:
        raise ValueError(f"Unknown claimed identity {claim!r}.")
    if measurements["status"] != "ok":
        return {
            "status": measurements["status"],
            "claimed_id": claim,
            "accepted": None,
            "distance": None,
            "threshold": model["threshold"],
        }
    return {"status": "ok", **verify_features(measurements, claim, model)}


def analyze_frame(frame, detector, timestamp_ms=None):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    if timestamp_ms is None:
        result = detector.detect(image)
    else:
        result = detector.detect_for_video(image, timestamp_ms)
    height, width = frame.shape[:2]
    measurements = analyze_hand(result.hand_landmarks, width, height)
    return measurements, result.hand_landmarks


def verify_image(image_path, detector, claim, model):
    _, _, decision = analyze_image(image_path, detector, claim, model)
    return decision


def analyze_image(image_path, detector, claim, model):
    frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not decode image {image_path}.")
    measurements, hands = analyze_frame(frame, detector)
    return frame, hands, decision_from_measurements(measurements, claim, model)


def verification_preview(frame, hands, decision, method, *, source="camera", model=None):
    original_height, original_width = frame.shape[:2]
    measurements = analyze_hand(hands, original_width, original_height)
    scale = min(1.0, 900 / original_width, 560 / original_height)
    if scale < 1:
        frame = cv2.resize(
            frame, (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    personal_demo = model is not None and model.get("model_kind") == "personal_demo"
    preview = make_preview(
        frame, hands,
        title="Personal Verification" if personal_demo else "Identity Verification",
        measurements=measurements,
    )
    height = preview.shape[0]
    preview = cv2.copyMakeBorder(
        preview, 0, 156, 0, max(0, 900 - preview.shape[1]),
        cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    method_label = "Finger lengths (pixels)" if method == "pixels" else "Finger / palm-width ratios"
    lines = [f"Claimed person: {decision['claimed_id'][:40]}  |  {method_label}"]
    color = (0, 220, 255)
    if decision["status"] == "ok":
        verdict = "ACCEPT" if decision["accepted"] else "REJECT"
        color = (70, 240, 125) if decision["accepted"] else (80, 100, 255)
        detail = f"Distance: {decision['distance']:.5f}   Threshold: {decision['threshold']:.5f}"
    else:
        verdict = "NO DECISION"
        detail = f"Capture status: {decision['status']}"
    cv2.putText(
        preview, verdict, (12, height + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        preview, detail, (210, height + 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA,
    )
    if source == "image":
        lines.append("Photo verification. Distance at or below threshold = ACCEPT.")
    else:
        lines.append(CAPTURE_MESSAGES.get(decision["status"], "Show your right palm facing the camera."))
    if personal_demo:
        lines.append("Personal demo: dataset threshold; webcam calibration pending.")
    else:
        lines.append("Exploratory scores using thresholds calibrated on dataset photos.")
    lines.append("Q / Esc closes the window.")
    for row, text in enumerate(lines):
        y = height + 23 if row == 0 else height + 71 + row * 21
        cv2.putText(
            preview, text, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )
    return preview


def show_image_verification(frame, hands, decision, model, *, image_name=""):
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if image_name:
            cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {image_name}")
        preview = verification_preview(frame, hands, decision, model["method"], source="image", model=model)
        cv2.resizeWindow(WINDOW_NAME, preview.shape[1], preview.shape[0])
        cv2.imshow(WINDOW_NAME, preview)
        while True:
            if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


def verify_camera(camera_index, detector, claim, model):
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        last_timestamp = -1
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera did not return a frame.")
            last_timestamp = max(last_timestamp + 1, monotonic_ns() // 1_000_000)
            measurements, hands = analyze_frame(frame, detector, last_timestamp)
            decision = decision_from_measurements(measurements, claim, model)
            cv2.imshow(
                WINDOW_NAME, verification_preview(frame, hands, decision, model["method"], model=model)
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Calibrated matcher JSON from evaluate.py")
    parser.add_argument("--claim", required=True, help="Enrolled subject ID to verify")
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--show", action="store_true", help="Display the image, landmarks, and verification decision")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Verify a single image and print the decision as JSON")
    source.add_argument("--camera", type=int, help="Verify live frames from an OpenCV camera index")
    args = parser.parse_args()
    if args.show and args.image is None:
        parser.error("--show requires --image; camera mode already displays a window.")

    try:
        model = load_model(args.model)
        if args.claim not in model["templates"]:
            raise ValueError(f"Unknown claimed identity {args.claim!r}.")
        if not args.landmarker.is_file():
            raise FileNotFoundError(f"Hand landmarker model missing at {args.landmarker}.")
        vision = mp.tasks.vision
        mode = vision.RunningMode.IMAGE if args.image is not None else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.landmarker)),
            running_mode=mode,
            num_hands=2,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            if args.image is not None:
                frame, hands, decision = analyze_image(args.image, detector, args.claim, model)
                print(json.dumps(decision, indent=2, allow_nan=False))
                if args.show:
                    show_image_verification(frame, hands, decision, model, image_name=args.image.name)
                return 0 if decision["status"] == "ok" else 2
            verify_camera(args.camera, detector, args.claim, model)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Verification error: {error}\n")
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
