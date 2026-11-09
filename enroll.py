"""Capture a personal hand template and try it on later webcam frames."""

import argparse
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp

from enrollment import EnrollmentSession, validate_user_id
from matching import load_model
from pilot import make_preview
from verify import (
    CAPTURE_MESSAGES,
    DEFAULT_LANDMARKER,
    analyze_frame,
    decision_from_measurements,
    verification_preview,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "outputs" / "evaluation" / "model_ratios.json"
WINDOW_NAME = "Personal Hand Enrollment"


def enrollment_preview(frame, hands, measurements, count, target, feedback):
    height, width = frame.shape[:2]
    scale = min(1.0, 900 / width, 560 / height)
    if scale < 1:
        frame = cv2.resize(
            frame, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    preview = make_preview(
        frame, hands, title="Personal Enrollment", measurements=measurements,
    )
    image_height = preview.shape[0]
    preview = cv2.copyMakeBorder(
        preview, 0, 142, 0, max(0, 900 - preview.shape[1]),
        cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    lines = [
        f"Saved samples {count}/{target}",
        "SPACE saves one valid sample. Remove and reposition your hand between samples.",
        feedback,
        "Personal demo - threshold transferred from dataset; webcam not calibrated.",
        "Q / Esc closes the window.",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            preview, text, (12, image_height + 25 + row * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return preview


def run_enrollment(
    camera_index, detector, user_id, output_root, reference_model, sample_count,
    reference_model_path=None,
):
    camera = cv2.VideoCapture(camera_index)
    model_path = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1.")
        session = EnrollmentSession(
            user_id, output_root, reference_model, sample_count=sample_count,
            reference_model_path=reference_model_path,
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        phase = "enrolling"
        feedback = "Show your right palm facing the camera and press SPACE when ready."
        personal_model = None
        last_timestamp = -1
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera did not return a frame.")
            now_ns = monotonic_ns()
            last_timestamp = max(last_timestamp + 1, now_ns // 1_000_000)
            measurements, hands = analyze_frame(frame, detector, last_timestamp)

            if phase == "verifying":
                decision = decision_from_measurements(measurements, user_id, personal_model)
                preview = verification_preview(
                    frame, hands, decision, personal_model["method"], model=personal_model,
                )
            else:
                preview = enrollment_preview(
                    frame, hands, measurements, session.count, session.target, feedback,
                )
                if phase == "remove_hand" and measurements["status"] == "no_hand":
                    phase = "verifying"
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if phase == "enrolling" and key == 32:
                update = session.add_sample(frame, measurements, now_ns / 1_000_000_000)
                if update["status"] == "saved":
                    feedback = "Sample saved. Remove and reposition your hand before the next sample."
                    if update["count"] == update["target"]:
                        model_path = session.finalize()
                        personal_model = load_model(model_path)
                        print(f"Personal model saved at {model_path}", flush=True)
                        feedback = "Enrollment saved. Remove your hand, then show it again for verification."
                        phase = "remove_hand"
                elif update["status"] == "capture_too_soon":
                    feedback = "Wait a moment and reposition your hand before the next sample."
                else:
                    feedback = CAPTURE_MESSAGES.get(
                        update["status"], "Sample not saved. Adjust your hand and try again.",
                    )
        return model_path
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Local user ID for this enrollment")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10, help="Number of manual captures, from 5 to 30")
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--output", type=Path, default=ROOT / "captures")
    args = parser.parse_args()
    if not 5 <= args.samples <= 30:
        parser.error("--samples must be between 5 and 30.")

    try:
        validate_user_id(args.user)
        reference = load_model(args.reference_model)
        if reference["method"] != "ratios":
            raise ValueError("Personal enrollment requires a ratio reference model.")
        if not args.landmarker.is_file():
            raise FileNotFoundError(f"Hand landmarker model missing at {args.landmarker}.")
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.landmarker)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            run_enrollment(
                args.camera, detector, args.user, args.output, reference, args.samples,
                reference_model_path=args.reference_model,
            )
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Enrollment error: {error}\n")
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
