"""Enroll and compare hand geometry, palm texture, and their calibrated fusion."""

import argparse
import json
import math
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
import numpy as np

from enrollment import validate_user_id
from geometry import RATIO_COLUMNS
from multimodal import load_model, load_reference, score_probe
from palm_enrollment import DEFAULT_CAPTURES, PalmEnrollment, latest_palm_model
from palm_trials import PalmTrialLog
from palmprint import analyze_palm
from palm_view import SAVE_BUTTON_BOUNDS, render_preview as palm_preview
from verify import CAPTURE_MESSAGES, DEFAULT_LANDMARKER, analyze_frame, snapshot_feedback


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "configs" / "palm_reference.json"
WINDOW_NAME = "Combined Hand Verification"


def analyze(frame, detector, timestamp=None, quality=None):
    geometry, hands = analyze_frame(frame, detector, timestamp)
    geometry["features"] = [geometry[name] for name in RATIO_COLUMNS] if geometry["status"] == "ok" else None
    palm = analyze_palm(frame, hands, quality=quality)
    palm["minimum_roi_pixels"] = (quality or {}).get("min_roi_pixels", 64)
    return geometry, palm, hands


def compare(geometry, palm, user_id, model):
    return score_probe(
        geometry["features"], palm["features"], user_id, model,
        geometry_status=geometry["status"], palm_status=palm["status"],
    )


def live_guidance(geometry, palm, phase, waiting_removal):
    if waiting_removal:
        return "Remove your hand completely, then show it again."
    if geometry["status"] != "ok":
        return CAPTURE_MESSAGES.get(geometry["status"], "Show one open right palm with every fingertip visible.")
    if palm["status"] == "palm_too_small":
        return f"Palm detail {palm['quality']['native_roi_pixels']:.0f}px. Move closer, keeping your whole hand in view."
    if palm["status"] == "palm_low_contrast":
        return "Improve palm lighting. Avoid strong shadows and glare."
    if palm["status"] != "ok":
        return f"Adjust the palm position: {palm['status']}."
    if phase == "enrolling":
        return "Ready. Press SPACE to save one personal reference image."
    return "Hand image ready. Live combined verification is running."




def run_camera(camera_index, detector, user_id, reference, model, captures_root=DEFAULT_CAPTURES,
               actual=None, distance_cm=None, condition="standard", scale=1.0, samples=10,
               camera_width=1280, camera_height=720):
    camera = cv2.VideoCapture(camera_index)
    log = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        session = None if model is not None else PalmEnrollment(user_id, reference, captures_root, samples)
        phase = "verifying" if model is not None else "enrolling"
        waiting_removal = False
        last_capture_ns = None
        timestamp = -1
        feedback = "Show one open right palm. Move closer if the palm input is too small."
        feedback_until_ns = 0
        show_details = False
        capture_clicked = False

        def on_mouse(event, x, y, _flags, _userdata):
            nonlocal capture_clicked
            left, top, right, bottom = SAVE_BUTTON_BOUNDS
            if (event == cv2.EVENT_LBUTTONUP and phase == "enrolling" and not waiting_removal
                    and left <= x <= right and top <= y <= bottom):
                capture_clicked = True

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1000, 730)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera did not return a frame.")
            if scale != 1:
                frame = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            now = monotonic_ns()
            timestamp = max(timestamp + 1, now // 1_000_000)
            geometry, palm, hands = analyze(frame, detector, timestamp, quality=(model or reference)["quality"])
            if waiting_removal and geometry["status"] == "no_hand":
                waiting_removal = False
                if phase == "remove_hand":
                    phase = "verifying"
                feedback = "Show the right palm again when ready."
            scores = compare(geometry, palm, user_id, model) if model is not None and phase == "verifying" else None
            guidance = live_guidance(geometry, palm, phase, waiting_removal)
            ready = geometry["status"] == palm["status"] == "ok" and not waiting_removal
            displayed_feedback = feedback if ready and now < feedback_until_ns else guidance
            display_phase = "remove_hand" if waiting_removal else phase
            enrollment_target = samples if phase in ("enrolling", "remove_hand") else 0
            preview = palm_preview(frame, hands, geometry, palm, scores, user_id, display_phase,
                                   session.count if session else 0, enrollment_target, displayed_feedback, actual, distance_cm, scale,
                                   show_details=show_details)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if capture_clicked:
                capture_clicked = False
                if phase == "enrolling" and not waiting_removal:
                    key = 32
            if key in (ord("d"), ord("D")):
                show_details = not show_details
            if key in (32, ord("s"), ord("S"), ord("t"), ord("T")):
                feedback_until_ns = now + 2_000_000_000
            if key in (ord("s"), ord("S")) and scores is not None:
                feedback = snapshot_feedback(preview, scores, model)
            trigger = (phase == "enrolling" and key == 32) or (phase == "verifying" and key in (ord("t"), ord("T")))
            if not trigger:
                continue
            if waiting_removal:
                feedback = "Remove your hand completely before the next sample or trial."
                continue
            if last_capture_ns is not None and now - last_capture_ns < 1_000_000_000:
                feedback = "Wait at least one second between samples or trials."
                continue
            if phase == "enrolling":
                status = session.add_sample(frame, geometry, palm)
                if status != "saved":
                    feedback = f"Sample not saved: {status}. Keep the palm clear and well lit."
                    continue
                last_capture_ns = now
                waiting_removal = True
                feedback = f"Saved {session.count}/{samples}. Remove your hand, then show it again."
                if session.count == samples:
                    path = session.finalize()
                    model = load_model(path)
                    phase = "remove_hand"
                    print(f"Palm profile saved at {path}", flush=True)
            elif actual is None:
                feedback = "Enter Actual person ID in the project menu before recording trials."
            else:
                if log is None:
                    log = PalmTrialLog(ROOT / "outputs" / "palm_trials", model, user_id, actual, distance_cm,
                                       condition, input_scale=scale)
                update = log.record(scores, palm["quality"])
                last_capture_ns = now
                waiting_removal = geometry["status"] == "ok" or palm["status"] == "ok"
                feedback = f"Trial {log.count} saved. Remove your hand before the next attempt."
                if update.get("export_error"):
                    feedback = "Trial saved. Report export pending. Close to retry."
    finally:
        try:
            camera.release()
            cv2.destroyAllWindows()
        finally:
            if log is not None:
                log.export()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--reenroll", action="store_true")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--actual")
    parser.add_argument("--distance-cm", type=float)
    parser.add_argument("--condition", default="standard")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize input digitally before detection (0.1 to 1.0)")
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        validate_user_id(args.user)
        if args.camera_width < 1 or args.camera_height < 1:
            raise ValueError("Camera width and height must be positive.")
        if args.actual is not None:
            validate_user_id(args.actual)
        if not math.isfinite(args.scale) or not 0.1 <= args.scale <= 1:
            raise ValueError("Input scale must be between 0.1 and 1.0.")
        if args.distance_cm is not None and (not math.isfinite(args.distance_cm) or args.distance_cm <= 0):
            raise ValueError("Enter a positive measured distance in centimeters.")
        if not 5 <= args.samples <= 30:
            raise ValueError("Collect between 5 and 30 samples.")
        if not args.condition.strip() or len(args.condition) > 80 or not args.condition.isprintable():
            raise ValueError("Enter a printable condition label of 1 to 80 characters.")
        model_path = args.model or (None if args.reenroll else latest_palm_model(args.user, args.captures))
        model = load_model(model_path) if model_path else None
        if model is not None and args.user not in model["templates"]:
            raise ValueError("The selected palm model does not contain this user.")
        if args.image is not None and model is None:
            raise ValueError("Enroll a palm profile first, or supply --model for image comparison.")
        reference = load_reference(args.reference) if model is None else None
        if not args.landmarker.is_file():
            raise FileNotFoundError("Download hand_landmarker.task using the setup instructions.")
        vision = mp.tasks.vision
        mode = vision.RunningMode.IMAGE if args.image is not None else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.landmarker)), running_mode=mode, num_hands=2)
        with vision.HandLandmarker.create_from_options(options) as detector:
            if args.image is None:
                run_camera(args.camera, detector, args.user, reference, model, args.captures,
                           args.actual, args.distance_cm, args.condition, args.scale, args.samples,
                           args.camera_width, args.camera_height)
            else:
                encoded = args.image.read_bytes()
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR) if encoded else None
                if frame is None:
                    raise ValueError("Could not decode the hand image.")
                if args.scale != 1:
                    frame = cv2.resize(frame, (max(1, round(frame.shape[1] * args.scale)), max(1, round(frame.shape[0] * args.scale))), interpolation=cv2.INTER_AREA)
                geometry, palm, hands = analyze(frame, detector, quality=model["quality"])
                scores = compare(geometry, palm, args.user, model)
                print(json.dumps(scores, indent=2, allow_nan=False))
                if args.show:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, 1000, 730)
                    show_details = False
                    feedback = "Photo comparison"
                    preview = palm_preview(frame, hands, geometry, palm, scores, args.user, "verifying", 0, 0,
                                           feedback, args.actual, args.distance_cm, args.scale, image_input=True,
                                           show_details=show_details)
                    try:
                        while True:
                            cv2.imshow(WINDOW_NAME, preview)
                            key = cv2.waitKey(50) & 0xFF
                            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                                break
                            if key in (ord("s"), ord("S")):
                                feedback = snapshot_feedback(preview, scores, model)
                            if key in (ord("d"), ord("D")):
                                show_details = not show_details
                            if key in (ord("s"), ord("S"), ord("d"), ord("D")):
                                preview = palm_preview(frame, hands, geometry, palm, scores, args.user, "verifying", 0, 0,
                                                       feedback, args.actual, args.distance_cm, args.scale, image_input=True,
                                                       show_details=show_details)
                    finally:
                        cv2.destroyAllWindows()
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Palm study error: {error}\n")
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
