import argparse
from itertools import pairwise
from math import dist
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp


MEASUREMENT_PATHS = (
    ("Thumb", (2, 3, 4)),
    ("Index", (5, 6, 7, 8)),
    ("Middle", (9, 10, 11, 12)),
    ("Ring", (13, 14, 15, 16)),
    ("Pinky", (17, 18, 19, 20)),
    ("Palm length", (0, 9)),
)
WINDOW_NAME = "Hand Geometry Pilot"
DEFAULT_MODEL = Path(__file__).with_name("hand_landmarker.task")


def measure_hand(points):
    """Measure paths in pixels and divide them by the palm width proxy."""
    palm_width = dist(points[5], points[17])
    if palm_width < 1:
        return []

    measurements = []
    for label, path in MEASUREMENT_PATHS:
        length = sum(dist(points[start], points[end]) for start, end in pairwise(path))
        measurements.append((label, length, length / palm_width))
    return measurements


def make_preview(frame, hands):
    height, width = frame.shape[:2]
    preview = cv2.copyMakeBorder(
        frame, 0, max(0, 240 - height), 0, 260, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )

    def label(text, x, y, scale=0.46):
        cv2.putText(
            preview, text, (width + x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (240, 240, 240), 1, cv2.LINE_AA,
        )

    label("Hand Geometry Pilot", 12, 27, 0.58)
    label("Q / Esc to close", 12, 224)
    if len(hands) != 1:
        label("Show one open hand.", 12, 64)
        return preview

    # x and y are normalized by different dimensions. Convert each axis to pixels.
    points = [(point.x * width, point.y * height) for point in hands[0]]
    if not all(0 <= x < width and 0 <= y < height for x, y in points):
        label("Keep the whole hand in frame.", 12, 64)
        return preview

    measurements = measure_hand(points)
    if not measurements:
        label("Move your hand closer.", 12, 64)
        return preview

    drawing_points = [(int(x), int(y)) for x, y in points]
    for edge in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(preview, drawing_points[edge.start], drawing_points[edge.end], (42, 200, 255), 2)
    for point in drawing_points:
        cv2.circle(preview, point, 4, (70, 240, 125), -1, cv2.LINE_AA)

    label("Feature", 12, 58)
    label("Pixels", 122, 58)
    label("Ratio", 198, 58)
    for row, (name, pixels, ratio) in enumerate(measurements):
        y = 83 + row * 21
        label(name, 12, y)
        label(f"{pixels:.1f}", 122, y)
        label(f"{ratio:.3f}", 198, y)

    return preview


def run_preview(camera_index, model_path):
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                last_timestamp = max(last_timestamp + 1, monotonic_ns() // 1_000_000)
                result = detector.detect_for_video(image, last_timestamp)
                cv2.imshow(WINDOW_NAME, make_preview(frame, result.hand_landmarks))

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Show webcam hand landmarks and geometry measurements.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MediaPipe .task model")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file missing: {args.model}. See the setup instructions in README.md.")

    try:
        run_preview(args.camera, args.model)
    except (RuntimeError, ValueError, cv2.error) as error:
        parser.exit(1, f"Pilot error: {error}\n")


if __name__ == "__main__":
    main()
