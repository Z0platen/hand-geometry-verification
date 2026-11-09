# Webcam Hand Geometry Verification

A Biometrics course project at Sapienza University of Rome. The goal is to verify a claimed identity from the geometry of an open hand captured by a standard webcam.

## Webcam pilot

The pilot shows hand landmarks, pixel lengths, and ratios to palm width in a live webcam window. It processes frames in memory for display.

Set up a fresh checkout on Windows with Python 3.12. Run these commands in a CMD terminal from the repository folder. In VS Code, select Command Prompt as the terminal profile.

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
curl.exe --fail --location --output hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Start the pilot with the following command.

```cmd
.venv\Scripts\python.exe pilot.py
```

Hold one open hand facing the camera with all fingertips visible. Move it closer and farther away while keeping the pose steady, then compare the pixel lengths and ratios. Press **Q** or **Esc** to close the window. If you have multiple cameras, use `--camera 1` to try the second one.

## Planned verification flow

1. Enroll a user by capturing several samples of the same hand in an open-palm pose.
2. Detect hand landmarks with MediaPipe and compute geometric features such as finger-length ratios and distances normalized by palm width.
3. Compare a new capture with the claimed user's enrolled template and display a similarity score and an accept or reject decision.

## Evaluation

- Compare pixel distances with scale-normalized features at different camera distances.
- Use separate capture sessions for enrollment and testing. Set the decision threshold with validation samples before evaluating the final test set.
- Report false accepts, false rejects, and captures where hand landmarks could not be detected.

## Project status

The webcam measurement pilot is the first milestone. Store local participant captures and biometric templates under `data/` or `captures/`, which Git ignores.

### Completed

- Added a webcam preview with 21 hand landmarks, pixel measurements, and ratios normalized by the distance between the index and little finger base joints.
- Added camera selection, missing-model and camera error messages, and Q/Esc exit controls.
- Pinned the tested MediaPipe and OpenCV dependencies and documented Windows CMD setup.
- Checked the downloaded 11K Hands archive against its metadata. Both contain 11,076 entries, including 2,813 right-palmar images from 188 people.
- Inspected a preliminary sample of 50 right-palmar images from 10 people. All 50 yielded in-frame landmarks needed for five finger measurements. The estimated wrist was outside the image in 31 cases, making palm-length measurements unsuitable for those images.

### Validation and next milestone

The checks so far cover landmark extraction, preview rendering, and the scale-invariance calculation. Identity verification performance will be evaluated after enrollment, matching, and threshold calibration are implemented. The live camera check is a manual step.

The next milestone is a reproducible dataset extractor that reads the ZIP and metadata, selects right-palmar images, and exports five finger measurements with their quality status. The same feature definition will be used for dataset and webcam comparisons. Session-based testing will require separate capture-session data.

Dataset source and citation are available on the [11K Hands project page](https://sites.google.com/view/11khands/home).

Reference: [MediaPipe Hand Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
