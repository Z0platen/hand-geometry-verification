# Webcam Hand Geometry Verification

A Biometrics course project at Sapienza University of Rome. The goal is to verify a claimed identity from the geometry of an open hand captured by a standard webcam.

## Prototype

1. Enroll a user by capturing several samples of the same hand in an open-palm pose.
2. Detect hand landmarks with MediaPipe and compute geometric features such as finger-length ratios and distances normalized by palm width.
3. Compare a new capture with the claimed user's enrolled template and display a similarity score and an accept or reject decision.

## Evaluation

- Compare pixel distances with scale-normalized features at different camera distances.
- Use separate capture sessions for enrollment and testing. Set the decision threshold with validation samples before evaluating the final test set.
- Report false accepts, false rejects, and captures where hand landmarks could not be detected.

## Project status

The project is at the planning stage. Store local participant captures and biometric templates under `data/` or `captures/`, which Git ignores.

Reference: [MediaPipe Hand Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
