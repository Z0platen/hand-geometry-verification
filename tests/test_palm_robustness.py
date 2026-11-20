import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from palm_robustness import MODES, read_test_probes, run, validate_scales


class PalmRobustnessTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        archive, model_path, splits, landmarker = (root / name for name in ("photos.zip", "model.json", "splits.csv", "hand.task"))
        encoded = cv2.imencode(".jpg", np.full((80, 120, 3), 120, np.uint8))[1].tobytes()
        with ZipFile(archive, "w") as file:
            file.writestr("hands/test.jpg", encoded)
        with splits.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image_name", "subject_id", "image_sha256", "split"])
            writer.writeheader()
            writer.writerows([
                {"image_name": "enrollment.jpg", "subject_id": "a", "image_sha256": hashlib.sha256(b"enrollment").hexdigest(), "split": "enrollment"},
                {"image_name": "test.jpg", "subject_id": "a", "image_sha256": hashlib.sha256(encoded).hexdigest(), "split": "test"},
            ])
        model = {"schema_version": 2, "templates": {"a": {}, "b": {}}, "quality": {},
                 "thresholds": dict.fromkeys(MODES, 0.5),
                 "split_manifest_sha256": hashlib.sha256(splits.read_bytes()).hexdigest()}
        model_path.write_text(json.dumps(model), encoding="utf-8")
        landmarker.write_bytes(b"landmarker fixture")
        return archive, model_path, splits, landmarker, model

    def test_scales_require_unique_finite_downsampling_values(self):
        self.assertEqual(validate_scales([1, 0.5, 0.25]), [1.0, 0.5, 0.25])
        for values in ([], [0], [1.1], [-0.5], [float("nan")], [True], [0.5, 0.5], ["bad"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_scales(values)

    def test_only_test_role_is_selected_and_manifest_binding_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, splits, _, model = self.fixture(folder)
            self.assertEqual([row["image_name"] for row in read_test_probes(splits, model)], ["test.jpg"])
            splits.write_text(splits.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_test_probes(splits, model)

    def test_cross_role_duplicate_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, splits, _, model = self.fixture(folder)
            rows = list(csv.DictReader(splits.read_text().splitlines()))
            rows[0]["image_sha256"] = rows[1]["image_sha256"]
            with splits.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model["split_manifest_sha256"] = hashlib.sha256(splits.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "leak"):
                read_test_probes(splits, model)

    def test_full_images_are_resized_and_common_valid_denominators_match(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, model_path, splits, landmarker, model = self.fixture(folder)
            output = Path(folder) / "results"
            shapes = []

            def analyze(frame, detector, model):
                shapes.append(frame.shape[:2])
                palm_ok = frame.shape[0] == 80
                return {"status": "ok"}, {"status": "ok" if palm_ok else "palm_too_small", "features": [1] if palm_ok else None, "quality": {"native_roi_pixels": frame.shape[0]}}

            def score(geometry, palm, claim, model, geometry_status="ok", palm_status="ok"):
                modes = {}
                for mode in MODES:
                    valid = mode == "geometry" or palm_status == "ok"
                    distance = 0.1 if claim == "a" else 0.9
                    modes[mode] = {"status": "ok" if valid else "component_unavailable", "accepted": distance <= 0.5 if valid else None, "distance": distance if valid else None, "threshold": 0.5}
                return {"claimed_id": claim, "modes": modes}

            with patch("palm_robustness.load_model", return_value=model), \
                    patch("palm_robustness.PreparedMatcher") as matcher, \
                    patch("palm_robustness.mp.tasks.vision.HandLandmarker.create_from_options"), \
                    patch("palm_robustness.analyze_probe", side_effect=analyze):
                matcher.return_value.score_all.side_effect = lambda geometry, palm, **kwargs: {
                    claim: score(geometry, palm, claim, model, **kwargs) for claim in model["templates"]
                }
                summary = run(archive, model_path, splits, output, [1, 0.5], landmarker)
            self.assertEqual(shapes, [(80, 120), (40, 60)])
            half = summary["results"][1]["methods"]
            self.assertEqual(half["geometry"]["conditional_valid"]["genuine_comparisons"], 1)
            self.assertEqual(half["geometry"]["conditional_valid"]["impostor_comparisons"], 1)
            self.assertEqual(half["palm"]["conditional_valid"]["acquisition_failures"], 1)
            for method in half.values():
                self.assertEqual(method["common_valid"]["valid_probes"], 0)
                self.assertIsNone(method["common_valid"]["fmr"])
                self.assertIsNone(method["common_valid"]["fnmr"])
            self.assertEqual(summary["thresholds"], model["thresholds"])
            self.assertTrue((output / "resolution_comparison.png").is_file())
            with (output / "summary.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 12)

    def test_wrong_held_out_image_hash_aborts_without_partial_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, model_path, splits, landmarker, model = self.fixture(folder)
            with ZipFile(archive, "w") as file:
                file.writestr("test.jpg", b"changed source")
            output = Path(folder) / "results"
            with patch("palm_robustness.load_model", return_value=model), \
                    patch("palm_robustness.PreparedMatcher"), \
                    patch("palm_robustness.mp.tasks.vision.HandLandmarker.create_from_options"):
                with self.assertRaisesRegex(ValueError, "content differs"):
                    run(archive, model_path, splits, output, [1], landmarker)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(folder).glob("results-build-*")))


if __name__ == "__main__":
    unittest.main()
