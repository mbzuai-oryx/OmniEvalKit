import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.pixmo_pointing.dataloader import _parse_points, aggregate_scores, compute_score
from datasets.pixmo_pointing.prompt import build_prompt, coordinate_format


class PixmoPointingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "image.png"
        Image.new("RGB", (101, 101)).save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def sample(self, points, masks, model_name="other_model"):
        mask_path = self.root / "masks.npz"
        np.savez_compressed(mask_path, **{f"mask_{idx}": mask for idx, mask in enumerate(masks)})
        return {
            "id": "sample",
            "image_path": str(self.image_path),
            "mask_path": str(mask_path),
            "points": [{"x": x, "y": y} for x, y in points],
            "model_name": model_name,
        }

    @staticmethod
    def mask_at(x, y, shape=(101, 101)):
        mask = np.zeros(shape, dtype=bool)
        mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
        return mask

    def test_single_point(self):
        result = compute_score(self.sample([(20, 30)], [self.mask_at(20, 30)]), "[[22, 30]]")
        self.assertTrue(result["correct_3px"])
        self.assertTrue(result["correct_5px"])
        self.assertTrue(result["mask_correct"])
        self.assertAlmostEqual(result["euclidean_distance_pixels"], 2.0)
        self.assertAlmostEqual(result["euclidean_distance_normalized"], 0.02)

    def test_multiple_points(self):
        sample = self.sample([(20, 30), (70, 80)], [self.mask_at(20, 30), self.mask_at(70, 80)])
        result = compute_score(sample, "[(70, 80), (20, 30)]")
        self.assertTrue(result["correct_3px"])
        self.assertEqual(result["precision_3px"], 1.0)
        self.assertEqual(result["recall_3px"], 1.0)
        self.assertEqual(result["euclidean_distance_pixels"], 0.0)
        self.assertEqual(result["euclidean_distance_normalized"], 0.0)

    def test_extra_prediction(self):
        result = compute_score(self.sample([(20, 30)], [self.mask_at(20, 30)]), "[[20, 30], [80, 80]]")
        self.assertFalse(result["correct_5px"])
        self.assertEqual(result["precision_5px"], 0.5)
        self.assertAlmostEqual(result["euclidean_distance_pixels"], 100 * np.sqrt(2) / 2)
        self.assertAlmostEqual(result["euclidean_distance_normalized"], np.sqrt(2) / 2)

    def test_missing_prediction(self):
        sample = self.sample([(20, 30), (70, 80)], [self.mask_at(20, 30), self.mask_at(70, 80)])
        result = compute_score(sample, "[[20, 30]]")
        self.assertFalse(result["correct_5px"])
        self.assertEqual(result["recall_5px"], 0.5)
        self.assertAlmostEqual(result["euclidean_distance_pixels"], 100 * np.sqrt(2) / 2)
        self.assertAlmostEqual(result["euclidean_distance_normalized"], np.sqrt(2) / 2)

    def test_duplicate_prediction(self):
        sample = self.sample([(20, 30), (70, 80)], [self.mask_at(20, 30), self.mask_at(70, 80)])
        result = compute_score(sample, "[[20, 30], [20, 30]]")
        self.assertFalse(result["correct_5px"])
        self.assertEqual(result["precision_5px"], 0.5)

    def test_zero_point_sample(self):
        empty_mask = np.zeros((101, 101), dtype=bool)
        sample = self.sample([], [empty_mask])
        empty = compute_score(sample, "[]")
        extra = compute_score(sample, "[[10, 10]]")
        malformed = compute_score(sample, "No target is visible.")
        self.assertTrue(empty["correct_3px"])
        self.assertTrue(empty["mask_correct"])
        self.assertFalse(extra["correct_3px"])
        self.assertFalse(malformed["correct_3px"])

    def test_model_coordinate_formats(self):
        cases = [
            ("qwen3vl30b_whisper", "[[250, 750]]", "normalized_1000"),
            ("qwen25vlomni_advance", "[[25, 75]]", "pixels"),
            ("minicpmv45_whisper", "[[0.25, 0.75]]", "normalized_1"),
            ("future_model", "[[25, 75]]", "pixels"),
        ]
        for model_name, prediction, expected_format in cases:
            with self.subTest(model_name=model_name):
                sample = self.sample([(25, 75)], [self.mask_at(25, 75)], model_name=model_name)
                result = compute_score(sample, prediction)
                self.assertEqual(result["coordinate_format"], expected_format)
                self.assertTrue(result["correct_3px"])
                self.assertTrue(result["mask_correct"])
                self.assertEqual(result["euclidean_distance_pixels"], 0.0)
                self.assertEqual(result["euclidean_distance_normalized"], 0.0)

    def test_different_image_and_mask_resolutions(self):
        self.image_path.unlink()
        Image.new("RGB", (201, 101)).save(self.image_path)
        mask = np.zeros((11, 21), dtype=bool)
        mask[7:10, 4:7] = True
        sample = self.sample([(25, 75)], [mask])
        result = compute_score(sample, "[[50, 75]]")
        self.assertTrue(result["correct_3px"])
        self.assertTrue(result["mask_correct"])

    def test_robust_parser_and_malformed_output(self):
        valid_cases = (
            ("[(.5, -2e1), (3., +4E-1)]", [[0.5, -20.0], [3.0, 0.4]]),
            ('{"points": [{"x": 1, "y": 2.5}]}', [[1.0, 2.5]]),
            ('[{"point_2d": [1e1, 2e1]}]', [[10.0, 20.0]]),
        )
        for text, expected in valid_cases:
            with self.subTest(text=text):
                points, valid = _parse_points(text)
                self.assertTrue(valid)
                self.assertEqual(points, expected)
        for text in ("The points are 1, 2 and 3, 4", "[[1, 2, 3]]", "[1, 2, 3, 4]"):
            with self.subTest(text=text):
                self.assertEqual(_parse_points(text), ([], False))

    def test_minimum_distance_bipartite_matching(self):
        sample = self.sample([(20, 30), (70, 80)], [self.mask_at(20, 30), self.mask_at(70, 80)])
        result = compute_score(sample, "[[71, 80], [19, 30]]")
        self.assertAlmostEqual(result["euclidean_distance_pixels"], 1.0)
        self.assertAlmostEqual(result["euclidean_distance_normalized"], 0.01)

    def test_prompt_formats(self):
        self.assertEqual(coordinate_format("qwen3vl30b_whisper"), "normalized_1000")
        self.assertEqual(coordinate_format("qwen25vlomni_advance"), "pixels")
        self.assertEqual(coordinate_format("minicpmo45"), "normalized_1")
        self.assertIn("0 to 1000", build_prompt("Locate it", model_name="qwen3vl30b_whisper"))
        self.assertIn("0 to 1", build_prompt("Locate it", model_name="minicpmo45"))
        self.assertIn("pixel coordinates", build_prompt("Locate it", model_name="future_model"))

    def test_aggregate_metrics(self):
        sample = self.sample([(20, 30)], [self.mask_at(20, 30)])
        results = [compute_score(sample, "[[20, 30]]"), compute_score(sample, "[[24, 30]]")]
        summary = aggregate_scores(results)
        expected = {
            "accuracy_3px", "accuracy_5px",
            "avg_precision_3px", "avg_recall_3px", "avg_f1_3px",
            "avg_precision_5px", "avg_recall_5px", "avg_f1_5px",
            "avg_euclidean_distance_normalized", "avg_euclidean_distance_pixels",
            "mask_accuracy", "avg_mask_f1",
        }
        self.assertTrue(expected.issubset(summary))
        self.assertEqual(summary["accuracy_3px"], 0.5)
        self.assertEqual(summary["accuracy_5px"], 1.0)
        self.assertAlmostEqual(summary["avg_euclidean_distance_pixels"], 2.0)
        self.assertAlmostEqual(summary["avg_euclidean_distance_normalized"], 0.02)


if __name__ == "__main__":
    unittest.main()
