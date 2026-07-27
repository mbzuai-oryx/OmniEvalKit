import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from datasets.pointarena_counting.dataloader import aggregate_scores, compute_score
from datasets.pointarena_counting.prompt import build_prompt, coordinate_format


class PointArenaCountingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mask_path = self.root / "mask.png"

    def tearDown(self):
        self.temp_dir.cleanup()

    def sample(self, boxes, model_name="future_model", width=101, height=101):
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
        cv2.imwrite(str(self.mask_path), mask)
        return {
            "id": "sample",
            "mask_path": str(self.mask_path),
            "image_width": width,
            "image_height": height,
            "target_region_count": len(boxes),
            "model_name": model_name,
        }

    def test_exact_count_and_correct_pointing(self):
        result = compute_score(self.sample([(10, 10, 20, 20), (60, 60, 70, 70)]), "[[15,15],[65,65]]")
        self.assertTrue(result["count_correct"])
        self.assertTrue(result["pointing_correct"])
        self.assertEqual(result["pointing_f1"], 1.0)

    def test_correct_count_wrong_locations(self):
        result = compute_score(self.sample([(10, 10, 20, 20), (60, 60, 70, 70)]), "[[30,30],[40,40]]")
        self.assertTrue(result["count_correct"])
        self.assertFalse(result["pointing_correct"])
        self.assertEqual(result["pointing_f1"], 0.0)

    def test_wrong_count_but_one_valid_point(self):
        result = compute_score(self.sample([(10, 10, 20, 20), (60, 60, 70, 70)]), "[[15,15]]")
        self.assertFalse(result["count_correct"])
        self.assertFalse(result["pointing_correct"])
        self.assertEqual(result["pointing_precision"], 1.0)
        self.assertEqual(result["pointing_recall"], 0.5)

    def test_extra_points(self):
        result = compute_score(self.sample([(10, 10, 20, 20)]), "[[15,15],[80,80]]")
        self.assertEqual(result["count_absolute_error"], 1)
        self.assertEqual(result["pointing_precision"], 0.5)
        self.assertFalse(result["pointing_correct"])

    def test_missing_points(self):
        result = compute_score(self.sample([(10, 10, 20, 20)]), "[]")
        self.assertEqual(result["pointing_recall"], 0.0)
        self.assertFalse(result["count_correct"])

    def test_duplicate_points_on_one_target(self):
        result = compute_score(self.sample([(10, 10, 20, 20), (60, 60, 70, 70)]), "[[15,15],[16,16]]")
        self.assertTrue(result["count_correct"])
        self.assertEqual(result["pointing_precision"], 0.5)
        self.assertEqual(result["pointing_recall"], 0.5)
        self.assertFalse(result["pointing_correct"])

    def test_out_of_bounds_prediction(self):
        result = compute_score(self.sample([(10, 10, 20, 20)]), "[[-1,15]]")
        self.assertEqual(result["invalid_points"], [[-1.0, 15.0]])
        self.assertTrue(result["count_correct"])
        self.assertFalse(result["pointing_correct"])

    def test_each_model_coordinate_format(self):
        cases = (
            ("qwen3vl30b_whisper", "[[150,150]]", "normalized_1000"),
            ("qwen25vlomni_advance", "[[15,15]]", "pixels"),
            ("minicpmo45", "[[0.15,0.15]]", "normalized_1"),
            ("future_model", "[[15,15]]", "pixels"),
        )
        for model_name, prediction, expected_format in cases:
            with self.subTest(model_name=model_name):
                result = compute_score(self.sample([(10, 10, 20, 20)], model_name), prediction)
                self.assertEqual(result["coordinate_format"], expected_format)
                self.assertTrue(result["pointing_correct"])

    def test_mask_with_multiple_target_regions(self):
        boxes = [(5, 5, 15, 15), (40, 40, 50, 50), (80, 80, 90, 90)]
        result = compute_score(self.sample(boxes), "[[10,10],[45,45],[85,85]]")
        self.assertEqual(result["mask_region_count"], 3)
        self.assertEqual(result["target_region_count"], 3)
        self.assertTrue(result["pointing_correct"])

    def test_aggregate_metrics_and_prompts(self):
        sample = self.sample([(10, 10, 20, 20)])
        results = [compute_score(sample, "[[15,15]]"), compute_score(sample, "[[80,80]]")]
        summary = aggregate_scores(results)
        self.assertEqual(summary["count_accuracy"], 1.0)
        self.assertEqual(summary["count_mae"], 0.0)
        self.assertEqual(summary["pointing_accuracy"], 0.5)
        self.assertEqual(coordinate_format("qwen3vl30b"), "normalized_1000")
        self.assertEqual(coordinate_format("minicpmv45_whisper"), "normalized_1")
        self.assertIn("0 to 1000", build_prompt("Point", model_name="qwen3vl30b"))
        self.assertIn("pixel coordinates", build_prompt("Point", model_name="qwen25vlomni_advance"))


if __name__ == "__main__":
    unittest.main()
