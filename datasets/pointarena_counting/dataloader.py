import ast
import json
import math
import re

import cv2
import numpy as np

from datasets._vision_video_common import load_jsonl_dataset
from datasets.pointarena_counting.prompt import coordinate_format


eval_type = "open"


def load_data(data_dir=None):
    return load_jsonl_dataset("pointarena_counting", data_dir)


load_dataset = load_data


def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    native_points, parse_valid = _parse_points(prediction)
    image_width = int(sample["image_width"])
    image_height = int(sample["image_height"])
    point_format = coordinate_format(sample.get("model_name"))
    pixel_points = _to_pixel_points(native_points, point_format, image_width, image_height)
    valid_points, invalid_points = _partition_points(pixel_points, image_width, image_height)

    labels, mask_region_count = _load_target_regions(sample["mask_path"])
    matched_regions = _matched_regions(valid_points, labels, image_width, image_height)
    predicted_count = len(native_points)
    target_region_count = int(sample["target_region_count"])
    matched_count = len(matched_regions)

    if predicted_count == 0 and mask_region_count == 0:
        pointing_precision = pointing_recall = pointing_f1 = 1.0 if parse_valid else 0.0
    else:
        pointing_precision = matched_count / predicted_count if predicted_count else 0.0
        pointing_recall = matched_count / mask_region_count if mask_region_count else 0.0
        pointing_f1 = (
            2 * pointing_precision * pointing_recall / (pointing_precision + pointing_recall)
            if pointing_precision + pointing_recall else 0.0
        )

    count_correct = predicted_count == target_region_count
    pointing_correct = bool(
        parse_valid
        and matched_count == predicted_count == mask_region_count
    )

    result = dict(sample)
    result.update({
        "raw_prediction": prediction,
        "prediction": prediction,
        "prediction_parse_valid": parse_valid,
        "coordinate_format": point_format,
        "parsed_native_points": native_points,
        "converted_pixel_points": pixel_points,
        "predicted_points": pixel_points,
        "invalid_points": invalid_points,
        "predicted_count": predicted_count,
        "target_region_count": target_region_count,
        "mask_region_count": mask_region_count,
        "count_correct": count_correct,
        "count_absolute_error": abs(predicted_count - target_region_count),
        "pointing_precision": pointing_precision,
        "pointing_recall": pointing_recall,
        "pointing_f1": pointing_f1,
        "pointing_correct": pointing_correct,
        "correct": pointing_correct,
    })
    return result


def aggregate_scores(results, eval_type="open"):
    return {
        "n_samples": len(results),
        "count_accuracy": _average_bool(results, "count_correct"),
        "count_mae": _average(results, "count_absolute_error"),
        "pointing_accuracy": _average_bool(results, "pointing_correct"),
        "avg_pointing_precision": _average(results, "pointing_precision"),
        "avg_pointing_recall": _average(results, "pointing_recall"),
        "avg_pointing_f1": _average(results, "pointing_f1"),
        "accuracy": _average_bool(results, "pointing_correct"),
        "avg_bleu1": 0.0,
        "avg_rouge_l": 0.0,
    }


def _parse_points(value):
    if isinstance(value, str):
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        if parsed is None:
            return [], False
    else:
        parsed = value

    try:
        return _coerce_points(parsed), True
    except (TypeError, ValueError, OverflowError):
        return [], False


def _coerce_points(value):
    if isinstance(value, dict):
        if "points" in value:
            value = value["points"]
        elif "point_2d" in value:
            value = [value["point_2d"]]
        elif "x" in value and "y" in value:
            value = [value]
        else:
            raise ValueError("Point object must contain points, point_2d, or x/y")
    if not isinstance(value, (list, tuple)):
        raise TypeError("Points must be a list or tuple")
    if not value:
        return []
    if len(value) == 2 and all(_is_number(item) for item in value):
        value = [value]

    points = []
    for item in value:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                pair = [item["x"], item["y"]]
            elif "point_2d" in item:
                pair = item["point_2d"]
            else:
                raise ValueError("Point object must contain x/y or point_2d")
        else:
            pair = item
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_is_number(number) for number in pair):
            raise ValueError("Each point must contain exactly two finite numbers")
        points.append([float(pair[0]), float(pair[1])])
    return points


def _is_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _to_pixel_points(points, point_format, image_width, image_height):
    x_scale = max(image_width - 1, 1)
    y_scale = max(image_height - 1, 1)
    if point_format == "normalized_1000":
        return [[x * x_scale / 1000.0, y * y_scale / 1000.0] for x, y in points]
    if point_format == "normalized_1":
        return [[x * x_scale, y * y_scale] for x, y in points]
    return [[x, y] for x, y in points]


def _partition_points(points, image_width, image_height):
    valid = []
    invalid = []
    for point in points:
        x, y = point
        (valid if 0 <= x <= image_width - 1 and 0 <= y <= image_height - 1 else invalid).append(point)
    return valid, invalid


def _load_target_regions(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read PointArena mask: {mask_path}")
    region_count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels, region_count - 1


def _matched_regions(points, labels, image_width, image_height):
    mask_height, mask_width = labels.shape
    matched = set()
    for x, y in points:
        mask_x = round(x * (mask_width - 1) / max(image_width - 1, 1))
        mask_y = round(y * (mask_height - 1) / max(image_height - 1, 1))
        region = int(labels[mask_y, mask_x])
        if region:
            matched.add(region)
    return matched


def _average(results, key):
    return sum(float(result.get(key, 0.0)) for result in results) / len(results) if results else 0.0


def _average_bool(results, key):
    return sum(bool(result.get(key)) for result in results) / len(results) if results else 0.0
