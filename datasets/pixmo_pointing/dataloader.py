import ast
import json
import math
import re

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from datasets._vision_video_common import load_jsonl_dataset
from datasets.pixmo_pointing.prompt import coordinate_format


eval_type = "open"


def load_data(data_dir=None):
    return load_jsonl_dataset("pixmo_pointing", data_dir, manifest_name="test_clean.jsonl")


load_dataset = load_data


def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    native_points, parse_valid = _parse_points(prediction)
    image_width, image_height = _image_size(sample)
    point_format = coordinate_format(sample.get("model_name"))
    predicted_points = _to_pixel_points(native_points, point_format, image_width, image_height)
    ground_truth_points = _ground_truth_pixel_points(sample, image_width, image_height)
    normalized_predictions = _normalize_pixel_points(predicted_points, image_width, image_height)
    normalized_targets = _normalize_pixel_points(ground_truth_points, image_width, image_height)
    euclidean_distance_normalized = _minimum_average_distance(
        normalized_predictions,
        normalized_targets,
        math.sqrt(2.0),
        parse_valid,
    )
    euclidean_distance_pixels = _minimum_average_distance(
        predicted_points,
        ground_truth_points,
        math.hypot(max(image_width - 1, 1), max(image_height - 1, 1)),
        parse_valid,
    )

    precision_3px, recall_3px, f1_3px, correct_3px = _distance_scores(
        predicted_points, ground_truth_points, 3.0, parse_valid
    )
    precision_5px, recall_5px, f1_5px, correct_5px = _distance_scores(
        predicted_points, ground_truth_points, 5.0, parse_valid
    )

    masks = _load_masks(sample.get("mask_path"))
    mask_precision, mask_recall, mask_f1, mask_correct = _mask_scores(
        predicted_points, masks, image_width, image_height, parse_valid
    )
    result = dict(sample)
    result.update({
        "raw_prediction": prediction,
        "prediction": prediction,
        "prediction_parse_valid": parse_valid,
        "parsed_native_points": native_points,
        "coordinate_format": point_format,
        "predicted_points": predicted_points,
        "predicted_points_pixels": predicted_points,
        "ground_truth_points_pixels": ground_truth_points,
        "euclidean_distance_normalized": euclidean_distance_normalized,
        "euclidean_distance_pixels": euclidean_distance_pixels,
        "precision_3px": precision_3px,
        "recall_3px": recall_3px,
        "f1_3px": f1_3px,
        "correct_3px": correct_3px,
        "precision_5px": precision_5px,
        "recall_5px": recall_5px,
        "f1_5px": f1_5px,
        "correct_5px": correct_5px,
        "mask_precision": mask_precision,
        "mask_recall": mask_recall,
        "mask_f1": mask_f1,
        "mask_correct": mask_correct,
        "pointing_precision": mask_precision,
        "pointing_recall": mask_recall,
        "pointing_f1": mask_f1,
        "correct": mask_correct,
    })
    return result


def aggregate_scores(results, eval_type="open"):
    count = len(results)
    return {
        "n_samples": count,
        "accuracy_3px": _average_bool(results, "correct_3px"),
        "accuracy_5px": _average_bool(results, "correct_5px"),
        "avg_precision_3px": _average(results, "precision_3px"),
        "avg_recall_3px": _average(results, "recall_3px"),
        "avg_f1_3px": _average(results, "f1_3px"),
        "avg_precision_5px": _average(results, "precision_5px"),
        "avg_recall_5px": _average(results, "recall_5px"),
        "avg_f1_5px": _average(results, "f1_5px"),
        "avg_euclidean_distance_normalized": _average(results, "euclidean_distance_normalized"),
        "avg_euclidean_distance_pixels": _average(results, "euclidean_distance_pixels"),
        "accuracy": _average_bool(results, "mask_correct"),
        "mask_accuracy": _average_bool(results, "mask_correct"),
        "avg_mask_f1": _average(results, "mask_f1"),
        "avg_pointing_f1": _average(results, "mask_f1"),
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
    except (TypeError, ValueError):
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
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _image_size(sample):
    if sample.get("image_width") and sample.get("image_height"):
        return int(sample["image_width"]), int(sample["image_height"])
    with Image.open(sample["image_path"]) as image:
        return image.size


def _to_pixel_points(points, point_format, image_width, image_height):
    x_scale = max(image_width - 1, 1)
    y_scale = max(image_height - 1, 1)
    if point_format == "normalized_1000":
        return [[x * x_scale / 1000.0, y * y_scale / 1000.0] for x, y in points]
    if point_format == "normalized_1":
        return [[x * x_scale, y * y_scale] for x, y in points]
    return [[x, y] for x, y in points]


def _ground_truth_pixel_points(sample, image_width, image_height):
    x_scale = max(image_width - 1, 1) / 100.0
    y_scale = max(image_height - 1, 1) / 100.0
    return [
        [float(point["x"]) * x_scale, float(point["y"]) * y_scale]
        for point in sample.get("points", [])
    ]


def _normalize_pixel_points(points, image_width, image_height):
    x_scale = max(image_width - 1, 1)
    y_scale = max(image_height - 1, 1)
    return [[x / x_scale, y / y_scale] for x, y in points]


def _minimum_average_distance(points, targets, unmatched_penalty, parse_valid):
    if not parse_valid:
        return float(unmatched_penalty)
    if not points and not targets:
        return 0.0

    size = max(len(points), len(targets))
    costs = np.full((size, size), float(unmatched_penalty), dtype=np.float64)
    for point_idx, point in enumerate(points):
        for target_idx, target in enumerate(targets):
            costs[point_idx, target_idx] = math.dist(point, target)

    row_indices, column_indices = linear_sum_assignment(costs)
    return float(costs[row_indices, column_indices].sum() / size)


def _load_masks(path):
    if not path:
        return []
    with np.load(path) as archive:
        masks = [np.asarray(archive[key], dtype=bool) for key in archive.files]
    return [mask for mask in masks if np.any(mask)]


def _distance_scores(points, targets, threshold, parse_valid):
    edges = [
        [idx for idx, target in enumerate(targets) if math.dist(point, target) <= threshold]
        for point in points
    ]
    return _matching_scores(edges, len(points), len(targets), parse_valid)


def _mask_scores(points, masks, image_width, image_height, parse_valid):
    edges = []
    for x, y in points:
        matched_masks = []
        if not (0.0 <= x <= image_width - 1 and 0.0 <= y <= image_height - 1):
            edges.append(matched_masks)
            continue
        for idx, mask in enumerate(masks):
            height, width = mask.shape
            px = round(x * (width - 1) / max(image_width - 1, 1))
            py = round(y * (height - 1) / max(image_height - 1, 1))
            if bool(mask[py, px]):
                matched_masks.append(idx)
        edges.append(matched_masks)
    return _matching_scores(edges, len(points), len(masks), parse_valid)


def _matching_scores(edges, prediction_count, target_count, parse_valid):
    if not prediction_count and not target_count:
        value = 1.0 if parse_valid else 0.0
        return value, value, value, bool(parse_valid)
    if not prediction_count or not target_count or not parse_valid:
        return 0.0, 0.0, 0.0, False

    assignments = {}

    def assign(point_idx, visited):
        for mask_idx in edges[point_idx]:
            if mask_idx in visited:
                continue
            visited.add(mask_idx)
            if mask_idx not in assignments or assign(assignments[mask_idx], visited):
                assignments[mask_idx] = point_idx
                return True
        return False

    matches = sum(assign(point_idx, set()) for point_idx in range(prediction_count))
    precision = matches / prediction_count
    recall = matches / target_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correct = matches == prediction_count == target_count
    return precision, recall, f1, correct


def _average(results, key):
    return sum(float(result.get(key, 0.0)) for result in results) / len(results) if results else 0.0


def _average_bool(results, key):
    return sum(bool(result.get(key)) for result in results) / len(results) if results else 0.0
