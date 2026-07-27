from datasets._vision_video_common import load_jsonl_dataset
from datasets.refcoco.prompt import coordinate_format
import re


eval_type = "open"


def load_data(data_dir=None):
    return load_jsonl_dataset("refcoco", data_dir)


load_dataset = load_data


def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    raw_predicted_box = _parse_box(prediction)
    predicted_box = _to_original_pixels(raw_predicted_box, sample)
    ground_truth_box = [float(value) for value in sample.get("bbox_xyxy", [])]
    iou = _box_iou(predicted_box, ground_truth_box) if predicted_box and len(ground_truth_box) == 4 else 0.0
    result = dict(sample)
    result.update({
        "prediction": prediction,
        "predicted_bbox_raw": raw_predicted_box,
        "predicted_bbox": predicted_box,
        "ground_truth_bbox": ground_truth_box,
        "iou": iou,
        "correct": iou >= 0.5,
    })
    return result


def aggregate_scores(results, eval_type="open"):
    count = len(results)
    return {
        "n_samples": count,
        "accuracy": sum(bool(result.get("correct")) for result in results) / count if count else 0.0,
        "avg_iou": sum(float(result.get("iou", 0.0)) for result in results) / count if count else 0.0,
        "avg_bleu1": 0.0,
        "avg_rouge_l": 0.0,
    }


def _parse_box(value):
    text = str(value)
    bracketed = re.search(r"\[([^\[\]]+)\]", text)
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        bracketed.group(1) if bracketed else text,
    )
    return [float(number) for number in numbers[:4]] if len(numbers) >= 4 else []


def _to_original_pixels(box, sample):
    if not box or coordinate_format(sample.get("model_name")) != "normalized_1000":
        return box
    width = float(sample["image_width"])
    height = float(sample["image_height"])
    return [
        box[0] * width / 1000.0,
        box[1] * height / 1000.0,
        box[2] * width / 1000.0,
        box[3] * height / 1000.0,
    ]


def _box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
