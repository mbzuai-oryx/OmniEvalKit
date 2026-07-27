from datasets._vision_video_common import load_jsonl_dataset
import re


eval_type = "open"


def load_data(data_dir=None):
    return load_jsonl_dataset("pixmo_count", data_dir, manifest_name="test_clean.jsonl")


load_dataset = load_data


def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    match = re.fullmatch(r"\s*(-?\d+)\s*", str(prediction))
    predicted = int(match.group(1)) if match else None
    expected = int(sample.get("answer"))
    result = dict(sample)
    result.update({
        "prediction": prediction,
        "predicted_count": predicted,
        "valid_count": predicted is not None,
        "correct": predicted == expected,
    })
    return result


def aggregate_scores(results, eval_type="open"):
    count = len(results)
    return {
        "n_samples": count,
        "accuracy": sum(bool(result.get("correct")) for result in results) / count if count else 0.0,
        "valid_output_rate": sum(bool(result.get("valid_count")) for result in results) / count if count else 0.0,
        "avg_bleu1": 0.0,
        "avg_rouge_l": 0.0,
    }
