from datasets._vision_video_common import load_jsonl_dataset


eval_type = "closed"


def load_data(data_dir=None):
    return load_jsonl_dataset("mme", data_dir)


load_dataset = load_data


def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    expected = str(sample.get("answer", "")).strip().lower()
    predicted = str(prediction).strip().lower()
    result = dict(sample)
    result.update({
        "prediction": prediction,
        "predicted_answer": predicted,
        "correct": predicted == expected,
    })
    return result


def aggregate_scores(results, eval_type="open"):
    count = len(results)
    categories = {}
    for category in sorted({str(result.get("category", "")) for result in results}):
        rows = [result for result in results if str(result.get("category", "")) == category]
        pairs = {}
        for row in rows:
            pairs.setdefault(str(row.get("pair_id", row.get("id"))), []).append(bool(row.get("correct")))
        accuracy = sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0
        accuracy_plus = sum(len(values) == 2 and all(values) for values in pairs.values()) / len(pairs) if pairs else 0.0
        categories[category] = {
            "accuracy": accuracy,
            "accuracy_plus": accuracy_plus,
            "score": 100.0 * (accuracy + accuracy_plus),
        }
    return {
        "n_samples": count,
        "accuracy": sum(bool(result.get("correct")) for result in results) / count if count else 0.0,
        "mme_score": sum(value["score"] for value in categories.values()),
        "category_scores": categories,
        "avg_bleu1": 0.0,
        "avg_rouge_l": 0.0,
    }
