import re
from collections import Counter


OPTION_RE = re.compile(r"(?:^|[^A-Za-z])([A-F])(?:[^A-Za-z]|$)", re.IGNORECASE)
OPTION_ANSWER_RE = re.compile(r"^\s*([A-I])[\.\)]\s+\S", re.IGNORECASE)
OPEN_ROUGE_L_CORRECT_THRESHOLD = 0.3


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _chartqa_normalize(text):
    return normalize_text(text).strip(" .,%")


def _first_number(text):
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def chartqa_relaxed_match(prediction, reference):
    pred_number = _first_number(prediction)
    ref_number = _first_number(reference)
    if pred_number is not None and ref_number is not None:
        if ref_number == 0:
            return abs(pred_number - ref_number) < 1e-6
        return abs(pred_number - ref_number) / abs(ref_number) <= 0.05
    return _chartqa_normalize(prediction) == _chartqa_normalize(reference)


def is_chartqa_sample(sample):
    return str(sample.get("id", "")).startswith("chartqa_")


def extract_option_letter(text):
    text = str(text).strip()
    for pattern in (
        r"^\s*([A-I])\s*$",
        r"^\s*([A-I])[\.\)]",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-I])\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = OPTION_RE.search(text)
    return match.group(1).upper() if match else ""


def extract_explicit_option_letter(text):
    text = str(text).strip()
    for pattern in (
        r"^\s*\(?([A-I])\)?\s*$",
        r"^\s*([A-I])[\.\)]",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-I])\)?\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def option_letter_match(prediction, ground_truth):
    predicted = extract_option_letter(prediction)
    expected = extract_option_letter(ground_truth)
    return bool(predicted and expected and predicted == expected)


def is_mcq_like_sample(sample, prediction=""):
    if sample.get("options"):
        return True
    return bool(OPTION_ANSWER_RE.match(str(sample.get("answer", ""))) and extract_explicit_option_letter(prediction))


def bleu1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            overlap += 1
            ref_counts[token] -= 1
    return overlap / len(pred_tokens)


def rouge_l(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    lengths = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        prev = 0
        for idx, ref_token in enumerate(ref_tokens, start=1):
            current = lengths[idx]
            if pred_token == ref_token:
                lengths[idx] = prev + 1
            else:
                lengths[idx] = max(lengths[idx], lengths[idx - 1])
            prev = current

    lcs = lengths[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reference_list(sample):
    references = sample.get("references")
    if references in (None, ""):
        references = []
    if isinstance(references, str):
        references = [references]
    elif not isinstance(references, (list, tuple)):
        references = []
    references = [str(reference).strip() for reference in references if str(reference).strip()]
    answer = str(sample.get("answer", "")).strip()
    if answer and normalize_text(answer) not in {normalize_text(reference) for reference in references}:
        references.insert(0, answer)
    return references or ([answer] if answer else [])


def best_open_reference_scores(prediction, references):
    if not references:
        return "", 0.0, 0.0, False
    best_reference = references[0]
    best_bleu = 0.0
    best_rouge = 0.0
    exact = False
    for reference in references:
        b1 = bleu1(prediction, reference)
        rl = rouge_l(prediction, reference)
        if normalize_text(prediction) == normalize_text(reference):
            exact = True
        if rl > best_rouge or (rl == best_rouge and b1 > best_bleu):
            best_reference = reference
            best_bleu = b1
            best_rouge = rl
    return best_reference, best_bleu, best_rouge, exact


def compute_score(sample, prediction, eval_type="closed", llm_judge_correct=None):
    result = dict(sample)
    result["prediction"] = prediction

    if eval_type == "closed":
        expected = extract_option_letter(sample.get("answer", ""))
        predicted = extract_option_letter(prediction)
        predicted_answer = predicted or str(prediction or "").strip()
        exact_match = normalize_text(prediction) == normalize_text(sample.get("answer", ""))
        letter_match = option_letter_match(prediction, sample.get("answer", ""))
        result.update(
            {
                "predicted_answer": predicted_answer,
                "option_letter_match": letter_match,
                "exact_match": exact_match,
                "correct": letter_match or exact_match,
            }
        )
        return result

    if eval_type == "open":
        references = reference_list(sample)
        reference, b1, rl, exact_match = best_open_reference_scores(prediction, references)
        chartqa_match = bool(is_chartqa_sample(sample) and any(chartqa_relaxed_match(prediction, ref) for ref in references))
        mcq_like = is_mcq_like_sample(sample, prediction)
        predicted = extract_explicit_option_letter(prediction) if mcq_like else ""
        predicted_answer = predicted or str(prediction or "").strip()
        expected_letters = [extract_option_letter(reference) for reference in references] if mcq_like else []
        letter_match = bool(predicted and predicted in expected_letters)
        if mcq_like:
            correct = letter_match or exact_match
        elif chartqa_match:
            correct = True
        elif llm_judge_correct is not None:
            correct = bool(llm_judge_correct)
        elif references:
            correct = exact_match or rl >= OPEN_ROUGE_L_CORRECT_THRESHOLD
        else:
            correct = False
        result.update(
            {
                "bleu1": b1,
                "rouge_l": rl,
                "matched_reference": reference,
                "predicted_answer": predicted_answer,
                "option_letter_match": letter_match if mcq_like else None,
                "llm_judge_correct": llm_judge_correct,
                "correct": correct,
            }
        )
        return result

    raise ValueError(f"Unsupported eval_type: {eval_type}")


def aggregate_scores(results, eval_type="closed"):
    n_samples = len(results)
    correct = sum(1 for result in results if result.get("correct"))
    summary = {
        "n_samples": n_samples,
        "accuracy": correct / n_samples if n_samples else 0.0,
        "avg_bleu1": 0.0,
        "avg_rouge_l": 0.0,
    }
    if eval_type == "open" and n_samples:
        summary["avg_bleu1"] = sum(result.get("bleu1", 0.0) for result in results) / n_samples
        summary["avg_rouge_l"] = sum(result.get("rouge_l", 0.0) for result in results) / n_samples
    return summary


evaluate_sample = compute_score
summarize_results = aggregate_scores
