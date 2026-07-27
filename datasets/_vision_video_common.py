import json
import re
from pathlib import Path

from utils.evaluate import normalize_text


SYSTEM_PROMPT_IMAGE_QA = (
    "You are a visual question answering evaluator. "
    "Use the provided image and answer the question concisely."
)

SYSTEM_PROMPT_DOC_QA = (
    "You are a document visual question answering evaluator. "
    "Read the provided image carefully and answer the question concisely."
)

SYSTEM_PROMPT_MATH = (
    "You are a visual math reasoning evaluator. "
    "Use the provided image when available and answer with the requested final answer format."
)

SYSTEM_PROMPT_VIDEO_QA = (
    "You are a video understanding evaluator. "
    "Use the provided video and answer with only the requested option letter."
)


def load_jsonl_dataset(dataset_name, data_dir=None, manifest_name="test.jsonl"):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / dataset_name
    manifest = root / manifest_name
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}. Run scripts/prepare_vlm_datasets.py {dataset_name} first.")

    samples = []
    with manifest.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            samples.append(_normalize(json.loads(line), root, dataset_name, idx))
    return samples


def build_open_prompt(question, options=None):
    return str(question).strip()


def build_mcq_prompt(question, options=None):
    question = str(question).strip()
    options = options or []
    if options:
        labels = [chr(ord("A") + idx) for idx in range(len(options))]
        choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
        return f"Question: {question}\n\nChoices:\n{choices}\n\nAnswer with only one option letter."
    return f"Question: {question}\n\nAnswer with only one option letter."


def compute_mcq_score(sample, prediction, eval_type="closed", llm_judge_correct=None):
    option_count = len(sample.get("options") or [])
    predicted = _extract_mcq_letter(prediction, option_count)
    expected = _extract_mcq_letter(sample.get("answer", ""), option_count)
    exact = normalize_text(prediction) == normalize_text(sample.get("answer", ""))
    result = dict(sample)
    result.update({
        "prediction": prediction,
        "predicted_answer": predicted or str(prediction).strip(),
        "option_letter_match": bool(predicted and expected and predicted == expected),
        "exact_match": exact,
        "correct": bool((predicted and expected and predicted == expected) or exact),
    })
    return result


def _extract_mcq_letter(value, option_count):
    text = str(value).strip()
    for pattern in (
        r"^\s*\(?([A-I])\)?(?:\s|[\.\):,;-]|$)",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-I])\)?\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if not option_count or ord(letter) - ord("A") < option_count:
                return letter
    return ""


def _normalize(record, root, dataset_name, idx):
    answer = _first(record, "answer", "reference", "label", "target") or ""
    options = _normalize_options(_first(record, "options", "choices", "candidates"))
    sample_id = _first(record, "id", "question_id", "questionId", "uid", "sample_id") or f"{dataset_name}_{idx:06d}"
    references = _first(record, "references", "answers")
    if references in (None, ""):
        references = [answer] if answer else []
    elif isinstance(references, str):
        references = [references]

    sample = dict(record)
    sample.update({
        "id": str(sample_id),
        "question": str(_first(record, "question", "prompt", "query") or "").strip(),
        "options": options,
        "answer": str(answer).strip(),
        "references": [str(item).strip() for item in references if str(item).strip()],
        "audio_path": _resolve_path(root, _first(record, "audio_path", "audio")),
        "video_path": _resolve_path(root, _first(record, "video_path", "video")),
        "image_path": _resolve_path(root, _first(record, "image_path", "image")),
        "mask_path": _resolve_path(root, record.get("mask_path")),
    })
    return sample


def _first(record, *keys):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(root, value):
    if value in (None, ""):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    for candidate in (root / path, root / "images" / path.name, root / "videos" / path.name):
        if candidate.exists():
            return str(candidate.resolve())
    return str((root / path).resolve())


def _normalize_options(options):
    if options in (None, ""):
        return []
    if isinstance(options, dict):
        options = [options[key] for key in sorted(options)]
    elif isinstance(options, str):
        options = re.split(r"\n|\|", options)
    return [_strip_option_prefix(option) for option in options if _strip_option_prefix(option)]


def _strip_option_prefix(option):
    text = "" if option is None else str(option).strip()
    return re.sub(r"^[A-I][\.\):]\s*", "", text, count=1)
