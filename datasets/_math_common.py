import json
import re
from pathlib import Path


SYSTEM_PROMPT_MATH = (
    "You are a math word problem evaluator. "
    "Solve the problem carefully and provide the final answer in the requested format."
)

FINAL_MARKER_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"-?(?:\d+(?:,\d{3})*|\d*?\.\d+)(?:/\d+(?:,\d{3})*(?:\.\d+)?)?")


def load_math_jsonl_dataset(dataset_name, data_dir=None, split="test"):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / dataset_name
    manifest = root / f"{split}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    return [_normalize_math_sample(record, root) for record in _read_jsonl(manifest)]


def build_gsm8k_prompt(question, options=None):
    return (
        f"Problem:\n{str(question or '').strip()}\n\n"
        "Solve the problem step by step. End your response with the final answer on its own line "
        "in this exact format:\n#### <answer>"
    )


def build_symbolic_math_prompt(question, options=None):
    return (
        f"Problem:\n{str(question or '').strip()}\n\n"
        "Solve the problem carefully. End your response with the final answer on its own line "
        "in this exact format:\n#### <answer>"
    )


def postprocess_gsm8k_prediction(prediction):
    return extract_final_answer(prediction)


def postprocess_symbolic_math_prediction(prediction):
    return extract_marked_or_boxed_answer(prediction)


def extract_marked_or_boxed_answer(text):
    text = str(text or "").strip()
    markers = FINAL_MARKER_RE.findall(text)
    if markers:
        return normalize_math_answer(markers[-1])
    boxed = extract_last_boxed(text)
    if boxed:
        return normalize_math_answer(boxed)
    return normalize_math_answer(text)


def extract_final_answer(text):
    text = str(text or "").strip()
    markers = FINAL_MARKER_RE.findall(text)
    if markers:
        return normalize_math_answer(markers[-1])
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_math_answer(numbers[-1])
    return normalize_math_answer(text)


def extract_last_boxed(text):
    token = r"\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(token), text)]
    for start in reversed(starts):
        idx = start + len(token)
        depth = 1
        chars = []
        while idx < len(text):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(char)
            idx += 1
    return ""


def normalize_math_answer(text):
    text = str(text or "").strip()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".")


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve_path(root, value):
    if value in (None, ""):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _normalize_math_sample(record, root):
    answer = normalize_math_answer(record.get("answer") or record.get("reference") or "")
    return {
        "id": str(record.get("id") or ""),
        "question": str(record.get("question") or record.get("prompt") or "").strip(),
        "options": [],
        "answer": answer,
        "reference": answer,
        "references": [answer],
        "audio_path": "",
        "video_path": "",
        "image_path": _resolve_path(root, record.get("image_path", "")),
        **record,
        "answer": answer,
        "reference": answer,
        "references": [answer],
        "image_path": _resolve_path(root, record.get("image_path", "")),
    }
