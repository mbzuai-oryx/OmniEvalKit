import json
import os
from pathlib import Path


SYSTEM_PROMPT_CODE = (
    "You are a code generation evaluator. "
    "Write correct, executable code for the requested programming task."
)


def load_code_jsonl_dataset(dataset_name, data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / dataset_name
    manifest = root / "test.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    return [_normalize_code_sample(record) for record in _read_jsonl(manifest)]


def load_multiple_dataset(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / "multiple"
    config = os.getenv("MULTIPLE_CONFIG", "").strip()
    manifests = []
    if config:
        manifests = [root / config / "test.jsonl"]
    else:
        manifests = sorted(root.glob("*/test.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No MultiPL-E manifests found under {root}")

    samples = []
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
        samples.extend(_normalize_code_sample(record) for record in _read_jsonl(manifest))
    return samples


def build_code_prompt(question, options=None):
    return (
        f"{str(question or '').strip()}\n\n"
        "Return only the completed code. Do not include Markdown fences, explanations, or extra text."
    )


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _normalize_code_sample(record):
    return {
        "id": str(record.get("id") or record.get("task_id") or ""),
        "question": str(record.get("question") or record.get("prompt") or "").strip(),
        "options": [],
        "answer": str(record.get("answer") or record.get("reference") or "").strip(),
        "reference": str(record.get("reference") or record.get("answer") or "").strip(),
        "references": [str(record.get("reference") or record.get("answer") or "").strip()],
        "audio_path": "",
        "video_path": "",
        "image_path": "",
        **record,
    }
