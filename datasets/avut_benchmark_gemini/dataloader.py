import json
import re
from pathlib import Path


DATASET_NAME = "avut_benchmark_gemini"
ANNOTATION_FILES = ('annotation/avut_benchmark_gemini.jsonl', 'avut_benchmark_gemini.jsonl')
eval_type = "closed"


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / DATASET_NAME
    if not root.exists():
        raise FileNotFoundError(f"Run data/download/{DATASET_NAME}.sh first")

    manifests = _find_manifests(root)
    if not manifests:
        raise FileNotFoundError(f"Run data/download/{DATASET_NAME}.sh first")

    samples = []
    for manifest in manifests:
        for record in _read_records(manifest):
            if isinstance(record, dict):
                samples.append(_normalize_record(record, root, manifest.parent, len(samples)))
    return samples


load_dataset = load_data


def _find_manifests(root):
    for name in ANNOTATION_FILES:
        path = root / name
        if path.exists():
            return [path]
    return sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))


def _read_records(path):
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.extend(_as_records(json.loads(line)))
        return records

    with path.open("r", encoding="utf-8") as f:
        return _as_records(json.load(f))


def _as_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "items", "annotations", "questions", "records"):
            if key in data:
                return _as_records(data[key])
        values = list(data.values())
        if values and all(isinstance(value, dict) for value in values):
            return values
    return [data]


def _normalize_record(record, root, base_dir, idx):
    audio_value = _media_value(record, ("audio_path", "audio", "WavPath", "wav_path"), ("audio_paths_dict",))
    video_value = _media_value(record, ("video_path", "video", "VideoPath", "videoPath"), ("video_paths_dict",))
    image_value = _media_value(record, ("image_path", "image", "ImagePath", "imagePath"), ("image_paths_dict",))
    answer = _first(record, ("answer", "gt_answer", "GT_Answer", "Answer", "label", "target"))
    question = _first(record, ("question", "Question", "query", "prompt", "instruction"))
    qa_id = _first(record, ("qa_id", "id", "sample_id", "question_id", "uid"))
    video_id = _first(record, ("video_id",))
    sample_id = f"{video_id}_{qa_id}" if qa_id is not None and video_id is not None else qa_id or video_id or f"{DATASET_NAME}_{idx}"

    return {
        "id": str(sample_id),
        "question": "" if question is None else str(question).strip(),
        "options": _normalize_options(_first(record, ("options", "choices", "Choice", "Choices"))),
        "answer": "" if answer is None else str(answer).strip(),
        "video_path": _resolve_path(root, base_dir, video_value),
        "audio_path": _resolve_path(root, base_dir, audio_value),
        "image_path": _resolve_path(root, base_dir, image_value),
        "task_type": "" if record.get("task_type") is None else str(record.get("task_type")).strip(),
    }


def _first(record, keys):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _media_value(record, keys, dict_keys):
    value = _first(record, keys)
    if value in (None, ""):
        for key in dict_keys:
            value = record.get(key)
            if value not in (None, ""):
                break
    if isinstance(value, dict):
        for item in value.values():
            if item not in (None, ""):
                return item
        return None
    if isinstance(value, list):
        return next((item for item in value if item not in (None, "")), None)
    return value


def _normalize_options(options):
    if options in (None, ""):
        return []
    if isinstance(options, dict):
        options = [options[key] for key in sorted(options)]
    elif isinstance(options, str):
        stripped = options.strip()
        if stripped.startswith("["):
            try:
                options = json.loads(stripped)
            except json.JSONDecodeError:
                options = re.split(r"\n|\|", options)
        else:
            options = re.split(r"\n|\|", options)

    normalized = []
    for option in options:
        if isinstance(option, dict):
            option = _first(option, ("text", "value", "option", "answer"))
        text = _strip_option_prefix(option)
        if text:
            normalized.append(text)
    return normalized


def _strip_option_prefix(option):
    text = "" if option is None else str(option).strip()
    return re.sub(r"^[A-Z][\.\)]\s*", "", text, count=1)


def _resolve_path(root, base_dir, value):
    if value in (None, ""):
        return None
    path = Path(str(value).strip())
    if path.is_absolute():
        return str(path)
    for anchor in (base_dir, root, Path.cwd()):
        candidate = (anchor / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((root / path).resolve())
