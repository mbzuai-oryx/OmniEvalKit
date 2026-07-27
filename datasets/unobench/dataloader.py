import json
import re
from pathlib import Path


DATASET_NAME = "unobench"
ANNOTATION_FILES = ('unobench.jsonl',)
# UNoBench mixes multiple-choice and open/free-form items; run_eval needs one mode, so use open while preserving options per sample.
eval_type = "open"


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
    found = [root / name for name in ANNOTATION_FILES if (root / name).exists()]
    return found or sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))


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
    question, embedded_options = _split_embedded_options(
        _first(record, ("question", "Question", "query", "prompt", "instruction"))
    )
    sample_id = _first(record, ("id", "sample_id", "question_id", "video_id", "uid")) or f"{DATASET_NAME}_{idx}"
    video_path = _resolve_path(root, base_dir, video_value)
    video_path = _drop_missing_paths(video_path)

    return {
        "id": str(sample_id),
        "question": "" if question is None else str(question).strip(),
        "options": _normalize_options(_first(record, ("options", "choices", "Choice", "Choices"))) or embedded_options,
        "answer": "" if answer is None else str(answer).strip(),
        "video_path": video_path,
        "audio_path": _resolve_path(root, base_dir, audio_value),
        "image_path": _resolve_path(root, base_dir, image_value),
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
        items = [item for item in value.values() if item not in (None, "")]
        if not items:
            return None
        return items[0] if len(items) == 1 else items
    if isinstance(value, list):
        items = [item for item in value if item not in (None, "")]
        if not items:
            return None
        return items[0] if len(items) == 1 else items
    return value


def _split_embedded_options(question):
    text = "" if question is None else str(question).strip()
    choice_re = re.compile(
        r"(?:^|\n)\s*([A-I])[\.\)]\s*(.*?)(?=(?:\n\s*[A-I][\.\)]\s*)|\n\s*(?:Please select|请从|请选择)|$)",
        re.DOTALL,
    )
    matches = list(choice_re.finditer(text))
    if len(matches) < 2:
        choice_re = re.compile(
            r"(?:^|\s)\(([A-I])\)\s*(.*?)(?=(?:\s+\([A-I]\)\s*)|$)",
            re.DOTALL,
        )
        matches = list(choice_re.finditer(text))
    if len(matches) < 2:
        return text, []

    options = [_strip_option_prefix(match.group(0)) for match in matches]
    question_text = text[: matches[0].start()].strip()
    tail = text[matches[-1].end() :].strip()
    tail = re.sub(r"^(?:Please select|请从|请选择).*", "", tail, flags=re.DOTALL).strip()
    if tail:
        question_text = f"{question_text}\n{tail}".strip()
    return question_text, [option for option in options if option]


def _normalize_options(options):
    if options in (None, ""):
        return []
    if isinstance(options, dict):
        options = [options[key] for key in sorted(options)]
    elif isinstance(options, str):
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
    return re.sub(r"^\(?[A-I]\)?[\.\)]\s+(?=.)", "", text, count=1)


def _resolve_path(root, base_dir, value):
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        resolved = [_resolve_path(root, base_dir, item) for item in value]
        return [path for path in resolved if path]
    path = Path(str(value).strip())
    if path.is_absolute():
        return str(path)
    for anchor in (base_dir, root, Path.cwd()):
        candidate = (anchor / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((root / path).resolve())


def _drop_missing_paths(value):
    if isinstance(value, list):
        existing = [path for path in value if Path(path).exists()]
        if not existing:
            return None
        return existing[0] if len(existing) == 1 else existing
    if value and not Path(value).exists():
        return None
    return value
