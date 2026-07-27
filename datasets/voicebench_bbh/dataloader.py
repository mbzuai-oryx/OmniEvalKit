import json
from pathlib import Path


eval_type = "closed"
_DATASET_NAME = "voicebench_bbh"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / _DATASET_NAME


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    manifest = root / "test.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"VoiceBench BBH manifest not found: {manifest}")

    samples = []
    with manifest.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            reference = _as_text(_coalesce(record, "reference", "answer"))
            sample_id = _coalesce(record, "name", "id", default=f"{_DATASET_NAME}_{idx}")
            samples.append(
                {
                    "id": str(sample_id),
                    "question": _as_text(_coalesce(record, "prompt", "question")),
                    "options": _normalize_options(_coalesce(record, "choices", "options", default=[])),
                    "answer": reference,
                    "audio_path": _resolve_audio_path(root, _coalesce(record, "WavPath", "audio_path")),
                    "video_path": "",
                    "image_path": "",
                    "reference": reference,
                }
            )
    return samples


def _coalesce(record, *keys, default=""):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(item) for item in value).strip()
    return str(value).strip()


def _normalize_options(options):
    if not options:
        return []
    if isinstance(options, dict):
        return [_as_text(value) for value in options.values()]
    if isinstance(options, (list, tuple)):
        return [_as_text(option) for option in options]
    return [_as_text(options)]


def _resolve_audio_path(root, wav_path):
    if not wav_path:
        return ""
    path = Path(str(wav_path))
    if path.exists():
        return str(path)
    candidate = root / "audio" / path.name
    if candidate.exists():
        return str(candidate.resolve())
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())
