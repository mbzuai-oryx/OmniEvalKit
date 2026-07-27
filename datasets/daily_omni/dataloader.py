import csv
import json
from pathlib import Path


eval_type = "closed"


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "daily_omni"
    if not root.exists():
        raise FileNotFoundError(
            f"Daily-Omni data directory not found: {root}. "
            "Pass --data_dir or place data under data/daily_omni."
        )

    manifest = _find_manifest(root)
    if manifest:
        records = _read_manifest(manifest)
        base_dir = manifest.parent
    else:
        records = _read_raw_daily_omni(root)
        base_dir = root

    return [_normalize_record(record, base_dir, idx) for idx, record in enumerate(records)]


def _find_manifest(root):
    for name in ("daily_omni.jsonl", "daily_omni.json", "daily_omni.csv"):
        path = root / name
        if path.exists():
            return path
    for suffix in ("*.jsonl", "*.json", "*.csv"):
        matches = sorted(root.glob(suffix))
        if matches:
            return matches[0]
    return None


def _read_manifest(path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "samples", "items"):
                if key in data:
                    return data[key]
        return data
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest type: {path}")


def _read_raw_daily_omni(root):
    videos_dir = root / "Videos"
    if not videos_dir.exists():
        return []
    records = []
    for qa_path in sorted(videos_dir.glob("*/QAs*.json")):
        records.extend(_read_json_array_fragments(qa_path))
    return records


def _read_json_array_fragments(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        pos = 0
        items = []
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            value, pos = decoder.raw_decode(text, pos)
            if isinstance(value, list):
                items.extend(value)
            else:
                items.append(value)
        return items


def _normalize_record(record, base_dir, idx):
    options = record.get("options") or record.get("choices") or record.get("Choice") or []
    if isinstance(options, str):
        options = [item.strip() for item in options.split("|") if item.strip()]
    options = [_strip_option_prefix(option) for option in options]

    video_path = record.get("video_path") or record.get("VideoPath") or ""
    audio_path = record.get("audio_path") or record.get("WavPath") or ""
    if not audio_path and video_path:
        audio_path = _sidecar_audio_path(base_dir / video_path)

    sample_id = record.get("id") or f"{record.get('video_id') or 'daily_omni'}_{idx}"
    return {
        "id": str(sample_id),
        "question": record.get("question") or record.get("Question") or "",
        "options": options,
        "answer": str(record.get("answer") or record.get("gt_answer") or record.get("Answer") or "").strip(),
        "video_path": _resolve_path(base_dir, video_path),
        "audio_path": _resolve_path(base_dir, audio_path),
    }


def _strip_option_prefix(option):
    option = str(option).strip()
    if len(option) > 2 and option[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and option[1] in {".", ")"}:
        return option[2:].strip()
    return option


def _resolve_path(base_dir, path):
    if not path:
        return ""
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _sidecar_audio_path(video_path):
    candidates = [video_path.with_suffix(".wav")]
    if video_path.stem.endswith("_video"):
        candidates.append(video_path.with_name(f"{video_path.stem[:-6]}_audio.wav"))
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(video_path.parents[2]))
            except ValueError:
                return str(candidate)
    return ""
