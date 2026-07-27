import json
import re
from pathlib import Path


DATASET_NAME = "avhbench"
eval_type = "closed"
CAPTION_TASK = "AV Captioning"


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / DATASET_NAME
    if not root.exists():
        raise FileNotFoundError(f"Run data/download/{DATASET_NAME}.sh first")

    manifest = _find_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"No AVHBench QA json found under {root}")

    samples = []
    for idx, record in enumerate(_read_records(manifest)):
        if not isinstance(record, dict):
            continue
        task = str(record.get("task", "")).strip()
        label = str(record.get("label", "")).strip()
        if task == CAPTION_TASK or _normalize_yes_no(label) not in {"Yes", "No"}:
            continue
        samples.append(_normalize_record(record, root, idx))
    return samples


load_dataset = load_data


def _find_manifest(root):
    candidates = [
        root / "avhbench_qa.json",
        root / "QA.json",
        root / "qa.json",
        root / "annotations.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    json_files = [
        path
        for path in sorted(root.rglob("*.json"))
        if "download" not in path.parts and not path.name.endswith(".metadata")
    ]
    return json_files[0] if json_files else None


def _read_records(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "items", "annotations", "questions", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def _normalize_record(record, root, idx):
    video_id = str(record.get("video_id", "")).strip()
    task = str(record.get("task", "")).strip()
    return {
        "id": f"{video_id}_{idx}",
        "question": str(record.get("text", "")).strip(),
        "options": [],
        "answer": _normalize_yes_no(record.get("label", "")),
        "video_path": _resolve_video_path(root, video_id),
        "audio_path": None,
        "image_path": None,
        "video_id": video_id,
        "task_type": task,
    }


def _normalize_yes_no(value):
    text = str(value or "").strip()
    if re.fullmatch(r"yes", text, flags=re.IGNORECASE):
        return "Yes"
    if re.fullmatch(r"no", text, flags=re.IGNORECASE):
        return "No"
    return text


def _resolve_video_path(root, video_id):
    if not video_id:
        return None
    names = [video_id, f"{video_id}.mp4"]
    for name in names:
        path = Path(name)
        if path.suffix:
            candidates = [
                root / path,
                root / "videos" / path,
                root / "video" / path,
                root / "AVHBench" / path,
                root / "AVHBench" / "videos" / path,
            ]
        else:
            candidates = []
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

    matches = sorted(root.rglob(f"{video_id}.mp4"))
    return str(matches[0].resolve()) if matches else str((root / "videos" / f"{video_id}.mp4").resolve())
