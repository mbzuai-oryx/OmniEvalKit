import os
import re
from pathlib import Path


DATA_ROOT_NAME = "videomathqa"


def load_videomathqa_split(parquet_name, id_key, data_dir=None):
    import pandas as pd

    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / DATA_ROOT_NAME
    if not root.exists():
        raise FileNotFoundError(f"Run data/download/{DATA_ROOT_NAME}.sh first")

    parquet_path = root / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing annotation parquet: {parquet_path}")

    samples = []
    for idx, record in enumerate(pd.read_parquet(parquet_path).to_dict(orient="records")):
        video_id = str(record.get("videoID", "")).strip()
        question_id = record.get("question_id", idx)
        sample_id = record.get(id_key) or question_id or idx
        question = str(record.get("question", "")).strip()
        subtitle = _subtitle_text(root, video_id)
        if subtitle:
            question = f"{question}\n\nVideo subtitle/transcript excerpt:\n{subtitle}"

        samples.append(
            {
                "id": f"{video_id}_{sample_id}",
                "question": question,
                "options": _normalize_options(record.get("options")),
                "answer": _normalize_answer(record.get("answer")),
                "video_path": _resolve_video_path(root, video_id),
                "audio_path": None,
                "image_path": None,
                "video_id": video_id,
                "question_id": str(question_id),
                "category": str(record.get("category", "")).strip(),
                "length": str(record.get("length", "")).strip(),
                "steps": str(record.get("steps", "") or ""),
            }
        )
    return samples


def _normalize_options(options):
    if options is None:
        return []
    if hasattr(options, "tolist"):
        options = options.tolist()
    if isinstance(options, str):
        options = [part for part in re.split(r"\n|\|", options) if part.strip()]
    normalized = []
    for option in options:
        text = str(option or "").strip()
        text = re.sub(r"^[A-Z][\.\)]\s*", "", text, count=1)
        if text:
            normalized.append(text)
    return normalized


def _normalize_answer(answer):
    text = str(answer or "").strip()
    match = re.search(r"([A-E])", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else text


def _resolve_video_path(root, video_id):
    for name in (f"{video_id}.mp4", video_id):
        path = Path(name)
        for anchor in (root / "videos", root / "video", root):
            candidate = anchor / path
            if candidate.exists():
                return str(candidate.resolve())
    return str((root / "videos" / f"{video_id}.mp4").resolve())


def _subtitle_text(root, video_id):
    if str(os.getenv("VIDEOMATHQA_INCLUDE_SUBTITLES", "True")).lower() not in {"1", "true", "yes", "y"}:
        return ""

    subtitle_path = root / "subtitles" / f"{video_id}.srt"
    if not subtitle_path.exists():
        return ""

    max_chars = int(os.getenv("VIDEOMATHQA_MAX_SUBTITLE_CHARS", "6000"))
    text = _parse_srt(subtitle_path)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[subtitle truncated]"
    return text


def _parse_srt(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = []
    seen = set()
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.isdigit() or "-->" in text:
            continue
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen:
            seen.add(text)
            lines.append(text)
    return "\n".join(lines)
