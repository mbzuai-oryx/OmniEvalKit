import json
import re
from pathlib import Path


eval_type = "closed"

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "voicebench_mmsu"
)
_LEGACY_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "voicebench"
    / "mmsu"
)


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else _default_data_dir()
    manifest = root / "test.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"VoiceBench MMSU manifest not found: {manifest}")

    samples = []
    with manifest.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            question, options = _parse_mmsu_prompt(record.get("prompt", ""))
            audio_path = _resolve_audio_path(root, record.get("WavPath", ""))
            sample_id = record.get("name") or f"voicebench_mmsu_{idx}"
            samples.append(
                {
                    "id": str(sample_id),
                    "question": question,
                    "options": options,
                    "answer": str(record.get("reference", "")).strip(),
                    "audio_path": audio_path,
                    "video_path": "",
                    "image_path": "",
                    "reference": str(record.get("reference", "")).strip(),
                    "raw_prompt": record.get("prompt", ""),
                }
            )
    return samples


def _default_data_dir():
    return _DEFAULT_DATA_DIR if (_DEFAULT_DATA_DIR / "test.jsonl").exists() else _LEGACY_DATA_DIR


def _parse_mmsu_prompt(prompt):
    prompt = re.sub(
        r"\n+What is the answer to the above multiple choice question\?.*$",
        "",
        prompt.strip(),
        flags=re.DOTALL,
    )
    question_lines = []
    options = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^[A-D]\.\s*(.*)$", line)
        if match:
            options.append(match.group(1).strip())
        elif options:
            options[-1] = f"{options[-1]} {line}"
        else:
            question_lines.append(line)
    return " ".join(question_lines).strip(), options


def _resolve_audio_path(root, wav_path):
    if not wav_path:
        return ""
    path = Path(wav_path)
    if path.exists():
        return str(path)
    candidate = root / "audio" / path.name
    if candidate.exists():
        return str(candidate.resolve())
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())
