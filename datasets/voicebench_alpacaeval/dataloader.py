import json
from pathlib import Path


eval_type = "open"

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "voicebench_alpacaeval"
)
_LEGACY_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "voicebench"
    / "alpacaeval"
)


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else _default_data_dir()
    manifest = root / "test.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"VoiceBench AlpacaEval manifest not found: {manifest}")

    samples = []
    with manifest.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = _resolve_audio_path(root, record.get("WavPath", ""))
            sample_id = record.get("name") or f"voicebench_alpacaeval_{idx}"
            samples.append(
                {
                    "id": str(sample_id),
                    "question": record.get("prompt", ""),
                    "options": [],
                    "answer": "",
                    "audio_path": audio_path,
                    "video_path": "",
                    "image_path": "",
                    "reference": "",
                }
            )
    return samples


def _default_data_dir():
    return _DEFAULT_DATA_DIR if (_DEFAULT_DATA_DIR / "test.jsonl").exists() else _LEGACY_DATA_DIR


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
