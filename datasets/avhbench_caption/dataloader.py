from datasets.avhbench.dataloader import CAPTION_TASK, _find_manifest, _read_records, _resolve_video_path
from pathlib import Path


DATASET_NAME = "avhbench"
eval_type = "open"


def load_data(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / DATASET_NAME
    if not root.exists():
        raise FileNotFoundError("Run data/download/avhbench.sh first")

    manifest = _find_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"No AVHBench QA json found under {root}")

    samples = []
    for idx, record in enumerate(_read_records(manifest)):
        if not isinstance(record, dict) or str(record.get("task", "")).strip() != CAPTION_TASK:
            continue
        video_id = str(record.get("video_id", "")).strip()
        answer = str(record.get("label", "")).strip()
        samples.append(
            {
                "id": f"{video_id}_{idx}",
                "question": str(record.get("text", "")).strip(),
                "options": [],
                "answer": answer,
                "references": [answer],
                "video_path": _resolve_video_path(root, video_id),
                "audio_path": None,
                "image_path": None,
                "video_id": video_id,
                "task_type": CAPTION_TASK,
            }
        )
    return samples


load_dataset = load_data
