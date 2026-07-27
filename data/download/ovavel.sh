#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET="ovavel"
HF_REPO="OmniEvalKit/omnievalkit-dataset"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/$DATASET"
LINK_PATH="$TARGET_ROOT/data/$DATASET"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: python or python3 is required." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT"

"$PYTHON" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$DATA_ROOT",
    allow_patterns=["$DATASET/*"],
)
print("Downloaded $DATASET parquet shards from $HF_REPO")
PY

"$PYTHON" - <<PY
import json
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path("$DATA_ROOT")
source_dir = root / "$DATASET"
manifest = root / "ovavel_test.jsonl"
parquet_files = sorted(source_dir.glob("test-*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No OVAVEL parquet shards found under {source_dir}")


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def as_str(value):
    value = scalar(value)
    return "" if value == "" else str(value)


def media_dict(value):
    return value if isinstance(value, dict) else {}


def write_bytes(path, data):
    if not path or not data:
        return ""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(path)


rows = 0
videos = 0
audios = 0

with manifest.open("w", encoding="utf-8") as out:
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=16):
            for record in batch.to_pylist():
                sample_id = as_str(record.get("sample_id")) or f"ovavel_{rows}"
                event_category = as_str(record.get("event_category"))
                video_info = media_dict(record.get("video"))
                audio_info = media_dict(record.get("audio"))

                video_path = as_str(video_info.get("path")) or as_str(record.get("VideoPath"))
                audio_path = ""

                if write_bytes(video_path, video_info.get("bytes")):
                    videos += 1

                audio_bytes = audio_info.get("bytes")
                if audio_bytes:
                    audio_path = f"audios/{sample_id}.wav"
                    write_bytes(audio_path, audio_bytes)
                    audios += 1

                question = (
                    f"Localize when the event '{event_category}' occurs in the video. "
                    "Return the 10-bin event presence vector."
                )

                item = {
                    "id": sample_id,
                    "sample_id": sample_id,
                    "question": question,
                    "answer": as_str(record.get("gt_label")),
                    "gt_label": as_str(record.get("gt_label")),
                    "event_category": event_category,
                    "dataset_type": as_str(record.get("dataset_type")),
                    "dataset_name": as_str(record.get("dataset_name")),
                    "cls_type": as_str(record.get("cls_type")),
                    "split": as_str(record.get("split")),
                    "duration": scalar(record.get("duration")),
                    "VideoPath": video_path,
                    "WavPath": audio_path,
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\\n")
                rows += 1

print(f"Wrote {manifest} with {rows} rows, {videos} videos, {audios} audio files")
sys.stdout.flush()
os._exit(0)
PY

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$LINK_PATH" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "OVAVEL data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"
echo "Done: $DATASET is available at $LINK_PATH -> $DATA_ROOT"
