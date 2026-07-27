#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET="avut_benchmark_human"
HF_REPO="OmniEvalKit/omnievalkit-dataset"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/avut-benchmark"
LINK_PATH="$TARGET_ROOT/data/$DATASET"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: python or python3 is required." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/annotation"

"$PYTHON" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$DATA_ROOT/raw",
    allow_patterns=["$DATASET/*"],
)
print("Downloaded $DATASET parquet shards from $HF_REPO")
PY

"$PYTHON" - <<PY
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path("$DATA_ROOT")
source_dir = root / "raw" / "$DATASET"
annotation = root / "annotation" / "$DATASET.jsonl"
root_manifest = root / "$DATASET.jsonl"
parquets = sorted(source_dir.glob("test-*.parquet"))

if not parquets:
    raise FileNotFoundError(f"No parquet shards found under {source_dir}")


def parse_json_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def media_dict(value):
    return value if isinstance(value, dict) else {}


def media_path(record, media, key):
    return media.get("path") or record.get(key) or ""


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
with annotation.open("w", encoding="utf-8") as out:
    for parquet_path in parquets:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=16):
            for record in batch.to_pylist():
                video_info = media_dict(record.pop("video", None))
                audio_info = media_dict(record.pop("audio", None))
                video_path = media_path(record, video_info, "VideoPath")
                audio_path = media_path(record, audio_info, "WavPath")

                if write_bytes(video_path, video_info.get("bytes")):
                    videos += 1
                if not audio_path and video_path and audio_info.get("bytes"):
                    audio_path = str(Path(video_path).with_suffix(".wav"))
                if write_bytes(audio_path, audio_info.get("bytes")):
                    audios += 1

                record["VideoPath"] = video_path
                record["WavPath"] = audio_path
                record["choices"] = parse_json_value(record.get("choices"))
                out.write(json.dumps(record, ensure_ascii=False) + "\\n")
                rows += 1

root_manifest.write_text(annotation.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Wrote {annotation} with {rows} rows, {videos} videos, {audios} audio files")
sys.stdout.flush()
os._exit(0)
PY

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$(readlink -f "$LINK_PATH")" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "AVUT data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"
echo "Done: $DATASET is available at $LINK_PATH -> $DATA_ROOT"
