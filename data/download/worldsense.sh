#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/worldsense"
LINK_PATH="$TARGET_ROOT/data/worldsense"
HF_REPO="honglyhly/WorldSense"

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
)
PY

mkdir -p "$DATA_ROOT/videos" "$DATA_ROOT/subtitles"

for archive in "$DATA_ROOT"/worldsense_videos_*.zip; do
  [ -e "$archive" ] || continue
  unzip -n -q "$archive" -d "$DATA_ROOT/videos"
done

if [ -f "$DATA_ROOT/worldsense_subtitles.zip" ]; then
  unzip -n -q "$DATA_ROOT/worldsense_subtitles.zip" -d "$DATA_ROOT/subtitles"
fi

"$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("$DATA_ROOT")
source = root / "worldsense_qa.json"
target = root / "worldsense.jsonl"

if not source.exists():
    raise FileNotFoundError(f"Missing official annotation file: {source}")

data = json.loads(source.read_text(encoding="utf-8"))
with target.open("w", encoding="utf-8") as out:
    for video_id, video_record in data.items():
        for task_name, task in sorted(video_record.items()):
            if not task_name.startswith("task") or not isinstance(task, dict):
                continue
            record = {
                "id": f"{video_id}_{task_name}",
                "dataset_type": "mcq",
                "dataset_name": "worldsense",
                "video_id": video_id,
                "question": task.get("question", ""),
                "choices": task.get("candidates", []),
                "gt_answer": task.get("answer", ""),
                "VideoPath": f"videos/{video_id}.mp4",
                "duration": video_record.get("duration", ""),
                "task_domain": task.get("task_domain", ""),
                "task_type": task.get("task_type", ""),
                "domain": video_record.get("domain", ""),
                "sub_category": video_record.get("sub_category", ""),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\\n")

print(f"Wrote {target}")
PY

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$(cd "$LINK_PATH" && pwd)" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "WorldSense data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"

echo "Done: worldsense is available at $LINK_PATH -> $DATA_ROOT"
echo "Manifest: $DATA_ROOT/worldsense.jsonl"
