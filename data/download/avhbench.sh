#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$TARGET_ROOT/data/avhbench"
DOWNLOAD_ROOT="$DATA_ROOT/downloads"
VIDEO_ID="10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7"
QA_ID="1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: python or python3 is required." >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_ROOT"

download_drive_file() {
  local file_id="$1"
  local output="$2"
  local url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -o "$output" "$url" && return 0
  fi

  "$PYTHON" -m gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$output"
}

download_drive_file "$VIDEO_ID" "$DOWNLOAD_ROOT/avhbench_videos"
download_drive_file "$QA_ID" "$DOWNLOAD_ROOT/avhbench_qa"

"$PYTHON" - <<PY
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

root = Path("$DATA_ROOT")
downloads = Path("$DOWNLOAD_ROOT")
videos = root / "videos"
videos.mkdir(parents=True, exist_ok=True)


def unpack_or_copy(path, target_dir, preferred_name=None):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target_dir)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(target_dir)
        return
    if preferred_name:
        shutil.copy2(path, root / preferred_name)


video_archive = downloads / "avhbench_videos"
qa_archive = downloads / "avhbench_qa"
unpack_or_copy(video_archive, videos)
unpack_or_copy(qa_archive, root, preferred_name="avhbench_qa.json")

for qa_path in sorted(root.rglob("*.json")):
    if "downloads" in qa_path.parts:
        continue
    try:
        data = json.loads(qa_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(data, list) and data and isinstance(data[0], dict) and "video_id" in data[0]:
        if qa_path.name != "avhbench_qa.json":
            shutil.copy2(qa_path, root / "avhbench_qa.json")
        break

print(f"AVHBench data prepared under {root}")
print(f"Videos found: {len(list(root.rglob('*.mp4')))}")
print(f"QA path: {root / 'avhbench_qa.json'}")
PY

echo "Done: AVHBench is available at $DATA_ROOT"
