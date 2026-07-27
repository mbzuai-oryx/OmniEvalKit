#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/video-holmes"
LINK_PATH="$TARGET_ROOT/data/video_holmes"
HF_REPO="TencentARC/Video-Holmes"

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

if [ -f "$DATA_ROOT/videos.zip" ]; then
  unzip -n -q "$DATA_ROOT/videos.zip" -d "$DATA_ROOT"
else
  echo "Error: missing $DATA_ROOT/videos.zip" >&2
  exit 1
fi

if [ -f "$DATA_ROOT/annotations.zip" ]; then
  unzip -n -q "$DATA_ROOT/annotations.zip" -d "$DATA_ROOT"
fi

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$(cd "$LINK_PATH" && pwd)" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "Video-Holmes data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"

echo "Done: video_holmes is available at $LINK_PATH -> $DATA_ROOT"
echo "Test annotation: $DATA_ROOT/test_Video-Holmes.json"
echo "Videos: $DATA_ROOT/videos_cropped"
