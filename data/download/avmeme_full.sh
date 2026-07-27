#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET="avmeme_full"
HF_REPO="naplab/AVMeme-Exam"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/avmeme"
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
)
print("Downloaded $HF_REPO to $DATA_ROOT")
PY

if [ -f "$DATA_ROOT/clips.zip" ]; then
  unzip -n -q "$DATA_ROOT/clips.zip" -d "$DATA_ROOT"
else
  echo "Error: expected archive missing: $DATA_ROOT/clips.zip" >&2
  exit 1
fi

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$(readlink -f "$LINK_PATH")" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "AVMeme data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"
echo "Done: $DATASET is available at $LINK_PATH -> $DATA_ROOT"
