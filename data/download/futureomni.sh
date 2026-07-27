#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$TARGET_ROOT/data/omni/raw_hf/futureomni"
LINK_PATH="$TARGET_ROOT/data/futureomni"
HF_REPO="OpenMOSS-Team/FutureOmni"

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

if [ -f "$DATA_ROOT/test_splitted_videos_part_aa" ] && ! find "$DATA_ROOT" -type f -name '*.mp4' | grep -q .; then
  ZIP_PATH="$DATA_ROOT/test_splitted_videos.zip"
  cat "$DATA_ROOT"/test_splitted_videos_part_* > "$ZIP_PATH"
  unzip -n -q "$ZIP_PATH" -d "$DATA_ROOT"
  rm -f "$ZIP_PATH"
fi

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ] && [ "$(cd "$LINK_PATH" && pwd)" != "$DATA_ROOT" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "FutureOmni data is available at: $DATA_ROOT" >&2
  exit 1
fi

ln -s "$DATA_ROOT" "$LINK_PATH"

echo "Done: futureomni is available at $LINK_PATH -> $DATA_ROOT"
echo "Test annotation: $DATA_ROOT/futureomni_test.json"
