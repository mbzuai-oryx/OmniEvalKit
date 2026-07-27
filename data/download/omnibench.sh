#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HF_DOWNLOADER="${HF_DOWNLOADER:-$TARGET_ROOT/scripts/hf_download.py}"
DATASET="omnibench"
RESTORED_ROOT="$TARGET_ROOT/data/omni/raw_hf/omnibench"
LINK_PATH="$TARGET_ROOT/data/omnibench"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: python or python3 is required." >&2
  exit 1
fi

if [ ! -f "$HF_DOWNLOADER" ]; then
  echo "Error: downloader not found: $HF_DOWNLOADER" >&2
  exit 1
fi

"$PYTHON" "$HF_DOWNLOADER" --datasets "$DATASET" --output_dir "$TARGET_ROOT/data" "$@"

if [ ! -d "$RESTORED_ROOT" ]; then
  echo "Error: expected restored dataset root missing: $RESTORED_ROOT" >&2
  exit 1
fi

if [ -L "$LINK_PATH" ]; then
  rm "$LINK_PATH"
elif [ -e "$LINK_PATH" ]; then
  echo "Error: refusing to replace existing non-symlink path: $LINK_PATH" >&2
  echo "Restored data is available at: $RESTORED_ROOT" >&2
  exit 1
fi

ln -s "$RESTORED_ROOT" "$LINK_PATH"
echo "Done: $DATASET is available at $LINK_PATH -> $RESTORED_ROOT"
