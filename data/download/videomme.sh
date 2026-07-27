#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HF_DOWNLOADER="${HF_DOWNLOADER:-$TARGET_ROOT/scripts/hf_download.py}"
DATASET="videomme"
RESTORED_ROOT="$TARGET_ROOT/data/omni/raw_hf/videomme"
LINK_PATH="$TARGET_ROOT/data/videomme"

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

# Video-MME videos may require the original lmms-lab/Video-MME video download.
# Pass --download_videos here or download the videos manually if the HF package
# only restores annotations/media references.
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
