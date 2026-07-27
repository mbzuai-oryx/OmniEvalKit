#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET="streamingbench_omni_fix"
RESTORED_ROOT="$TARGET_ROOT/data/omni/raw_hf/streamingbench"
SOURCE_ANNOTATION="$RESTORED_ROOT/streamingbench_omni_fix.jsonl"
LINK_PATH="$TARGET_ROOT/data/streamingbench_omni_fix"

# Manual/local-conversion only: no packaged HF config was found for this
# dataset variant. This wrapper exposes the locally converted StreamingBench
# root after the expected annotation exists.
echo "No packaged Hugging Face config was found for $DATASET."
echo "Create or copy the local converted annotation to: $SOURCE_ANNOTATION"
echo "Expected media prefix: $RESTORED_ROOT/video/"

if [ ! -f "$SOURCE_ANNOTATION" ]; then
  echo "Skipping: source annotation is not available yet."
  exit 0
fi

if [ ! -d "$RESTORED_ROOT/video" ]; then
  echo "Warning: expected video directory is missing: $RESTORED_ROOT/video" >&2
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
