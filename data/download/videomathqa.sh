#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$TARGET_ROOT/data/videomathqa"
HF_REPO="MBZUAI/VideoMathQA"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: python or python3 is required." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

"$PYTHON" - <<PY
from huggingface_hub import hf_hub_download

for filename in [
    "videomathqa_mcq_test.parquet",
    "videomathqa_mbin_test.parquet",
    "subtitles.zip",
    "videos.zip",
    "README.md",
]:
    path = hf_hub_download(
        "$HF_REPO",
        repo_type="dataset",
        filename=filename,
        local_dir="$DATA_ROOT",
    )
    print(path)
PY

unzip -n -q "$DATA_ROOT/subtitles.zip" -d "$DATA_ROOT"
unzip -n -q "$DATA_ROOT/videos.zip" -d "$DATA_ROOT"

echo "Done: VideoMathQA is available at $DATA_ROOT"
