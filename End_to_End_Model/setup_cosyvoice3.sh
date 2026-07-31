#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

end_to_end_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
omnievalkit_root="$(cd "${end_to_end_root}/.." && pwd)"
cosyvoice_repo="${end_to_end_root}/third_party/CosyVoice"
model_dir="${omnievalkit_root}/pretrained_models/Fun-CosyVoice3-0.5B"

if [[ ! -d "${cosyvoice_repo}/cosyvoice" ]]; then
    mkdir -p "${end_to_end_root}/third_party"
    git clone --depth 1 --recursive \
        https://github.com/FunAudioLLM/CosyVoice.git \
        "${cosyvoice_repo}"
fi

# Install only the missing inference dependencies. In particular, do not let
# CosyVoice's CUDA-oriented requirements replace this environment's ROCm torch.
python -m pip install \
    "gdown==5.1.0" \
    "matplotlib==3.10.5" \
    "pyworld==0.3.4"

python -m pip install --no-deps \
    "inflect==7.3.1" \
    "typeguard==4.6.0" \
    "conformer==0.3.2" \
    "x-transformers==2.11.24" \
    "einx==0.4.3" \
    "frozendict==2.4.7" \
    "loguru==0.7.3" \
    "wget==3.2" \
    "lightning==2.2.4" \
    "pytorch-lightning==2.2.4" \
    "torchmetrics==1.3.2" \
    "lightning-utilities==0.11.2"

mkdir -p "${model_dir}"
MODEL_DIR="${model_dir}" python -s - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir=os.environ["MODEL_DIR"],
)
PY

echo "CosyVoice3 is ready at ${model_dir}"
