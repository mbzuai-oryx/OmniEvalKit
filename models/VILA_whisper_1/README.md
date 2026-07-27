# NVILA-8B-Video + Whisper

This model package evaluates `Efficient-Large-Model/NVILA-8B-Video` through NVIDIA's official VILA inference API and optionally adds Whisper transcript text.

Runtime components:

- external `Efficient-Large-Model/NVILA-8B-Video` checkpoint or Hugging Face model ID
- `VILA`: official `NVlabs/VILA` source
- `scaling_on_scales`: official `s2wrapper` dependency

Checkpoints are intentionally not included in this repository.

Pinned revisions:

- NVILA checkpoint: `0a88e33bf3da24379d560774cbf1e9567a9fe658`
- VILA source: `0f1426e8da9181e6e6653e10bc15f62d515fa2f6`
- scaling_on_scales: `9c008a37540e761f53574b488979db6e49a64312`

Run with:

```bash
CUDA_VISIBLE_DEVICES=3 HIP_VISIBLE_DEVICES=3 \
RESUME=True \
MODEL=VILA_whisper_1 \
MODEL_PATH=Efficient-Large-Model/NVILA-8B-Video \
VILA_DEVICE_MAP=cuda:0 \
VILA_USE_ASR=True \
VILA_ASR_DEVICE=cuda \
VILA_WHISPER_USE_EMBEDDED_VIDEO_AUDIO=True \
VILA_NUM_VIDEO_FRAMES=64 \
EVAL_DATASETS="pointarena_counting,pixmo_pointing" \
OUTPUT_DIR=results/VILA_whisper_1 \
LLM_JUDGE=False \
TEMPERATURE=0 \
MAX_NEW_TOKENS=1024 \
bash eval.sh
```

`CUDA_VISIBLE_DEVICES=3` maps physical GPU 3 to logical `cuda:0` inside Python.
