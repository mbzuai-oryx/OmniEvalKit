# NVILA-8B-HD-Video + Whisper

This package evaluates `nvidia/NVILA-8B-HD-Video` with NVIDIA's documented
`AutoProcessor` and `AutoModel` inference flow. Whisper optionally adds audio
transcripts for audio/video datasets.

Runtime components:

- external `nvidia/NVILA-8B-HD-Video` checkpoint or Hugging Face model ID
- `AutoGaze`: NVIDIA AutoGaze source
- external AutoGaze checkpoint or Hugging Face model ID

Checkpoints are intentionally not included in this repository.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
RESUME=False \
VILA_DEVICE_MAP=cuda:0 \
VILA_USE_ASR=True \
VILA_ASR_DEVICE=cuda \
VILA_WHISPER_USE_EMBEDDED_VIDEO_AUDIO=True \
VILA_NUM_VIDEO_FRAMES=64 \
VILA_NUM_VIDEO_FRAMES_THUMBNAIL=32 \
VILA_MAX_TILES_VIDEO=24 \
VILA_MAX_BATCH_SIZE_AUTOGAZE=1 \
VILA_MAX_BATCH_SIZE_SIGLIP=1 \
MODEL=VILA_whisper \
MODEL_PATH=nvidia/NVILA-8B-HD-Video \
EVAL_DATASETS="chartqa" \
OUTPUT_DIR=results/VILA_whisper \
LLM_JUDGE=False \
TEMPERATURE=0 \
MAX_NEW_TOKENS=4096 \
bash eval.sh
```

With one physical GPU exposed through `CUDA_VISIBLE_DEVICES`, address it as
logical `cuda:0` inside Python.
