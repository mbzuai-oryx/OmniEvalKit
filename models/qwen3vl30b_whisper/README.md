# qwen3vl30b_whisper

Qwen3-VL-30B-A3B-Instruct evaluator with Whisper ASR transcript injection.

Qwen3-VL is used as a vision-language model. Audio files and embedded video
audio are transcribed with Whisper, then inserted into the text prompt before
the original question.

Useful environment variables:

- `QWEN3VL30B_USE_ASR=True`
- `QWEN3VL30B_ASR_DEVICE=cuda`
- `QWEN3VL30B_VIDEO_FPS=1.0`
- `QWEN3VL30B_VIDEO_NUM_FRAMES=128`
- `QWEN3VL30B_TRANSCRIBE_VIDEO_AUDIO=True`
- `QWEN3VL30B_DEVICE_MAP=auto`
- `QWEN3VL30B_TORCH_DTYPE=auto`
- `QWEN3VL30B_ATTN_IMPLEMENTATION=sdpa`
