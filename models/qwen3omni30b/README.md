# qwen3omni30b

Native Qwen3-Omni-30B-A3B-Instruct evaluator.

This adapter uses the model's native audio/image/video/text input path. Whisper
is not used by default because Qwen3-Omni is a full omni model.

Useful environment variables:

- `QWEN3OMNI_USE_AUDIO_IN_VIDEO=True`
- `QWEN3OMNI_NUM_VIDEO_FRAMES=32`
- `QWEN3OMNI_DEVICE_MAP=auto`
- `QWEN3OMNI_TORCH_DTYPE=auto`
- `QWEN3OMNI_ATTN_IMPLEMENTATION=sdpa`
- `QWEN3OMNI_DISABLE_TALKER=True`
- `QWEN3OMNI_MAX_AUDIO_SECONDS=`
