# Qwen2.5-Omni 3B

This model folder loads `Qwen/Qwen2.5-Omni-3B` through the shared Omni wrapper.

The wrapper:
- uses `Qwen2_5OmniForConditionalGeneration`
- uses `Qwen2_5OmniProcessor`
- uses `qwen_omni_utils.process_mm_info` when available
- falls back to local OpenCV/PyAV/ffmpeg media decoding when `qwen_omni_utils` is unavailable
- passes `use_audio_in_video` consistently through preprocessing, processor, and generation for embedded video audio
- runs text-only generation with `generation_mode="text"` and `return_audio=False`

Model name for evaluation:

```bash
MODEL=qwen25omni
MODEL_PATH=Qwen/Qwen2.5-Omni-3B
```
