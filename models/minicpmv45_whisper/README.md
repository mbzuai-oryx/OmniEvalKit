# MiniCPM-V 4.5 + Whisper

This folder evaluates `openbmb/MiniCPM-V-4_5` as a vision-language model and adds Whisper ASR transcript text for audio evidence.

The MiniCPM-V side follows the Hugging Face usage pattern:

- `AutoModel.from_pretrained(..., trust_remote_code=True, attn_implementation="sdpa")`
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- images/video frames are passed directly in `msgs`
- video samples also pass `temporal_ids`
- image-only samples use the model/processor default image slicing (`max_slice_nums=9`) unless `MINICPMV45_MAX_SLICE_NUMS` is set
- video samples default to `MINICPMV45_VIDEO_MAX_SLICE_NUMS=1`
- video sampling follows the MiniCPM-V 4.5 dynamic packing recipe, so `MINICPMV45_VIDEO_MAX_FRAMES=180` with `MINICPMV45_VIDEO_MAX_PACKING=3` can decode up to 540 packed frames

Model name for evaluation:

```bash
MODEL=minicpmv45_whisper
MODEL_PATH=openbmb/MiniCPM-V-4_5
```

Common settings:

```bash
MINICPMV45_USE_ASR=True
MINICPMV45_ASR_MODEL=openai/whisper-large-v3-turbo
MINICPMV45_ASR_DEVICE=cuda
MINICPMV45_VIDEO_FPS=5
MINICPMV45_VIDEO_MAX_FRAMES=180
```

For convenience this wrapper also accepts the existing `GEMMA4E2B_ASR_*` and `QWEN25VL_ASR_*` environment variables as ASR fallbacks.
