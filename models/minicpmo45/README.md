# MiniCPM-o 4.5

This folder loads `openbmb/MiniCPM-o-4_5` through its Hugging Face remote `AutoModel` code and calls the model's documented `chat` API.

The wrapper:
- uses `AutoModel.from_pretrained(..., trust_remote_code=True)`
- passes images, audio, and video as MiniCPM native chat content objects
- interleaves video frames with sidecar audio chunks via `get_video_frame_audio_segments(video_path, audio_path=...)`
- defaults to omni understanding modules with `init_vision=True`, `init_audio=True`, and `init_tts=False`
- defaults to text-only output with `generate_audio=False`
- includes a small compatibility shim for Transformers 5 dev builds that expect `all_tied_weights_keys`
- restores the native Transformers 5 Qwen3 generation input preparation at runtime
- defaults to `MINICPMO45_NUM_BEAMS=1` for deterministic evaluation and Transformers 5 stability
- routes vision settings by modality:
  - image/image+audio tasks: `MINICPMO45_IMAGE_USE_IMAGE_ID=True`, `MINICPMO45_IMAGE_MAX_SLICE_NUMS=9`
  - video/video+audio omni tasks: `MINICPMO45_VIDEO_USE_IMAGE_ID=False`, `MINICPMO45_VIDEO_MAX_SLICE_NUMS=1`

Model name for evaluation:

```bash
MODEL=minicpmo45
MODEL_PATH=openbmb/MiniCPM-o-4_5
```

Example:

```bash
MINICPMO45_INIT_TTS=False \
MINICPMO45_VIDEO_USE_FFMPEG=False \
MODEL=minicpmo45 \
MODEL_PATH=openbmb/MiniCPM-o-4_5 \
bash eval.sh
```

The official model card recommends these core dependencies for non-TTS evaluation:

```bash
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils>=1.0.5"
```

In this ROCm environment, keep `PYTHONNOUSERSITE=1` set so Python does not pick up incompatible user-site packages before the conda environment packages.

For document/image OCR benchmarks such as DocVQA, do not force `MINICPMO45_MAX_SLICE_NUMS=1`.
That setting compresses a full document page into one low-detail visual slice and can strongly reduce VLM/OCR accuracy.

Recommended modality groups:
- image VLM: `chartqa`, `docvqa`, `infographicvqa`, `mathverse_mini`, `mathvista`, `ocrbench`, `textvqa`
- video VLM: `longvideobench_val`, `lvbench`, `motionbench`, `videomme`, `videomme_short`
- omni audio-video: `daily_omni`, `avut_benchmark_human`, `avut_benchmark_gemini`, `av_odyssey`, `omnibench`, `ovobench`, `streamingbench_*`, `jointavbench`, `futureomni`

Global overrides still work:
- `MINICPMO45_MAX_SLICE_NUMS` overrides both image and video slice defaults.
- `MINICPMO45_USE_IMAGE_ID` overrides both image and video image-id defaults.
