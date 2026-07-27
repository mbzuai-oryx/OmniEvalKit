# Gemma 4 E2B + Whisper

This wrapper is copied from `models/gemma4e2b` and adds only Whisper transcript
text. It does not use SenseVoice or Mellow.

```bash
MODEL=gemma4e2b_whisper
MODEL_PATH=google/gemma-4-E2B-it
GEMMA4E2B_USE_ASR=True
GEMMA4E2B_ASR_DEVICE=cuda
```

The wrapper still passes native media to Gemma. If an explicit audio file exists,
it adds an `Audio file transcript` block. If a video has an audio stream, it adds
a `Video audio transcript` block from the original video before video resizing.
