# OmniVinci

Adapter for `nvidia/omnivinci`.

Provide the external checkpoint explicitly; it is not included:

```bash
OMNIVINCI_MODEL_PATH=/path/to/omnivinci-checkpoint
```

Useful runtime knobs:

```bash
OMNIVINCI_NUM_VIDEO_FRAMES=128
OMNIVINCI_LOAD_AUDIO_IN_VIDEO=True
OMNIVINCI_AUDIO_LENGTH=max_3600
OMNIVINCI_TORCH_DTYPE=float16
OMNIVINCI_DEVICE_MAP=auto
```
