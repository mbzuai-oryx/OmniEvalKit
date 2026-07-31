# Qwen2.5-VL Omni Advance

Run from the repository root:

```bash
python3 End_to_End_Model/run_qwen25vlomni_advance.py \
  --video examples/video.mp4 \
  --audio examples/audio.wav \
  --question "What happens in this video?" \
  --output-mode both \
  --output-audio outputs/answer.wav
```

Output modes are `text`, `audio`, and `both`. For `audio` or `both`, prepare
CosyVoice3 first:

```bash
bash End_to_End_Model/setup_cosyvoice3.sh
```

`--video`, `--audio`, and `--image` are optional. Repeat `--audio` or `--image`
to provide multiple files.

```bash
python3 End_to_End_Model/run_qwen25vlomni_advance.py --help
```
