import argparse
import sys
from pathlib import Path

from .cosyvoice3_tts import CosyVoice3TTS, remove_user_site_from_path
from .pipeline import OUTPUT_MODES, run_pipeline


END_TO_END_ROOT = Path(__file__).resolve().parent
OMNIEVALKIT_ROOT = END_TO_END_ROOT.parent
DEFAULT_COSYVOICE_REPO = END_TO_END_ROOT / "third_party" / "CosyVoice"
DEFAULT_COSYVOICE_MODEL = OMNIEVALKIT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
DEFAULT_TTS_PROMPT_AUDIO = DEFAULT_COSYVOICE_REPO / "asset" / "zero_shot_prompt.wav"
DEFAULT_TTS_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. Answer the user's question using the "
    "provided visual and audio information."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run qwen25vlomni_advance with video, audio, images, and a question, "
            "then return text, CosyVoice3 speech, or both."
        )
    )
    parser.add_argument("--question", required=True, help="Question to ask about the supplied media.")
    parser.add_argument("--video", help="Optional input video path.")
    parser.add_argument(
        "--audio",
        action="append",
        help="Optional input audio path. Repeat --audio to supply multiple files.",
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Optional input image path. Repeat --image to supply multiple files.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model-path", default=None, help="Qwen2.5-VL checkpoint path or model ID.")
    parser.add_argument(
        "--disable-asr",
        action="store_true",
        help="Do not transcribe input audio or the video's audio track.",
    )
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default="text")
    parser.add_argument("--output-audio", default=str(END_TO_END_ROOT / "outputs" / "answer.wav"))
    parser.add_argument("--cosyvoice-model-dir", default=str(DEFAULT_COSYVOICE_MODEL))
    parser.add_argument("--cosyvoice-repo", default=str(DEFAULT_COSYVOICE_REPO))
    parser.add_argument("--tts-prompt-audio", default=str(DEFAULT_TTS_PROMPT_AUDIO))
    parser.add_argument(
        "--tts-prompt-text",
        default=DEFAULT_TTS_PROMPT_TEXT,
        help="Exact transcript of --tts-prompt-audio for zero-shot voice cloning.",
    )
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--tts-fp16", action="store_true")
    return parser


def _is_url(value):
    return str(value).startswith(("http://", "https://"))


def _normalize_question(question):
    """Allow shell arguments containing literal ``\n`` sequences."""
    return str(question).replace("\\n", "\n")


def _resolve_media_paths(parser, values, label):
    if not values:
        return None
    if isinstance(values, str):
        values = [values]

    resolved = []
    for value in values:
        if _is_url(value):
            resolved.append(value)
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
        resolved.append(str(path))
    return resolved


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    videos = _resolve_media_paths(parser, args.video, "Video")
    video_path = videos[0] if videos else None
    audio_paths = _resolve_media_paths(parser, args.audio, "Audio")
    image_paths = _resolve_media_paths(parser, args.image, "Image")

    remove_user_site_from_path()
    from .model import Model

    model = Model(model_path=args.model_path)
    model.use_asr = not args.disable_asr

    tts = None
    if args.output_mode in {"audio", "both"}:
        tts = CosyVoice3TTS(
            model_dir=args.cosyvoice_model_dir,
            cosyvoice_repo=args.cosyvoice_repo,
            prompt_audio=args.tts_prompt_audio,
            prompt_text=args.tts_prompt_text,
            speed=args.tts_speed,
            fp16=args.tts_fp16,
        )

    result = run_pipeline(
        model,
        question=_normalize_question(args.question),
        video_path=video_path,
        audio_path=audio_paths,
        image_path=image_paths,
        system_prompt=args.system_prompt,
        output_mode=args.output_mode,
        output_audio_path=args.output_audio,
        tts=tts,
    )

    if args.output_mode in {"text", "both"}:
        print(f"Answer:\n{result.answer}")
    if result.audio_path is not None:
        print(f"Audio: {result.audio_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
