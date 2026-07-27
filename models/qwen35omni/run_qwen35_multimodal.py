"""Run Qwen3.5-2B for T2T, I2T, transcript-backed A2T, V2T, and AV2T."""

from __future__ import annotations

import argparse
import gc
import json
import os
import types
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_VIDEO_NUM_FRAMES = 16
DEFAULT_VIDEO_FPS = 1.0


def read_text(value: str | None, file_path: str | None, label: str) -> str:
    if value and file_path:
        raise ValueError(f"Use either --{label} or --{label}-file, not both.")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return (value or "").strip()


def as_video_url(video: str) -> str:
    parsed = urlparse(video)
    if parsed.scheme in {"http", "https", "file", "data"}:
        return video
    return Path(video).expanduser().resolve().as_uri()


def is_local_path(value: str) -> bool:
    return urlparse(value).scheme not in {"http", "https", "file", "data"}


def require_inputs(args: argparse.Namespace) -> None:
    if args.mode in {"a2t", "av2t"} and not (args.transcript or args.transcript_file):
        if not (args.audio or args.video):
            if args.mode == "av2t":
                raise ValueError(
                    "--mode av2t requires --video plus an auto-generated transcript/audio, "
                    "or --transcript/--transcript-file."
                )
            raise ValueError(
                "--mode a2t requires --audio/--video for automatic ASR, "
                "or --transcript/--transcript-file."
            )
    if args.mode in {"v2t", "av2t"} and not args.video:
        raise ValueError(f"--mode {args.mode} requires --video.")
    if args.mode == "i2t" and not args.image:
        raise ValueError("--mode i2t requires --image.")
    if args.video and is_local_path(args.video):
        video_path = Path(args.video).expanduser()
        if not video_path.is_file():
            raise ValueError(f"--video must point to a local video file: {video_path}")
    if args.audio and is_local_path(args.audio):
        audio_path = Path(args.audio).expanduser()
        if not audio_path.is_file():
            raise ValueError(f"--audio must point to a local audio file: {audio_path}")
    if args.image and is_local_path(args.image):
        image_path = Path(args.image).expanduser()
        if not image_path.is_file():
            raise ValueError(f"--image must point to a local image file: {image_path}")
    if not args.query and not args.instruction:
        raise ValueError("Provide --query, --instruction, or both.")


def compose_text(args: argparse.Namespace, transcript: str, context_text: str) -> str:
    parts: list[str] = []
    if args.instruction:
        parts.append(f"Instruction:\n{args.instruction.strip()}")
    if context_text:
        parts.append(f"Text context:\n{context_text}")
    if transcript:
        parts.append(f"Audio transcript:\n{transcript}")
    if args.query:
        parts.append(f"Query:\n{args.query.strip()}")
    return "\n\n".join(parts)


def is_hf_asr_model(model_id: str) -> bool:
    model_path = Path(model_id).expanduser()
    return (model_path.is_dir() and (model_path / "config.json").is_file()) or "/" in model_id


def get_asr_input_path(args: argparse.Namespace) -> str:
    return args.audio or args.video


def decode_audio_with_pyav(media_path: str) -> dict[str, Any]:
    import av
    import numpy as np

    chunks = []
    with av.open(media_path) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"no audio stream found in {media_path}")

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                chunk = resampled.to_ndarray()
                chunks.append(chunk.reshape(-1))

    if not chunks:
        raise ValueError(f"no audio samples decoded from {media_path}")

    audio = np.concatenate(chunks)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    return {"array": audio, "sampling_rate": 16000}


def transcribe_with_hf_whisper(args: argparse.Namespace, device: str) -> str:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    local_model_path = Path(args.asr_model).expanduser()
    model_path = str(local_model_path.resolve()) if local_model_path.exists() else args.asr_model
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device == "cuda" else -1,
    )
    try:
        audio = decode_audio_with_pyav(str(Path(get_asr_input_path(args)).expanduser().resolve()))
        result = asr(
            audio,
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
        return result["text"].strip()
    finally:
        del asr
        del processor
        del model


def transcribe_with_openai_whisper(args: argparse.Namespace, device: str) -> str:
    import whisper

    try:
        model = whisper.load_model(args.asr_model, device=device)
    except Exception as exc:
        raise SystemExit(
            f"error: failed to load Whisper ASR model {args.asr_model!r}: {exc}"
        ) from exc
    try:
        try:
            result = model.transcribe(
                str(Path(get_asr_input_path(args)).expanduser().resolve()),
                fp16=device == "cuda",
            )
        except Exception as exc:
            raise SystemExit(f"error: failed to transcribe audio with Whisper: {exc}") from exc
        return result["text"].strip()
    finally:
        del model


def transcribe_video_audio(args: argparse.Namespace) -> None:
    if args.mode not in {"a2t", "av2t"} or args.transcript or args.transcript_file:
        return

    import torch

    device = args.asr_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if is_hf_asr_model(args.asr_model):
            args.transcript = transcribe_with_hf_whisper(args, device)
        else:
            args.transcript = transcribe_with_openai_whisper(args, device)
    except Exception as exc:
        raise SystemExit(
            f"error: failed to transcribe audio with ASR model {args.asr_model!r}: {exc}"
        ) from exc
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_openai_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    transcript = read_text(args.transcript, args.transcript_file, "transcript")
    context_text = read_text(args.text, args.text_file, "text")
    text = compose_text(args, transcript, context_text)

    messages: list[dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt.strip()})

    if args.mode == "i2t":
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": as_video_url(args.image)}},
            {"type": "text", "text": text},
        ]
        messages.append({"role": "user", "content": content})
    elif args.mode in {"v2t", "av2t"}:
        content: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {"url": as_video_url(args.video)},
                "num_frames": args.video_num_frames,
                "fps": args.video_fps,
            },
            {"type": "text", "text": text},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


def build_transformers_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    transcript = read_text(args.transcript, args.transcript_file, "transcript")
    context_text = read_text(args.text, args.text_file, "text")
    text = compose_text(args, transcript, context_text)

    messages: list[dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt.strip()})

    if args.mode == "i2t":
        content: list[dict[str, Any]] = [
            {"type": "image", "image": str(Path(args.image).expanduser().resolve())},
            {"type": "text", "text": text},
        ]
        messages.append({"role": "user", "content": content})
    elif args.mode in {"v2t", "av2t"}:
        content: list[dict[str, Any]] = [
            {
                "type": "video",
                "video": str(Path(args.video).expanduser().resolve()),
                "num_frames": args.video_num_frames,
                "fps": args.video_fps,
            },
            {"type": "text", "text": text},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


def run_openai(args: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    from openai import OpenAI

    extra_body: dict[str, Any] = {"top_k": args.top_k}
    if args.enable_thinking:
        extra_body["enable_thinking"] = True
    if args.mode in {"v2t", "av2t"}:
        extra_body["mm_processor_kwargs"] = {
            "fps": args.video_fps,
            "do_sample_frames": True,
        }

    client = OpenAI(
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


def force_video_backend(processor: Any, backend: str) -> None:
    """Work around Transformers falling back to torchvision video decoding."""
    if not hasattr(processor, "video_processor"):
        return

    from transformers.video_utils import load_video

    def fetch_videos(self: Any, video_url_or_urls: Any, sample_indices_fn: Any = None) -> Any:
        if isinstance(video_url_or_urls, list):
            return list(
                zip(
                    *[
                        self.fetch_videos(item, sample_indices_fn=sample_indices_fn)
                        for item in video_url_or_urls
                    ]
                )
            )
        return load_video(
            video_url_or_urls,
            backend=backend,
            sample_indices_fn=sample_indices_fn,
        )

    processor.video_processor.fetch_videos = types.MethodType(
        fetch_videos,
        processor.video_processor,
    )


def force_qwen35_video_tokens(processor: Any) -> None:
    """Make Qwen3.5 video placeholders match Qwen3_5 model expectations."""
    if not hasattr(processor, "video_processor") or not hasattr(processor, "video_token"):
        return

    def replace_video_token(self: Any, video_inputs: dict[str, Any], video_idx: int) -> str:
        merge_length = self.video_processor.merge_size**2
        num_frames = int(video_inputs["video_grid_thw"][video_idx][0])
        frame_seqlen = int(video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length)
        metadata = video_inputs["video_metadata"][video_idx]
        metadata.fps = 24 if metadata.fps is None else metadata.fps
        timestamps = self._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            self.video_processor.temporal_patch_size,
        )

        video_placeholder = ""
        for frame_idx in range(num_frames):
            video_placeholder += f"<{timestamps[frame_idx]:.1f} seconds>"
            video_placeholder += (
                self.vision_start_token
                + self.video_token * frame_seqlen
                + self.vision_end_token
            )
        return video_placeholder

    processor.replace_video_token = types.MethodType(replace_video_token, processor)


def validate_video_token_alignment(processor: Any, inputs: Any) -> None:
    if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
        return

    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    actual_tokens = int((inputs["input_ids"] == video_token_id).sum())
    merge_length = processor.video_processor.merge_size**2
    expected_tokens = int((inputs["video_grid_thw"].prod(dim=1) // merge_length).sum())
    if actual_tokens != expected_tokens:
        raise ValueError(
            "Video token alignment failed before generation: "
            f"input has {actual_tokens} video tokens but processor produced "
            f"{expected_tokens} video feature slots."
        )
    if "mm_token_type_ids" in inputs:
        expected_groups = int(inputs["video_grid_thw"][:, 0].sum())
        actual_groups = 0
        for token_types in inputs["mm_token_type_ids"]:
            in_video_group = False
            for token_type in token_types.tolist():
                if token_type == 2 and not in_video_group:
                    actual_groups += 1
                    in_video_group = True
                elif token_type != 2:
                    in_video_group = False
        if actual_groups != expected_groups:
            raise ValueError(
                "Video rope grouping failed before generation: "
                f"input has {actual_groups} video token groups but video_grid_thw "
                f"requires {expected_groups} temporal groups."
            )


def validate_image_token_alignment(processor: Any, inputs: Any) -> None:
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        return

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    actual_tokens = int((inputs["input_ids"] == image_token_id).sum())
    merge_length = processor.image_processor.merge_size**2
    expected_tokens = int((inputs["image_grid_thw"].prod(dim=1) // merge_length).sum())
    if actual_tokens != expected_tokens:
        raise ValueError(
            "Image token alignment failed before generation: "
            f"input has {actual_tokens} image tokens but processor produced "
            f"{expected_tokens} image feature slots."
        )


def run_transformers(args: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model)
    if args.mode in {"v2t", "av2t"}:
        force_video_backend(processor, args.video_backend)
        force_qwen35_video_tokens(processor)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    template_kwargs: dict[str, Any] = {}
    if args.enable_thinking:
        template_kwargs["enable_thinking"] = True
    processor_kwargs: dict[str, Any] = {}
    if args.mode in {"v2t", "av2t"}:
        processor_kwargs = {
            "num_frames": args.video_num_frames,
            "fps": None,
            "do_sample_frames": True,
        }
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
        **template_kwargs,
    )
    validate_image_token_alignment(processor, inputs)
    validate_video_token_alignment(processor, inputs)
    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

    generated_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs["input_ids"], output_ids)
    ]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-2B for T2T, I2T, transcript-backed A2T, V2T, and AV2T."
    )
    parser.add_argument("--mode", choices=["t2t", "i2t", "a2t", "v2t", "av2t"], required=True)
    parser.add_argument("--backend", choices=["openai", "transformers"], default="openai")
    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument("--system-prompt")
    parser.add_argument("--instruction")
    parser.add_argument("--query")
    parser.add_argument("--text")
    parser.add_argument("--text-file")
    parser.add_argument("--transcript")
    parser.add_argument("--transcript-file")
    parser.add_argument("--audio")
    parser.add_argument("--video")
    parser.add_argument("--image")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-device", default="auto")

    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--video-num-frames", type=int, default=DEFAULT_VIDEO_NUM_FRAMES)
    parser.add_argument(
        "--video-backend",
        choices=["pyav", "torchcodec", "torchvision", "opencv", "decord"],
        default="pyav",
        help="Video decoder backend for local Transformers inference.",
    )

    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--print-payload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        require_inputs(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    transcribe_video_audio(args)

    if args.backend == "openai":
        messages = build_openai_messages(args)
    else:
        messages = build_transformers_messages(args)

    if args.print_payload:
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        return

    if args.backend == "openai":
        answer = run_openai(args, messages)
    else:
        answer = run_transformers(args, messages)

    print(answer)


if __name__ == "__main__":
    main()
