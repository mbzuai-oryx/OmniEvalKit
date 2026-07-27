import hashlib
import logging
import os
import subprocess
from pathlib import Path


MAX_IMAGE_SIZE = 1344


class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("GEMMA4E2B_MODEL_PATH", "google/gemma-4-E2B")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("GEMMA4E2B_TOP_P", "0.8"))
        self.device_map = os.getenv("GEMMA4E2B_DEVICE_MAP", "auto")
        self.torch_dtype = os.getenv("GEMMA4E2B_TORCH_DTYPE", "auto")
        self.attn_implementation = os.getenv("GEMMA4E2B_ATTN_IMPLEMENTATION", "sdpa")
        self.video_num_frames = self._optional_int(os.getenv("GEMMA4E2B_VIDEO_NUM_FRAMES", ""))
        self.video_default_frames = int(os.getenv("GEMMA4E2B_VIDEO_DEFAULT_FRAMES", "32"))
        self.video_fps = float(os.getenv("GEMMA4E2B_VIDEO_FPS", "1.0"))
        self.video_min_pixels = int(os.getenv("GEMMA4E2B_VIDEO_MIN_PIXELS", str(256 * 256)))
        self.video_max_pixels = int(os.getenv("GEMMA4E2B_VIDEO_MAX_PIXELS", str(1024 * 1024)))
        self.audio_sampling_rate = int(os.getenv("GEMMA4E2B_AUDIO_SAMPLING_RATE", "16000"))
        self.processor = None
        self.model = None
        self.supports_audio = True
        self._video_metadata = None
        self._load()

    @staticmethod
    def _optional_int(value):
        value = str(value or "").strip()
        return int(value) if value else None

    def _load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            from transformers import AutoModelForMultimodalLM

            model_cls = AutoModelForMultimodalLM
            self.supports_audio = True
        except ImportError:
            model_cls = AutoModelForImageTextToText
            self.supports_audio = False

        kwargs = {
            "device_map": self.device_map,
        }
        if self.torch_dtype:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = model_cls.from_pretrained(self.model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_path, padding_side="left")
        self.model.eval()

    @staticmethod
    def _path_value(path):
        if not path:
            return None
        path = str(path).strip()
        if path.startswith(("http://", "https://")):
            return path
        return str(Path(path).expanduser().resolve())

    @classmethod
    def _path_values(cls, paths):
        if not paths:
            return []
        if isinstance(paths, (list, tuple)):
            return [cls._path_value(path) for path in paths if path]
        return [cls._path_value(paths)]

    @staticmethod
    def _is_remote_path(path):
        return isinstance(path, str) and path.startswith(("http://", "https://"))

    @staticmethod
    def _media_cache_dir():
        cache_dir = Path(os.getenv("GEMMA4E2B_MEDIA_CACHE", "/tmp/gemma4e2b_media_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @classmethod
    def _cache_path(cls, path, suffix):
        source = str(path)
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        return cls._media_cache_dir() / f"{Path(source).stem}_{digest}{suffix}"

    @staticmethod
    def _scaled_size(width, height, max_size=MAX_IMAGE_SIZE):
        longest = max(width, height)
        if longest <= max_size:
            return width, height
        scale = max_size / float(longest)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return new_width, new_height

    @staticmethod
    def _pixel_bounded_size(width, height, min_pixels, max_pixels):
        if min_pixels > max_pixels:
            min_pixels = max_pixels
        pixels = width * height
        scale = 1.0
        if pixels > max_pixels:
            scale = (max_pixels / float(pixels)) ** 0.5
        elif pixels < min_pixels:
            scale = (min_pixels / float(pixels)) ** 0.5
        if scale == 1.0:
            return width, height
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return new_width, new_height

    @staticmethod
    def _even_size(width, height):
        return max(2, width - (width % 2)), max(2, height - (height % 2))

    @classmethod
    def _prepare_image_path(cls, image_path):
        if not image_path or cls._is_remote_path(image_path):
            return image_path

        from PIL import Image

        path = Path(image_path)
        with Image.open(path) as image:
            width, height = image.size
            new_width, new_height = cls._scaled_size(width, height)
            if (new_width, new_height) == (width, height):
                return str(path)

            output_path = cls._cache_path(path, ".jpg")
            if output_path.exists():
                return str(output_path)

            resized = image.convert("RGB").resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized.save(output_path, format="JPEG", quality=95)
            logging.info("Resized image for ROCm: %s %sx%s -> %sx%s", path, width, height, new_width, new_height)
            return str(output_path)

    @classmethod
    def _prepare_image_paths(cls, image_paths):
        return [cls._prepare_image_path(path) for path in image_paths]

    @classmethod
    def _video_dimensions(cls, video_path):
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip().splitlines()
        if not output or "x" not in output[0]:
            return None
        parts = [part.strip() for part in output[0].split("x") if part.strip()]
        if len(parts) < 2:
            return 1280, 720
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 1280, 720

    def _prepare_video_path(self, video_path):
        if not video_path or self._is_remote_path(video_path):
            return video_path

        dimensions = self._video_dimensions(video_path)
        if not dimensions:
            return video_path
        width, height = dimensions
        new_width, new_height = self._scaled_size(width, height)
        new_width, new_height = self._pixel_bounded_size(
            new_width,
            new_height,
            self.video_min_pixels,
            self.video_max_pixels,
        )
        new_width, new_height = self._even_size(new_width, new_height)
        if (new_width, new_height) == (width, height):
            return video_path

        output_path = self._cache_path(video_path, ".mp4")
        if output_path.exists():
            if output_path.stat().st_size > 0 and self._video_dimensions(str(output_path)):
                return str(output_path)
            output_path.unlink(missing_ok=True)

        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"scale={new_width}:{new_height}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(output_path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            logging.warning("Failed to resize video for ROCm, using original: %s: %s", video_path, completed.stderr.strip())
            return video_path
        logging.info("Resized video for ROCm: %s %sx%s -> %sx%s", video_path, width, height, new_width, new_height)
        return str(output_path)

    @staticmethod
    def _sample_indices(total_frames, num_frames):
        if total_frames <= num_frames:
            return set(range(total_frames))
        if num_frames <= 1:
            return {0}
        return {
            int(round(index * (total_frames - 1) / (num_frames - 1)))
            for index in range(num_frames)
        }

    def _target_frame_count(self, stream, container, total_frames):
        if self.video_num_frames is not None:
            return self.video_num_frames

        duration = None
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000

        if duration and self.video_fps > 0:
            target_frames = max(1, int(round(duration * self.video_fps)))
        else:
            target_frames = self.video_default_frames
        if total_frames:
            target_frames = min(target_frames, total_frames)
        return target_frames

    def _set_video_metadata(self, frames, original_fps=None, frame_indices=None):
        from transformers.video_utils import VideoMetadata

        if not frames:
            self._video_metadata = None
            return

        width, height = frames[0].size
        if self.video_num_frames is None:
            fps = self.video_fps
            indices = list(range(len(frames)))
        else:
            fps = original_fps
            indices = frame_indices or list(range(len(frames)))
        self._video_metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=fps,
            width=width,
            height=height,
            duration=(len(frames) / fps) if fps else None,
            frames_indices=indices,
        )

    def _decode_video_frames(self, video_path):
        if not video_path or self._is_remote_path(video_path):
            return video_path

        import av
        from PIL import Image

        with av.open(video_path) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {video_path}")

            total_frames = int(stream.frames or 0)
            original_fps = float(stream.average_rate) if stream.average_rate else None
            num_frames = self._target_frame_count(stream, container, total_frames)
            if total_frames:
                target_indices = sorted(self._sample_indices(total_frames, num_frames))
                frames = []
                for idx, frame in enumerate(container.decode(stream)):
                    if idx in target_indices:
                        frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                    if len(frames) >= len(target_indices):
                        break
                if frames:
                    self._set_video_metadata(frames, original_fps, target_indices)
                    return frames

            frames = [Image.fromarray(frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
        if not frames:
            raise ValueError(f"No video frames decoded from {video_path}")
        indices = sorted(self._sample_indices(len(frames), num_frames))
        sampled_frames = [frames[index] for index in indices]
        self._set_video_metadata(sampled_frames, frame_indices=indices)
        return sampled_frames

    def _prepare_video_value(self, video_path):
        prepared_path = self._prepare_video_path(video_path)
        return self._decode_video_frames(prepared_path)

    @staticmethod
    def _is_gpu_kernel_error(exc):
        text = str(exc)
        return "invalid configuration" in text or "hipError" in text or "AcceleratorError" in text

    @staticmethod
    def _suppress_hallucination(transcript: str) -> str:
        if not transcript or not transcript.strip():
            return "[no audio content]"
        pilcrow_ratio = transcript.count("¶") / len(transcript)
        if pilcrow_ratio > 0.3:
            return "[non-speech audio]"
        return transcript

    @staticmethod
    def _build_text(query):
        return query.strip()

    def _build_messages(self, video_path, image_paths, audio_paths, query, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "path": image_path})
        if video_path:
            content.append({"type": "video", "video": video_path})
        if audio_paths:
            if self.supports_audio:
                for audio_path in audio_paths:
                    content.append({"type": "audio", "path": audio_path})
            else:
                logging.warning("Gemma4E2B local model class does not support native audio; skipping %s audio file(s).", len(audio_paths))

        text = self._build_text(query)
        if content:
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": text})
        return messages

    def _direct_processor_inputs(self, messages):
        media_tokens = []
        images = []
        videos = []
        audio = []
        text_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{role.capitalize()}:\n{content.strip()}")
                continue

            for item in content:
                item_type = item.get("type")
                if item_type == "image":
                    images.append(item.get("path") or item.get("url") or item.get("image"))
                    media_tokens.append(getattr(self.processor, "image_token", "<image>"))
                elif item_type == "video":
                    videos.append(item.get("video") or item.get("path") or item.get("url"))
                    media_tokens.append(getattr(self.processor, "video_token", "<|video|>"))
                elif item_type == "audio":
                    audio.append(item.get("path") or item.get("url") or item.get("audio"))
                    media_tokens.append(getattr(self.processor, "audio_token", "<|audio|>"))
                elif item_type == "text":
                    text_parts.append(f"{role.capitalize()}:\n{item.get('text', '').strip()}")

        prompt = "\n".join(media_tokens + text_parts + ["Assistant:"])
        kwargs = {
            "text": [prompt],
            "images": [image for image in images if image] or None,
            "videos": [video for video in videos if video] or None,
            "audio": [item for item in audio if item] or None,
            "return_tensors": "pt",
            "sampling_rate": self.audio_sampling_rate,
        }
        if self.video_num_frames is not None:
            kwargs["num_frames"] = self.video_num_frames
        return self.processor(**kwargs)

    def _apply_chat_template(self, messages):
        if not getattr(self.processor, "chat_template", None):
            return self._direct_processor_inputs(messages)

        videos_kwargs = {"do_sample_frames": False}
        if self._video_metadata is not None:
            videos_kwargs["video_metadata"] = [self._video_metadata]
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "processor_kwargs": {
                "sampling_rate": self.audio_sampling_rate,
                "videos_kwargs": videos_kwargs,
            },
        }
        if self.video_num_frames is not None:
            kwargs["num_frames"] = self.video_num_frames
        return self.processor.apply_chat_template(messages, enable_thinking=True, **kwargs)

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        self._video_metadata = None
        resolved_video = self._prepare_video_value(self._path_value(video_path))
        resolved_images = self._prepare_image_paths(self._path_values(image_path))
        resolved_audio_paths = self._path_values(audio_path)

        messages = self._build_messages(
            resolved_video,
            resolved_images,
            resolved_audio_paths,
            query,
            system_prompt,
        )
        inputs = self._apply_chat_template(messages)
        input_len = inputs["input_ids"].shape[-1]
        device = getattr(self.model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            inputs = inputs.to(device, dtype=dtype)
        else:
            inputs = inputs.to(device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            if self._is_gpu_kernel_error(exc):
                sample_id = sample.get("id") if sample else "unknown"
                logging.warning("Skipping sample %s: ROCm kernel error on this input — %s", sample_id, exc)
                return "[ERROR: GPU kernel config]"
            raise

        decoded = self.processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
