import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path


MAX_IMAGE_SIZE = 1344


class Model:
    WHISPER_LANGUAGE_NAMES = {
        "<|zh|>": "Chinese",
        "<|en|>": "English",
        "<|yue|>": "Cantonese",
        "<|ja|>": "Japanese",
        "<|ko|>": "Korean",
        "<|fr|>": "French",
        "<|de|>": "German",
        "<|es|>": "Spanish",
        "<|it|>": "Italian",
        "<|pt|>": "Portuguese",
        "<|ru|>": "Russian",
        "<|ar|>": "Arabic",
        "<|hi|>": "Hindi",
    }

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
        self.use_asr = os.getenv("GEMMA4E2B_USE_ASR", os.getenv("QWEN25VL_USE_ASR", "True")).lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.asr_model_path = os.getenv(
            "GEMMA4E2B_ASR_MODEL",
            os.getenv(
                "QWEN25VL_ASR_MODEL",
                "openai/whisper-large-v3-turbo",
            ),
        )
        self.asr_device = os.getenv("GEMMA4E2B_ASR_DEVICE", os.getenv("QWEN25VL_ASR_DEVICE", "cuda"))
        self.asr_min_language_confidence = float(
            os.getenv("GEMMA4E2B_ASR_MIN_LANGUAGE_CONFIDENCE", os.getenv("QWEN25VL_ASR_MIN_LANGUAGE_CONFIDENCE", "0.65"))
        )
        self.asr_non_speech_text = os.getenv("GEMMA4E2B_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.processor = None
        self.model = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
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

    def _load_asr(self):
        if self.asr is not None:
            return self.asr

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = torch.float16 if torch.cuda.is_available() and str(self.asr_device).startswith("cuda") else torch.float32
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.asr_model.to(self.asr_device)
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_path)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            torch_dtype=dtype,
            device=self.asr_device,
        )
        return self.asr

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

    def _load_audio_for_asr(self, path):
        try:
            return self._load_audio_with_ffmpeg(path, 16000)
        except Exception:
            return self._decode_audio_with_pyav(path, 16000)

    def _whisper_analysis(self, audio_path):
        array = self._load_audio_for_asr(audio_path)
        if self._is_silent_audio(array):
            return {
                "language": "",
                "language_confidence": 0.0,
                "transcript": "",
            }

        language_token, language_confidence = self._detect_spoken_language(array)
        language = self._format_whisper_language(language_token)
        if language_confidence < self.asr_min_language_confidence:
            return {
                "language": language,
                "language_confidence": language_confidence,
                "transcript": "",
            }

        asr = self._load_asr()
        audio = {
            "array": array,
            "sampling_rate": 16000,
        }
        duration_seconds = len(audio["array"]) / audio["sampling_rate"]
        needs_long_form = duration_seconds > 30
        result = asr(
            audio,
            return_timestamps=needs_long_form,
            chunk_length_s=30 if needs_long_form else None,
            stride_length_s=3 if needs_long_form else None,
            generate_kwargs={
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_prev_tokens": False,
            },
        )
        transcript = result.get("text", "").strip()
        if not self._should_include_transcript(transcript):
            transcript = ""
        return {
            "language": language,
            "language_confidence": language_confidence,
            "transcript": transcript,
        }

    def _transcribe_media_sources(self, audio_paths, video_path):
        blocks = []
        for audio_path in audio_paths:
            block = self._format_whisper_block("Audio file", self._whisper_analysis(audio_path))
            if block:
                blocks.append(block)

        if video_path and self._has_audio_stream(video_path):
            block = self._format_whisper_block("Video audio", self._whisper_analysis(video_path))
            if block:
                blocks.append(block)
        return "\n\n".join(blocks)

    @staticmethod
    def _format_whisper_block(label, whisper):
        transcript = (whisper or {}).get("transcript", "").strip()
        if not transcript:
            return ""

        lines = [f"{label} transcript:"]
        language = whisper.get("language", "")
        confidence = whisper.get("language_confidence", 0.0)
        if language:
            lines.append(f"Audio Language: {language} ({confidence * 100:.2f}%)")
        lines.append("Whisper Transcript:")
        lines.append(transcript)
        return "\n".join(lines)

    def _build_text(self, query, transcript=""):
        parts = []
        if transcript:
            parts.append(transcript)
        parts.append(str(query or "").strip())
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _has_audio_stream(media_path):
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(media_path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            return False
        return bool(completed.stdout.decode("utf-8", errors="replace").strip())

    @staticmethod
    def _is_silent_audio(array):
        if array is None or len(array) == 0:
            return True
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
        peak = float(np.max(np.abs(array))) if len(array) else 0.0
        return rms < 1e-4 and peak < 1e-3

    def _detect_spoken_language(self, array):
        chunks = self._language_detection_chunks(array)
        best_token = ""
        best_confidence = 0.0
        for chunk in chunks:
            token, confidence = self._detect_language_for_chunk(chunk)
            if confidence > best_confidence:
                best_token = token
                best_confidence = confidence
        return best_token, best_confidence

    @staticmethod
    def _language_detection_chunks(array, sampling_rate=16000, chunk_seconds=30):
        chunk_size = sampling_rate * chunk_seconds
        if len(array) <= chunk_size:
            return [array]
        starts = [0, max((len(array) - chunk_size) // 2, 0), max(len(array) - chunk_size, 0)]
        chunks = []
        seen = set()
        for start in starts:
            if start in seen:
                continue
            seen.add(start)
            chunks.append(array[start : start + chunk_size])
        return chunks

    def _detect_language_for_chunk(self, array):
        import torch

        self._load_asr()
        processor = self.asr_processor
        model = self.asr_model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        inputs = processor(
            array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            decoder_input_ids = torch.tensor(
                [[model.config.decoder_start_token_id]],
                device=device,
            )
            decoder_out = model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
            logits = model.proj_out(decoder_out.last_hidden_state[:, -1, :])[0]

        lang_token_ids = self._get_language_token_ids(processor.tokenizer, device)
        if lang_token_ids.numel() == 0:
            return "", 1.0

        lang_probs = torch.softmax(logits[lang_token_ids].float(), dim=-1)
        best = torch.argmax(lang_probs)
        token_id = lang_token_ids[best].item()
        token = processor.tokenizer.convert_ids_to_tokens(token_id)
        return token, float(lang_probs[best])

    def _get_language_token_ids(self, tokenizer, device):
        import torch

        if self._asr_language_token_ids is None:
            language_ids = []
            for token, token_id in tokenizer.get_vocab().items():
                if token.startswith("<|") and token.endswith("|>"):
                    lang = token[2:-2]
                    if len(lang) in (2, 3) and lang.isalpha():
                        language_ids.append(token_id)
            self._asr_language_token_ids = sorted(language_ids)
        return torch.tensor(self._asr_language_token_ids, device=device, dtype=torch.long)

    @classmethod
    def _format_whisper_language(cls, language_token):
        if not language_token:
            return ""
        return cls.WHISPER_LANGUAGE_NAMES.get(language_token, language_token.strip("<|>"))

    def _should_include_transcript(self, transcript):
        transcript = (transcript or "").strip()
        return bool(transcript) and transcript != self.asr_non_speech_text and not self._is_bad_transcript(transcript)

    @staticmethod
    def _is_bad_transcript(text, threshold=0.5):
        text = (text or "").strip()
        if not text:
            return True
        lowered = text.lower()
        if "[blank_audio]" in lowered or "blank audio" in lowered:
            return True
        bad_chars = sum(text.count(char) for char in ("\u00b6", "\u266a", "\u2669", "\u266b", "\ufffd"))
        if bad_chars and bad_chars / max(len(text), 1) >= threshold:
            return True
        compact = "".join(text.split())
        if len(compact) >= 40:
            most_common = max(compact.count(char) for char in set(compact))
            if most_common / len(compact) >= 0.8:
                return True
        words = re.findall(r"[\w']+", lowered, flags=re.UNICODE)
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.20:
                return True
            most_common_word = max(words.count(word) for word in set(words))
            if most_common_word / len(words) >= 0.50:
                return True
        return False

    @staticmethod
    def _load_audio_with_ffmpeg(path, sampling_rate):
        import numpy as np

        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "-",
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio = np.frombuffer(completed.stdout, dtype=np.float32)
        if audio.size:
            return audio.copy()
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed with code {completed.returncode}: {error}")
        raise ValueError(f"No audio decoded from {path}")

    @staticmethod
    def _decode_audio_with_pyav(path, sampling_rate):
        import av
        import numpy as np

        chunks = []
        with av.open(path) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise ValueError(f"No audio stream found in {path}")

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=sampling_rate,
            )
            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            raise ValueError(f"No audio samples decoded from {path}")

        audio = np.concatenate(chunks)
        if np.issubdtype(audio.dtype, np.integer):
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    def _build_messages(self, video_path, image_paths, audio_paths, query, system_prompt, transcript=""):
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

        text = self._build_text(query, transcript)
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
        return self.processor.apply_chat_template(messages, **kwargs)

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        self._video_metadata = None
        resolved_video_path = self._path_value(video_path)
        resolved_video = self._prepare_video_value(resolved_video_path)
        resolved_images = self._prepare_image_paths(self._path_values(image_path))
        resolved_audio_paths = self._path_values(audio_path)
        transcript = ""
        if self.use_asr:
            transcript = self._transcribe_media_sources(resolved_audio_paths, resolved_video_path)

        messages = self._build_messages(
            resolved_video,
            resolved_images,
            resolved_audio_paths,
            query,
            system_prompt,
            transcript,
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
