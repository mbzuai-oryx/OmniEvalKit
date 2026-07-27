import logging
import os
import re
import subprocess
import warnings
from pathlib import Path


class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("QWEN3OMNI_MODEL_PATH")
        if not self.model_path:
            raise ValueError("Set --model_path or QWEN3OMNI_MODEL_PATH to a checkpoint path or model ID")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("QWEN3OMNI_TOP_P", "0.8"))
        self.repetition_penalty = float(os.getenv("QWEN3OMNI_REPETITION_PENALTY", "1.0"))
        self.num_video_frames = int(os.getenv("QWEN3OMNI_NUM_VIDEO_FRAMES", "32"))
        self.video_fps = float(os.getenv("QWEN3OMNI_VIDEO_FPS", "1.0"))
        self.use_audio_in_video = self._str_to_bool(os.getenv("QWEN3OMNI_USE_AUDIO_IN_VIDEO", "True"))
        self.disable_talker = self._str_to_bool(os.getenv("QWEN3OMNI_DISABLE_TALKER", "True"))
        self.device_map = os.getenv("QWEN3OMNI_DEVICE_MAP", "auto")
        self.torch_dtype = os.getenv("QWEN3OMNI_TORCH_DTYPE", "auto")
        self.attn_implementation = os.getenv("QWEN3OMNI_ATTN_IMPLEMENTATION", "sdpa")
        self.max_memory = os.getenv("QWEN3OMNI_MAX_MEMORY", "").strip()
        self.image_max_pixels = os.getenv("QWEN3OMNI_IMAGE_MAX_PIXELS", "").strip()
        max_audio_seconds = os.getenv("QWEN3OMNI_MAX_AUDIO_SECONDS", "").strip()
        self.max_audio_seconds = float(max_audio_seconds) if max_audio_seconds else None
        self.processor = None
        self.model = None
        self._load()

    @staticmethod
    def _str_to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _load(self):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        kwargs = {
            "dtype": self._resolve_dtype(torch, self.torch_dtype),
            "device_map": self.device_map,
        }
        if self.max_memory:
            kwargs["max_memory"] = self._parse_max_memory(self.max_memory)
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        self._set_image_max_pixels()
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.model_path, **kwargs)
        if self.disable_talker and hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.eval()

    @staticmethod
    def _resolve_dtype(torch, dtype_name):
        dtype_name = str(dtype_name or "auto").strip().lower()
        if dtype_name in {"", "auto"}:
            return "auto"
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return aliases.get(dtype_name, "auto")

    @staticmethod
    def _parse_max_memory(value):
        max_memory = {}
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            key, memory = item.split(":", 1)
            key = key.strip()
            memory = memory.strip()
            max_memory[int(key) if key.isdigit() else key] = memory
        return max_memory

    def _set_image_max_pixels(self):
        if not self.image_max_pixels:
            return
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        if size is None:
            return
        size["longest_edge"] = int(self.image_max_pixels)

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
    def _sanitize_indexed_media_placeholders(text):
        def replace(match):
            return f"[{match.group(1)} {match.group(2)}]"

        text = str(text or "").strip()
        return re.sub(r"<(audio|image|video)_(\d+)>", replace, text)

    @staticmethod
    def _video_codec(path):
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip().splitlines()
        return output[0].strip().lower() if output else None

    @staticmethod
    def _sample_indices(total_frames, num_frames):
        import numpy as np

        if total_frames <= 0:
            return [0] * num_frames
        return np.round(np.linspace(0, max(total_frames - 1, 0), num_frames)).astype(int).tolist()

    @staticmethod
    def _load_video_frames_with_pyav(path, num_frames):
        import av
        import numpy as np

        frames = []
        with av.open(path) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {path}")

            total_frames = int(stream.frames or 0)
            if total_frames:
                target_indices = set(Model._sample_indices(total_frames, num_frames))
                for index, frame in enumerate(container.decode(stream)):
                    if index in target_indices:
                        frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= len(target_indices):
                        break
            else:
                decoded_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
                indices = Model._sample_indices(len(decoded_frames), num_frames)
                frames = [decoded_frames[index] for index in indices if index < len(decoded_frames)]

        if not frames:
            raise ValueError(f"No frames decoded from video file: {path}")

        while len(frames) < min(num_frames, 2):
            frames.append(frames[-1])
        return np.stack(frames)

    @staticmethod
    def _load_video_frames_with_cv2(path, num_frames):
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = Model._sample_indices(total_frames, num_frames)
        frames = []
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()

        if not frames:
            raise ValueError(f"No frames decoded from video file: {path}")
        while len(frames) < min(num_frames, 2):
            frames.append(frames[-1])
        return np.stack(frames)

    def _load_video_frames(self, path):
        if self._video_codec(path) == "av1":
            return self._load_video_frames_with_pyav(path, self.num_video_frames)
        try:
            return self._load_video_frames_with_cv2(path, self.num_video_frames)
        except Exception:
            return self._load_video_frames_with_pyav(path, self.num_video_frames)

    @staticmethod
    def _load_image(path):
        from PIL import Image

        return Image.open(path).convert("RGB")

    def _load_audio(self, path):
        sampling_rate = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        ffmpeg_error = None
        try:
            audio = self._load_audio_with_ffmpeg(path, sampling_rate)
        except Exception as exc:
            ffmpeg_error = exc
            try:
                audio = self._decode_audio_with_pyav(path, sampling_rate)
            except Exception:
                try:
                    import librosa

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
                except Exception as audio_exc:
                    raise RuntimeError(f"Failed to decode audio: {path}. ffmpeg error: {ffmpeg_error}") from audio_exc

        if self.max_audio_seconds is not None:
            max_samples = int(sampling_rate * self.max_audio_seconds)
            audio = audio[:max_samples]
        return audio

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
            str(path),
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

            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            raise ValueError(f"No audio samples decoded from {path}")

        audio = np.concatenate(chunks)
        if np.issubdtype(audio.dtype, np.integer):
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    def _build_messages(self, system_prompt, image_paths, video_path, audio_paths, query, use_audio_in_video):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": str(system_prompt).strip()}]})

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            content.append({"type": "video", "video": video_path})
        if not (use_audio_in_video and video_path):
            for audio_path in audio_paths:
                content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": self._sanitize_indexed_media_placeholders(query)})
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_inputs(self, messages, image_paths, video_path, audio_paths, use_audio_in_video):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_input = [self._load_image(path) for path in image_paths] or None
        video_input = self._load_video_frames(video_path) if video_path else None

        audio_inputs = []
        effective_use_audio_in_video = use_audio_in_video
        if use_audio_in_video and video_path:
            try:
                audio_inputs.append(self._load_audio(video_path))
            except Exception as exc:
                logging.warning("Could not decode embedded video audio for %s; continuing without it: %s", video_path, exc)
                effective_use_audio_in_video = False
        if not (effective_use_audio_in_video and video_path):
            audio_inputs.extend(self._load_audio(path) for path in audio_paths)

        audio_input = None
        if len(audio_inputs) == 1:
            audio_input = audio_inputs[0]
        elif audio_inputs:
            audio_input = audio_inputs

        inputs = self.processor(
            text=text,
            audio=audio_input,
            images=image_input,
            videos=video_input,
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "fps": self.video_fps,
                "use_audio_in_video": effective_use_audio_in_video,
            },
        )
        return inputs, effective_use_audio_in_video

    def _input_device(self):
        import torch

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_inputs(self, inputs):
        inputs = inputs.to(self._input_device())
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            inputs = inputs.to(dtype)
        return inputs

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        del sample
        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)
        use_audio_in_video = bool(resolved_video and self.use_audio_in_video and not resolved_audio_paths)

        messages = self._build_messages(
            system_prompt,
            resolved_images,
            resolved_video,
            resolved_audio_paths,
            query,
            use_audio_in_video,
        )
        inputs, effective_use_audio_in_video = self._prepare_inputs(
            messages,
            resolved_images,
            resolved_video,
            resolved_audio_paths,
            use_audio_in_video,
        )
        inputs = self._move_inputs(inputs)

        generate_kwargs = {
            **inputs,
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": effective_use_audio_in_video,
            "thinker_max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        text_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        sequences = text_outputs.sequences if hasattr(text_outputs, "sequences") else text_outputs
        generated_ids = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], sequences)]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
