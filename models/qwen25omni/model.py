import logging
import os
import subprocess
import time
import warnings
from pathlib import Path


class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("QWEN25OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-3B")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.use_audio_in_video = self._str_to_bool(os.getenv("QWEN25OMNI_USE_AUDIO_IN_VIDEO", "True"))
        self.processor = None
        self.model = None
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        self._load()

    @staticmethod
    def _str_to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _load(self):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
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
            path,
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
            return list(range(num_frames))
        return np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int).tolist()

    @staticmethod
    def _load_video_frames_with_pyav(path, num_frames=8):
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
        return np.stack(frames)

    @staticmethod
    def _load_video_frames(path, num_frames=8):
        import cv2

        if Model._video_codec(path) == "av1":
            return Model._load_video_frames_with_pyav(path, num_frames)

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = Model._sample_indices(total_frames, num_frames)

        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()

        if not frames:
            return Model._load_video_frames_with_pyav(path, num_frames)

        import numpy as np

        return np.stack(frames)

    def _load_audio(self, path):
        sampling_rate = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        ffmpeg_error = None
        try:
            return self._load_audio_with_ffmpeg(path, sampling_rate)
        except Exception as exc:
            ffmpeg_error = exc

        try:
            return self._decode_audio_with_pyav(path, sampling_rate)
        except Exception:
            pass

        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
            return audio
        except Exception as exc:
            raise RuntimeError(f"Failed to decode audio: {path}. ffmpeg error: {ffmpeg_error}") from exc

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

    def _load_image(self, path):
        from PIL import Image

        return Image.open(path).convert("RGB")

    @staticmethod
    def _build_messages(system_prompt, image_paths, video_path, audio_paths, query):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            content.append({"type": "video", "video": video_path})
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})
        return messages

    def _official_processor_inputs(self, messages, use_audio_in_video):
        from qwen_omni_utils import process_mm_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        return self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )

    def _fallback_processor_inputs(self, messages, resolved_images, resolved_video, resolved_audio_paths, use_audio_in_video):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        video_input = self._load_video_frames(resolved_video) if resolved_video else None
        image_input = [self._load_image(path) for path in resolved_images] or None
        audio_inputs = []
        effective_use_audio_in_video = use_audio_in_video
        if use_audio_in_video and resolved_video:
            try:
                audio_inputs.append(self._load_audio(resolved_video))
            except Exception as exc:
                logging.warning("Could not decode embedded video audio for %s; continuing without it: %s", resolved_video, exc)
                effective_use_audio_in_video = False
        audio_inputs.extend(self._load_audio(path) for path in resolved_audio_paths)

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
            use_audio_in_video=effective_use_audio_in_video,
            videos_kwargs={"fps": 1.0},
        )
        return inputs, effective_use_audio_in_video

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        vlm_start = time.perf_counter()
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
        )

        try:
            inputs = self._official_processor_inputs(messages, use_audio_in_video)
            effective_use_audio_in_video = use_audio_in_video
        except ImportError:
            inputs, effective_use_audio_in_video = self._fallback_processor_inputs(
                messages,
                resolved_images,
                resolved_video,
                resolved_audio_paths,
                use_audio_in_video,
            )

        device = getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            inputs = inputs.to(dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_mode="text",
                use_audio_in_video=effective_use_audio_in_video,
                return_audio=False,
                thinker_max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        text_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        sequences = text_outputs.sequences if hasattr(text_outputs, "sequences") else text_outputs
        generated_ids = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], sequences)]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_vlm_inference_time_sec = time.perf_counter() - vlm_start
        # print("Final Ans => ", decoded[0].strip())
        return decoded[0].strip() if decoded else ""
