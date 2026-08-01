import os
import re
import subprocess
import warnings
from pathlib import Path


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
        self.model_path = model_path or os.getenv("QWEN3VL30B_MODEL_PATH")
        if not self.model_path:
            raise ValueError("Set --model_path or QWEN3VL30B_MODEL_PATH to a checkpoint path or model ID")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("QWEN3VL30B_TOP_P", "0.8"))
        self.repetition_penalty = float(os.getenv("QWEN3VL30B_REPETITION_PENALTY", "1.0"))
        self.video_fps = float(os.getenv("QWEN3VL30B_VIDEO_FPS", "1.0"))
        video_num_frames = os.getenv("QWEN3VL30B_VIDEO_NUM_FRAMES", "").strip()
        self.video_num_frames = int(video_num_frames) if video_num_frames else None
        self.device_map = os.getenv("QWEN3VL30B_DEVICE_MAP", "auto")
        self.torch_dtype = os.getenv("QWEN3VL30B_TORCH_DTYPE", "auto")
        self.attn_implementation = os.getenv("QWEN3VL30B_ATTN_IMPLEMENTATION", "sdpa")
        self.max_memory = os.getenv("QWEN3VL30B_MAX_MEMORY", "").strip()
        self.offload_folder = os.getenv("QWEN3VL30B_OFFLOAD_FOLDER", "").strip()
        self.use_asr = os.getenv("QWEN3VL30B_USE_ASR", os.getenv("QWEN25VL_USE_ASR", "True")).lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.transcribe_video_audio = os.getenv("QWEN3VL30B_TRANSCRIBE_VIDEO_AUDIO", "True").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.asr_model_path = os.getenv(
            "QWEN3VL30B_ASR_MODEL",
            os.getenv(
                "QWEN25VL_ASR_MODEL",
                "openai/whisper-large-v3-turbo",
            ),
        )
        self.asr_device = os.getenv("QWEN3VL30B_ASR_DEVICE", os.getenv("QWEN25VL_ASR_DEVICE", "cuda"))
        self.asr_min_language_confidence = float(
            os.getenv("QWEN3VL30B_ASR_MIN_LANGUAGE_CONFIDENCE", os.getenv("QWEN25VL_ASR_MIN_LANGUAGE_CONFIDENCE", "0.65"))
        )
        self.asr_non_speech_text = os.getenv("QWEN3VL30B_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.processor = None
        self.model = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        kwargs = {
            "dtype": self._resolve_dtype(torch, self.torch_dtype),
            "device_map": self.device_map,
        }
        if self.max_memory:
            kwargs["max_memory"] = self._parse_max_memory(self.max_memory)
        if self.offload_folder:
            os.makedirs(self.offload_folder, exist_ok=True)
            kwargs["offload_folder"] = self.offload_folder
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(self.model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._patch_video_placeholder_token()
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
    def _register_pyav_video_reader():
        try:
            import av
            import numpy as np
            import torch
            import qwen_vl_utils.vision_process as vision_process
        except Exception:
            return

        if "pyav" in vision_process.VIDEO_READER_BACKENDS:
            vision_process.FORCE_QWENVL_VIDEO_READER = "pyav"
            vision_process.VIDEO_READER_BACKENDS["torchvision"] = vision_process.VIDEO_READER_BACKENDS["pyav"]
            vision_process.get_video_reader_backend.cache_clear()
            return

        def _read_video_pyav(ele):
            video_path = ele["video"]
            if isinstance(video_path, str) and video_path.startswith("file://"):
                video_path = video_path[7:]

            frames = []
            with av.open(video_path) as container:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if stream is None:
                    raise ValueError(f"No video stream found in {video_path}")

                video_fps = float(stream.average_rate or stream.base_rate or 1.0)
                start_time = ele.get("video_start", 0.0) or 0.0
                end_time = ele.get("video_end", None)
                for frame in container.decode(stream):
                    frame_time = frame.time
                    if frame_time is not None:
                        if frame_time < start_time:
                            continue
                        if end_time is not None and frame_time > end_time:
                            break
                    frames.append(frame.to_ndarray(format="rgb24"))

            if not frames:
                raise ValueError(f"No video frames decoded from {video_path}")

            frame_factor = int(getattr(vision_process, "FRAME_FACTOR", 2))
            if len(frames) < frame_factor:
                frames.extend([frames[-1]] * (frame_factor - len(frames)))

            total_frames = len(frames)
            nframes = vision_process.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
            indices = torch.linspace(0, total_frames - 1, nframes).round().long()
            sampled = [frames[index] for index in indices.tolist()]
            video = torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2)
            sample_fps = nframes / max(total_frames, 1e-6) * video_fps
            video_metadata = {
                "fps": video_fps,
                "frames_indices": indices,
                "total_num_frames": total_frames,
                "video_backend": "pyav",
            }
            return video, video_metadata, sample_fps

        vision_process.VIDEO_READER_BACKENDS["pyav"] = _read_video_pyav
        vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_pyav
        vision_process.FORCE_QWENVL_VIDEO_READER = "pyav"
        vision_process.get_video_reader_backend.cache_clear()

    @staticmethod
    def _register_transformers_pyav_video_reader():
        try:
            from transformers.video_processing_utils import BaseVideoProcessor
            from transformers.video_utils import VideoMetadata, load_video
        except Exception:
            return

        if getattr(BaseVideoProcessor, "_omnievalkit_pyav_fetch", False):
            return

        def load_video_robust(video_path, sample_indices_fn):
            try:
                return load_video(video_path, backend="pyav", sample_indices_fn=sample_indices_fn)
            except Exception as pyav_exc:
                try:
                    return load_video_with_cv2(video_path, sample_indices_fn)
                except Exception as cv2_exc:
                    raise ValueError(
                        f"Could not decode any frames from video: {video_path}. "
                        f"pyav error: {pyav_exc}; cv2 fallback error: {cv2_exc}"
                    ) from cv2_exc

        def load_video_with_cv2(video_path, sample_indices_fn):
            import cv2
            import numpy as np

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError("cv2 could not open video")

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            metadata = VideoMetadata(
                total_num_frames=max(frame_count, 1),
                fps=fps,
                duration=float(frame_count / fps) if frame_count and fps else 0.0,
                video_backend="cv2",
                height=height,
                width=width,
            )
            indices = sample_indices_fn(metadata=metadata)
            indices = sorted({int(index) for index in indices if int(index) >= 0})

            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while len(frames) < max(len(indices), 1):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            capture.release()

            if not frames:
                raise ValueError("cv2 decoded zero frames")

            if len(frames) < max(len(indices), 1):
                frames.extend([frames[-1]] * (max(len(indices), 1) - len(frames)))
            metadata.frames_indices = indices[: len(frames)]
            return np.stack(frames), metadata

        def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
            if isinstance(video_url_or_urls, list):
                return list(zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
            return load_video_robust(video_url_or_urls, sample_indices_fn=sample_indices_fn)

        BaseVideoProcessor.fetch_videos = fetch_videos
        BaseVideoProcessor._omnievalkit_pyav_fetch = True

    def _patch_video_placeholder_token(self):
        if not hasattr(self.processor, "replace_video_token"):
            return

        def replace_video_token(processor, video_inputs, video_idx):
            merge_length = processor.video_processor.merge_size**2
            num_frames = video_inputs["video_grid_thw"][video_idx][0]
            frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
            metadata = video_inputs["video_metadata"][video_idx]

            if metadata.fps is None:
                metadata.fps = 24
            timestamps = processor._calculate_timestamps(
                metadata.frames_indices,
                metadata.fps,
                processor.video_processor.temporal_patch_size,
            )

            video_placeholder = ""
            for frame_idx in range(num_frames):
                video_placeholder += f"<{timestamps[frame_idx]:.1f} seconds>"
                video_placeholder += (
                    processor.vision_start_token
                    + processor.video_token * frame_seqlen
                    + processor.vision_end_token
                )
            return video_placeholder

        self.processor.replace_video_token = replace_video_token.__get__(self.processor, type(self.processor))

    def _load_asr(self):
        if self.asr is not None:
            return self.asr

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = self.asr_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        is_cuda = str(device).startswith("cuda")
        dtype = torch.float16 if is_cuda else torch.float32
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.asr_model.to(device)
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_path)
        pipeline_device = 0
        if is_cuda and ":" in str(device):
            pipeline_device = int(str(device).rsplit(":", 1)[-1])
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            torch_dtype=dtype,
            device=pipeline_device if is_cuda else -1,
        )
        return self.asr

    @staticmethod
    def _decode_audio_with_pyav(audio_path, sampling_rate=16000):
        import av
        import numpy as np

        chunks = []
        with av.open(audio_path) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise ValueError(f"No audio stream found in {audio_path}")

            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            raise ValueError(f"No audio samples decoded from {audio_path}")

        audio = np.concatenate(chunks)
        if np.issubdtype(audio.dtype, np.integer):
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    @staticmethod
    def _load_audio_with_ffmpeg(audio_path, sampling_rate=16000):
        import numpy as np

        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            str(audio_path),
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
        array = np.frombuffer(completed.stdout, dtype=np.float32)
        if array.size:
            return array.copy()
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed with code {completed.returncode}: {error}")
        raise ValueError(f"No audio decoded from {audio_path}")

    def _load_audio_for_asr(self, path):
        ffmpeg_error = None
        try:
            return self._load_audio_with_ffmpeg(path, 16000)
        except Exception as exc:
            ffmpeg_error = exc
        try:
            return self._decode_audio_with_pyav(path, 16000)
        except Exception:
            pass
        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                array, _ = librosa.load(path, sr=16000, mono=True)
            return array
        except Exception as exc:
            raise RuntimeError(f"Failed to decode audio for ASR: {path}. ffmpeg error: {ffmpeg_error}") from exc

    def _whisper_analysis(self, audio_path):
        array = self._load_audio_for_asr(audio_path)
        if self._is_silent_audio(array):
            return {"language": "", "language_confidence": 0.0, "transcript": ""}

        language_token, language_confidence = self._detect_spoken_language(array)
        language = self._format_whisper_language(language_token)
        if language_confidence < self.asr_min_language_confidence:
            return {"language": language, "language_confidence": language_confidence, "transcript": ""}

        asr = self._load_asr()
        audio = {"array": array, "sampling_rate": 16000}
        duration_seconds = len(array) / 16000
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
        return {"language": language, "language_confidence": language_confidence, "transcript": transcript}

    def _transcribe_media_sources(self, audio_paths, video_path):
        blocks = []
        seen = set()
        for audio_path in audio_paths:
            block = self._format_whisper_block("Audio file", self._whisper_analysis(audio_path))
            normalized = self._normalize_transcript_for_dedupe(block)
            if block and normalized not in seen:
                seen.add(normalized)
                blocks.append(block)

        if self.transcribe_video_audio and video_path and self._has_audio_stream(video_path):
            try:
                block = self._format_whisper_block("Video audio", self._whisper_analysis(video_path))
            except Exception as exc:
                print(f"ASR skipped video audio: {video_path}: {exc}")
            else:
                normalized = self._normalize_transcript_for_dedupe(block)
                if block and normalized not in seen:
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
        inputs = processor(array, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
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
    def _format_whisper_language(cls, token):
        if not token:
            return ""
        return cls.WHISPER_LANGUAGE_NAMES.get(token, token.strip("<|>"))

    @staticmethod
    def _is_silent_audio(array):
        if array is None or len(array) == 0:
            return True
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
        peak = float(np.max(np.abs(array))) if len(array) else 0.0
        return rms < 1e-4 and peak < 1e-3

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

    def _should_include_transcript(self, transcript):
        transcript = (transcript or "").strip()
        return bool(transcript) and transcript != self.asr_non_speech_text and not self._is_bad_transcript(transcript)

    @staticmethod
    def _normalize_transcript_for_dedupe(transcript):
        return re.sub(r"\s+", " ", (transcript or "").strip().lower())

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

    @classmethod
    def _build_text(cls, query, transcript):
        parts = []
        if transcript:
            parts.append(transcript)
        parts.append(cls._sanitize_indexed_media_placeholders(query))
        return "\n\n".join(part for part in parts if part)

    def _build_messages(self, video_path, image_paths, query, system_prompt, transcript):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt).strip()})

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            content.append({"type": "video", "video": video_path})

        text = self._build_text(query, transcript)
        content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_inputs(self, messages):
        self._register_pyav_video_reader()
        self._register_transformers_pyav_video_reader()
        processor_kwargs = {}
        if self._messages_have_video(messages):
            if self.video_num_frames is not None:
                processor_kwargs["videos_kwargs"] = {"num_frames": self.video_num_frames, "fps": None}
            else:
                processor_kwargs["videos_kwargs"] = {"fps": self.video_fps}
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        return inputs.to(self._input_device())

    @staticmethod
    def _messages_have_video(messages):
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                if any(item.get("type") == "video" for item in content if isinstance(item, dict)):
                    return True
        return False

    def _input_device(self):
        import torch

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        del sample
        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)

        transcript = ""
        if self.use_asr:
            transcript = self._transcribe_media_sources(resolved_audio_paths, resolved_video)

        messages = self._build_messages(resolved_video, resolved_images, query, system_prompt, transcript)
        inputs = self._prepare_inputs(messages)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**generate_kwargs)

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
