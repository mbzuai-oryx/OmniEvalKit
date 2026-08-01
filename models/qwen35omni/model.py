import hashlib
import logging
import os
import re
import subprocess
import warnings
from pathlib import Path

from .run_qwen35_multimodal import (
    force_qwen35_video_tokens,
    force_video_backend,
    validate_image_token_alignment,
    validate_video_token_alignment,
)

MAX_IMAGE_SIZE = 1344


class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("QWEN35OMNI_MODEL_PATH", "Qwen/Qwen3.5-2B")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("QWEN35OMNI_TOP_P", "0.8"))
        video_num_frames = os.getenv("QWEN35OMNI_VIDEO_NUM_FRAMES", "").strip()
        self.video_num_frames = int(video_num_frames) if video_num_frames else None
        self.video_fps = float(os.getenv("QWEN35OMNI_VIDEO_FPS", "1.0"))
        self.video_min_pixels = int(os.getenv("QWEN35OMNI_VIDEO_MIN_PIXELS", str(256 * 256)))
        self.video_max_pixels = int(os.getenv("QWEN35OMNI_VIDEO_MAX_PIXELS", str(1024 * 1024)))
        self.video_backend = os.getenv("QWEN35OMNI_VIDEO_BACKEND", "pyav")
        self.device_map = os.getenv("QWEN35OMNI_DEVICE_MAP", "auto")
        self.dtype_name = os.getenv("QWEN35OMNI_DTYPE", "bfloat16")
        self.use_asr = os.getenv("QWEN35OMNI_USE_ASR", "False").lower() in {"1", "true", "yes", "y"}
        self.asr_model_path = os.getenv(
            "QWEN35OMNI_ASR_MODEL",
            "openai/whisper-large-v3-turbo",
        )
        self.asr_device = os.getenv("QWEN35OMNI_ASR_DEVICE", "cpu")
        self.asr_min_language_confidence = float(os.getenv("QWEN35OMNI_ASR_MIN_LANGUAGE_CONFIDENCE", "0.50"))
        self.asr_non_speech_text = os.getenv("QWEN35OMNI_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.processor = None
        self.model = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
        self.delegate = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if getattr(config, "model_type", None) == "qwen2_5_omni":
            from models.qwen25omni.model import Model as Qwen25OmniModel

            self.delegate = Qwen25OmniModel(model_path=self.model_path)
            return

        dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float16
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        force_video_backend(self.processor, self.video_backend)
        force_qwen35_video_tokens(self.processor)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
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
        cache_dir = Path(os.getenv("QWEN35OMNI_MEDIA_CACHE", "/tmp/qwen35omni_media_cache"))
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

    @staticmethod
    def _video_readable(video_path):
        try:
            import av

            with av.open(video_path) as container:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if stream is None:
                    return False
                next(container.decode(stream))
            return True
        except Exception:
            return False

    @classmethod
    def _prepare_video_path(cls, video_path):
        if not video_path or cls._is_remote_path(video_path):
            return video_path

        dimensions = cls._video_dimensions(video_path)
        if not dimensions:
            return video_path
        width, height = dimensions
        new_width, new_height = cls._scaled_size(width, height)
        new_width = max(2, new_width - (new_width % 2))
        new_height = max(2, new_height - (new_height % 2))
        if (new_width, new_height) == (width, height):
            return video_path

        output_path = cls._cache_path(video_path, ".mp4")
        if output_path.exists():
            if (
                output_path.stat().st_size > 0
                and cls._video_dimensions(str(output_path))
                and cls._video_readable(str(output_path))
            ):
                return str(output_path)
            output_path.unlink(missing_ok=True)

        temp_output_path = output_path.with_suffix(".tmp.mp4")
        temp_output_path.unlink(missing_ok=True)
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
            str(temp_output_path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (
            completed.returncode != 0
            or not temp_output_path.exists()
            or temp_output_path.stat().st_size == 0
            or not cls._video_dimensions(str(temp_output_path))
            or not cls._video_readable(str(temp_output_path))
        ):
            temp_output_path.unlink(missing_ok=True)
            logging.warning("Failed to resize video for ROCm, using original: %s: %s", video_path, completed.stderr.strip())
            return video_path
        temp_output_path.replace(output_path)
        logging.info("Resized video for ROCm: %s %sx%s -> %sx%s", video_path, width, height, new_width, new_height)
        return str(output_path)

    @staticmethod
    def _is_gpu_kernel_error(exc):
        text = str(exc)
        return "invalid configuration" in text or "hipError" in text or "AcceleratorError" in text

    @staticmethod
    def _sample_audio_type(sample):
        if not sample:
            return ""
        return str(sample.get("audio_type", "") or "").strip().lower()

    def _use_native_audio(self, audio_type, audio_count):
        return False

    def _load_asr(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if self.asr is not None:
            return self.asr

        device = self.asr_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        is_cuda = device.startswith("cuda")
        torch_dtype = torch.float16 if is_cuda else torch.float32
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.asr_model.to(device)
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_path)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=0 if is_cuda else -1,
        )
        return self.asr

    # def _transcribe_audio(self, audio_path):
    #     from .run_qwen35_multimodal import decode_audio_with_pyav

    #     asr = self._load_asr()
    #     audio = decode_audio_with_pyav(audio_path)
    #     result = asr(
    #         audio,
    #         # return_timestamps=False,
    #         return_timestamps=False,
    #         generate_kwargs={"task": "transcribe"},
    #     )
    #     return result["text"].strip()


    def _transcribe_audio(self, audio_path, with_timestamps=False):
        asr = self._load_asr()

        array = self._load_audio_for_asr(audio_path)
        if self._is_silent_audio(array):
            return self.asr_non_speech_text

        language_token, language_confidence = self._detect_spoken_language(array)
        if language_confidence < self.asr_min_language_confidence:
            print(
                f"ASR skipped non-speech/uncertain audio: language={language_token or 'unknown'} "
                f"confidence={language_confidence:.3f}"
            )
            return self.asr_non_speech_text

        if with_timestamps:
            return self._transcribe_audio_with_timestamps(array)

        audio = {
            "array": array,
            "sampling_rate": 16000,
        }

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
        transcript = result["text"].strip()
        transcript = self._suppress_hallucination(transcript)
        return self.asr_non_speech_text if self._is_bad_transcript(transcript) else transcript

    @staticmethod
    def _suppress_hallucination(transcript: str) -> str:
        """
        If Whisper output is >30% pilcrow characters, it hallucinated.
        Replace with a neutral placeholder instead of injecting garbage.
        """
        if not transcript or not transcript.strip():
            return "[no audio content]"
        pilcrow_ratio = transcript.count("¶") / len(transcript)
        if pilcrow_ratio > 0.3:
            return "[non-speech audio]"
        return transcript

    def _transcribe_audio_with_timestamps(self, array):
        asr = self._load_asr()
        audio = {
            "array": array,
            "sampling_rate": 16000,
        }
        result = asr(
            audio,
            return_timestamps="word",
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
        transcript = self._suppress_hallucination(transcript)
        if transcript in {"[no audio content]", "[non-speech audio]"}:
            return transcript
        if self._is_bad_transcript(transcript):
            return self.asr_non_speech_text

        timestamped = self._sentence_timestamp_transcript(result.get("chunks", []))
        return timestamped or transcript

    @staticmethod
    def _sentence_timestamp_transcript(chunks):
        sentences = []
        words = []
        start_time = None
        end_time = None
        last_end_time = -1.0
        sentence_end_re = re.compile(r"[.!?。！？।؟]+$")

        def add_sentence(sentence, st, et):
            nonlocal last_end_time
            if st < last_end_time - 0.25:
                return
            sentences.append(f"[{st:.2f}s-{et:.2f}s] : {sentence}")
            last_end_time = et

        for item in chunks or []:
            word = item.get("text", "")
            timestamp = item.get("timestamp")
            if not timestamp:
                continue
            st, et = timestamp
            if st is None or et is None:
                continue
            if start_time is None:
                start_time = float(st)
            words.append(word)
            end_time = float(et)
            if sentence_end_re.search(word.strip()):
                sentence = Model._clean_timestamp_sentence(words)
                if sentence:
                    add_sentence(sentence, start_time, end_time)
                words = []
                start_time = None
                end_time = None

        if words and start_time is not None and end_time is not None:
            sentence = Model._clean_timestamp_sentence(words)
            if sentence:
                add_sentence(sentence, start_time, end_time)
        return "\n".join(sentences)

    @staticmethod
    def _clean_timestamp_sentence(words):
        sentence = "".join(words).strip()
        return re.sub(r"\s+", " ", sentence)

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

    @staticmethod
    def _is_bad_transcript(text, threshold=0.5):
        text = (text or "").strip()
        if not text:
            return True
        lowered = text.lower()
        if "[blank_audio]" in lowered or "blank audio" in lowered:
            return True
        bad_chars = sum(text.count(char) for char in ("¶", "♪", "♩", "♫", "�"))
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
    def _load_audio_for_asr(audio_path):
        ffmpeg_error = None
        try:
            return Model._load_audio_with_ffmpeg(audio_path)
        except Exception as exc:
            ffmpeg_error = exc

        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                array, _ = librosa.load(audio_path, sr=16000, mono=True)
            return array
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode audio for ASR: {audio_path}. ffmpeg error: {ffmpeg_error}"
            ) from exc

    @staticmethod
    def _load_audio_with_ffmpeg(audio_path):
        import numpy as np

        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            audio_path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
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

    def _build_text(self, query, transcript, transcript_has_timestamps=False):
        parts = []
        if transcript:
            if transcript.startswith("Audio 1 transcript"):
                parts.append(transcript)
            else:
                transcript_label = "Audio transcript with timestamps" if transcript_has_timestamps else "Audio transcript"
                parts.append(f"{transcript_label}:\n{transcript}")
        parts.append(query.strip())
        return "\n\n".join(parts)

    def _build_messages(
        self,
        video_path,
        image_paths,
        audio_paths,
        query,
        system_prompt,
        transcript,
        transcript_has_timestamps=False,
        use_native_audio=False,
    ):
        text = self._build_text(query, transcript, transcript_has_timestamps=transcript_has_timestamps)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            video_content = {"type": "video", "video": video_path, "fps": self.video_fps}
            if self.video_num_frames is not None:
                video_content["num_frames"] = self.video_num_frames
            content.append(video_content)
        if use_native_audio:
            for audio_path in audio_paths:
                content.append({"type": "audio", "audio": audio_path})
        if content:
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": text})
        return messages

    def _transcribe_audio_paths(self, audio_paths, with_timestamps=False):
        transcripts = [
            self._transcribe_audio(audio_path, with_timestamps=with_timestamps)
            for audio_path in audio_paths
        ]
        if len(transcripts) == 1:
            return transcripts[0]
        return "\n\n".join(
            f"Audio {idx} transcript:\n{transcript}"
            for idx, transcript in enumerate(transcripts, start=1)
        )

    @staticmethod
    def _needs_timestamp_transcript(sample):
        return bool(sample) and str(sample.get("task_type", "")).strip() == "Audio Event Location"

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        if self.delegate is not None:
            return self.delegate.run_inference(audio_path, video_path, query, system_prompt, image_path=image_path, sample=sample)

        resolved_video = self._prepare_video_path(self._path_value(video_path))
        resolved_images = self._prepare_image_paths(self._path_values(image_path))
        resolved_audio_paths = self._path_values(audio_path)
        use_native_audio = self._use_native_audio(
            self._sample_audio_type(sample),
            len(resolved_audio_paths),
        )
        transcript = ""
        transcript_has_timestamps = self._needs_timestamp_transcript(sample)
        if self.use_asr and resolved_audio_paths and not use_native_audio:
            transcript = self._transcribe_audio_paths(
                resolved_audio_paths,
                with_timestamps=transcript_has_timestamps,
            )

        messages = self._build_messages(
            resolved_video,
            resolved_images,
            resolved_audio_paths,
            query,
            system_prompt,
            transcript,
            transcript_has_timestamps=transcript_has_timestamps,
            use_native_audio=use_native_audio,
        )
        processor_kwargs = {}
        if resolved_video:
            processor_kwargs = {
                "videos_kwargs": {
                    "fps": self.video_fps,
                    "do_sample_frames": True,
                    "size": {
                        "shortest_edge": self.video_min_pixels,
                        "longest_edge": self.video_max_pixels,
                    },
                },
            }
            if self.video_num_frames is not None:
                processor_kwargs["videos_kwargs"]["num_frames"] = self.video_num_frames

        # print('messages => ', messages)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        
        validate_image_token_alignment(self.processor, inputs)
        validate_video_token_alignment(self.processor, inputs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        # with torch.no_grad():
        #     output_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_new_tokens,
        #         temperature=self.temperature,
        #         top_p=self.top_p,
        #         do_sample=self.temperature > 0,
        #     )

        ## Non-thinking mode for vision tasks:
        # with torch.no_grad():
        #     output_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_new_tokens,           # keep short
        #         do_sample=True,
        #         temperature=0.7, 
        #         top_p=0.9,      
        #         min_p=0.0,       
        #         top_k=20,        
        #         # presence_penalty=1.5, 
        #         repetition_penalty=1.0,       # ← replaces presence_penalty (not supported in HF)
        #     )

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.30,
                    no_repeat_ngram_size=4,
                )
        except Exception as exc:
            if self._is_gpu_kernel_error(exc):
                sample_id = sample.get("id") if sample else "unknown"
                logging.warning("Skipping sample %s: ROCm kernel error on this input — %s", sample_id, exc)
                return "[ERROR: GPU kernel config]"
            raise

        generated_ids = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # print("Final Ans => ", decoded[0].strip())
        return decoded[0].strip() if decoded else ""


# QWEN35OMNI_USE_ASR=True MODEL=qwen35omni MODEL_PATH=Qwen/Qwen3.5-2B EVAL_DATASETS=daily_omni OUTPUT_DIR=results/qwen35_omni bash eval.sh
