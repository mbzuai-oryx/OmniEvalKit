import os
import re
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile


class Model:
    SENSEVOICE_LANGUAGE_TAGS = {
        "<|zh|>": "Chinese (Mandarin)",
        "<|en|>": "English",
        "<|yue|>": "Cantonese",
        "<|ja|>": "Japanese",
        "<|ko|>": "Korean",
        "<|nospeech|>": "No Speech",
    }
    SENSEVOICE_EMOTION_TAGS = {
        "<|HAPPY|>": "Happy",
        "<|SAD|>": "Sad",
        "<|ANGRY|>": "Angry",
        "<|NEUTRAL|>": "Neutral",
        "<|FEARFUL|>": "Fearful",
        "<|DISGUSTED|>": "Disgusted",
        "<|SURPRISED|>": "Surprised",
    }
    SENSEVOICE_EVENT_TAGS = {
        "<|BGM|>": "Background Music",
        "<|Speech|>": "Speech",
        "<|Applause|>": "Applause",
        "<|Laughter|>": "Laughter",
        "<|Cry|>": "Crying",
        "<|Sneeze|>": "Sneeze",
        "<|Breath|>": "Breath",
        "<|Cough|>": "Cough",
    }
    SENSEVOICE_LANGUAGE_TOKEN_IDS = {
        "zh": 24884,
        "en": 24885,
        "yue": 24888,
        "ja": 24892,
        "ko": 24896,
        "nospeech": 24992,
    }
    SENSEVOICE_RICH_TOKEN_IDS = {
        "language": {
            "zh": 24884,
            "en": 24885,
            "yue": 24888,
            "ja": 24892,
            "ko": 24896,
            "nospeech": 24992,
        },
        "emotion": {
            "Happy": 25001,
            "Sad": 25002,
            "Angry": 25003,
            "Neutral": 25004,
            "unknown": 25009,
        },
        "event": {
            "Speech": 24993,
            "Background Music": 24995,
            "Laughter": 24997,
            "Applause": 24999,
        },
    }
    SENSEVOICE_RICH_POSITIONS = {
        "language": 0,
        "emotion": 1,
        "event": 2,
    }
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
    MELLOW_AUDIO_PROMPT = (
        "Describe this audio clip — what type of audio it is, mention any speech, "
        "background music and prominent sound events, summarize your overall understanding of it."
    )

    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("QWEN25VL_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("QWEN25VL_TOP_P", "0.8"))
        self.repetition_penalty = float(os.getenv("QWEN25VL_REPETITION_PENALTY", "1.0"))
        self.video_fps = float(os.getenv("QWEN25VL_VIDEO_FPS", "1.0"))
        video_num_frames = os.getenv("QWEN25VL_VIDEO_NUM_FRAMES", "").strip()
        self.video_num_frames = int(video_num_frames) if video_num_frames else None
        self.device_map = os.getenv("QWEN25VL_DEVICE_MAP", "auto")
        self.torch_dtype = os.getenv("QWEN25VL_TORCH_DTYPE", "auto")
        self.attn_implementation = os.getenv("QWEN25VL_ATTN_IMPLEMENTATION", "")
        self.use_asr = os.getenv("QWEN25VL_USE_ASR", "False").lower() in {"1", "true", "yes", "y"}
        self.asr_model_path = os.getenv(
            "QWEN25VL_ASR_MODEL",
            "openai/whisper-large-v3-turbo",
        )
        self.asr_device = os.getenv("QWEN25VL_ASR_DEVICE", "cuda")
        self.asr_language = os.getenv("QWEN25VL_ASR_LANGUAGE", "").strip()
        self.asr_min_language_confidence = float(os.getenv("QWEN25VL_ASR_MIN_LANGUAGE_CONFIDENCE", "0.65"))
        self.asr_non_speech_text = os.getenv("QWEN25VL_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.use_sensevoice = os.getenv("QWEN25VL_ADVANCE_USE_SENSEVOICE", "False").lower() in {"1", "true", "yes", "y"}
        self.sensevoice_model_path = os.getenv(
            "QWEN25VL_SENSEVOICE_MODEL",
            "FunAudioLLM/SenseVoiceSmall",
        )
        self.sensevoice_device = os.getenv("QWEN25VL_SENSEVOICE_DEVICE", self.asr_device)
        self.sensevoice_tag_threshold = float(os.getenv("QWEN25VL_SENSEVOICE_TAG_THRESHOLD", "0.65"))
        self.use_mellow = os.getenv("QWEN25VL_USE_MELLOW", "False").lower() in {"1", "true", "yes", "y"}
        self.mellow_src = Path(os.getenv("QWEN25VL_MELLOW_SRC", "Mellow-src"))
        self.mellow_dir = Path(os.getenv("QWEN25VL_MELLOW_DIR", "mellow"))
        self.mellow_checkpoint_path = Path(
            os.getenv("QWEN25VL_MELLOW_CHECKPOINT", str(self.mellow_dir / "v0_s.ckpt"))
        )
        self.mellow_smollm_path = Path(
            os.getenv("QWEN25VL_MELLOW_SMOLLM", "SmolLM2-135M")
        )
        self.mellow_max_len = int(os.getenv("QWEN25VL_MELLOW_MAX_LEN", "192"))
        self.mellow_top_p = float(os.getenv("QWEN25VL_MELLOW_TOP_P", "0.8"))
        self.mellow_temperature = float(os.getenv("QWEN25VL_MELLOW_TEMPERATURE", "1.0"))
        self.parallel_audio_analysis = os.getenv("QWEN25VL_ADVANCE_PARALLEL_AUDIO", "True").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.processor = None
        self.model = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
        self.sensevoice = None
        self.mellow = None
        self._load()

    def _load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
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
    def _sample_audio_type(sample):
        if not sample:
            return ""
        return str(sample.get("audio_type", "") or "").strip().lower()

    @staticmethod
    def _decode_audio_with_pyav(audio_path):
        import av
        import numpy as np

        chunks = []
        with av.open(audio_path) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise ValueError(f"No audio stream found in {audio_path}")

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=16000,
            )
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
    def _register_pyav_video_reader():
        import av
        import numpy as np
        import torch
        import qwen_vl_utils.vision_process as vision_process

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
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=total_frames,
                video_fps=video_fps,
            )
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

    def _load_sensevoice(self):
        if self.sensevoice is not None:
            return self.sensevoice

        from funasr import AutoModel

        device = self.sensevoice_device
        if device == "auto":
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"

        self.sensevoice = AutoModel(
            model=self.sensevoice_model_path,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
        )
        return self.sensevoice

    def _load_mellow(self):
        if self.mellow is not None:
            return self.mellow

        import numpy as np
        import soundfile as sf
        import torch
        import yaml

        if str(self.mellow_src) not in sys.path:
            sys.path.insert(0, str(self.mellow_src))

        import mellow.wrapper as mellow_wrapper
        from mellow import MellowWrapper

        original_hf_hub_download = mellow_wrapper.hf_hub_download

        def local_hf_hub_download(repo_id, filename, *args, **kwargs):
            local_files = {
                "v0_s.ckpt": self.mellow_checkpoint_path,
                "v0.ckpt": self.mellow_dir / "v0.ckpt",
                "v0.yaml": self.mellow_dir / "v0.yaml",
                "config.json": self.mellow_dir / "config.json",
            }
            if filename in local_files:
                return str(local_files[filename])
            return original_hf_hub_download(repo_id, filename, *args, **kwargs)

        def soundfile_load(path):
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = np.asarray(audio).T
            return torch.from_numpy(audio), sample_rate

        mellow_wrapper.hf_hub_download = local_hf_hub_download
        mellow_wrapper.torchaudio.load = soundfile_load
        self._patch_mellow_config(mellow_wrapper, yaml)

        cuda = torch.cuda.is_available()
        device = 0 if cuda else "cpu"
        mellow = MellowWrapper(
            config="v0",
            model="v0_s",
            device=device,
            use_cuda=cuda,
        )
        mellow.model = mellow.model.float()
        if not hasattr(mellow.tokenizer, "encode_plus"):
            def encode_plus_compat(*args, **kwargs):
                if kwargs.pop("pad_to_max_length", False):
                    kwargs["padding"] = "max_length"
                return mellow.tokenizer(*args, **kwargs)

            mellow.tokenizer.encode_plus = encode_plus_compat

        self.mellow = mellow
        return self.mellow

    def _patch_mellow_config(self, mellow_wrapper, yaml):
        config_path = self.mellow_src / "mellow" / "config" / "v0.yaml"
        with config_path.open() as handle:
            config = yaml.safe_load(handle)

        config["data"]["tokenizer_type"] = str(self.mellow_smollm_path)
        config["model"]["decoder"]["text_decoder"] = str(self.mellow_smollm_path)

        tmp_dir = Path(tempfile.mkdtemp(prefix="mellow_cfg_"))
        patched_path = tmp_dir / "v0.yaml"
        with patched_path.open("w") as handle:
            yaml.safe_dump(config, handle)
        mellow_wrapper.files = lambda package: tmp_dir
        return patched_path

    def _mellow_description(self, audio_path):
        tmp_path = None
        try:
            input_path, tmp_path = self._mellow_input_path(audio_path)
            mellow = self._load_mellow()
            response = mellow.generate(
                examples=[[str(input_path), str(input_path), self.MELLOW_AUDIO_PROMPT]],
                max_len=self.mellow_max_len,
                top_p=self.mellow_top_p,
                temperature=self.mellow_temperature,
            )
            description = str(response[0]).strip() if response else ""
            return "" if self._is_bad_transcript(description) else description
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _mellow_input_path(self, audio_path):
        self._require_audio_stream(audio_path)
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            audio_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            tmp_path,
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed to extract Mellow audio: {error}")
        return tmp_path, tmp_path

    def _whisper_analysis(self, audio_path, with_timestamps=False):
        array = self._load_audio_for_asr(audio_path)
        if self._is_silent_audio(array):
            return {
                "language": "",
                "language_confidence": 0.0,
                "transcript": "",
                "skipped": "silent audio",
            }

        language_token, language_confidence = self._detect_spoken_language(array)
        language = self._format_whisper_language(language_token)
        if language_confidence < self.asr_min_language_confidence:
            return {
                "language": language,
                "language_confidence": language_confidence,
                "transcript": "",
                "skipped": "uncertain speech/language",
            }

        transcript = self._transcribe_audio_with_timestamps(array) if with_timestamps else self._transcribe_audio_array(array)
        if not self._should_include_transcript(transcript):
            transcript = ""
        return {
            "language": language,
            "language_confidence": language_confidence,
            "transcript": transcript,
            "skipped": "" if transcript else "empty or unreliable transcript",
        }

    def _transcribe_audio_array(self, array):
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
            generate_kwargs=self._asr_generate_kwargs(
                {
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_prev_tokens": False,
                }
            ),
        )
        transcript = result["text"].strip()
        return self.asr_non_speech_text if self._is_bad_transcript(transcript) else transcript

    def _asr_generate_kwargs(self, kwargs):
        if self.asr_language:
            kwargs = dict(kwargs)
            kwargs["language"] = self.asr_language
        return kwargs

    @classmethod
    def _format_whisper_language(cls, language_token):
        if not language_token:
            return ""
        return cls.WHISPER_LANGUAGE_NAMES.get(language_token, language_token.strip("<|>"))

    def _sensevoice_analysis(self, audio_path):
        tmp_path = None
        input_path = audio_path
        try:
            input_path, tmp_path = self._sensevoice_input_path(audio_path)
            model = self._load_sensevoice()
            rich_probs = self._get_sensevoice_rich_probs(model, input_path)
            return self._sensevoice_cues_from_probs(rich_probs)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _sensevoice_input_path(self, audio_path):
        suffix = Path(audio_path).suffix.lower()
        if suffix in {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}:
            return audio_path, None

        self._require_audio_stream(audio_path)
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            audio_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            tmp_path,
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed to extract SenseVoice audio: {error}")
        return tmp_path, tmp_path

    @staticmethod
    def _require_audio_stream(media_path):
        if Model._has_audio_stream(media_path):
            return
        raise RuntimeError(f"No audio stream found in media file: {media_path}")

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
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffprobe failed while checking audio stream for {media_path}: {error}")
        return bool(completed.stdout.decode("utf-8", errors="replace").strip())

    def _get_sensevoice_rich_probs(self, auto_model, audio_file, topk=5):
        import torch
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        asr_model = auto_model.model
        kwargs = auto_model.kwargs
        frontend = kwargs["frontend"]
        tokenizer = kwargs["tokenizer"]
        model_device = kwargs["device"]

        audio_sample_list = load_audio_text_image_video(
            audio_file,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )
        speech, speech_lengths = extract_fbank(
            audio_sample_list,
            data_type=kwargs.get("data_type", "sound"),
            frontend=frontend,
        )
        speech = speech.to(device=model_device)
        speech_lengths = speech_lengths.to(device=model_device)

        language_query = asr_model.embed(
            torch.LongTensor([[asr_model.lid_dict["auto"]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        textnorm_query = asr_model.embed(
            torch.LongTensor([[asr_model.textnorm_dict["withitn"]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = asr_model.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(speech.size(0), 1, 1)
        speech = torch.cat((language_query, event_emo_query, speech), dim=1)
        speech_lengths += 3

        with torch.no_grad():
            encoder_out, _ = asr_model.encoder(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            ctc_log_probs = asr_model.ctc.log_softmax(encoder_out)

        results = {}
        for name, token_ids in self.SENSEVOICE_RICH_TOKEN_IDS.items():
            ids = torch.tensor(list(token_ids.values()), device=ctc_log_probs.device)
            log_probs = ctc_log_probs[0, self.SENSEVOICE_RICH_POSITIONS[name], ids]
            probs = torch.softmax(log_probs, dim=0).detach().cpu().tolist()
            ranked = sorted(zip(token_ids.keys(), probs), key=lambda item: item[1], reverse=True)
            results[name] = ranked[:topk]
        return results

    def _sensevoice_cues_from_probs(self, rich_probs):
        emotions = [
            label
            for label, probability in rich_probs.get("emotion", [])
            if label != "unknown" and probability >= self.sensevoice_tag_threshold
        ]
        events = [
            label
            for label, probability in rich_probs.get("event", [])
            if probability >= self.sensevoice_tag_threshold
        ]
        return {
            "emotions": emotions,
            "events": events,
        }

    # def _transcribe_audio(self, audio_path):
    #     asr = self._load_asr()
    #     audio = {
    #         "array": self._decode_audio_with_pyav(audio_path),
    #         "sampling_rate": 16000,
    #     }
    #     result = asr(
    #         audio,
    #         # return_timestamps=True,
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
            raise RuntimeError(
                f"ASR rejected non-speech/uncertain audio: language={language_token or 'unknown'} "
                f"confidence={language_confidence:.3f} for {audio_path}"
            )

        if with_timestamps:
            return self._transcribe_audio_with_timestamps(array)

        audio = {
            "array": array,
            "sampling_rate": 16000,
        }
        
        # Check duration and set return_timestamps accordingly
        duration_seconds = len(audio["array"]) / audio["sampling_rate"]
        needs_long_form = duration_seconds > 30
        
        result = asr(
            audio,
            return_timestamps=needs_long_form,   # True only when needed
            chunk_length_s=30 if needs_long_form else None,
            stride_length_s=3 if needs_long_form else None,
            generate_kwargs=self._asr_generate_kwargs(
                {
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_prev_tokens": False,
                }
            ),
        )
        
        # result["text"] exists in both modes
        transcript = result["text"].strip()
        return self.asr_non_speech_text if self._is_bad_transcript(transcript) else transcript

    def _transcribe_audio_with_timestamps(self, array):
        asr = self._load_asr()
        audio = {
            "array": array,
            "sampling_rate": 16000,
        }
        result = asr(
            audio,
            return_timestamps="word",
            generate_kwargs=self._asr_generate_kwargs(
                {
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_prev_tokens": False,
                }
            ),
        )
        transcript = result.get("text", "").strip()
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
        pyav_error = None
        try:
            return Model._load_audio_with_ffmpeg(audio_path)
        except Exception as exc:
            ffmpeg_error = exc

        try:
            return Model._decode_audio_with_pyav(audio_path)
        except Exception as exc:
            pyav_error = exc

        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                array, _ = librosa.load(audio_path, sr=16000, mono=True)
            return array
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode audio for ASR: {audio_path}. "
                f"ffmpeg error: {ffmpeg_error}; pyav error: {pyav_error}; librosa error: {exc}"
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
            if re.match(r"^(Audio \d+|Audio file|Video audio) (transcript|analysis)", transcript):
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
    ):
        text = self._build_text(query, transcript, transcript_has_timestamps=transcript_has_timestamps)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            video_content = {"type": "video", "video": video_path}
            if self.video_num_frames is not None:
                video_content["nframes"] = self.video_num_frames
            else:
                video_content["fps"] = self.video_fps
            content.append(video_content)
        if content:
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return messages

    def _transcribe_audio_paths(self, audio_paths, with_timestamps=False):
        transcripts = []
        for audio_path in audio_paths:
            transcript = self._transcribe_audio(audio_path, with_timestamps=with_timestamps)
            if self._should_include_transcript(transcript):
                transcripts.append(transcript)
        if not transcripts:
            return ""
        if len(transcripts) == 1:
            return transcripts[0]
        return "\n\n".join(
            f"Audio {idx} transcript:\n{transcript}"
            for idx, transcript in enumerate(transcripts, start=1)
        )

    def _transcribe_media_sources(self, audio_paths, video_path, with_timestamps=False):
        blocks = []
        for audio_path in audio_paths:
            block = self._analyze_media_source("Audio file", audio_path, with_timestamps=with_timestamps)
            if block:
                blocks.append(block)

        if video_path:
            if self._has_audio_stream(video_path):
                block = self._analyze_media_source("Video audio", video_path, with_timestamps=with_timestamps)
                if block:
                    blocks.append(block)

        return "\n\n".join(blocks)

    def _analyze_media_source(self, label, audio_path, with_timestamps=False):
        if self.parallel_audio_analysis and (self.use_sensevoice or self.use_mellow):
            max_workers = 1 + int(self.use_sensevoice) + int(self.use_mellow)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                whisper_future = executor.submit(self._whisper_analysis, audio_path, with_timestamps)
                sensevoice_future = executor.submit(self._sensevoice_analysis, audio_path) if self.use_sensevoice else None
                mellow_future = executor.submit(self._mellow_description, audio_path) if self.use_mellow else None
                whisper = self._future_result(whisper_future, "Whisper")
                sensevoice = self._future_result(sensevoice_future, "SenseVoice") if sensevoice_future else None
                mellow = self._future_result(mellow_future, "Mellow") if mellow_future else ""
        else:
            whisper = self._future_result(lambda: self._whisper_analysis(audio_path, with_timestamps), "Whisper")
            sensevoice = self._future_result(lambda: self._sensevoice_analysis(audio_path), "SenseVoice") if self.use_sensevoice else None
            mellow = self._future_result(lambda: self._mellow_description(audio_path), "Mellow") if self.use_mellow else ""
        return self._format_audio_analysis_block(label, whisper, sensevoice, mellow)

    @staticmethod
    def _future_result(future_or_callable, name):
        try:
            if callable(future_or_callable):
                return future_or_callable()
            return future_or_callable.result()
        except Exception as exc:
            raise RuntimeError(f"{name} analysis failed") from exc

    def _format_audio_analysis_block(self, label, whisper, sensevoice, mellow=""):
        lines = []
        whisper_transcript = (whisper or {}).get("transcript", "").strip()
        sensevoice_line = self._format_sensevoice_cue_line(sensevoice)
        mellow = (mellow or "").strip()
        if not whisper_transcript and not sensevoice_line and not mellow:
            return ""

        lines.append(f"{label} analysis:")
        if whisper:
            language = whisper.get("language", "")
            confidence = whisper.get("language_confidence", 0.0)
            if language:
                lines.append(f"- highest language & probability: {language} ({confidence * 100:.2f}%)")
            if whisper_transcript:
                lines.append("- Whisper Transcript:")
                lines.append(whisper_transcript)

        if sensevoice_line:
            lines.append(sensevoice_line)
        if mellow:
            lines.append(f"Audio Description: {mellow}")

        return "\n".join(lines)

    @staticmethod
    def _has_sensevoice_tags(sensevoice):
        if not sensevoice:
            return False
        return bool(sensevoice.get("emotions") or sensevoice.get("events"))

    @staticmethod
    def _format_sensevoice_cue_line(sensevoice):
        if not sensevoice:
            return ""
        pieces = []
        if sensevoice.get("emotions"):
            pieces.append(f"Audio Emotion: {', '.join(sensevoice['emotions'])}")
        if sensevoice.get("events"):
            pieces.append(f"Audio Events: {', '.join(sensevoice['events'])}")
        return "; ".join(pieces)

    def _should_include_transcript(self, transcript):
        transcript = (transcript or "").strip()
        return bool(transcript) and transcript != self.asr_non_speech_text and not self._is_bad_transcript(transcript)

    @staticmethod
    def _normalize_transcript_for_dedupe(transcript):
        return re.sub(r"\s+", " ", (transcript or "").strip().lower())

    def _prepare_inputs(self, messages):
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "qwen25vlomni requires qwen-vl-utils. Install it in the active environment with "
                "`pip install qwen-vl-utils`."
            ) from exc

        self._register_pyav_video_reader()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        if isinstance(video_kwargs.get("fps"), list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _needs_timestamp_transcript(sample):
        if not sample:
            return False
        task_type = str(sample.get("task_type", "")).strip().lower()
        if task_type == "audio event location":
            return True

        question = str(sample.get("question", "") or "")
        options = "\n".join(str(option) for option in sample.get("options", []) or [])
        text = f"{question}\n{options}"
        timestamp_patterns = (
            r"\b(?:at|around|near|from|between|after|before|during|by)\s+\d{1,2}:\d{2}(?::\d{2})?\b",
            r"\b\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?\b",
            r"\b(?:at|around|near|from|between|after|before|during|by)\s+\d+(?:\.\d+)?\s*s\b",
            r"\bstart\s*time\s*:\s*\d+(?:\.\d+)?\s*s\s*,\s*end\s*time\s*:\s*\d+(?:\.\d+)?\s*s\b",
            r"\btimestamps?\b",
            r"\bwhat moments frame\b",
            r"\bstart and (?:stop|end)\b",
            r"\bbeginning and ending moments\b",
        )
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in timestamp_patterns)

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)
        transcript = ""
        transcript_has_timestamps = self._needs_timestamp_transcript(sample)
        if self.use_asr:
            transcript = self._transcribe_media_sources(
                resolved_audio_paths,
                resolved_video,
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
        )
        # print("messages => ", messages)
        inputs = self._prepare_inputs(messages)
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_new_tokens,
        #         temperature=self.temperature,
        #         top_p=self.top_p,
        #         repetition_penalty=self.repetition_penalty,
        #         do_sample=self.temperature > 0,
        #     )

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        if self.repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = self.repetition_penalty

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print("Final Ans => ", decoded[0].strip())
        return decoded[0].strip() if decoded else ""

## For Video+Audio+Text
# CUDA_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=2 \
# QWEN25VL_USE_ASR=True \
# QWEN25VL_ASR_DEVICE=cuda \
# MODEL=qwen25vlomni \
# EVAL_DATASETS=daily_omni \
# OUTPUT_DIR=results/qwen25vlomni \
# LLM_JUDGE=False \
# TEMPERATURE=0 \
# MAX_NEW_TOKENS=512 \
# bash eval.sh

## For Video+Text
# CUDA_VISIBLE_DEVICES=3 HIP_VISIBLE_DEVICES=3 \
# MODEL=qwen25vlomni \
# EVAL_DATASETS=daily_omni \
# OUTPUT_DIR=results/qwen25vl \
# LLM_JUDGE=False \
# TEMPERATURE=0 \
# MAX_NEW_TOKENS=512 \
# bash eval.sh
