import os
import re
import subprocess
import sys
import time
import types
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
        self.package_dir = Path(__file__).resolve().parent
        self.default_checkpoint = self.package_dir / "model_checkpoint"
        self.default_autogaze_checkpoint = self.package_dir / "autogaze_checkpoint"
        self.local_autogaze_source = self.package_dir / "AutoGaze"
        self.local_python_deps = self.package_dir / "python_deps"
        self.local_vila_source = self.package_dir / "VILA"
        self.local_s2_source = self.package_dir / "scaling_on_scales"
        self.model_path = model_path or os.getenv("VILA_MODEL_PATH") or self._default_model_path()
        self.autogaze_model_path = os.getenv("VILA_AUTOGAZE_MODEL_PATH") or self._default_autogaze_model_path()
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("VILA_TOP_P", "0.8"))
        self.device_map = os.getenv("VILA_DEVICE_MAP", "cuda:0")
        self.torch_dtype = os.getenv("VILA_TORCH_DTYPE", "bfloat16")
        self.attn_implementation = os.getenv("VILA_ATTN_IMPLEMENTATION", "sdpa")
        self.num_video_frames = int(os.getenv("VILA_NUM_VIDEO_FRAMES", "128"))
        self.num_video_frames_thumbnail = int(os.getenv("VILA_NUM_VIDEO_FRAMES_THUMBNAIL", "64"))
        self.max_tiles_video = int(os.getenv("VILA_MAX_TILES_VIDEO", "48"))
        self.max_batch_size_autogaze = int(os.getenv("VILA_MAX_BATCH_SIZE_AUTOGAZE", "16"))
        self.max_batch_size_siglip = int(os.getenv("VILA_MAX_BATCH_SIZE_SIGLIP", "32"))
        self.gazing_ratio_tile = self._float_or_list(os.getenv("VILA_GAZING_RATIO_TILE", ""), [0.2] + [0.06] * 15)
        self.gazing_ratio_thumbnail = self._float_or_list(os.getenv("VILA_GAZING_RATIO_THUMBNAIL", "1"), 1)
        self.autogaze_fallback_ratio = float(os.getenv("VILA_AUTOGAZE_FALLBACK_RATIO", "0.06"))
        self.task_loss_requirement_tile = self._optional_float(os.getenv("VILA_TASK_LOSS_REQUIREMENT_TILE", "0.6"))
        self.task_loss_requirement_thumbnail = self._optional_float(
            os.getenv("VILA_TASK_LOSS_REQUIREMENT_THUMBNAIL", "")
        )
        self.use_asr = self._str_to_bool(os.getenv("VILA_USE_ASR", os.getenv("QWEN25VL_USE_ASR", "True")))
        self.whisper_use_embedded_video_audio = self._str_to_bool(
            os.getenv("VILA_WHISPER_USE_EMBEDDED_VIDEO_AUDIO", "True")
        )
        self.asr_model_path = os.getenv(
            "VILA_ASR_MODEL",
            os.getenv(
                "QWEN25VL_ASR_MODEL",
                "openai/whisper-large-v3-turbo",
            ),
        )
        self.asr_device = os.getenv("VILA_ASR_DEVICE", os.getenv("QWEN25VL_ASR_DEVICE", "cuda"))
        self.asr_language = os.getenv("VILA_ASR_LANGUAGE", os.getenv("QWEN25VL_ASR_LANGUAGE", "")).strip()
        self.asr_min_language_confidence = float(
            os.getenv("VILA_ASR_MIN_LANGUAGE_CONFIDENCE", os.getenv("QWEN25VL_ASR_MIN_LANGUAGE_CONFIDENCE", "0.65"))
        )
        self.asr_chunk_seconds = float(os.getenv("VILA_ASR_CHUNK_SECONDS", os.getenv("QWEN25VL_ASR_CHUNK_SECONDS", "25")))
        self.asr_non_speech_text = os.getenv("VILA_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.processor = None
        self.model = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        self._load()

    def _default_model_path(self):
        if (self.default_checkpoint / "config.json").exists():
            return str(self.default_checkpoint)
        return "nvidia/NVILA-8B-HD-Video"

    def _default_autogaze_model_path(self):
        if (self.default_autogaze_checkpoint / "config.json").exists():
            return str(self.default_autogaze_checkpoint)
        return "nvidia/AutoGaze"

    @staticmethod
    def _str_to_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _optional_float(value):
        value = str(value or "").strip()
        if not value or value.lower() in {"none", "null"}:
            return None
        return float(value)

    @classmethod
    def _float_or_list(cls, value, default):
        value = str(value or "").strip()
        if not value:
            return default
        if "," in value:
            return cls._float_list(value, default)
        return float(value)

    @staticmethod
    def _float_list(value, default):
        value = str(value or "").strip()
        if not value:
            return default
        return [float(item.strip()) for item in value.split(",") if item.strip()]

    def _load(self):
        self._remove_user_site_packages()
        self._add_local_dependency_paths()
        self._patch_transformers_tied_weights()
        self._patch_transformers_siglip_initializers()
        self._patch_autogaze_generation_cache()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "VILA_whisper requires Transformers and the bundled AutoGaze source."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            autogaze_model_id=self.autogaze_model_path,
            num_video_frames=self.num_video_frames,
            num_video_frames_thumbnail=self.num_video_frames_thumbnail,
            max_tiles_video=self.max_tiles_video,
            gazing_ratio_tile=self.gazing_ratio_tile,
            gazing_ratio_thumbnail=self.gazing_ratio_thumbnail,
            task_loss_requirement_tile=self.task_loss_requirement_tile,
            task_loss_requirement_thumbnail=self.task_loss_requirement_thumbnail,
            max_batch_size_autogaze=self.max_batch_size_autogaze,
            trust_remote_code=True,
        )
        self._patch_processor_autogaze_fallback()
        self._patch_processor_empty_media_padding()

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
            "max_batch_size_siglip": self.max_batch_size_siglip,
        }
        if self.torch_dtype:
            model_kwargs["dtype"] = self.torch_dtype if self.torch_dtype == "auto" else getattr(torch, self.torch_dtype)
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()

    def _patch_processor_autogaze_fallback(self):
        original = getattr(self.processor, "_run_autogaze_batched", None)
        if original is None or getattr(self.processor, "_omnieval_autogaze_fallback_patched", False):
            return

        def _run_autogaze_batched(processor, all_videos, autogaze_device, cpu_device, gazing_ratio, task_loss_requirement):
            try:
                return original(all_videos, autogaze_device, cpu_device, gazing_ratio, task_loss_requirement)
            except (RuntimeError, IndexError, ValueError) as exc:
                text = str(exc)
                known_transformers5_issue = (
                    "cannot reshape tensor of 0 elements" in text
                    or "cache_position" in text
                    or "inputs_embeds" in text
                    or "index -1 is out of bounds" in text
                )
                if not known_transformers5_issue:
                    raise

                import torch

                total = all_videos.shape[0]
                frames = all_videos.shape[1]
                patch_size = getattr(processor, "target_patch_size", 16)
                target_scales = getattr(processor, "target_scales", [56, 112, 224, 448])
                total_patches_per_frame = sum((scale // patch_size) ** 2 for scale in target_scales)
                counts = self._fallback_gazing_counts(gazing_ratio, frames, total_patches_per_frame, cpu_device)
                per_frame_pos = [
                    torch.arange(int(count.item()), device=cpu_device, dtype=torch.long)
                    for count in counts
                ]
                gazing_pos_1d = torch.cat(per_frame_pos, dim=0)
                gazing_pos = gazing_pos_1d.unsqueeze(0).expand(total, -1).contiguous()
                if_padded = torch.zeros(total, gazing_pos.shape[1], device=cpu_device, dtype=torch.bool)
                num_gazing = counts.unsqueeze(0).expand(total, -1).contiguous()
                return gazing_pos, if_padded, num_gazing

        self.processor._run_autogaze_batched = types.MethodType(_run_autogaze_batched, self.processor)
        self.processor._omnieval_autogaze_fallback_patched = True

    def _patch_processor_empty_media_padding(self):
        original = getattr(self.processor, "_preprocess_text", None)
        if original is None or getattr(self.processor, "_omnieval_empty_media_padding_patched", False):
            return

        def _preprocess_text(
            processor,
            text,
            *,
            image_token_padding_strategy,
            video_token_padding_strategy,
            **kwargs,
        ):
            if video_token_padding_strategy == [[]]:
                video_token_padding_strategy = []
            if image_token_padding_strategy == [[]]:
                image_token_padding_strategy = []
            if hasattr(original, "__func__"):
                return original.__func__(
                    processor,
                    text,
                    image_token_padding_strategy=image_token_padding_strategy,
                    video_token_padding_strategy=video_token_padding_strategy,
                    **kwargs,
                )
            return original(
                text,
                image_token_padding_strategy=image_token_padding_strategy,
                video_token_padding_strategy=video_token_padding_strategy,
                **kwargs,
            )

        self.processor._preprocess_text = types.MethodType(_preprocess_text, self.processor)
        self.processor._omnieval_empty_media_padding_patched = True

    def _fallback_gazing_counts(self, gazing_ratio, frames, total_patches_per_frame, device):
        import torch

        if gazing_ratio is None:
            ratios = torch.full((frames,), self.autogaze_fallback_ratio, device=device, dtype=torch.float32)
        elif isinstance(gazing_ratio, (list, tuple)):
            values = [float(value) for value in gazing_ratio]
            if not values:
                values = [self.autogaze_fallback_ratio]
            if len(values) < frames:
                values.extend([values[-1]] * (frames - len(values)))
            ratios = torch.tensor(values[:frames], device=device, dtype=torch.float32)
        else:
            ratios = torch.full((frames,), float(gazing_ratio), device=device, dtype=torch.float32)

        ratios = ratios.clamp(min=1.0 / max(total_patches_per_frame, 1), max=1.0)
        return torch.ceil(ratios * total_patches_per_frame).to(torch.long).clamp(min=1)

    def _add_local_dependency_paths(self):
        for path in (
            self.local_python_deps,
            self.local_autogaze_source,
            self.local_s2_source,
            self.local_vila_source,
        ):
            if path.exists():
                value = str(path)
                if value not in sys.path:
                    sys.path.insert(0, value)

    @staticmethod
    def _remove_user_site_packages():
        if not Model._str_to_bool(os.getenv("VILA_DISABLE_USER_SITE", "True")):
            return
        try:
            import site

            user_sites = site.getusersitepackages()
        except Exception:
            return
        if isinstance(user_sites, str):
            user_sites = [user_sites]
        user_sites = {str(Path(path).resolve()) for path in user_sites}
        sys.path[:] = [
            path
            for path in sys.path
            if str(Path(path or ".").resolve()) not in user_sites
        ]

    @staticmethod
    def _patch_transformers_siglip_initializers():
        try:
            import torch
            from transformers import initialization as init
            import transformers.models.siglip.modeling_siglip as siglip
        except Exception:
            return

        aliases = {
            "_trunc_normal_": getattr(init, "trunc_normal_", torch.nn.init.trunc_normal_),
            "trunc_normal_tf_": getattr(init, "trunc_normal_", torch.nn.init.trunc_normal_),
            "variance_scaling_": getattr(init, "_variance_scaling", None),
            "lecun_normal_": getattr(init, "lecun_normal_", None),
            "default_flax_embed_init": getattr(init, "default_flax_embed_init_", None),
        }
        for name, value in aliases.items():
            if value is not None and not hasattr(siglip, name):
                setattr(siglip, name, value)

    @staticmethod
    def _patch_autogaze_generation_cache():
        try:
            import torch
            from autogaze.models.autogaze.modeling_llama_multi_token_pred import LlamaForCausalLM_MultiTokenPred
        except Exception:
            return
        original_prepare_inputs = getattr(LlamaForCausalLM_MultiTokenPred, "prepare_inputs_for_generation", None)
        original_update_kwargs = getattr(
            LlamaForCausalLM_MultiTokenPred,
            "_update_model_kwargs_for_generation",
            None,
        )

        def _get_initial_cache_position(self, *args):
            if len(args) == 2:
                input_ids, model_kwargs = args
                cur_len = input_ids.shape[-1]
                device = input_ids.device
            else:
                cur_len, device, model_kwargs = args

            if model_kwargs.get("cache_position") is not None:
                return model_kwargs

            past_key_values = model_kwargs.get("past_key_values")
            past_length = 0
            if past_key_values is not None:
                try:
                    past_length = past_key_values.get_seq_length()
                except AttributeError:
                    try:
                        past_length = past_key_values[0][0].shape[-2]
                    except Exception:
                        past_length = 0

            model_kwargs["cache_position"] = torch.arange(
                cur_len,
                dtype=torch.long,
                device=device,
            ) + past_length
            return model_kwargs

        LlamaForCausalLM_MultiTokenPred._get_initial_cache_position = _get_initial_cache_position

        if original_update_kwargs is not None and not getattr(
            LlamaForCausalLM_MultiTokenPred,
            "_omnieval_update_kwargs_patched",
            False,
        ):

            def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
                cache_position = model_kwargs.get("cache_position")
                if cache_position is not None and cache_position.numel() == 0:
                    attention_mask = model_kwargs.get("attention_mask")
                    length = int(attention_mask.shape[-1]) if attention_mask is not None else 1
                    model_kwargs["cache_position"] = torch.arange(
                        max(length, 1),
                        dtype=torch.long,
                        device=cache_position.device,
                    )
                return original_update_kwargs(self, outputs, model_kwargs, *args, **kwargs)

            LlamaForCausalLM_MultiTokenPred._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
            LlamaForCausalLM_MultiTokenPred._omnieval_update_kwargs_patched = True

        if original_prepare_inputs is not None and not getattr(
            LlamaForCausalLM_MultiTokenPred,
            "_omnieval_prepare_inputs_patched",
            False,
        ):

            def prepare_inputs_for_generation(
                self,
                input_ids,
                next_sequence_length=None,
                past_key_values=None,
                attention_mask=None,
                inputs_embeds=None,
                is_first_iteration=False,
                **kwargs,
            ):
                if next_sequence_length == 0:
                    next_sequence_length = inputs_embeds.shape[1] if inputs_embeds is not None else None
                if inputs_embeds is not None and not is_first_iteration:
                    past_length = 0
                    if past_key_values is not None:
                        try:
                            past_length = past_key_values.get_seq_length()
                        except AttributeError:
                            try:
                                past_length = past_key_values[0][0].shape[-2]
                            except Exception:
                                past_length = 0
                    is_first_iteration = past_length == 0
                return original_prepare_inputs(
                    self,
                    input_ids,
                    next_sequence_length=next_sequence_length,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    is_first_iteration=is_first_iteration,
                    **kwargs,
                )

            LlamaForCausalLM_MultiTokenPred.prepare_inputs_for_generation = prepare_inputs_for_generation
            LlamaForCausalLM_MultiTokenPred._omnieval_prepare_inputs_patched = True

    @staticmethod
    def _patch_transformers_tied_weights():
        try:
            from transformers.modeling_utils import PreTrainedModel
        except Exception:
            return
        if hasattr(PreTrainedModel, "all_tied_weights_keys"):
            return

        @property
        def all_tied_weights_keys(self):
            keys = getattr(self, "_all_tied_weights_keys", None)
            if keys is None:
                keys = getattr(self, "_tied_weights_keys", None) or {}
            if isinstance(keys, dict):
                return keys
            return {key: key for key in keys}

        @all_tied_weights_keys.setter
        def all_tied_weights_keys(self, value):
            self._all_tied_weights_keys = value

        PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

    def _load_asr(self):
        if self.asr is not None:
            return self.asr

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = self.asr_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() and str(device).startswith("cuda") else torch.float32
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_path,
            torch_dtype=dtype,
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
            torch_dtype=dtype,
            device=0 if str(device).startswith("cuda") else -1,
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

        if self.asr_language:
            language_token, language_confidence = self._forced_asr_language(), 1.0
        else:
            language_token, language_confidence = self._detect_spoken_language(array)
        language = self._format_whisper_language(language_token)
        if language_confidence < self.asr_min_language_confidence:
            return {
                "language": language,
                "language_confidence": language_confidence,
                "transcript": "",
            }

        transcript = self._transcribe_audio_array(array)
        if not self._should_include_transcript(transcript):
            transcript = ""
        return {
            "language": language,
            "language_confidence": language_confidence,
            "transcript": transcript,
        }

    def _transcribe_audio_array(self, array):
        transcripts = []
        for chunk in self._asr_chunks(array):
            if self._is_silent_audio(chunk):
                continue
            result = self._load_asr()(
                {
                    "array": chunk,
                    "sampling_rate": 16000,
                },
                return_timestamps=False,
                generate_kwargs=self._asr_generate_kwargs(),
            )
            transcript = result.get("text", "").strip()
            if self._should_include_transcript(transcript):
                transcripts.append(transcript)
        return " ".join(transcripts).strip()

    def _asr_chunks(self, array, sampling_rate=16000):
        chunk_seconds = max(1.0, float(self.asr_chunk_seconds))
        chunk_size = int(sampling_rate * chunk_seconds)
        if len(array) <= chunk_size:
            return [array]
        return [array[start : start + chunk_size] for start in range(0, len(array), chunk_size)]

    def _asr_generate_kwargs(self):
        kwargs = {
            "task": "transcribe",
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_prev_tokens": False,
        }
        if self.asr_language:
            kwargs["language"] = self.asr_language
        return kwargs

    def _forced_asr_language(self):
        language = self.asr_language.strip().lower()
        if language.startswith("<|") and language.endswith("|>"):
            return language
        return f"<|{language}|>"

    def _transcribe_media_sources(self, audio_paths, video_path):
        entries = []
        for audio_path in audio_paths:
            entry = self._format_whisper_entry("audio file", self._whisper_analysis(audio_path))
            if entry:
                entries.append(entry)

        if self.whisper_use_embedded_video_audio and video_path and self._has_audio_stream(video_path):
            entry = self._format_whisper_entry("embedded video audio", self._whisper_analysis(video_path))
            if entry:
                entries.append(entry)
        if not entries:
            return ""
        return "Audio transcript:\n" + "\n\n".join(entries)

    @staticmethod
    def _format_whisper_entry(label, whisper):
        transcript = (whisper or {}).get("transcript", "").strip()
        if not transcript:
            return ""

        lines = [f"Whisper Audio Transcript ({label}):"]
        language = whisper.get("language", "")
        confidence = whisper.get("language_confidence", 0.0)
        if language:
            lines.append(f"Audio Language: {language} ({confidence * 100:.2f}%)")
        lines.append(transcript)
        return "\n".join(lines)

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

    @staticmethod
    def _build_text(query, transcript=""):
        parts = []
        if transcript:
            parts.append(transcript)
        parts.append(Model._sanitize_indexed_media_placeholders(query))
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _sanitize_indexed_media_placeholders(text):
        def replace(match):
            return f"[{match.group(1)} {match.group(2)}]"

        text = str(text or "").strip()
        return re.sub(r"<(audio|image|video)_(\d+)>", replace, text)

    @staticmethod
    def _load_image(path):
        from PIL import Image

        return Image.open(path).convert("RGB")

    def _load_video_frames(self, path):
        try:
            return self._load_video_frames_with_cv2(path, self.num_video_frames)
        except Exception:
            return self._load_video_frames_with_pyav(path, self.num_video_frames)

    @staticmethod
    def _sample_indices(total_frames, num_frames):
        import numpy as np

        if total_frames <= 0:
            return [0] * num_frames
        return np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int).tolist()

    @classmethod
    def _load_video_frames_with_cv2(cls, path, num_frames):
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = cls._sample_indices(frame_count, num_frames)
        frames_by_index = {}
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_by_index[index] = Image.fromarray(frame)
        capture.release()

        frames = [frames_by_index[index] for index in indices if index in frames_by_index]
        if not frames:
            raise ValueError(f"Could not extract any frames from video: {path}")
        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        return frames

    @classmethod
    def _load_video_frames_with_pyav(cls, path, num_frames):
        import av
        from PIL import Image

        decoded = []
        with av.open(path) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {path}")
            for frame in container.decode(stream):
                decoded.append(Image.fromarray(frame.to_ndarray(format="rgb24")))

        if not decoded:
            raise ValueError(f"Could not extract any frames from video: {path}")

        indices = cls._sample_indices(len(decoded), num_frames)
        frames = [decoded[min(index, len(decoded) - 1)] for index in indices]
        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        return frames

    def _processor_call_no_video(self, **processor_kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        normalized_text, normalized_images, _ = self.processor._normalize_inputs(
            text=processor_kwargs.pop("text"),
            images=processor_kwargs.pop("images", None),
            videos=None,
        )
        images_inputs, image_strategy = (
            self.processor._preprocess_images(normalized_images, **processor_kwargs)
            if normalized_images
            else (BatchFeature(), [])
        )
        if normalized_images:
            image_strategy = self._image_padding_strategy_from_model(images_inputs)
        text_inputs = self.processor._preprocess_text(
            normalized_text,
            image_token_padding_strategy=image_strategy,
            video_token_padding_strategy=[],
            **processor_kwargs,
        )
        return BatchFeature(
            {
                **text_inputs,
                **images_inputs,
            }
        )

    def _image_padding_strategy_from_model(self, images_inputs):
        import torch

        shuffle_num = 9
        try:
            default_gazing = self.model._make_default_gazing_info(1, 1, torch.device("cpu"))
            patches_per_tile = int(default_gazing["num_gazing_each_frame"][0].item())
        except Exception:
            patch_size = getattr(self.processor, "target_patch_size", 16)
            target_scales = getattr(self.processor, "target_scales", [56, 112, 224, 448])
            patches_per_tile = sum((scale // patch_size) ** 2 for scale in target_scales)

        def ceil_div(value, divisor):
            return (int(value) + divisor - 1) // divisor

        strategy = []
        for num_tiles in images_inputs.get("num_spatial_tiles_each_image", []):
            tile_tokens = ceil_div(int(num_tiles) * patches_per_tile, shuffle_num)
            thumb_tokens = ceil_div(patches_per_tile, shuffle_num)
            strategy.append([tile_tokens + thumb_tokens])
        return strategy

    def _media_tokens(self, image_count, has_video):
        tokenizer = self.processor.tokenizer
        tokens = []
        image_token = getattr(tokenizer, "image_token", "<image>")
        video_token = getattr(tokenizer, "video_token", "<video>")
        tokens.extend([image_token] * image_count)
        if has_video:
            tokens.append(video_token)
        return "\n".join(tokens)

    def _build_prompt(self, query, system_prompt, transcript, image_count, has_video):
        text_query = self._build_text(query, transcript)
        parts = []
        media_tokens = self._media_tokens(image_count, has_video)
        if media_tokens:
            parts.append(media_tokens)
        if system_prompt:
            parts.append(f"System instruction:\n{system_prompt.strip()}")
        parts.append(text_query)
        return "\n\n".join(part for part in parts if part)

    def _model_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)
        images = [self._load_image(path) for path in resolved_images]

        transcript = ""
        if self.use_asr:
            whisper_start = time.perf_counter()
            transcript = self._transcribe_media_sources(resolved_audio_paths, resolved_video)
            self.last_whisper_inference_time_sec = time.perf_counter() - whisper_start

        prompt = self._build_prompt(
            query=query,
            system_prompt=system_prompt,
            transcript=transcript,
            image_count=len(images),
            has_video=bool(resolved_video),
        )
        processor_kwargs = {
            "text": prompt,
            "images": images or None,
            "return_tensors": "pt",
        }

        vlm_start = time.perf_counter()
        if resolved_video:
            processor_kwargs["videos"] = resolved_video
            inputs = self.processor(**processor_kwargs)
        else:
            inputs = self._processor_call_no_video(**processor_kwargs)
        model_device = self._model_device()
        inputs = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        self.last_vlm_inference_time_sec = time.perf_counter() - vlm_start
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )[0].strip()
