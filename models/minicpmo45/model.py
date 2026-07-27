import logging
import os
import time
from pathlib import Path


class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("MINICPMO45_MODEL_PATH", "openbmb/MiniCPM-o-4_5")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("MINICPMO45_TOP_P", "0.8"))
        self.num_beams = int(os.getenv("MINICPMO45_NUM_BEAMS", "1"))
        self.attn_implementation = os.getenv("MINICPMO45_ATTN_IMPLEMENTATION", "sdpa")
        self.torch_dtype_name = os.getenv("MINICPMO45_TORCH_DTYPE", "bfloat16")
        self.device = os.getenv("MINICPMO45_DEVICE", "")
        self.init_vision = self._str_to_bool(os.getenv("MINICPMO45_INIT_VISION", "True"))
        self.init_audio = self._str_to_bool(os.getenv("MINICPMO45_INIT_AUDIO", "True"))
        self.init_tts = self._str_to_bool(os.getenv("MINICPMO45_INIT_TTS", "False"))
        self.enable_thinking = self._str_to_bool(os.getenv("MINICPMO45_ENABLE_THINKING", "False"))
        self.use_tts_template = self._str_to_bool(os.getenv("MINICPMO45_USE_TTS_TEMPLATE", "False"))
        self.generate_audio = self._str_to_bool(os.getenv("MINICPMO45_GENERATE_AUDIO", "False"))
        self.omni_mode = os.getenv("MINICPMO45_OMNI_MODE", "").strip()
        self.use_image_id = self._optional_bool(os.getenv("MINICPMO45_USE_IMAGE_ID", ""))
        self.max_slice_nums = self._optional_int(os.getenv("MINICPMO45_MAX_SLICE_NUMS", ""))
        self.image_use_image_id = self._optional_bool(os.getenv("MINICPMO45_IMAGE_USE_IMAGE_ID", "True"))
        self.image_max_slice_nums = self._optional_int(os.getenv("MINICPMO45_IMAGE_MAX_SLICE_NUMS", "9"))
        self.video_use_image_id = self._optional_bool(os.getenv("MINICPMO45_VIDEO_USE_IMAGE_ID", "False"))
        self.video_max_slice_nums = self._optional_int(os.getenv("MINICPMO45_VIDEO_MAX_SLICE_NUMS", "1"))
        self.video_stack_frames = int(os.getenv("MINICPMO45_VIDEO_STACK_FRAMES", "1"))
        self.video_use_ffmpeg = self._str_to_bool(os.getenv("MINICPMO45_VIDEO_USE_FFMPEG", "False"))
        self.video_use_audio = os.getenv("MINICPMO45_VIDEO_USE_AUDIO", "").strip()
        self.interleave_video_audio = self._str_to_bool(os.getenv("MINICPMO45_INTERLEAVE_VIDEO_AUDIO", "True"))
        self.model = None
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        self._load()

    @staticmethod
    def _str_to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @classmethod
    def _optional_bool(cls, value):
        value = str(value or "").strip()
        if not value:
            return None
        return cls._str_to_bool(value)

    @staticmethod
    def _optional_int(value):
        value = str(value or "").strip()
        return int(value) if value else None

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

    def _torch_dtype(self, torch):
        dtype_name = str(self.torch_dtype_name or "").strip().lower()
        if dtype_name in {"", "auto"}:
            return "auto"
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_name in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported MINICPMO45_TORCH_DTYPE: {self.torch_dtype_name}")

    @staticmethod
    def _transformers_major_version(transformers):
        version = getattr(transformers, "__version__", "0")
        major = str(version).split(".", 1)[0]
        return int(major) if major.isdigit() else 0

    @staticmethod
    def _patch_transformers5_tied_weights(transformers):
        if Model._transformers_major_version(transformers) < 5:
            return

        from transformers.modeling_utils import PreTrainedModel

        if hasattr(PreTrainedModel, "all_tied_weights_keys"):
            return

        def get_all_tied_weights_keys(self):
            keys = self.__dict__.get("_all_tied_weights_keys_compat")
            if keys is None:
                keys = getattr(self, "_tied_weights_keys", None) or []
            if isinstance(keys, dict):
                return keys
            return {key: None for key in keys}

        def set_all_tied_weights_keys(self, value):
            self.__dict__["_all_tied_weights_keys_compat"] = value

        PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)

    @staticmethod
    def _patch_transformers5_cache_api(transformers):
        if Model._transformers_major_version(transformers) < 5:
            return

        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            return

        if hasattr(DynamicCache, "seen_tokens"):
            return

        def seen_tokens(self):
            return self.get_seq_length()

        DynamicCache.seen_tokens = property(seen_tokens)

    def _load(self):
        try:
            import torch
            import transformers
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-o-4_5 requires torch, transformers, accelerate, torchaudio, "
                "and minicpmo-utils. The model card recommends: "
                'pip install "transformers==4.51.0" accelerate '
                '"torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" '
                '"minicpmo-utils>=1.0.5"'
            ) from exc

        self._patch_transformers5_tied_weights(transformers)
        self._patch_transformers5_cache_api(transformers)
        dtype_key = "dtype" if self._transformers_major_version(transformers) >= 5 else "torch_dtype"
        kwargs = {
            "trust_remote_code": True,
            "init_vision": self.init_vision,
            "init_audio": self.init_audio,
            "init_tts": self.init_tts,
            dtype_key: self._torch_dtype(torch),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        self.model.eval()
        self._patch_minicpm_prepare_inputs_for_generation()
        self._patch_minicpm_whisper_encoder_layer()

        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if target_device and target_device.lower() != "auto":
            self.model = self.model.to(target_device)

    def _patch_minicpm_prepare_inputs_for_generation(self):
        try:
            import transformers
        except ImportError:
            return
        if self._transformers_major_version(transformers) < 5:
            return

        llm = getattr(self.model, "llm", None)
        if llm is None or getattr(llm, "_omnieval_native_prepare_inputs_patch", False):
            return

        native_prepare_inputs = type(llm).prepare_inputs_for_generation.__get__(llm, type(llm))
        llm.prepare_inputs_for_generation = native_prepare_inputs
        llm._omnieval_native_prepare_inputs_patch = True

    def _patch_minicpm_whisper_encoder_layer(self):
        try:
            import transformers
        except ImportError:
            return
        if self._transformers_major_version(transformers) < 5:
            return

        apm = getattr(self.model, "apm", None)
        layers = getattr(apm, "layers", None)
        if not layers:
            return

        layer_cls = type(layers[0])
        if getattr(layer_cls, "_omnieval_transformers5_attention_patch", False):
            return

        def forward(
            layer_self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            output_attentions=False,
            past_key_values=None,
            use_cache=False,
        ):
            import torch
            from torch import nn

            residual = hidden_states
            hidden_states = layer_self.self_attn_layer_norm(hidden_states)
            attn_outputs = layer_self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
            )
            hidden_states = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            next_past_key_values = attn_outputs[2] if len(attn_outputs) > 2 else past_key_values

            hidden_states = nn.functional.dropout(
                hidden_states,
                p=layer_self.dropout,
                training=layer_self.training,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer_self.final_layer_norm(hidden_states)
            hidden_states = layer_self.activation_fn(layer_self.fc1(hidden_states))
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=layer_self.activation_dropout,
                training=layer_self.training,
            )
            hidden_states = layer_self.fc2(hidden_states)
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=layer_self.dropout,
                training=layer_self.training,
            )
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (attn_weights,)
            if use_cache:
                outputs += (next_past_key_values,)
            return outputs

        layer_cls.forward = forward
        layer_cls._omnieval_transformers5_attention_patch = True

    def _video_should_include_audio(self, audio_paths):
        if self.video_use_audio:
            return self._str_to_bool(self.video_use_audio)
        return not audio_paths

    def _omni_mode(self, has_video, has_audio):
        if self.omni_mode:
            return self._str_to_bool(self.omni_mode)
        return bool(has_video and has_audio)

    @staticmethod
    def _load_image(image_path):
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _load_audio(audio_path):
        import librosa

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        return audio

    def _load_video_contents(self, video_path, audio_paths):
        from minicpmo.utils import get_video_frame_audio_segments

        segmented_audio_path = audio_paths[0] if self.interleave_video_audio and audio_paths else None
        use_video_audio = bool(segmented_audio_path) or self._video_should_include_audio(audio_paths)
        video_frames, audio_segments, stacked_frames = self._extract_video_segments(
            get_video_frame_audio_segments,
            video_path,
            segmented_audio_path,
            use_audio=use_video_audio,
            use_ffmpeg=self.video_use_ffmpeg,
        )

        contents = []
        for index, frame in enumerate(video_frames):
            contents.append(frame)
            if use_video_audio and audio_segments is not None and index < len(audio_segments):
                contents.append(audio_segments[index])
            if stacked_frames is not None and index < len(stacked_frames) and stacked_frames[index] is not None:
                contents.append(stacked_frames[index])
        return contents, 1 if segmented_audio_path else 0

    def _extract_video_segments(self, get_video_frame_audio_segments, video_path, audio_path, use_audio, use_ffmpeg):
        try:
            return get_video_frame_audio_segments(
                video_path,
                audio_path=audio_path,
                stack_frames=self.video_stack_frames,
                use_ffmpeg=use_ffmpeg,
                adjust_audio_length=use_audio,
            )
        except Exception as first_error:
            if use_audio and not audio_path:
                logging.warning(
                    "Audio extraction failed for %s; retrying MiniCPM video extraction without audio.",
                    video_path,
                )
                try:
                    return get_video_frame_audio_segments(
                        video_path,
                        stack_frames=self.video_stack_frames,
                        use_ffmpeg=use_ffmpeg,
                        adjust_audio_length=False,
                    )
                except Exception:
                    if use_ffmpeg:
                        raise

            if use_ffmpeg:
                raise first_error

            logging.warning("Decord failed for %s; retrying MiniCPM video extraction with ffmpeg.", video_path)
            try:
                return get_video_frame_audio_segments(
                    video_path,
                    audio_path=audio_path,
                    stack_frames=self.video_stack_frames,
                    use_ffmpeg=True,
                    adjust_audio_length=use_audio,
                )
            except Exception:
                if use_audio and not audio_path:
                    logging.warning(
                        "ffmpeg audio extraction failed for %s; retrying local frame-only extraction.",
                        video_path,
                    )
                    return self._load_video_frames_only(video_path), None, None
                raise

    def _load_video_frames_only(self, video_path):
        try:
            return self._load_video_frames_only_pyav(video_path)
        except Exception:
            return self._load_video_frames_only_cv2(video_path)

    def _target_video_frame_count(self, duration):
        if duration and duration > 0:
            return max(1, min(64, int(round(duration))))
        return 8

    @staticmethod
    def _sample_indices(total_frames, count):
        if total_frames <= 0:
            return []
        if count >= total_frames:
            return list(range(total_frames))
        if count <= 1:
            return [0]
        return [round(index * (total_frames - 1) / (count - 1)) for index in range(count)]

    def _load_video_frames_only_pyav(self, video_path):
        import av
        from PIL import Image

        with av.open(video_path) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {video_path}")

            duration = None
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = container.duration / 1_000_000

            total_frames = int(stream.frames or 0)
            target_count = self._target_video_frame_count(duration)
            if total_frames:
                target_indices = set(self._sample_indices(total_frames, target_count))
                frames = []
                for index, frame in enumerate(container.decode(stream)):
                    if index in target_indices:
                        frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                    if len(frames) >= len(target_indices):
                        break
            else:
                decoded = [Image.fromarray(frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
                indices = self._sample_indices(len(decoded), target_count)
                frames = [decoded[index] for index in indices]

        if not frames:
            raise ValueError(f"No video frames decoded from {video_path}")
        return frames

    def _load_video_frames_only_cv2(self, video_path):
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        duration = (total_frames / fps) if total_frames and fps else None
        indices = self._sample_indices(total_frames, self._target_video_frame_count(duration))

        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        capture.release()

        if not frames:
            raise ValueError(f"No video frames decoded from {video_path}")
        return frames

    def _build_content(self, audio_paths, video_path, image_paths, query):
        if audio_paths and not video_path and not image_paths:
            content = [query]
            for audio_path in audio_paths:
                content.append(self._load_audio(audio_path))
            return content

        content = []
        for image_path in image_paths:
            content.append(self._load_image(image_path))
        used_audio_count = 0
        if video_path:
            video_contents, used_audio_count = self._load_video_contents(video_path, audio_paths)
            content.extend(video_contents)
        for audio_path in audio_paths[used_audio_count:]:
            content.append(self._load_audio(audio_path))
        content.append(query)
        return content

    def _vision_generation_kwargs(self, has_video, has_images):
        if has_video:
            default_use_image_id = self.video_use_image_id
            default_max_slice_nums = self.video_max_slice_nums
        elif has_images:
            default_use_image_id = self.image_use_image_id
            default_max_slice_nums = self.image_max_slice_nums
        else:
            default_use_image_id = None
            default_max_slice_nums = None
        return {
            "use_image_id": self.use_image_id if self.use_image_id is not None else default_use_image_id,
            "max_slice_nums": self.max_slice_nums if self.max_slice_nums is not None else default_max_slice_nums,
        }

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        vlm_start = time.perf_counter()
        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)
        video_includes_audio = bool(resolved_video and self._video_should_include_audio(resolved_audio_paths))
        has_audio = bool(resolved_audio_paths or video_includes_audio)

        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [system_prompt.strip()],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self._build_content(
                    resolved_audio_paths,
                    resolved_video,
                    resolved_images,
                    query,
                ),
            }
        )

        generation_kwargs = {
            "msgs": messages,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "use_tts_template": self.use_tts_template,
            "enable_thinking": self.enable_thinking,
            "generate_audio": self.generate_audio,
            "omni_mode": self._omni_mode(bool(resolved_video), has_audio),
            "num_beams": self.num_beams,
        }
        generation_kwargs.update(self._vision_generation_kwargs(bool(resolved_video), bool(resolved_images)))
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        answer = self.model.chat(**generation_kwargs)
        self.last_vlm_inference_time_sec = time.perf_counter() - vlm_start
        if answer is None:
            return ""
        return str(answer).strip()
