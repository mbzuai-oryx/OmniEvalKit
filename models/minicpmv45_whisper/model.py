import logging
import math
import os
import time

import numpy as np

from models.minicpmo45_whisper.model import Model as MiniCPMO45WhisperModel


class Model(MiniCPMO45WhisperModel):
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("MINICPMV45_MODEL_PATH", "openbmb/MiniCPM-V-4_5")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.top_p = float(os.getenv("MINICPMV45_TOP_P", "0.8"))
        self.num_beams = int(os.getenv("MINICPMV45_NUM_BEAMS", "3"))
        self.attn_implementation = os.getenv("MINICPMV45_ATTN_IMPLEMENTATION", "sdpa")
        self.torch_dtype_name = os.getenv("MINICPMV45_TORCH_DTYPE", "bfloat16")
        self.device = os.getenv("MINICPMV45_DEVICE", "")
        self.device_map = os.getenv("MINICPMV45_DEVICE_MAP", "").strip()
        self.batch_3d_resampler = self._optional_bool(os.getenv("MINICPMV45_BATCH_3D_RESAMPLER", ""))
        self.enable_thinking = self._str_to_bool(os.getenv("MINICPMV45_ENABLE_THINKING", "False"))
        self.use_image_id = self._optional_bool(os.getenv("MINICPMV45_USE_IMAGE_ID", ""))
        self.max_slice_nums = self._optional_int(os.getenv("MINICPMV45_MAX_SLICE_NUMS", ""))
        self.video_max_slice_nums = self._optional_int(os.getenv("MINICPMV45_VIDEO_MAX_SLICE_NUMS", "1"))
        self.max_inp_length = self._optional_int(os.getenv("MINICPMV45_MAX_INP_LENGTH", ""))
        self.video_fps = float(os.getenv("MINICPMV45_VIDEO_FPS", "5"))
        self.video_max_frames = int(os.getenv("MINICPMV45_VIDEO_MAX_FRAMES", "180"))
        self.video_max_packing = int(os.getenv("MINICPMV45_VIDEO_MAX_PACKING", "3"))
        force_packing = os.getenv("MINICPMV45_FORCE_PACKING", "").strip()
        self.force_packing = int(force_packing) if force_packing else None
        self.video_time_scale = float(os.getenv("MINICPMV45_VIDEO_TIME_SCALE", "0.1"))
        self.use_asr = self._str_to_bool(
            os.getenv("MINICPMV45_USE_ASR", os.getenv("GEMMA4E2B_USE_ASR", os.getenv("QWEN25VL_USE_ASR", "True")))
        )
        self.whisper_use_embedded_video_audio = self._str_to_bool(
            os.getenv("MINICPMV45_WHISPER_USE_EMBEDDED_VIDEO_AUDIO", "True")
        )
        self.asr_model_path = os.getenv(
            "MINICPMV45_ASR_MODEL",
            os.getenv(
                "GEMMA4E2B_ASR_MODEL",
                os.getenv(
                    "QWEN25VL_ASR_MODEL",
                    "/vast/users/imran.razzak/Document/Qwen-omni/Qwen-omni3.5/whisper-large-v3-turbo-hf",
                ),
            ),
        )
        self.asr_device = os.getenv("MINICPMV45_ASR_DEVICE", "cuda")
        self.asr_language = os.getenv(
            "MINICPMV45_ASR_LANGUAGE",
            os.getenv(
                "MINICPMO45_ASR_LANGUAGE",
                os.getenv("GEMMA4E2B_ASR_LANGUAGE", os.getenv("QWEN25VL_ASR_LANGUAGE", "")),
            ),
        ).strip()
        self.asr_min_language_confidence = float(
            os.getenv(
                "MINICPMV45_ASR_MIN_LANGUAGE_CONFIDENCE",
                os.getenv("GEMMA4E2B_ASR_MIN_LANGUAGE_CONFIDENCE", os.getenv("QWEN25VL_ASR_MIN_LANGUAGE_CONFIDENCE", "0.65")),
            )
        )
        self.asr_chunk_seconds = float(
            os.getenv(
                "MINICPMV45_ASR_CHUNK_SECONDS",
                os.getenv(
                    "MINICPMO45_ASR_CHUNK_SECONDS",
                    os.getenv("GEMMA4E2B_ASR_CHUNK_SECONDS", os.getenv("QWEN25VL_ASR_CHUNK_SECONDS", "25")),
                ),
            )
        )
        self.asr_non_speech_text = os.getenv("MINICPMV45_ASR_NON_SPEECH_TEXT", "[Non-speech audio]")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.asr = None
        self.asr_model = None
        self.asr_processor = None
        self._asr_language_token_ids = None
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        self._load()

    def _load(self):
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-V-4_5 requires torch, transformers, accelerate, decord, pillow, "
                "and the Hugging Face remote model code."
            ) from exc

        self._patch_transformers5_tied_weights(transformers)
        self._patch_transformers5_cache_api(transformers)
        dtype_key = "dtype" if self._transformers_major_version(transformers) >= 5 else "torch_dtype"
        kwargs = {
            "trust_remote_code": True,
            dtype_key: self._torch_dtype(torch),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.device_map:
            kwargs["device_map"] = self.device_map
        if self.batch_3d_resampler is not None:
            kwargs["batch_3d_resampler"] = self.batch_3d_resampler

        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self._patch_minicpmv_tokenizer(self.processor.tokenizer)
        self.model.eval()

        if self.device_map:
            logging.warning("MiniCPM-V device map: %s", getattr(self.model, "hf_device_map", {}))
        else:
            target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if target_device and target_device.lower() != "auto":
                self.model = self.model.to(target_device)

    @staticmethod
    def _patch_minicpmv_tokenizer(tokenizer):
        token_attrs = {
            "bos_id": getattr(tokenizer, "bos_token_id", None) or "<|im_start|>",
            "eos_id": getattr(tokenizer, "eos_token_id", None) or "<|im_end|>",
            "eot_id": "<|im_end|>",
            "im_start_id": "<image>",
            "im_end_id": "</image>",
            "slice_start_id": "<slice>",
            "slice_end_id": "</slice>",
        }
        for attr, value in token_attrs.items():
            if hasattr(tokenizer, attr):
                continue
            token_id = value if isinstance(value, int) else tokenizer.convert_tokens_to_ids(value)
            if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(f"Could not resolve MiniCPM-V tokenizer token for {attr}: {value}")
            setattr(tokenizer, attr, token_id)

    def _load_image(self, image_path):
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def _encode_video(self, video_path):
        try:
            return self._encode_video_with_decord(video_path)
        except Exception as exc:
            logging.warning("MiniCPM-V decord video loading failed for %s; using frame-only fallback: %s", video_path, exc)
            frames = self._load_video_frames_only(video_path)
            return frames, self._fallback_temporal_ids(len(frames))

    def _encode_video_with_decord(self, video_path):
        from decord import VideoReader, cpu
        from PIL import Image

        reader = VideoReader(video_path, ctx=cpu(0))
        if len(reader) <= 0:
            raise ValueError(f"No frames decoded from video file: {video_path}")

        fps = max(float(reader.get_avg_fps()), 1.0)
        duration = max(float(len(reader)) / fps, self.video_time_scale)
        sampled_frame_count, packing_nums = self._video_sample_count(duration, fps, len(reader))
        if self.force_packing:
            packing_nums = min(max(1, self.force_packing), max(1, self.video_max_packing))
        frame_idx = self._uniform_sample_midpoint(len(reader), sampled_frame_count)
        frame_ts = frame_idx / fps
        temporal_ids = self._build_temporal_ids(frame_ts, duration, packing_nums)

        batch = reader.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in batch]
        return frames, temporal_ids

    def _video_sample_count(self, duration, source_fps, total_frames):
        max_frames = max(1, self.video_max_frames)
        max_packing = max(1, self.video_max_packing)
        target_fps = max(float(self.video_fps), 0.0)
        if target_fps <= 0:
            return 1, 1

        if target_fps * int(duration) <= max_frames:
            packing_nums = 1
            sample_count = round(min(target_fps, round(source_fps)) * min(max_frames, duration))
        else:
            packing_nums = math.ceil(duration * target_fps / max_frames)
            if packing_nums <= max_packing:
                sample_count = round(duration * target_fps)
            else:
                sample_count = round(max_frames * max_packing)
                packing_nums = max_packing

        sample_count = max(1, min(int(sample_count), total_frames))
        return sample_count, packing_nums

    @staticmethod
    def _uniform_sample_midpoint(total_frames, sample_count):
        if total_frames <= 0:
            return np.array([], dtype=int)
        if sample_count >= total_frames:
            return np.arange(total_frames, dtype=int)
        gap = total_frames / sample_count
        return np.array([int(index * gap + gap / 2) for index in range(sample_count)], dtype=int)

    def _build_temporal_ids(self, frame_ts, duration=None, packing_nums=None):
        scale = max(float(self.video_time_scale), 1e-6)
        if duration:
            scale_values = np.arange(0, duration, scale)
            if scale_values.size == 0:
                scale_values = np.array([0.0])
            scale_ids = self._map_to_nearest_scale(frame_ts, scale_values) / scale
            scale_ids = scale_ids.astype(np.int32)
        else:
            scale_ids = np.maximum(0, np.rint(frame_ts / scale).astype(np.int32))
        group_size = max(1, int(packing_nums or self.video_max_packing))
        groups = []
        for start in range(0, len(scale_ids), group_size):
            groups.append(scale_ids[start : start + group_size].tolist())
        return groups

    @staticmethod
    def _map_to_nearest_scale(values, scale_values):
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.asarray(scale_values)[:, None])
            _, indices = tree.query(np.asarray(values)[:, None])
            return np.asarray(scale_values)[indices]
        except Exception:
            values = np.asarray(values)
            scale_values = np.asarray(scale_values)
            indices = np.abs(values[:, None] - scale_values[None, :]).argmin(axis=1)
            return scale_values[indices]

    def _fallback_temporal_ids(self, frame_count):
        ids = np.arange(max(frame_count, 1), dtype=int)
        groups = []
        for start in range(0, len(ids), max(1, self.video_max_packing)):
            groups.append(ids[start : start + self.video_max_packing].tolist())
        return groups

    def _build_content(self, video_path, image_paths, query):
        content = []
        temporal_ids = None
        for image_path in image_paths:
            content.append(self._load_image(image_path))
        if video_path:
            frames, temporal_ids = self._encode_video(video_path)
            content.extend(frames)
        content.append(query)
        return content, temporal_ids

    def _chat_max_slice_nums(self, has_video):
        if self.max_slice_nums is not None:
            return self.max_slice_nums
        if has_video:
            return self.video_max_slice_nums
        return None

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        self.last_whisper_inference_time_sec = 0.0
        self.last_vlm_inference_time_sec = 0.0
        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)

        transcript = ""
        if self.use_asr:
            whisper_start = time.perf_counter()
            transcript = self._transcribe_media_sources(resolved_audio_paths, resolved_video)
            self.last_whisper_inference_time_sec = time.perf_counter() - whisper_start

        vlm_start = time.perf_counter()
        text_query = self._build_text(query, transcript)
        content, temporal_ids = self._build_content(resolved_video, resolved_images, text_query)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})

        generation_kwargs = {
            "msgs": messages,
            "tokenizer": self.tokenizer,
            "processor": self.processor,
            "stream": False,
            "max_new_tokens": self.max_new_tokens,
            "sampling": self.temperature > 0,
            "enable_thinking": self.enable_thinking,
            "use_image_id": self.use_image_id,
            "max_slice_nums": self._chat_max_slice_nums(bool(resolved_video)),
            "num_beams": self.num_beams,
        }
        if temporal_ids:
            generation_kwargs["temporal_ids"] = temporal_ids
        if self.max_inp_length:
            generation_kwargs["max_inp_length"] = self.max_inp_length
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        answer = self.model.chat(**generation_kwargs)
        self.last_vlm_inference_time_sec = time.perf_counter() - vlm_start
        if answer is None:
            return ""
        if not isinstance(answer, str):
            answer = "".join(str(chunk) for chunk in answer)
        return str(answer).strip()
