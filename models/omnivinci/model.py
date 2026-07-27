import copy
import importlib.util
import os
import sys
import site
import types
from pathlib import Path


class Model:
    def __init__(self, model_path=None):
        self.package_dir = Path(__file__).resolve().parent
        self.local_python_deps = self.package_dir / "python_deps"
        configured_path = model_path or os.getenv("OMNIVINCI_MODEL_PATH")
        if not configured_path:
            raise ValueError("Set --model_path or OMNIVINCI_MODEL_PATH to the external OmniVinci checkpoint")
        self.model_path = str(Path(configured_path).expanduser().resolve())
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.top_p = self._optional_float(os.getenv("OMNIVINCI_TOP_P", ""))
        self.num_video_frames = self._optional_int(os.getenv("OMNIVINCI_NUM_VIDEO_FRAMES", "128"))
        self.load_audio_in_video = self._str_to_bool(os.getenv("OMNIVINCI_LOAD_AUDIO_IN_VIDEO", "True"))
        self.audio_length = os.getenv("OMNIVINCI_AUDIO_LENGTH", "max_3600")
        self.torch_dtype_name = os.getenv("OMNIVINCI_TORCH_DTYPE", "float16")
        self.attn_implementation = os.getenv("OMNIVINCI_ATTN_IMPLEMENTATION", "sdpa")
        self.device_map = os.getenv("OMNIVINCI_DEVICE_MAP", "auto")
        self.low_cpu_mem_usage = self._str_to_bool(os.getenv("OMNIVINCI_LOW_CPU_MEM_USAGE", "True"))
        self.include_video_audio_with_separate_audio = self._str_to_bool(
            os.getenv("OMNIVINCI_INCLUDE_VIDEO_AUDIO_WITH_SEPARATE_AUDIO", "False")
        )
        self.model = None
        self.processor = None
        self.generation_config = None
        self._load()

    @staticmethod
    def _str_to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _optional_int(value):
        value = str(value or "").strip()
        return int(value) if value else None

    @staticmethod
    def _optional_float(value):
        value = str(value or "").strip()
        return float(value) if value else None

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
        raise ValueError(f"Unsupported OMNIVINCI_TORCH_DTYPE: {self.torch_dtype_name}")

    @staticmethod
    def _transformers_major_version(transformers):
        version = getattr(transformers, "__version__", "0")
        major = str(version).split(".", 1)[0]
        return int(major) if major.isdigit() else 0

    def _load(self):
        self._disable_user_site_packages()
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        self._patch_transformers_compat()
        self._patch_flash_attention_requests()
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"OmniVinci checkpoint not found: {self.model_path}")
        if self.local_python_deps.exists() and str(self.local_python_deps) not in sys.path:
            sys.path.insert(0, str(self.local_python_deps))
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._torch_dtype(torch),
            "device_map": self.device_map,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if not self.device_map:
            kwargs.pop("device_map", None)

        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        self._force_model_dtype(torch)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        except FileNotFoundError:
            self.processor = self._load_local_processor()
        self._align_projector_dtype()
        self._patch_image_processor_crop_size()
        self._patch_process_image_crop_size()
        self._patch_load_video_fallback()
        self.model.eval()
        self._apply_runtime_config()
        self.generation_config = copy.deepcopy(self.model.default_generation_config)

    def _load_local_processor(self):
        package_name = "_omnieval_omnivinci_checkpoint"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [self.model_path]
            package.__package__ = package_name
            sys.modules[package_name] = package

        module_name = f"{package_name}.auto_processor"
        module_path = Path(self.model_path) / "auto_processor.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.VILAProcessor.from_pretrained(self.model_path)

    def _patch_image_processor_crop_size(self):
        image_processor = getattr(self.processor, "image_processor", None)
        self._ensure_image_processor_crop_size(image_processor)

    @staticmethod
    def _ensure_image_processor_crop_size(image_processor):
        if image_processor is None:
            return None
        crop_size = getattr(image_processor, "crop_size", None)
        if crop_size is not None:
            return crop_size
        size = getattr(image_processor, "size", None)
        try:
            height = size["height"]
            width = size["width"]
        except (TypeError, KeyError):
            return None
        image_processor.crop_size = {"height": height, "width": width}
        return image_processor.crop_size

    def _patch_process_image_crop_size(self):
        processor_module = sys.modules.get(type(self.processor).__module__)
        if processor_module is None:
            return
        process_image = getattr(processor_module, "process_image", None)
        if process_image is None or getattr(process_image, "_omnieval_crop_size_patch", False):
            return

        def patched_process_image(image_file, data_args, image_folder, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None):
            Model._ensure_image_processor_crop_size(getattr(data_args, "image_processor", None))
            return process_image(
                image_file,
                data_args,
                image_folder,
                enable_dynamic_res=enable_dynamic_res,
                enable_dynamic_s2=enable_dynamic_s2,
                max_tiles=max_tiles,
            )

        patched_process_image._omnieval_crop_size_patch = True
        processor_module.process_image = patched_process_image
        mm_utils_module = sys.modules.get(getattr(process_image, "__module__", ""))
        if mm_utils_module is not None:
            mm_utils_module.process_image = patched_process_image

    def _patch_load_video_fallback(self):
        media_module_name = f"{type(self.processor).__module__.rsplit('.', 1)[0]}.media"
        media_module = sys.modules.get(media_module_name)
        if media_module is None:
            return
        load_video = getattr(media_module, "_load_video", None)
        if load_video is None or getattr(load_video, "_omnieval_video_fallback_patch", False):
            return

        def patched_load_video(video_path, *, num_frames, config, load_aud=False):
            if not load_aud and self._video_frame_count(video_path) <= num_frames:
                frames = self._load_video_frames(video_path, num_frames)
                return frames, None, self._video_info(video_path, frames, None, num_frames)

            try:
                frames, aud_feature, video_info = load_video(
                    video_path,
                    num_frames=num_frames,
                    config=config,
                    load_aud=load_aud,
                )
            except ValueError as exc:
                if load_aud or "has no frames" not in str(exc):
                    raise
                frames = self._load_video_frames_with_pyav(video_path, num_frames)
                return frames, None, self._video_info(video_path, frames, None, num_frames)
            if frames:
                return frames, aud_feature, video_info

            frames = self._load_video_frames(video_path, num_frames)
            return frames, aud_feature, self._video_info(video_path, frames, aud_feature, num_frames, video_info)

        patched_load_video._omnieval_video_fallback_patch = True
        media_module._load_video = patched_load_video

    @staticmethod
    def _sample_indices(total_frames, num_frames):
        import numpy as np

        if total_frames <= 0:
            return [0] * num_frames
        return np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int).tolist()

    @classmethod
    def _load_video_frames(cls, path, num_frames):
        try:
            return cls._load_video_frames_with_cv2(path, num_frames)
        except Exception:
            return cls._load_video_frames_with_pyav(path, num_frames)

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

    @staticmethod
    def _video_frame_times(path, num_frames):
        try:
            import cv2

            capture = cv2.VideoCapture(path)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            capture.release()
            if fps > 0:
                return [index / fps for index in range(num_frames)]
        except Exception:
            pass
        return list(range(num_frames))

    @staticmethod
    def _video_frame_count(path):
        try:
            import cv2

            capture = cv2.VideoCapture(path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            capture.release()
            return frame_count
        except Exception:
            return 0

    @classmethod
    def _video_info(cls, path, frames, aud_feature, expected_frame_count, video_info=None):
        video_info = video_info or {}
        video_info.setdefault("video_path", path)
        video_info.setdefault("has_audio", aud_feature is not None)
        video_info["video_frame_times"] = cls._video_frame_times(path, len(frames))
        video_info.setdefault("expected_frame_count", expected_frame_count)
        return video_info

    def _align_projector_dtype(self):
        target_dtype = getattr(self.model, "dtype", None)
        if target_dtype is None:
            return
        for getter_name in ("get_mm_projector", "get_sound_mm_projector", "get_speech_mm_projector"):
            getter = getattr(self.model, getter_name, None)
            if getter is None:
                continue
            projector = getter()
            if projector is not None:
                projector.to(dtype=target_dtype)

    def _force_model_dtype(self, torch):
        target_dtype = self._torch_dtype(torch)
        if isinstance(target_dtype, str):
            return
        self.model.to(dtype=target_dtype)
        self.model.config.torch_dtype = target_dtype
        self.model.config.model_dtype = target_dtype

    def _patch_flash_attention_requests(self):
        replacement = self.attn_implementation
        if replacement == "flash_attention_2":
            return

        def patch_from_pretrained(model_cls):
            if model_cls is None or getattr(model_cls, "_omnieval_attn_patch", False):
                return
            original = model_cls.from_pretrained.__func__

            def patched_from_pretrained(cls, *args, **kwargs):
                if kwargs.get("attn_implementation") == "flash_attention_2":
                    kwargs["attn_implementation"] = replacement
                return original(cls, *args, **kwargs)

            model_cls.from_pretrained = classmethod(patched_from_pretrained)
            model_cls._omnieval_attn_patch = True

        try:
            from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        except ImportError:
            SiglipVisionModel = None
        patch_from_pretrained(SiglipVisionModel)

        try:
            from transformers import Qwen2AudioEncoder
        except ImportError:
            Qwen2AudioEncoder = None
        patch_from_pretrained(Qwen2AudioEncoder)

    def _apply_runtime_config(self, use_audio_in_video=None):
        effective_audio = self.load_audio_in_video if use_audio_in_video is None else use_audio_in_video
        self.model.config.load_audio_in_video = effective_audio
        self.processor.config.load_audio_in_video = effective_audio
        if self.num_video_frames is not None and self.num_video_frames > 0:
            self.model.config.num_video_frames = self.num_video_frames
            self.processor.config.num_video_frames = self.num_video_frames
        if str(self.audio_length).strip() != "-1":
            self.model.config.audio_chunk_length = self.audio_length
            self.processor.config.audio_chunk_length = self.audio_length

    def _build_messages(self, system_prompt, image_paths, video_path, audio_paths, query, use_audio_in_video):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        if video_path:
            content.append({"type": "video", "video": video_path})
        if not use_audio_in_video:
            for audio_path in audio_paths:
                content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": str(query or "").strip()})
        messages.append({"role": "user", "content": content})
        return messages

    def _model_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def run_inference(self, audio_path, video_path, query, system_prompt, image_path=None, sample=None):
        import torch

        resolved_video = self._path_value(video_path)
        resolved_images = self._path_values(image_path)
        resolved_audio_paths = self._path_values(audio_path)
        use_audio_in_video = bool(
            resolved_video
            and self.load_audio_in_video
            and (self.include_video_audio_with_separate_audio or not resolved_audio_paths)
        )

        self._apply_runtime_config(use_audio_in_video)
        messages = self._build_messages(
            system_prompt,
            resolved_images,
            resolved_video,
            resolved_audio_paths,
            query,
            use_audio_in_video,
        )
        conversation = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor([conversation])
        self._patch_multi_sound_audio_info(inputs)

        device = self._model_device()
        input_ids = inputs.input_ids.to(device)

        generation_config = copy.deepcopy(self.generation_config)
        generation_kwargs = {"max_new_tokens": self.max_new_tokens, "max_length": 99999999}
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p
        else:
            generation_kwargs["do_sample"] = False
        generation_config.update(**generation_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                media=getattr(inputs, "media", None),
                media_config=getattr(inputs, "media_config", None),
                generation_config=generation_config,
            )

        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        input_length = input_ids.shape[1]
        generated_ids = output_ids[:, input_length:] if output_ids.shape[1] > input_length else output_ids
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    @staticmethod
    def _patch_multi_sound_audio_info(inputs):
        media = getattr(inputs, "media", None)
        if not media:
            return
        sounds = media.get("sound")
        audio_info = media.get("audio_info")
        if not isinstance(sounds, list) or not isinstance(audio_info, list):
            return
        if len(sounds) <= 1 or len(audio_info) != 1:
            return
        first = audio_info[0]
        if isinstance(first, list) and len(first) == len(sounds):
            media["audio_info"] = [[item] for item in first]

    @staticmethod
    def _disable_user_site_packages():
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        user_site = getattr(site, "USER_SITE", None)
        if not user_site:
            return
        user_site = Path(user_site).resolve().as_posix()
        sys.path[:] = [
            entry
            for entry in sys.path
            if not entry or not Path(entry).resolve().as_posix().startswith(user_site)
        ]
        for name, module in list(sys.modules.items()):
            if not name.startswith(("huggingface_hub", "transformers")):
                continue
            module_file = getattr(module, "__file__", "")
            if module_file and Path(module_file).resolve().as_posix().startswith(user_site):
                del sys.modules[name]

    @staticmethod
    def _patch_transformers_compat():
        try:
            import contextlib
            import transformers.modeling_utils as modeling_utils
            from transformers.modeling_utils import PreTrainedModel
        except ImportError:
            return

        if not hasattr(modeling_utils, "no_init_weights"):
            modeling_utils.no_init_weights = contextlib.nullcontext

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
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
