import site
import sys
from pathlib import Path


COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def remove_user_site_from_path():
    """Prevent ~/.local packages from shadowing the active conda environment."""
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    normalized_user_sites = {str(Path(path).resolve()) for path in user_sites}
    sys.path[:] = [
        path
        for path in sys.path
        if not path or str(Path(path).resolve()) not in normalized_user_sites
    ]


def load_wav_with_soundfile(wav_path, target_sample_rate):
    """Load audio without TorchCodec, whose CUDA wheels do not support ROCm."""
    import soundfile
    import torch
    from torchaudio.functional import resample

    audio, sample_rate = soundfile.read(
        str(wav_path),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(audio.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = resample(waveform, sample_rate, target_sample_rate)
    return waveform


class CosyVoice3TTS:
    def __init__(
        self,
        *,
        model_dir,
        cosyvoice_repo,
        prompt_audio,
        prompt_text,
        speed=1.0,
        fp16=False,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.cosyvoice_repo = Path(cosyvoice_repo).expanduser().resolve()
        self.prompt_audio = Path(prompt_audio).expanduser().resolve()
        self.prompt_text = self._normalize_prompt_text(prompt_text)
        self.speed = float(speed)
        self.fp16 = bool(fp16)
        self._model = None

    @staticmethod
    def _normalize_prompt_text(prompt_text):
        prompt_text = str(prompt_text or "").strip()
        if not prompt_text:
            raise ValueError("CosyVoice3 prompt text must not be empty.")
        if "<|endofprompt|>" in prompt_text:
            return prompt_text
        return f"{COSYVOICE3_PROMPT_PREFIX}{prompt_text}"

    def _validate_paths(self):
        config_path = self.model_dir / "cosyvoice3.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"CosyVoice3 checkpoint is missing {config_path}. "
                "Run scripts/setup_cosyvoice3.sh first."
            )
        if not (self.cosyvoice_repo / "cosyvoice").is_dir():
            raise FileNotFoundError(
                f"CosyVoice source was not found under {self.cosyvoice_repo}. "
                "Run scripts/setup_cosyvoice3.sh first."
            )
        if not self.prompt_audio.is_file():
            raise FileNotFoundError(f"CosyVoice3 prompt audio was not found: {self.prompt_audio}")
        if self.speed <= 0:
            raise ValueError("CosyVoice3 speed must be greater than zero.")

    def _load_model(self):
        if self._model is not None:
            return self._model

        self._validate_paths()
        remove_user_site_from_path()
        matcha_repo = self.cosyvoice_repo / "third_party" / "Matcha-TTS"
        for import_path in (matcha_repo, self.cosyvoice_repo):
            import_path = str(import_path)
            if import_path not in sys.path:
                sys.path.insert(0, import_path)

        from cosyvoice.cli import frontend
        from cosyvoice.cli.cosyvoice import AutoModel

        frontend.load_wav = load_wav_with_soundfile

        self._model = AutoModel(model_dir=str(self.model_dir), fp16=self.fp16)
        # Transformers 5 honors the checkpoint's bfloat16 config while the
        # CosyVoice projection layers load as float32. Keep the whole LLM on
        # one dtype to avoid a float/bfloat16 matmul mismatch.
        if self.fp16:
            self._model.model.llm.half()
        else:
            self._model.model.llm.float()
        return self._model

    def synthesize(self, text, output_path):
        text = str(text or "").strip()
        if not text:
            raise ValueError("Cannot synthesize an empty answer.")

        model = self._load_model()
        chunks = []
        for output in model.inference_zero_shot(
            text,
            self.prompt_text,
            str(self.prompt_audio),
            stream=False,
            speed=self.speed,
        ):
            speech = output.get("tts_speech")
            if speech is not None and speech.numel():
                if speech.ndim == 1:
                    speech = speech.unsqueeze(0)
                chunks.append(speech.detach().cpu())

        if not chunks:
            raise RuntimeError("CosyVoice3 returned no audio samples.")

        import torch
        import soundfile

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = torch.cat(chunks, dim=-1)
        soundfile.write(
            str(output_path),
            waveform.squeeze(0).numpy(),
            model.sample_rate,
            subtype="PCM_16",
        )
        return output_path
