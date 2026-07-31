from pathlib import Path

import pytest

from End_to_End_Model.cli import _normalize_question
from End_to_End_Model.cosyvoice3_tts import (
    COSYVOICE3_PROMPT_PREFIX,
    CosyVoice3TTS,
    load_wav_with_soundfile,
)
from End_to_End_Model.pipeline import run_pipeline


class FakeModel:
    def __init__(self, answer="The player celebrates with teammates."):
        self.answer = answer
        self.calls = []

    def run_inference(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, output_path):
        self.calls.append((text, output_path))
        return Path(output_path)


def test_text_mode_runs_model_without_tts():
    model = FakeModel()

    result = run_pipeline(
        model,
        question="What happens next?",
        video_path="match.mp4",
        audio_path=["commentary.wav"],
        output_mode="text",
    )

    assert result.answer == "The player celebrates with teammates."
    assert result.audio_path is None
    assert model.calls[0]["video_path"] == "match.mp4"
    assert model.calls[0]["audio_path"] == ["commentary.wav"]


@pytest.mark.parametrize("output_mode", ["audio", "both"])
def test_audio_modes_synthesize_the_generated_answer(tmp_path, output_mode):
    model = FakeModel()
    tts = FakeTTS()
    output_path = tmp_path / "answer.wav"

    result = run_pipeline(
        model,
        question="What happens next?",
        output_mode=output_mode,
        output_audio_path=output_path,
        tts=tts,
    )

    assert result.audio_path == output_path.resolve()
    assert tts.calls == [("The player celebrates with teammates.", output_path)]


def test_audio_mode_requires_tts(tmp_path):
    with pytest.raises(ValueError, match="TTS synthesizer"):
        run_pipeline(
            FakeModel(),
            question="What happens next?",
            output_mode="audio",
            output_audio_path=tmp_path / "answer.wav",
        )


def test_cosyvoice3_adds_required_prompt_prefix():
    assert (
        CosyVoice3TTS._normalize_prompt_text("Reference transcript.")
        == f"{COSYVOICE3_PROMPT_PREFIX}Reference transcript."
    )
    already_prefixed = f"{COSYVOICE3_PROMPT_PREFIX}Reference transcript."
    assert CosyVoice3TTS._normalize_prompt_text(already_prefixed) == already_prefixed


def test_soundfile_loader_mixes_to_mono_and_resamples(tmp_path):
    import numpy as np
    import soundfile

    input_path = tmp_path / "stereo.wav"
    soundfile.write(input_path, np.zeros((800, 2), dtype=np.float32), 8000)

    waveform = load_wav_with_soundfile(input_path, 16000)

    assert waveform.shape[0] == 1
    assert 1590 <= waveform.shape[1] <= 1610


def test_cosyvoice3_saves_all_generated_chunks(tmp_path):
    import soundfile
    import torch

    class FakeCosyVoice:
        sample_rate = 24000

        @staticmethod
        def inference_zero_shot(*args, **kwargs):
            yield {"tts_speech": torch.zeros(1, 4)}
            yield {"tts_speech": torch.zeros(1, 6)}

    tts = CosyVoice3TTS(
        model_dir=tmp_path,
        cosyvoice_repo=tmp_path,
        prompt_audio=tmp_path / "voice.wav",
        prompt_text="Reference transcript.",
    )
    tts._load_model = lambda: FakeCosyVoice()
    output_path = tmp_path / "answer.wav"

    assert tts.synthesize("Answer text.", output_path) == output_path.resolve()
    assert soundfile.info(output_path).frames == 10


def test_literal_newlines_from_shell_are_normalized():
    assert _normalize_question("Question\\nA. First\\nB. Second") == "Question\nA. First\nB. Second"
