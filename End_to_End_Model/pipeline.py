from dataclasses import dataclass
from pathlib import Path
from typing import Optional


OUTPUT_MODES = ("text", "audio", "both")


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    audio_path: Optional[Path] = None


def run_pipeline(
    model,
    *,
    question,
    video_path=None,
    audio_path=None,
    image_path=None,
    system_prompt="",
    output_mode="text",
    output_audio_path=None,
    tts=None,
):
    """Run multimodal inference and optionally synthesize the answer."""
    question = str(question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    output_mode = str(output_mode).strip().lower()
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of: {', '.join(OUTPUT_MODES)}")

    answer = model.run_inference(
        audio_path=audio_path,
        video_path=video_path,
        image_path=image_path,
        query=question,
        system_prompt=system_prompt,
        sample=None,
    )
    answer = str(answer or "").strip()
    if not answer:
        raise RuntimeError("The multimodal model returned an empty answer.")

    synthesized_path = None
    if output_mode in {"audio", "both"}:
        if tts is None:
            raise ValueError("A TTS synthesizer is required for audio output.")
        if not output_audio_path:
            raise ValueError("output_audio_path is required for audio output.")
        synthesized_path = Path(tts.synthesize(answer, output_audio_path)).resolve()

    return PipelineResult(answer=answer, audio_path=synthesized_path)
