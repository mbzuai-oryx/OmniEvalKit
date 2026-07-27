from datasets._audio_common import build_audio_only_option_prompt


SYSTEM_PROMPT = (
    "You are evaluating an audio multiple-choice question. "
    "Listen to the audio and answer with only one option letter."
)


build_prompt = build_audio_only_option_prompt
