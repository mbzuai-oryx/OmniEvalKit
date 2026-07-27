from datasets._audio_common import build_audio_only_open_qa_prompt


SYSTEM_PROMPT = (
    "You are an instruction-following assistant. "
    "Answer the spoken instruction directly and concisely."
)


build_prompt = build_audio_only_open_qa_prompt
