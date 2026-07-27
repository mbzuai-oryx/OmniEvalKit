SYSTEM_PROMPT = (
    "You are an instruction-following assistant. "
    "Answer the spoken instruction directly and concisely."
)


def build_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the instruction; answer directly; do not repeat."
