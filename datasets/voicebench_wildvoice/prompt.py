SYSTEM_PROMPT = (
    "You are an instruction-following assistant. "
    "Answer the given spoken instruction directly and naturally."
)


def build_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the instruction; answer naturally; do not repeat."
