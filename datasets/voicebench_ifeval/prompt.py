SYSTEM_PROMPT = (
    "You are an instruction-following assistant. "
    "Follow every instruction in the spoken prompt exactly."
)


def build_prompt(question, options=None):
    return "Use the Whisper Transcript as the instruction; follow it exactly; do not repeat."
