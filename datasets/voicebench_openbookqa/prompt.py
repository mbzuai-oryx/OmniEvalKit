SYSTEM_PROMPT = (
    "You are evaluating a spoken multiple-choice science question. "
    "Listen to the audio and answer with only one option letter."
)


def build_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the MCQ question; answer only option letter; do not repeat."
