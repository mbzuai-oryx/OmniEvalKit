import re


SYSTEM_PROMPT = (
    "You are evaluating a spoken reasoning question. "
    "Listen to the audio and give only the final answer."
)

YES_NO_ANSWER_RE = re.compile(r"^\s*the\s+answer\s+is\s*:\s*(yes|no)\s*\.?\s*$", re.IGNORECASE)


def build_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the reasoning question; answer only final answer; do not repeat."


def postprocess_prediction(prediction):
    prediction = str(prediction).strip()
    match = YES_NO_ANSWER_RE.match(prediction)
    if match:
        return match.group(1).capitalize()
    return prediction
