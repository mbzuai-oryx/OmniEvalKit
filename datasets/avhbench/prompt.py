import re


SYSTEM_PROMPT = (
    "You are an audio-visual hallucination benchmark evaluator. "
    "Use both the visual content and the audio content of the provided video. "
    "Answer the question exactly as requested."
)


def build_prompt(question, options=None):
    del options
    return (
        f"Question:\n{str(question or '').strip()}\n\n"
        "Answer with only one word: Yes or No."
    )


def postprocess_prediction(prediction):
    text = str(prediction or "").strip()
    for pattern in (
        r"^\s*(yes|no)\b",
        r"(?:answer|final answer)\s*(?:is|:)?\s*(yes|no)\b",
        r"\b(yes|no)\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    return text
