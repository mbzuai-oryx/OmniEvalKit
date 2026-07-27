import re


SYSTEM_PROMPT = (
    "You are a video mathematical reasoning evaluator. "
    "Use the video, audio, subtitles, and question text to solve the problem. "
    "Return exactly one option letter."
)


def build_prompt(question, options=None):
    labels = list("ABCDE")
    used_labels = labels[: len(options or [])] or labels
    choices = "\n".join(f"{label}. {option}" for label, option in zip(used_labels, options or []))
    label_text = ", ".join(used_labels)
    return (
        f"Question:\n{str(question or '').strip()}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Answer with only one option letter ({label_text})."
    )


def postprocess_prediction(prediction):
    text = str(prediction or "").strip()
    for pattern in (
        r"^\s*([A-E])\s*$",
        r"^\s*([A-E])[\.\)]",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-E])\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    standalone = re.findall(r"(?:^|[^A-Za-z])([A-E])(?:[^A-Za-z]|$)", text, re.IGNORECASE)
    return standalone[-1].upper() if standalone else text
