import re


SYSTEM_PROMPT = (
    "You are an audio-visual task evaluator. "
    "Use the provided media and text evidence. "
    "Follow the requested answer format exactly."
)
SUPPRESS_MODEL_STDOUT = True
SHOW_MODEL_MESSAGES = True
MEDIA_TOKEN_RE = re.compile(r"<(audio|image|video)_(\d+)>")


def _clean_question(question):
    text = MEDIA_TOKEN_RE.sub(lambda match: f" [{match.group(1)} {match.group(2)}] ", str(question or ""))
    return re.sub(r"[ \t]+", " ", text).strip()


def build_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])

    labels = ["A", "B", "C", "D"]
    choices = "\n".join(
        f"{label}. {option}"
        for label, option in zip(labels, options or [])
    )
    label_text = ", ".join(labels[: len(options or [])] or labels)
    return (
        f"Question:\n{_clean_question(question)}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Answer with only one option letter ({label_text})."
    )


def postprocess_prediction(prediction):
    prediction = str(prediction).strip()
    for pattern in (
        r"</answer>\s*([A-D])\b",
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=>)?\s*([A-D])\b",
        r"^\s*([A-D])\s*$",
        r"^\s*([A-D])[\.\)]",
    ):
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    option_number = re.search(r"(?:option|choice)\s*(?:number\s*)?([1-4])\b", prediction, re.IGNORECASE)
    if option_number:
        return "ABCD"[int(option_number.group(1)) - 1]

    standalone = re.findall(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", prediction, re.IGNORECASE)
    return standalone[-1].upper() if standalone else prediction
