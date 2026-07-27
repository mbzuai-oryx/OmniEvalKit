import re


SYSTEM_PROMPT = (
    "You are an audio-visual task evaluator. "
    "Use the provided media and text evidence. "
    "Follow the requested answer format exactly."
)


MEDIA_TOKEN_RE = re.compile(r"<(audio|image|video)_(\d+)>")


def _clean_question(question):
    text = MEDIA_TOKEN_RE.sub(lambda match: f" [{match.group(1)} {match.group(2)}] ", str(question or ""))
    return re.sub(r"[ \t]+", " ", text).strip()


def build_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])

    labels = ["A", "B", "C", "D", "E", "F"]
    used_labels = labels[: len(options or [])]
    choices = "\n".join(
        f"{label}. {option}"
        for label, option in zip(used_labels, options or [])
    )
    label_text = ", ".join(used_labels)
    return (
        f"Question:\n{_clean_question(question)}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Answer with only one option letter ({label_text})."
    )
