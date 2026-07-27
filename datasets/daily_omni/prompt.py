SYSTEM_PROMPT = (
    "You are evaluating an audio-visual multiple-choice question. "
    "Use the provided media and answer with only one option letter."
)


def build_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])

    labels = ["A", "B", "C", "D"]
    options = "\n".join(
        f"{label}. {option}"
        for label, option in zip(labels, options or [])
    )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{options}\n\n"
        "Answer with only the letter (A, B, C, or D)"
    )
