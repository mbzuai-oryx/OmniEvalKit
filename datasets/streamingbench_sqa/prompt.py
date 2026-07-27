SYSTEM_PROMPT = (
    "You are evaluating situated real-time QA. "
    "Choose the option matching the observed moment. "
    "Return exactly one option letter."
)


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
    return (
        f"Question: {question}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with only A, B, C, or D."
    )
