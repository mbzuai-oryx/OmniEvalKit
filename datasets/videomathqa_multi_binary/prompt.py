from datasets.videomathqa_mcq.prompt import SYSTEM_PROMPT, postprocess_prediction


def build_prompt(question, options=None):
    labels = ["A", "B"]
    choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, options or []))
    return (
        f"Question:\n{str(question or '').strip()}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with only one option letter (A or B)."
    )
