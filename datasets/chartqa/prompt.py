SYSTEM_PROMPT = (
    "You are a chart question answering evaluator. "
    "Read the chart carefully and answer with only the final answer."
)


def build_prompt(question, options=None):
    return f"{str(question).strip()}\nAnswer with only the final answer. Do not include explanation."
