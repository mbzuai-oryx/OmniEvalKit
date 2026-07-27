import re


SYSTEM_PROMPT = "Answer the image question with only yes or no."


def build_prompt(question, options=None):
    return str(question).strip()


def postprocess_prediction(prediction):
    match = re.match(r"\s*(yes|no)\b", str(prediction), re.IGNORECASE)
    return match.group(1).capitalize() if match else str(prediction).strip()
