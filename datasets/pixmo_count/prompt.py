import re


SYSTEM_PROMPT = "Count the requested objects in the image. Return only the integer count."


def build_prompt(question, options=None):
    return f"{str(question).strip()}\nAnswer with only one integer."


def postprocess_prediction(prediction):
    match = re.search(r"-?\d+", str(prediction))
    return match.group(0) if match else str(prediction).strip()
