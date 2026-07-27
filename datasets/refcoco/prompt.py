import re


SYSTEM_PROMPT = "Locate the referred object in the image."


def coordinate_format(model_name):
    normalized_name = re.sub(r"[^a-z0-9]", "", str(model_name or "").lower())
    return "normalized_1000" if normalized_name.startswith("qwen3vl") else "pixels"


def build_prompt(question, options=None, model_name=None):
    expression = str(question).strip()
    normalized_name = re.sub(r"[^a-z0-9]", "", str(model_name or "").lower())

    if coordinate_format(model_name) == "normalized_1000":
        return (
            f"Locate: {expression}\n"
            "Return only {\"bbox_2d\": [x_min, y_min, x_max, y_max]}, with coordinates normalized "
            "to integers from 0 to 1000 relative to the image width and height."
        )
    if normalized_name.startswith("qwen25vl"):
        return (
            f"Locate: {expression}\n"
            "Return only {\"bbox_2d\": [x_min, y_min, x_max, y_max]} using original-image pixel coordinates."
        )
    if normalized_name.startswith("minicpm"):
        return (
            f"Locate: {expression}\n"
            "Return only [x_min, y_min, x_max, y_max] using original-image pixel coordinates."
        )
    return (
        f"Locate: {expression}\n"
        "Return only [x_min, y_min, x_max, y_max] using original-image pixel coordinates."
    )


def postprocess_prediction(prediction):
    text = str(prediction)
    bracketed = re.search(r"\[([^\[\]]+)\]", text)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", bracketed.group(1) if bracketed else text)
    if len(numbers) < 4:
        return text.strip()
    return "[" + ", ".join(numbers[:4]) + "]"
