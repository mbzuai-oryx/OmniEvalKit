import re

SYSTEM_PROMPT = (
    "You are a visual point-localization model. "
    "Identify every visible instance matching the request."
)


def coordinate_format(model_name):
    normalized_name = re.sub(r"[^a-z0-9]", "", str(model_name or "").lower())
    if normalized_name.startswith("qwen3vl"):
        return "normalized_1000"
    if normalized_name.startswith("minicpm"):
        return "normalized_1"
    return "pixels"


def build_prompt(question, options=None, model_name=None):
    question = str(question).strip()
    output_format = coordinate_format(model_name)
    if output_format == "normalized_1000":
        coordinates = "coordinates normalized from 0 to 1000"
    elif output_format == "normalized_1":
        coordinates = "coordinates normalized from 0 to 1"
    else:
        coordinates = "original-image pixel coordinates"
    return (
        f"{question}\n"
        f"Return only [[x1, y1], [x2, y2], ...] using {coordinates}. "
        "Return exactly one point inside each requested target, no extra points, and [] when no target is present."
    )
