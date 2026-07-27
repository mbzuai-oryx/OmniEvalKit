import re


SYSTEM_PROMPT = "Point to every requested target in the image."


def coordinate_format(model_name):
    normalized_name = re.sub(r"[^a-z0-9]", "", str(model_name or "").lower())
    if normalized_name.startswith("qwen3vl"):
        return "normalized_1000"
    if normalized_name.startswith("minicpm"):
        return "normalized_1"
    return "pixels"


def build_prompt(question, options=None, model_name=None):
    point_format = coordinate_format(model_name)
    if point_format == "normalized_1000":
        coordinates = "coordinates normalized from 0 to 1000 relative to the image width and height"
    elif point_format == "normalized_1":
        coordinates = "coordinates normalized from 0 to 1 relative to the image width and height"
    else:
        coordinates = "original-image pixel coordinates"
    return (
        f"{str(question).strip()}\n"
        f"Return only [[x1, y1], [x2, y2], ...] using {coordinates}. "
        "Return exactly one point inside each requested target and no extra points."
    )
