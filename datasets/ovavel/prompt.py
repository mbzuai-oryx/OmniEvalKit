# import re


# SYSTEM_PROMPT = (
#     "You are an audio-visual event localization assistant. "
#     "Given a video and a event, split the video into 10 equal temporal bins 1s each. "
#     "For each bin, output 1 if the event is present, otherwise 0 on that timestamp. "
#     "Return only a 10-element Python list of 0/1 integers."
# )


# def build_prompt(question, options=None):
#     if isinstance(question, dict):
#         question = question.get("question", "")

#     event = _extract_event(question)
#     return (
#         f"Event: {event}\n"
#         "Task: Localize the event in the video.\n"
#         "Output: [x, x, x, x, x, x, x, x, x, x]\n"
#         "Each x must be 0 or 1."
#     )


# def _extract_event(question):
#     text = "" if question is None else str(question).strip()
#     match = re.search(r"event ['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return text





import re


SYSTEM_PROMPT = (
    "You are an event localization model. "
    "For this 10-second video, output one value per second. "
    "Use 1 if the event occurs during that second, else 0. "
    "Return only one Python list with exactly 10 values."
)


def build_prompt(question, options=None):
    if isinstance(question, dict):
        question = question.get("question", "")

    event = _extract_event(question)

    return (
        f"Event: {event}\n"
        "Video duration: 10 seconds.\n"
        "Output event presence for seconds 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10.\n"
        "Use 1 for present, 0 for absent.\n"
        "Format: [0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1]\n"
        "Return only the list."
    )


def _extract_event(question):
    text = "" if question is None else str(question).strip()

    match = re.search(r"event ['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text