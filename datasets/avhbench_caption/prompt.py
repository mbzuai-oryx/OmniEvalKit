SYSTEM_PROMPT = (
    "You are an audio-visual captioning evaluator. "
    "Use both the visual content and the audio content of the provided video."
)


def build_prompt(question, options=None):
    del options
    return (
        f"{str(question or '').strip()}\n"
        "Answer with one concise sentence describing both what is visible and what is audible."
    )


def postprocess_prediction(prediction):
    return str(prediction or "").strip()
