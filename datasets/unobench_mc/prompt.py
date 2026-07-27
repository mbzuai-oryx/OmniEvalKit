import re


SYSTEM_PROMPT = (
    "You are an audio-visual task evaluator. "
    "Use the provided media and text evidence. "
    "Follow the requested answer format exactly."
)


MEDIA_TOKEN_RE = re.compile(r"<(audio|image|video)_(\d+)>")
CHOICE_RE = re.compile(
    r"(?:^|\n)\s*([A-I])[\.\)]\s*(.*?)(?=(?:\n\s*[A-I][\.\)]\s*)|\n\s*(?:Please select|请从|请选择)|$)",
    re.DOTALL,
)
ANSWER_RE = re.compile(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", re.IGNORECASE)


def _replace_media_tokens(text):
    def repl(match):
        kind, index = match.groups()
        return f" [{kind} {index}] "

    return MEDIA_TOKEN_RE.sub(repl, str(text or ""))


def _strip_option_prefix(option):
    return re.sub(r"^\s*\(?[A-I]\)?[\.\)]\s+(?=.)", "", str(option or "").strip(), count=1)


def _split_embedded_options(question):
    text = str(question or "").strip()
    matches = list(CHOICE_RE.finditer(text))
    if len(matches) < 2:
        inline_choice_re = re.compile(
            r"(?:^|\s)\(([A-I])\)\s*(.*?)(?=(?:\s+\([A-I]\)\s*)|$)",
            re.DOTALL,
        )
        matches = list(inline_choice_re.finditer(text))
    if len(matches) < 2:
        return text, []

    options = [_strip_option_prefix(match.group(0)) for match in matches]
    question_text = text[: matches[0].start()].strip()
    tail = text[matches[-1].end() :].strip()
    tail = re.sub(r"^(?:Please select|请从|请选择).*", "", tail, flags=re.DOTALL).strip()
    if tail:
        question_text = f"{question_text}\n{tail}".strip()
    return question_text, [option for option in options if option]


def _clean_question(question):
    text = _replace_media_tokens(question)
    return re.sub(r"[ \t]+", " ", text).strip()


def _normalize_options(options):
    if not options:
        return []
    if isinstance(options, dict):
        options = [options[key] for key in sorted(options)]
    if isinstance(options, str):
        options = re.split(r"\n|\|", options)
    return [_strip_option_prefix(option) for option in options if str(option or "").strip()]


def _format_choices(options):
    labels = "ABCDEFGHI"[: len(options)]
    return "\n".join(f"{label}. {option}" for label, option in zip(labels, options))


def build_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])

    question, embedded_options = _split_embedded_options(question)
    options = _normalize_options(options) or embedded_options
    question = _clean_question(question)
    labels = ", ".join("ABCDEFGHI"[: len(options) or 4])

    if re.fullmatch(r"Answer the question in the following audio:?\s*(?:\[audio\s+\d+\])?", question, re.IGNORECASE):
        question = "Answer the multiple-choice question asked in the provided audio."

    choices = f"\n\nChoices:\n{_format_choices(options)}" if options else ""
    return (
        f"Question:\n{question}"
        f"{choices}\n\n"
        f"Answer with only one option letter ({labels})."
    )


def postprocess_prediction(prediction):
    text = str(prediction or "").strip()
    for pattern in (
        r"^\s*([A-D])\s*$",
        r"^\s*([A-D])[\.\)]",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = ANSWER_RE.search(text)
    return match.group(1).upper() if match else text
