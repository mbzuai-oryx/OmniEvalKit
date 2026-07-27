import json
import re
from pathlib import Path


FORCE_AUDIO_ONLY_DATASETS = {"meld"}


SYSTEM_PROMPT_QA = (
    "You are an audio question-answering evaluator. "
    "Use the provided audio evidence and follow the requested answer format exactly."
)

SYSTEM_PROMPT_CAPTION = (
    "You are an audio captioning evaluator. "
    "Describe the provided audio accurately and concisely."
)

SYSTEM_PROMPT_ASR = (
    "You are an automatic speech recognition evaluator. "
    "If an audio transcript is provided, output that transcript only. "
    "Otherwise, transcribe only the speech content from the provided audio."
)

SYSTEM_PROMPT_AST = (
    "You are a speech translation evaluator. "
    "Translate the speech from the provided audio and output only the translation."
)

SYSTEM_PROMPT_CLASSIFICATION = (
    "You are an audio classification evaluator. "
    "Classify the provided audio and follow the requested answer format exactly."
)

SYSTEM_PROMPT_VIDEO_CAPTION = (
    "You are a video captioning evaluator. "
    "Describe the provided video clip accurately and concisely."
)


def load_jsonl_dataset(dataset_name, data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / dataset_name
    manifest = root / "test.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest}. "
            f"Run scripts/prepare_audio_datasets.py {dataset_name} first."
        )

    samples = []
    with manifest.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(normalize_sample(record, root, dataset_name, idx))
    return samples


def normalize_sample(record, root, dataset_name, idx):
    question = first(record, "question", "Question", "prompt", "instruction", "query", "question_text") or ""
    options = normalize_options(first(record, "options", "choices", "Choices"))
    answer = normalize_answer(first(record, "answer", "reference", "Answer", "answers"), options)
    references = normalize_references(record, answer, options)
    sample_id = first(record, "id", "name", "question_id", "key", "save_name") or f"{dataset_name}_{idx}"
    audio_path = resolve_path(root, first(record, "audio_path", "WavPath", "wav_path", "AudioPath"))
    video_path = "" if dataset_name in FORCE_AUDIO_ONLY_DATASETS else resolve_path(
        root,
        first(record, "video_path", "VideoPath", "videoPath"),
    )
    audio_type = str(first(record, "audio_type", "modality") or "speech").strip().lower()

    return {
        "id": str(sample_id),
        "question": str(question).strip(),
        "options": options,
        "answer": answer,
        "audio_path": audio_path,
        "video_path": video_path,
        "image_path": "",
        "reference": answer,
        "references": references,
        "audio_type": audio_type,
    }


def first(record, *keys):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def resolve_path(root, value):
    if value in (None, ""):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    for candidate in (root / path, root / "audio" / path.name):
        if candidate.exists():
            return str(candidate.resolve())
    return str((root / path).resolve())


def normalize_options(options):
    if options in (None, ""):
        return []
    if isinstance(options, str):
        parsed = parse_json_maybe(options)
        options = parsed if parsed is not options else re.split(r"\n|\|", options)
    if isinstance(options, dict):
        options = [options[key] for key in sorted(options)]

    normalized = []
    for option in options:
        if isinstance(option, dict):
            option = first(option, "text", "value", "option", "answer")
        text = strip_option_prefix(option)
        if text:
            normalized.append(text)
    return normalized


def normalize_answer(answer, options=None):
    if isinstance(answer, str):
        parsed = parse_json_maybe(answer)
        if parsed is not answer:
            answer = parsed
    if isinstance(answer, dict):
        for key in ("text", "answer", "value", "aliases"):
            if key in answer:
                answer = answer[key]
                break
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    answer = "" if answer is None else str(answer).strip()
    letter = answer_letter(answer, options or [])
    return letter or answer


def normalize_references(record, answer, options=None):
    references = []
    for key in ("references", "reference", "answers", "answer", "Answer"):
        value = record.get(key)
        references.extend(expand_reference_value(value, options))
    for key in ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5"):
        references.extend(expand_reference_value(record.get(key), options))
    references.extend(expand_reference_value(answer, options))

    deduped = []
    seen = set()
    for reference in references:
        normalized = normalize_text(reference)
        if normalized and normalized not in seen:
            deduped.append(reference)
            seen.add(normalized)
    return deduped


def expand_reference_value(value, options=None):
    if value in (None, ""):
        return []
    if isinstance(value, str):
        parsed = parse_json_maybe(value)
        if parsed is not value:
            value = parsed
    if isinstance(value, dict):
        for key in ("text", "answer", "value", "aliases"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (list, tuple)):
        references = []
        for item in value:
            references.extend(expand_reference_value(item, options))
        return references
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    letter = answer_letter(text, options or [])
    return [letter or text]


def answer_letter(answer, options):
    match = re.match(r"^\s*([A-I])(?:[\.\)]|\s*$)", str(answer), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    normalized = normalize_text(answer)
    for idx, option in enumerate(options):
        if normalize_text(option) == normalized:
            return "ABCDEFGHI"[idx]
    return ""


def build_closed_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])

    options = normalize_options(options)
    choices = ""
    labels = "ABCDEFGHI"[: len(options) or 4]
    if options:
        choices = "\n\nChoices:\n" + "\n".join(
            f"{label}. {option}" for label, option in zip(labels, options)
        )
    return (
        f"Question:\n{str(question).strip()}"
        f"{choices}\n\n"
        f"Answer with only one option letter ({', '.join(labels)})."
    )


def build_open_qa_prompt(question, options=None):
    if isinstance(question, dict):
        question = question.get("question", "")
    return (
        f"Question:\n{str(question).strip()}\n\n"
        "Answer the question directly and concisely."
    )


def build_audio_only_open_qa_prompt(question=None, options=None):
    return (
        "Use the Whisper Transcript as the question and answer it directly. "
        "Do not repeat the question."
    )


def build_audio_only_instruction_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the question and follow its instruction exactly."


def build_audio_only_direct_instruction_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the question and answer its instruction directly."


def build_audio_only_natural_instruction_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the question and answer its instruction directly and naturally."


def build_audio_only_closed_prompt(question=None, options=None):
    return (
        "Use the Whisper Transcript as the question and answer it. "
        "If it is multiple-choice, answer with only the option letter; "
        "otherwise answer with only the final answer."
    )


def build_audio_only_final_answer_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the question and answer with only the final answer."


def build_audio_only_option_prompt(question=None, options=None):
    return "Use the Whisper Transcript as the question and answer with only the option letter."


def build_caption_prompt(question=None, options=None):
    return "Describe this audio clip in one concise sentence."


def build_asr_prompt(question=None, options=None):
    return (
        "If an Audio transcript or Whisper Transcript is provided above, output that transcript exactly. "
        "Otherwise, transcribe the speech in the audio exactly. Output only the transcript."
    )


def build_ast_prompt(question, options=None):
    if isinstance(question, dict):
        question = question.get("question", "")
    return f"{str(question).strip()}\n\nOutput only the translation."


def build_classification_prompt(question, options=None):
    if isinstance(question, dict):
        sample = question
        question = sample.get("question", "")
        options = sample.get("options", [])
    options = normalize_options(options)
    if options:
        labels = "ABCDEFGHI"[: len(options)]
        choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
        return (
            f"Question:\n{str(question).strip()}\n\n"
            f"Choices:\n{choices}\n\n"
            f"Answer with only one option letter ({', '.join(labels)})."
        )
    return f"Question:\n{str(question).strip()}\n\nAnswer with only the class label."


def build_video_caption_prompt(question=None, options=None):
    if isinstance(question, dict):
        question = question.get("question", "")
    question = str(question or "Please describe what is happening in this video clip.").strip()
    return f"Question:\n{question}\n\nOutput only the concise caption."


def postprocess_option_prediction(prediction):
    text = str(prediction or "").strip()
    for pattern in (
        r"^\s*\(?([A-I])\)?\s*$",
        r"^\s*([A-I])[\.\)]",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-I])\)?\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text


def parse_json_maybe(text):
    try:
        return json.loads(text)
    except Exception:
        return text


def strip_option_prefix(value):
    text = "" if value is None else str(value).strip()
    return re.sub(r"^\s*\(?[A-I]\)?[\.\)]\s+(?=.)", "", text, count=1)


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())
