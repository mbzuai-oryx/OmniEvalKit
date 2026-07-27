#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


VOICEBENCH_REPO = "hlt-lab/voicebench"
OMNIEVAL_REPO = "OmniEvalKit/omnievalkit-dataset"


DATASETS = {
    "gigaspeech_test": {"repo": "AudioLLMs/gigaspeech_test", "prefix": "data/"},
    "wenetspeech_test_net": {"repo": OMNIEVAL_REPO, "prefix": "wenetspeech_test_net/"},
    "wenetspeech_test_meeting": {"repo": OMNIEVAL_REPO, "prefix": "wenetspeech_test_meeting/"},
    "librispeech_test_clean": {"repo": OMNIEVAL_REPO, "prefix": "librispeech_test_clean/"},
    "librispeech_test_other": {"repo": OMNIEVAL_REPO, "prefix": "librispeech_test_other/"},
    "librispeech_dev_clean": {"repo": OMNIEVAL_REPO, "prefix": "librispeech_dev_clean/"},
    "librispeech_dev_other": {"repo": OMNIEVAL_REPO, "prefix": "librispeech_dev_other/"},
    "commonvoice_en": {"repo": OMNIEVAL_REPO, "prefix": "commonvoice_en/"},
    "commonvoice_zh": {"repo": OMNIEVAL_REPO, "prefix": "commonvoice_zh/"},
    "commonvoice_yue": {"repo": OMNIEVAL_REPO, "prefix": "commonvoice_yue/"},
    "commonvoice_fr": {"repo": OMNIEVAL_REPO, "prefix": "commonvoice_fr/"},
    "aishell1_test": {"repo": OMNIEVAL_REPO, "prefix": "aishell1_test/"},
    "aishell2_test": {"repo": OMNIEVAL_REPO, "prefix": "aishell2_test/"},
    "kespeech_test": {"repo": OMNIEVAL_REPO, "prefix": "kespeech_test/"},
    "voxpopuli_en": {"repo": OMNIEVAL_REPO, "prefix": "voxpopuli_en/"},
    "fleurs_zh": {"repo": OMNIEVAL_REPO, "prefix": "fleurs_zh/"},
    "fleurs_en": {"repo": OMNIEVAL_REPO, "prefix": "fleurs_en/"},
    "peoples_speech_test": {"repo": OMNIEVAL_REPO, "prefix": "peoples_speech_test/"},
    "tedlium3_test": {"repo": OMNIEVAL_REPO, "prefix": "tedlium3_test/"},
    "voicebench_alpacaeval": {"repo": VOICEBENCH_REPO, "prefix": "alpacaeval/"},
    "voicebench_alpacaeval_full": {"repo": VOICEBENCH_REPO, "prefix": "alpacaeval_full/"},
    "voicebench_bbh": {"repo": VOICEBENCH_REPO, "prefix": "bbh/"},
    "voicebench_mmsu": {"repo": VOICEBENCH_REPO, "prefix": "mmsu/"},
    "voicebench_openbookqa": {"repo": VOICEBENCH_REPO, "prefix": "openbookqa/"},
    "voicebench_advbench": {"repo": VOICEBENCH_REPO, "prefix": "advbench/"},
    "voicebench_commoneval": {"repo": VOICEBENCH_REPO, "prefix": "commoneval/"},
    "voicebench_ifeval": {"repo": VOICEBENCH_REPO, "prefix": "ifeval/"},
    "voicebench_sdqa": {"repo": VOICEBENCH_REPO, "prefix": "sd-qa/"},
    "voicebench_wildvoice": {"repo": VOICEBENCH_REPO, "prefix": "wildvoice/"},
    "voice_cmmlu": {"repo": OMNIEVAL_REPO, "prefix": "voice_cmmlu/"},
    "mmau_test_mini": {"repo": OMNIEVAL_REPO, "prefix": "mmau_test_mini/"},
    "mmsu_bench": {"repo": OMNIEVAL_REPO, "prefix": "mmsu_bench/"},
    "mmar_bench": {"repo": OMNIEVAL_REPO, "prefix": "mmar_bench/"},
    "audio_web_questions": {"repo": OMNIEVAL_REPO, "prefix": "audio_web_questions/"},
    "audio_trivia_qa": {"repo": OMNIEVAL_REPO, "prefix": "audio_trivia_qa/"},
    "audiocaps_test": {"repo": "AudioLLMs/audiocaps_test", "prefix": "data/"},
    "clothocaption_test": {"repo": OMNIEVAL_REPO, "prefix": "clothocaption_test/"},
    "wavcaps_audioset_sl": {"repo": OMNIEVAL_REPO, "prefix": "wavcaps_audioset_sl/"},
    "wavcaps_freesound": {"repo": OMNIEVAL_REPO, "prefix": "wavcaps_freesound/"},
    "wavcaps_soundbible": {"repo": OMNIEVAL_REPO, "prefix": "wavcaps_soundbible/"},
    "covost2_zh_en": {"repo": OMNIEVAL_REPO, "prefix": "covost2_zh_en/"},
    "covost2_en_zh": {"repo": OMNIEVAL_REPO, "prefix": "covost2_en_zh/"},
    "vocalsound": {"repo": OMNIEVAL_REPO, "prefix": "vocalsound/"},
    "meld": {"repo": OMNIEVAL_REPO, "prefix": "meld/"},
    "livesports3k_cc": {"repo": OMNIEVAL_REPO, "prefix": "livesports3k_cc/"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS) + ["all"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--raw-root", default="data/raw_hf_audio")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    names = sorted(DATASETS) if "all" in args.datasets else args.datasets
    for name in names:
        prepare_dataset(name, Path(args.data_root), Path(args.raw_root), args.batch_size)


def prepare_dataset(name, data_root, raw_root, batch_size):
    cfg = DATASETS[name]
    out_dir = data_root / name
    audio_dir = out_dir / "audio"
    video_dir = out_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "test.jsonl"
    done_ids = _read_done_ids(manifest_path)
    mode = "a" if done_ids else "w"

    files = _list_parquet_files(cfg["repo"], cfg["prefix"])
    if not files:
        raise RuntimeError(f"No parquet files found for {name}: {cfg['repo']}/{cfg['prefix']}")

    print(f"[{name}] {len(files)} parquet files -> {out_dir}", flush=True)
    with manifest_path.open(mode, encoding="utf-8") as manifest:
        row_index = 0
        for filename in files:
            parquet_path = _download_file(cfg["repo"], filename, raw_root)
            print(f"[{name}] reading {filename}", flush=True)
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    record_id = _record_id(name, row, row_index)
                    row_index += 1
                    if record_id in done_ids:
                        continue
                    normalized = _normalize_record(name, row, record_id, audio_dir, video_dir)
                    manifest.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    manifest.flush()
                    done_ids.add(record_id)
    print(f"[{name}] wrote {len(done_ids)} records to {manifest_path}", flush=True)


def _list_parquet_files(repo_id, prefix):
    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    files = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet")
    ]
    return sorted(files)


def _download_file(repo_id, filename, raw_root):
    local_dir = raw_root / _safe_name(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
    )


def _read_done_ids(manifest_path):
    if not manifest_path.exists():
        return set()
    done = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(str(json.loads(line).get("id")))
    return done


def _normalize_record(dataset_name, row, record_id, audio_dir, video_dir):
    audio_path = _materialize_audio(row, record_id, audio_dir)
    video_path = _materialize_video(row, record_id, video_dir)
    question = _question_text(dataset_name, row)
    options = _options(row)
    if _is_classification_dataset(dataset_name) and not options:
        options = _classification_options(dataset_name)
    answer = _target_text(dataset_name, row, options)
    if _is_caption_dataset(dataset_name):
        question = "Describe this audio clip in one concise sentence."
    if _is_asr_dataset(dataset_name):
        question = "Transcribe the speech in the audio exactly."
    if _is_ast_dataset(dataset_name):
        question = _ast_question(dataset_name)
    if _is_classification_dataset(dataset_name):
        question = _classification_question(dataset_name)
    if _is_duplex_video_dataset(dataset_name):
        question = _first(row, "question", "Question", "prompt") or "Please describe what is happening in this video clip."

    record = {
        "id": record_id,
        "name": record_id,
        "question": "" if question is None else str(question).strip(),
        "prompt": "" if question is None else str(question).strip(),
        "options": options,
        "choices": options,
        "answer": answer,
        "reference": answer,
        "WavPath": _relative_to_parent(audio_path, audio_dir.parent) if audio_path else "",
        "audio_path": _relative_to_parent(audio_path, audio_dir.parent) if audio_path else "",
        "VideoPath": _relative_to_parent(video_path, video_dir.parent) if video_path else "",
        "video_path": _relative_to_parent(video_path, video_dir.parent) if video_path else "",
        "image_path": "",
    }

    for key, value in row.items():
        if key in record or _is_media_key(key):
            continue
        if _json_safe_scalar(value):
            record[key] = value
    return record


def _question_text(dataset_name, row):
    return _first(row, "question", "Question", "prompt", "instruction", "query", "question_text") or ""


def _target_text(dataset_name, row, options):
    if _is_caption_dataset(dataset_name) or _is_duplex_video_dataset(dataset_name):
        return _caption_answer(row)
    if _is_asr_dataset(dataset_name):
        return _asr_answer(row)
    if _is_ast_dataset(dataset_name):
        return _translation_answer(row)
    if _is_classification_dataset(dataset_name):
        return _classification_answer(row, options)
    return _answer(row, options)


def _materialize_audio(row, record_id, audio_dir):
    audio_obj = _first(row, "audio", "context", "question_audio", "audio_bytes")
    wav_path = _first(row, "WavPath", "wav_path", "audio_path", "save_name")

    audio_bytes = None
    source_path = None
    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes")
        source_path = audio_obj.get("path")
    elif isinstance(audio_obj, (bytes, bytearray)):
        audio_bytes = bytes(audio_obj)
    elif isinstance(audio_obj, str):
        source_path = audio_obj

    suffix = _audio_suffix(source_path or wav_path)
    basename = _safe_name(record_id)
    target_name = basename if basename.lower().endswith(suffix) else f"{basename}{suffix}"
    target = audio_dir / target_name
    if audio_bytes:
        if not target.exists():
            target.write_bytes(audio_bytes)
        return str(target.resolve())

    if wav_path:
        return str(Path(str(wav_path)).as_posix())
    return ""


def _materialize_video(row, record_id, video_dir):
    video_obj = _first(row, "video", "Video", "video_bytes")
    video_path = _first(row, "VideoPath", "video_path", "videoPath")
    video_bytes = None
    source_path = None
    if isinstance(video_obj, dict):
        video_bytes = video_obj.get("bytes")
        source_path = video_obj.get("path")
    elif isinstance(video_obj, (bytes, bytearray)):
        video_bytes = bytes(video_obj)
    elif isinstance(video_obj, str):
        source_path = video_obj

    suffix = Path(str(source_path or video_path or "")).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        suffix = ".mp4"
    basename = _safe_name(record_id)
    target_name = basename if basename.lower().endswith(suffix) else f"{basename}{suffix}"
    target = video_dir / target_name
    if video_bytes:
        if not target.exists():
            target.write_bytes(video_bytes)
        return str(target.resolve())
    if video_path:
        return str(Path(str(video_path)).as_posix())
    return ""


def _record_id(dataset_name, row, idx):
    value = _first(row, "id", "name", "question_id", "key", "conversation_hash", "save_name")
    event_id = _first(row, "event_id")
    if event_id not in (None, ""):
        video_id = _first(row, "video_id", "id", "name") or dataset_name
        return f"{_safe_name(video_id)}_{_safe_name(event_id)}_{idx:06d}"
    if value in (None, ""):
        value = f"{dataset_name}_{idx:06d}"
    return f"{_safe_name(value)}_{idx:06d}"


def _options(row):
    choices = _first(row, "choices", "options", "choice", "Choices")
    if choices is None and all(key in row for key in ("A", "B", "C", "D")):
        choices = [row.get("A"), row.get("B"), row.get("C"), row.get("D")]
    if choices is None and all(key in row for key in ("choice_a", "choice_b", "choice_c", "choice_d")):
        choices = [row.get("choice_a"), row.get("choice_b"), row.get("choice_c"), row.get("choice_d")]
    return _normalize_options(choices)


def _normalize_options(choices):
    if choices in (None, ""):
        return []
    if isinstance(choices, str):
        parsed = _parse_json_maybe(choices)
        if parsed is not choices:
            choices = parsed
        else:
            choices = re.split(r"\n|\|", choices)
    if isinstance(choices, dict):
        choices = [choices[key] for key in sorted(choices)]
    normalized = []
    for choice in choices:
        text = _strip_option_prefix(choice)
        if text:
            normalized.append(text)
    return normalized


def _answer(row, options):
    answer = _first(
        row,
        "answer",
        "answers",
        "Answer",
        "reference",
        "target",
        "label",
        "answer_gt",
        "gt_answer",
    )
    if isinstance(answer, str):
        parsed = _parse_json_maybe(answer)
        if isinstance(parsed, (list, tuple)):
            answer = parsed
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    answer = "" if answer is None else str(answer).strip()
    letter = _answer_letter(answer, options)
    return letter or answer


def _caption_answer(row):
    answer = _first(row, "caption", "caption_1", "answer", "description", "gt_answer", "event_title")
    if isinstance(answer, str):
        parsed = _parse_json_maybe(answer)
        if isinstance(parsed, list) and parsed:
            answer = parsed[0]
    return "" if answer is None else str(answer).strip()


def _is_caption_dataset(dataset_name):
    return dataset_name in {
        "audiocaps_test",
        "clothocaption_test",
        "wavcaps_audioset_sl",
        "wavcaps_freesound",
        "wavcaps_soundbible",
    }


def _asr_answer(row):
    answer = _first(
        row,
        "text",
        "sentence",
        "transcript",
        "transcription",
        "normalized_text",
        "label",
        "text_tn",
        "target",
        "answer",
    )
    return "" if answer is None else str(answer).strip()


def _translation_answer(row):
    answer = _first(row, "translation", "sentence_translation", "target", "answer", "text", "text_en", "text_zh", "tgt_text")
    return "" if answer is None else str(answer).strip()


def _classification_answer(row, options):
    answer = _first(row, "label", "emotion", "Emotion", "class", "category", "answer", "target", "text")
    answer = "" if answer is None else str(answer).strip()
    return _answer_letter(answer, options) or answer


def _ast_question(dataset_name):
    if dataset_name == "covost2_zh_en":
        return "Translate the spoken Chinese audio into English."
    if dataset_name == "covost2_en_zh":
        return "Translate the spoken English audio into Chinese."
    return "Translate the speech in the audio."


def _classification_question(dataset_name):
    if dataset_name == "vocalsound":
        return "Classify the vocal sound in the audio."
    if dataset_name == "meld":
        return "Classify the speaker emotion in the audio."
    return "Classify the audio."


def _classification_options(dataset_name):
    if dataset_name == "vocalsound":
        return ["cough", "sigh", "throatclearing", "sneeze", "laughter", "sniff"]
    if dataset_name == "meld":
        return ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    return []


def _is_asr_dataset(dataset_name):
    return dataset_name in {
        "gigaspeech_test",
        "wenetspeech_test_net",
        "wenetspeech_test_meeting",
        "librispeech_test_clean",
        "librispeech_test_other",
        "librispeech_dev_clean",
        "librispeech_dev_other",
        "commonvoice_en",
        "commonvoice_zh",
        "commonvoice_yue",
        "commonvoice_fr",
        "aishell1_test",
        "aishell2_test",
        "kespeech_test",
        "voxpopuli_en",
        "fleurs_zh",
        "fleurs_en",
        "peoples_speech_test",
        "tedlium3_test",
    }


def _is_ast_dataset(dataset_name):
    return dataset_name in {"covost2_zh_en", "covost2_en_zh"}


def _is_classification_dataset(dataset_name):
    return dataset_name in {"vocalsound", "meld"}


def _is_duplex_video_dataset(dataset_name):
    return dataset_name == "livesports3k_cc"


def _answer_letter(answer, options):
    match = re.match(r"^\s*([A-I])(?:[\.\)]|\s*$)", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    normalized_answer = _normalize_text(answer)
    for idx, option in enumerate(options):
        if _normalize_text(option) == normalized_answer:
            return "ABCDEFGHI"[idx]
    return ""


def _first(row, *keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_json_maybe(text):
    try:
        return json.loads(text)
    except Exception:
        return text


def _strip_option_prefix(value):
    text = "" if value is None else str(value).strip()
    return re.sub(r"^\s*\(?[A-I]\)?[\.\)]\s+(?=.)", "", text, count=1)


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _safe_name(value):
    text = str(value).strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180] or "sample"


def _audio_suffix(path):
    suffix = Path(str(path or "")).suffix.lower()
    return suffix if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"} else ".wav"


def _relative_to_parent(path, parent):
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(parent.resolve()))
    except Exception:
        return str(path)


def _is_media_key(key):
    lowered = str(key).lower()
    return lowered in {"audio", "context", "question_audio", "audio_bytes", "video", "video_bytes"} or lowered.endswith("_bytes")


def _json_safe_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


if __name__ == "__main__":
    main()
