#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")


def download(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


def parquet_rows(path):
    return pq.read_table(path).to_pylist()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def extract_final_answer(answer):
    matches = FINAL_ANSWER_RE.findall(str(answer or "").strip())
    if matches:
        return normalize_final_answer(matches[-1])
    return normalize_final_answer(answer)


def normalize_final_answer(text):
    text = str(text or "").strip()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_symbolic_answer(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_gsm8k(record, idx, config_name, split):
    solution = str(record.get("answer") or "").strip()
    final_answer = extract_final_answer(solution)
    return {
        "id": f"gsm8k_{config_name}_{split}_{idx:05d}",
        "question": str(record.get("question") or "").strip(),
        "prompt": str(record.get("question") or "").strip(),
        "answer": final_answer,
        "reference": final_answer,
        "references": [final_answer],
        "solution": solution,
        "source_dataset": "openai/gsm8k",
        "config": config_name,
        "split": split,
        "audio_path": "",
        "video_path": "",
        "image_path": "",
        "options": [],
        "original": record,
    }


def prepare_config(config_name, split="test"):
    filename = f"{config_name}/{split}-00000-of-00001.parquet"
    records = parquet_rows(download("openai/gsm8k", filename))
    out_name = "gsm8k" if config_name == "main" else f"gsm8k_{config_name}"
    out_dir = DATA_ROOT / out_name
    count = write_jsonl(
        out_dir / f"{split}.jsonl",
        (normalize_gsm8k(record, idx, config_name, split) for idx, record in enumerate(records)),
    )
    dump_json(
        out_dir / "dataset_info.json",
        {
            "source": "openai/gsm8k",
            "source_file": filename,
            "config": config_name,
            "split": split,
            "rows": count,
            "answer_field": "final answer extracted from text after '####'",
        },
    )
    return out_name, count


def normalize_math500(record, idx):
    answer = normalize_symbolic_answer(record.get("answer") or "")
    return {
        "id": str(record.get("unique_id") or f"math500_{idx:05d}"),
        "question": str(record.get("problem") or "").strip(),
        "prompt": str(record.get("problem") or "").strip(),
        "answer": answer,
        "reference": answer,
        "references": [answer],
        "solution": str(record.get("solution") or "").strip(),
        "subject": str(record.get("subject") or "").strip(),
        "level": record.get("level"),
        "source_dataset": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "audio_path": "",
        "video_path": "",
        "image_path": "",
        "options": [],
        "original": record,
    }


def prepare_math500():
    filename = "test.jsonl"
    local_path = download("HuggingFaceH4/MATH-500", filename)
    with Path(local_path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    out_dir = DATA_ROOT / "math500"
    count = write_jsonl(out_dir / "test.jsonl", (normalize_math500(record, idx) for idx, record in enumerate(records)))
    dump_json(
        out_dir / "dataset_info.json",
        {
            "source": "HuggingFaceH4/MATH-500",
            "source_file": filename,
            "config": "default",
            "split": "test",
            "rows": count,
            "answer_field": "answer",
        },
    )
    return "math500", count


def _theoremqa_image_path(record, idx, out_dir):
    picture = record.get("Picture")
    if not picture:
        return ""
    image_bytes = picture.get("bytes") if isinstance(picture, dict) else None
    if not image_bytes:
        return ""
    source_path = str(picture.get("path") or f"image_{idx:05d}.png")
    suffix = Path(source_path).suffix or ".png"
    stem = Path(source_path).stem or f"image_{idx:05d}"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{idx:05d}_{stem}{suffix}"
    if not image_path.exists():
        image_path.write_bytes(image_bytes)
    return str(image_path.relative_to(out_dir))


def normalize_theoremqa(record, idx, out_dir):
    answer = normalize_symbolic_answer(record.get("Answer") or "")
    image_path = _theoremqa_image_path(record, idx, out_dir)
    question = str(record.get("Question") or "").strip()
    return {
        "id": f"theoremqa_{idx:05d}",
        "question": question,
        "prompt": question,
        "answer": answer,
        "reference": answer,
        "references": [answer],
        "answer_type": str(record.get("Answer_type") or "").strip(),
        "source_dataset": "TIGER-Lab/TheoremQA",
        "config": "default",
        "split": "test",
        "audio_path": "",
        "video_path": "",
        "image_path": image_path,
        "options": [],
        "original": {
            "Question": record.get("Question"),
            "Answer": record.get("Answer"),
            "Answer_type": record.get("Answer_type"),
            "Picture": {
                "path": record.get("Picture", {}).get("path") if record.get("Picture") else None,
                "has_bytes": bool(record.get("Picture", {}).get("bytes")) if record.get("Picture") else False,
            },
        },
    }


def prepare_theoremqa():
    files = HfApi().list_repo_files("TIGER-Lab/TheoremQA", repo_type="dataset")
    parquet_files = sorted(path for path in files if path.endswith(".parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(f"Expected exactly one TheoremQA parquet file, found {parquet_files}")
    filename = parquet_files[0]
    records = parquet_rows(download("TIGER-Lab/TheoremQA", filename))
    out_dir = DATA_ROOT / "theoremqa"
    count = write_jsonl(
        out_dir / "test.jsonl",
        (normalize_theoremqa(record, idx, out_dir) for idx, record in enumerate(records)),
    )
    n_images = sum(1 for record in records if record.get("Picture") and record["Picture"].get("bytes"))
    dump_json(
        out_dir / "dataset_info.json",
        {
            "source": "TIGER-Lab/TheoremQA",
            "source_file": filename,
            "config": "default",
            "split": "test",
            "rows": count,
            "images": n_images,
            "answer_field": "Answer",
        },
    )
    return "theoremqa", count


def parse_args():
    parser = argparse.ArgumentParser(description="Download and normalize GSM8K test splits for OmniEvalKit.")
    parser.add_argument("--configs", default="main,socratic", help="Comma-separated subset of: main, socratic")
    parser.add_argument(
        "--datasets",
        default="gsm8k",
        help="Comma-separated subset of: gsm8k, math500, theoremqa",
    )
    parser.add_argument("--split", default="test", choices=["test", "train"])
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    requested = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    if "gsm8k" in requested:
        for config_name in [item.strip() for item in args.configs.split(",") if item.strip()]:
            out_name, count = prepare_config(config_name, split=args.split)
            summary[out_name] = count
    if "math500" in requested or "math-500" in requested:
        out_name, count = prepare_math500()
        summary[out_name] = count
    if "theoremqa" in requested:
        out_name, count = prepare_theoremqa()
        summary[out_name] = count
    dump_json(DATA_ROOT / "math_datasets_info.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
