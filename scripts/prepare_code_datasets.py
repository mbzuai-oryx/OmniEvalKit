#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


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


def read_json_or_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parquet_rows(path):
    table = pq.read_table(path)
    return table.to_pylist()


def download(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


def base_record(record_id, question, answer, source, config, language):
    return {
        "id": str(record_id),
        "question": str(question or "").strip(),
        "prompt": str(question or "").strip(),
        "answer": str(answer or "").strip(),
        "reference": str(answer or "").strip(),
        "source_dataset": source,
        "config": config,
        "split": "test",
        "language": language,
        "audio_path": "",
        "video_path": "",
        "image_path": "",
        "options": [],
    }


def normalize_mbpp(record, config_name):
    task_id = record.get("task_id")
    tests = list(record.get("test_list") or [])
    challenge_tests = list(record.get("challenge_test_list") or [])
    setup_code = record.get("test_setup_code") or "\n".join(record.get("test_imports") or [])
    prompt_parts = [str(record.get("text") or record.get("prompt") or "").strip()]
    if setup_code:
        prompt_parts.append("Required imports/setup:\n" + setup_code.strip())
    if tests:
        prompt_parts.append("Your solution must pass these tests:\n" + "\n".join(tests))
    prompt = "\n\n".join(part for part in prompt_parts if part)
    row = base_record(
        record_id=f"mbpp_{config_name}_{task_id}",
        question=prompt,
        answer=record.get("code") or "",
        source="Muennighoff/mbpp",
        config=config_name,
        language="python",
    )
    row.update(
        {
            "task_id": task_id,
            "canonical_solution": row["answer"],
            "test": "\n".join(tests),
            "test_list": tests,
            "test_setup_code": setup_code,
            "challenge_test_list": challenge_tests,
            "original": record,
        }
    )
    return row


def normalize_mbppplus(record):
    task_id = record.get("task_id")
    public_tests = list(record.get("test_list") or [])
    setup_code = "\n".join(record.get("test_imports") or [])
    prompt_parts = [str(record.get("prompt") or "").strip()]
    if setup_code:
        prompt_parts.append("Required imports/setup:\n" + setup_code.strip())
    if public_tests:
        prompt_parts.append("Your solution must pass these public tests:\n" + "\n".join(public_tests))
    prompt = "\n\n".join(part for part in prompt_parts if part)
    row = base_record(
        record_id=f"mbppplus_{task_id}",
        question=prompt,
        answer=record.get("code") or "",
        source="evalplus/mbppplus",
        config="default",
        language="python",
    )
    row.update(
        {
            "task_id": task_id,
            "canonical_solution": row["answer"],
            "test": "\n".join(public_tests),
            "test_list": public_tests,
            "test_setup_code": setup_code,
            "plus_test": record.get("test") or "",
            "source_file": record.get("source_file") or "",
            "original": record,
        }
    )
    return row


def normalize_humaneval(record):
    task_id = str(record.get("task_id") or "")
    row = base_record(
        record_id=task_id,
        question=record.get("prompt") or "",
        answer=record.get("canonical_solution") or "",
        source="openai/openai_humaneval",
        config="openai_humaneval",
        language="python",
    )
    row.update(
        {
            "task_id": task_id,
            "canonical_solution": row["answer"],
            "test": record.get("test") or "",
            "entry_point": record.get("entry_point") or "",
            "original": record,
        }
    )
    return row


def normalize_multiple(record, config_name):
    task_id = str(record.get("name") or record.get("task_id") or record.get("id") or "")
    language = config_name.rsplit("-", 1)[-1] if "-" in config_name else ""
    prompt = record.get("prompt") or record.get("translation_prompt") or ""
    answer = (
        record.get("canonical_solution")
        or record.get("completion")
        or record.get("code")
        or record.get("answer")
        or ""
    )
    row = base_record(
        record_id=f"{config_name}_{task_id}".strip("_"),
        question=prompt,
        answer=answer,
        source="nuprl/MultiPL-E",
        config=config_name,
        language=language,
    )
    tests = record.get("tests") or record.get("test") or record.get("unit_tests") or ""
    row.update(
        {
            "task_id": task_id,
            "canonical_solution": row["answer"],
            "test": "\n".join(str(item) for item in tests) if isinstance(tests, list) else str(tests or ""),
            "entry_point": record.get("entry_point") or "",
            "original": record,
        }
    )
    return row


def prepare_mbpp():
    files = {
        "full": "data/mbpp.jsonl",
        "sanitized": "data/sanitized-mbpp.json",
    }
    summary = {}
    for config_name, filename in files.items():
        local_path = download("Muennighoff/mbpp", filename)
        records = read_json_or_jsonl(local_path)
        out_name = "mbpp" if config_name == "full" else "mbpp_sanitized"
        out_dir = DATA_ROOT / out_name
        count = write_jsonl(out_dir / "test.jsonl", (normalize_mbpp(row, config_name) for row in records))
        dump_json(
            out_dir / "dataset_info.json",
            {
                "source": "Muennighoff/mbpp",
                "source_file": filename,
                "config": config_name,
                "split": "test",
                "rows": count,
            },
        )
        summary[out_name] = count
    return summary


def prepare_humaneval():
    filename = "openai_humaneval/test-00000-of-00001.parquet"
    local_path = download("openai/openai_humaneval", filename)
    records = parquet_rows(local_path)
    out_dir = DATA_ROOT / "openai_humaneval"
    count = write_jsonl(out_dir / "test.jsonl", (normalize_humaneval(row) for row in records))
    dump_json(
        out_dir / "dataset_info.json",
        {
            "source": "openai/openai_humaneval",
            "source_file": filename,
            "config": "openai_humaneval",
            "split": "test",
            "rows": count,
        },
    )
    return {"openai_humaneval": count}


def prepare_mbppplus():
    files = HfApi().list_repo_files("evalplus/mbppplus", repo_type="dataset")
    parquet_files = sorted(path for path in files if path.endswith(".parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(f"Expected exactly one mbppplus parquet file, found {parquet_files}")
    filename = parquet_files[0]
    local_path = download("evalplus/mbppplus", filename)
    records = parquet_rows(local_path)
    out_dir = DATA_ROOT / "mbppplus"
    count = write_jsonl(out_dir / "test.jsonl", (normalize_mbppplus(row) for row in records))
    dump_json(
        out_dir / "dataset_info.json",
        {
            "source": "evalplus/mbppplus",
            "source_file": filename,
            "config": "default",
            "split": "test",
            "rows": count,
            "note": "test_list contains public tests; plus_test contains the EvalPlus test suite.",
        },
    )
    return {"mbppplus": count}


def prepare_multiple():
    files = HfApi().list_repo_files("nuprl/MultiPL-E", repo_type="dataset")
    parquet_files = sorted(path for path in files if path.endswith("/test-00000-of-00001.parquet"))
    summary = {}
    base_dir = DATA_ROOT / "multiple"
    for filename in parquet_files:
        config_name = filename.split("/", 1)[0]
        local_path = download("nuprl/MultiPL-E", filename)
        records = parquet_rows(local_path)
        out_dir = base_dir / config_name
        count = write_jsonl(out_dir / "test.jsonl", (normalize_multiple(row, config_name) for row in records))
        dump_json(
            out_dir / "dataset_info.json",
            {
                "source": "nuprl/MultiPL-E",
                "source_file": filename,
                "config": config_name,
                "split": "test",
                "rows": count,
            },
        )
        summary[config_name] = count
    dump_json(
        base_dir / "dataset_info.json",
        {
            "source": "nuprl/MultiPL-E",
            "n_configs": len(summary),
            "rows_by_config": summary,
            "total_rows": sum(summary.values()),
        },
    )
    return {f"multiple/{key}": value for key, value in summary.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and normalize code-generation datasets.")
    parser.add_argument(
        "--datasets",
        default="mbpp,mbppplus,humaneval,multiple",
        help="Comma-separated subset of: mbpp, mbppplus, humaneval, multiple",
    )
    return parser.parse_args()


def main():
    requested = {item.strip().lower() for item in parse_args().datasets.split(",") if item.strip()}
    summary = {}
    if "mbpp" in requested:
        summary.update(prepare_mbpp())
    if "mbppplus" in requested or "mbpp_plus" in requested:
        summary.update(prepare_mbppplus())
    if "humaneval" in requested or "openai_humaneval" in requested:
        summary.update(prepare_humaneval())
    if "multiple" in requested or "multipl-e" in requested:
        summary.update(prepare_multiple())
    dump_json(DATA_ROOT / "code_datasets_info.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
