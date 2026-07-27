#!/usr/bin/env python3
"""Post-process imaging benchmark predictions with an LLM-as-a-judge.

The script rewrites each predictions.jsonl in place, preserving all existing
fields and adding/overwriting only ``llm_judge_correct``. It also updates each
dataset summary.json with llm_judge_accuracy, llm_judge_total, and
llm_judge_correct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results_imaging"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_DATASETS = {
    "chartqa",
    "docvqa",
    "infographicvqa",
    "longvideobench_val",
    "lvbench",
    "mathverse_mini",
    "mathvista",
    "motionbench",
    "ocrbench",
    "textvqa",
    "videomme",
    "videomme_short",
}
CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


@dataclass
class Record:
    index: int
    sample: dict[str, Any]
    prompt: str


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9.%$:/+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(value: Any) -> list[float]:
    numbers = []
    for match in NUMBER_RE.findall(str(value or "")):
        raw = match.replace(",", "")
        is_percent = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            number = float(raw)
        except ValueError:
            continue
        if is_percent:
            numbers.append(number)
            numbers.append(number / 100.0)
        else:
            numbers.append(number)
    return numbers


def numbers_close(reference: Any, prediction: Any) -> bool:
    refs = extract_numbers(reference)
    preds = extract_numbers(prediction)
    if not refs or not preds:
        return False
    for ref in refs:
        for pred in preds:
            tol = max(0.05, abs(ref) * 0.03)
            if math.isclose(ref, pred, rel_tol=0.03, abs_tol=tol):
                return True
    return False


def extract_choice(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D", "E"}:
        return text.upper()
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"final\s*(?:answer|choice)?\s*(?:is|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"^\s*[\(\[]?\s*([A-E])\s*[\)\]]?[\s\.:,-]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else ""


def references(sample: dict[str, Any]) -> list[str]:
    refs = []
    answer = sample.get("answer")
    if isinstance(answer, list):
        refs.extend(str(item) for item in answer)
    elif answer is not None:
        refs.append(str(answer))
    extra = sample.get("references")
    if isinstance(extra, list):
        refs.extend(str(item) for item in extra)
    elif extra:
        refs.append(str(extra))
    seen = set()
    unique = []
    for ref in refs:
        key = normalize_text(ref)
        if key and key not in seen:
            unique.append(ref)
            seen.add(key)
    return unique


def option_map(sample: dict[str, Any]) -> dict[str, str]:
    options = sample.get("options") or []
    mapping: dict[str, str] = {}
    if isinstance(options, dict):
        iterable = options.items()
    else:
        iterable = enumerate(options)
    for key, value in iterable:
        if isinstance(key, int):
            letter = chr(ord("A") + key)
        else:
            letter = str(key).strip().upper()[:1]
        if letter in {"A", "B", "C", "D", "E"}:
            text = str(value)
            text = re.sub(r"^\s*[A-E]\s*[:.)-]\s*", "", text, flags=re.IGNORECASE)
            mapping[letter] = text.strip()
    return mapping


def cheap_semantic_judge(dataset: str, sample: dict[str, Any]) -> bool:
    """Deterministic fallback for smoke tests or unavailable LLM backends."""
    prediction = str(sample.get("prediction") or "")
    if not prediction.strip():
        return False

    refs = references(sample)
    pred_norm = normalize_text(prediction)
    ref_norms = [normalize_text(ref) for ref in refs]
    expected_choice = extract_choice(sample.get("answer"))
    predicted_choice = extract_choice(prediction)
    if expected_choice and predicted_choice:
        return expected_choice == predicted_choice

    opts = option_map(sample)
    if expected_choice and expected_choice in opts:
        expected_text = normalize_text(opts[expected_choice])
        if expected_text and expected_text in pred_norm:
            return True

    if dataset in {"chartqa", "mathverse_mini", "mathvista"} and refs:
        return any(numbers_close(ref, prediction) for ref in refs)

    for ref_norm in ref_norms:
        if not ref_norm:
            continue
        if pred_norm == ref_norm or ref_norm in pred_norm or pred_norm in ref_norm:
            return True
    return False


def build_prompt(model: str, dataset: str, sample: dict[str, Any]) -> str:
    rubric = (
        "You are an exact but fair LLM-as-a-judge for multimodal evaluation. "
        "Decide whether the prediction correctly answers the question compared "
        "with the reference answer. Judge semantically, not by exact wording. "
        "For chart/math datasets, verify numeric correctness, units, rounding, "
        "comparisons, and final reasoning. For OCR/TextVQA/DocVQA/InfographicVQA, "
        "allow minor OCR, punctuation, casing, and formatting differences, but "
        "penalize wrong or missing text/entities. For video/motion datasets, "
        "judge whether the predicted event/action/temporal answer matches the "
        "reference. If the reference is a multiple-choice letter, the prediction "
        "is correct only when it chooses that option or unambiguously states the "
        "same option text. Output exactly [[CORRECT]] or [[INCORRECT]]."
    )
    opts = sample.get("options")
    options_text = ""
    if opts:
        options_text = "\nOptions: " + json.dumps(opts, ensure_ascii=False)
    return (
        f"{rubric}\n\n"
        f"Model: {model}\n"
        f"Dataset: {dataset}\n"
        f"Question: {sample.get('question', '')}{options_text}\n"
        f"Reference answer: {sample.get('answer', '')}\n"
        f"Prediction: {sample.get('prediction', '')}\n\n"
        "Judgment:"
    )


class VLLMJudge:
    def __init__(self, model_name: str, temperature: float, max_model_len: int):
        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=8,
            stop=["\n"],
        )
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
        )

    def judge_batch(self, prompts: list[str]) -> list[bool]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [parse_judgment(out.outputs[0].text if out.outputs else "") for out in outputs]


class TransformersJudge:
    def __init__(self, model_name: str, temperature: float, max_model_len: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def judge_batch(self, prompts: list[str]) -> list[bool]:
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            texts.append(text)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
        ).to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        responses = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [parse_judgment(response) for response in responses]


class HeuristicJudge:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def judge_batch(self, records: list[Record]) -> list[bool]:
        return [cheap_semantic_judge(self.dataset, record.sample) for record in records]


def parse_judgment(text: str) -> bool:
    upper = text.upper()
    if "[[CORRECT]]" in upper:
        return True
    if "[[INCORRECT]]" in upper:
        return False
    return bool(re.search(r"\bCORRECT\b", upper)) and not bool(re.search(r"\bINCORRECT\b", upper))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def update_summary(path: Path, total: int, correct: int) -> None:
    summary_path = path.parent / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary["llm_judge_accuracy"] = correct / total if total else 0.0
    summary["llm_judge_total"] = total
    summary["llm_judge_correct"] = correct
    fd, tmp_name = tempfile.mkstemp(prefix="summary.json.", suffix=".tmp", dir=str(summary_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, summary_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def discover_prediction_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("predictions.jsonl")
        if path.parent.name in TARGET_DATASETS
    )


def process_file(path: Path, root: Path, judge: Any, backend: str, batch_size: int) -> tuple[str, str, int, int, float]:
    dataset = path.parent.name
    model = path.parent.parent.name
    rows = read_jsonl(path)
    total = len(rows)
    correct = 0
    records = [
        Record(index=i, sample=row, prompt=build_prompt(model, dataset, row))
        for i, row in enumerate(rows)
    ]
    judgments: list[bool] = []
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        if backend == "heuristic":
            judged = judge.judge_batch(batch)
        else:
            judged = judge.judge_batch([record.prompt for record in batch])
        judgments.extend(judged)
        print(f"{model}/{dataset}: judged {min(start + batch_size, total)}/{total}", flush=True)

    for row, is_correct in zip(rows, judgments):
        row["llm_judge_correct"] = bool(is_correct)
        correct += int(bool(is_correct))
    write_jsonl_atomic(path, rows)
    update_summary(path, total, correct)
    accuracy = correct / total if total else 0.0
    rel = path.relative_to(root)
    print(f"updated {rel}: {correct}/{total} = {accuracy:.4f}", flush=True)
    return model, dataset, total, correct, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm", "transformers", "heuristic"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit-files", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    files = discover_prediction_files(args.root)
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No predictions.jsonl files found under {args.root}")

    if args.backend == "vllm":
        judge = VLLMJudge(args.model, args.temperature, args.max_model_len)
    elif args.backend == "transformers":
        judge = TransformersJudge(args.model, args.temperature, args.max_model_len)
    else:
        judge = None

    results = []
    for path in files:
        dataset = path.parent.name
        file_judge = HeuristicJudge(dataset) if args.backend == "heuristic" else judge
        results.append(process_file(path, args.root, file_judge, args.backend, args.batch_size))

    print("\n| Model | Dataset | Total | Correct | LLM Judge Accuracy |")
    print("|---|---|---:|---:|---:|")
    for model, dataset, total, correct, accuracy in results:
        print(f"| {model} | {dataset} | {total} | {correct} | {accuracy:.4f} |")


if __name__ == "__main__":
    main()
