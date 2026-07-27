#!/usr/bin/env python3
"""Agent-rubric judge for selected imaging/video result JSONL files.

This script does not call an external LLM or local model. It applies the
judging rubric directly to question/answer/prediction fields and rewrites each
predictions.jsonl in place, adding or overwriting only llm_judge_correct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results_imaging"
TARGET_DATASETS = {
    "longvideobench_val",
    "lvbench",
    "mathverse_mini",
    "motionbench",
    "ocrbench",
    "videomme",
    "videomme_short",
}
LETTERS = {"A", "B", "C", "D", "E"}
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\\n", " ")
    )
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9.%:/+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_alnum(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def token_f1(a: str, b: str) -> float:
    ta = normalize_text(a).split()
    tb = normalize_text(b).split()
    if not ta or not tb:
        return 0.0
    used = [False] * len(tb)
    match = 0
    for token in ta:
        for i, other in enumerate(tb):
            if not used[i] and token == other:
                used[i] = True
                match += 1
                break
    precision = match / len(tb)
    recall = match / len(ta)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    for raw in NUMBER_RE.findall(str(value or "")):
        raw = raw.replace(",", "")
        percent = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            number = float(raw)
        except ValueError:
            continue
        numbers.append(number)
        if percent:
            numbers.append(number / 100.0)
    return numbers


def numbers_close(reference: Any, prediction: Any) -> bool:
    refs = extract_numbers(reference)
    preds = extract_numbers(prediction)
    if not refs or not preds:
        return False
    for ref in refs:
        for pred in preds:
            tolerance = max(0.05, abs(ref) * 0.03)
            if math.isclose(ref, pred, rel_tol=0.03, abs_tol=tolerance):
                return True
    return False


def final_segment(prediction: str) -> str:
    text = str(prediction or "").strip()
    patterns = [
        r"(?:final\s+answer|therefore|thus|so the answer|answer)\s*(?:is|:)?\s*(.+)$",
        r"(?:the correct option|correct answer|choice)\s*(?:is|:)?\s*(.+)$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text[-300:]


def extract_explicit_letter(prediction: Any) -> str:
    text = str(prediction or "").strip()
    if not text:
        return ""
    simple = re.sub(r"^[\s\(\[]+|[\s\)\].,:;]+$", "", text).upper()
    if simple in LETTERS:
        return simple
    segment = final_segment(text)
    patterns = [
        r"(?:final\s+answer|answer|option|choice|correct\s+option|correct\s+answer)\s*(?:is|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"^\s*[\(\[]?\s*([A-E])\s*[\)\]]?\s*(?:[.)\]:-]|$)",
        r"\b([A-E])\b\s*(?:is\s+the\s+correct|is\s+correct)",
    ]
    for source in (segment, text[:120]):
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return ""


def parse_options_from_question(question: Any) -> dict[str, str]:
    text = str(question or "")
    options: dict[str, str] = {}
    pattern = re.compile(
        r"(?:^|\n)\s*([A-E])\s*[:.)]\s*(.*?)(?=(?:\n\s*[A-E]\s*[:.)]\s*)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        letter = match.group(1).upper()
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        if value:
            options[letter] = value
    return options


def option_text_matches(expected_text: str, prediction: str) -> bool:
    expected_norm = normalize_text(expected_text)
    pred_norm = normalize_text(prediction)
    if not expected_norm or not pred_norm:
        return False
    if expected_norm in pred_norm:
        return True
    if numbers_close(expected_text, final_segment(prediction)):
        return True
    if len(expected_norm) >= 12 and token_f1(expected_text, prediction) >= 0.78:
        return True
    return False


def ocr_text_correct(answer: Any, prediction: Any) -> bool:
    ans = str(answer or "").strip()
    pred = str(prediction or "").strip()
    if not ans or not pred:
        return False
    ans_norm = normalize_text(ans)
    pred_norm = normalize_text(pred)
    if ans_norm == pred_norm:
        return True
    if ans_norm and re.search(rf"(?<![a-z0-9]){re.escape(ans_norm)}(?![a-z0-9])", pred_norm):
        return True
    ans_compact = compact_alnum(ans)
    pred_compact = compact_alnum(pred)
    if not ans_compact or not pred_compact:
        return False
    if ans_compact == pred_compact:
        return True
    if len(ans_compact) <= 5:
        return levenshtein(ans_compact, pred_compact) <= 1
    if ans_compact in pred_compact:
        return True
    distance = levenshtein(ans_compact, pred_compact)
    similarity = 1.0 - distance / max(len(ans_compact), len(pred_compact))
    return similarity >= 0.88 or token_f1(ans, pred) >= 0.85


def judge_sample(dataset: str, sample: dict[str, Any]) -> bool:
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    prediction = sample.get("prediction", "")
    if not str(prediction or "").strip():
        return False

    expected_letter = str(answer or "").strip().upper()
    if expected_letter in LETTERS:
        predicted_letter = extract_explicit_letter(prediction)
        if predicted_letter:
            return predicted_letter == expected_letter
        options = parse_options_from_question(question)
        expected_text = options.get(expected_letter, "")
        if expected_text:
            return option_text_matches(expected_text, str(prediction))
        return False

    if dataset == "ocrbench":
        return ocr_text_correct(answer, prediction)

    if numbers_close(answer, final_segment(str(prediction))):
        return True

    answer_norm = normalize_text(answer)
    prediction_norm = normalize_text(prediction)
    if answer_norm and (answer_norm == prediction_norm or answer_norm in prediction_norm):
        return True
    return len(answer_norm) >= 12 and token_f1(str(answer), str(prediction)) >= 0.82


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def update_summary(dataset_dir: Path, total: int, correct: int) -> None:
    summary_path = dataset_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary["llm_judge_accuracy"] = correct / total if total else 0.0
    summary["llm_judge_total"] = total
    summary["llm_judge_correct"] = correct
    fd, tmp_name = tempfile.mkstemp(prefix="summary.json.", suffix=".tmp", dir=str(dataset_dir))
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


def discover(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("predictions.jsonl")
        if path.parent.name in TARGET_DATASETS
    )


def process_file(root: Path, path: Path) -> tuple[str, str, int, int, float]:
    model = path.parent.parent.name
    dataset = path.parent.name
    rows = read_jsonl(path)
    correct = 0
    for row in rows:
        is_correct = judge_sample(dataset, row)
        row["llm_judge_correct"] = bool(is_correct)
        correct += int(is_correct)
    write_jsonl_atomic(path, rows)
    update_summary(path.parent, len(rows), correct)
    accuracy = correct / len(rows) if rows else 0.0
    print(f"updated {path.relative_to(root)}: {correct}/{len(rows)} = {accuracy:.4f}", flush=True)
    return model, dataset, len(rows), correct, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    results = [process_file(args.root, path) for path in discover(args.root)]

    print("\n| Model | Dataset | Total | Correct | LLM Judge Accuracy |")
    print("|---|---|---:|---:|---:|")
    for model, dataset, total, correct, accuracy in results:
        print(f"| {model} | {dataset} | {total} | {correct} | {accuracy:.4f} |")


if __name__ == "__main__":
    main()
