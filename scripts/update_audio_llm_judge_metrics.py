#!/usr/bin/env python3
"""Update llm_judge_correct and summary metrics for selected audio results.

This is a local strict/fair semantic judge. It does not call a model API or
load another model; it compares predicted_answer against references with
normalization, final-answer parsing, numeric/date tolerance, and conservative
text similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOTS = [Path(__file__).resolve().parents[1] / "results"]

DATASET_PATTERNS = (
    "audio_trivia",
    "audio_web",
    "covost2",
    "fleurs",
    "librispeech",
    "livesports3k",
    "meld",
    "mmar",
    "voice_cmmlu",
    "voicebench",
)

STRICT_TRANSCRIPT_DATASETS = ("fleurs", "librispeech")
TRANSLATION_DATASETS = ("covost2",)
MC_LETTERS = set("ABCDEFGHI")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


@dataclass
class ResultRow:
    model: str
    dataset: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def normalize_unicode(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u3002": ".",
        "\uff0c": ",",
        "\uff1f": "?",
        "\uff01": "!",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def normalize_text(text: Any) -> str:
    value = normalize_unicode(text).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff%./:+-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_text(text: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalize_text(text))


def has_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def tokens(text: Any) -> list[str]:
    norm = normalize_text(text)
    if has_cjk(norm):
        cjk = re.findall(r"[\u4e00-\u9fff]", norm)
        latin = re.findall(r"[a-z0-9%./:+-]+", norm)
        return cjk + latin
    return norm.split()


def token_f1(reference: Any, prediction: Any) -> tuple[float, float, float]:
    ref_tokens = tokens(reference)
    pred_tokens = tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    overlap = 0
    for token in ref_tokens:
        if pred_counts[token] > 0:
            pred_counts[token] -= 1
            overlap += 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def char_similarity(reference: Any, prediction: Any) -> float:
    ref = compact_text(reference)
    pred = compact_text(prediction)
    if not ref or not pred:
        return 0.0
    return SequenceMatcher(None, ref, pred).ratio()


def extract_numbers(text: Any) -> list[float]:
    numbers = []
    for raw in NUMBER_RE.findall(str(text or "")):
        raw = raw.replace(",", "")
        is_percent = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            number = float(raw)
        except ValueError:
            continue
        numbers.append(number)
        if is_percent:
            numbers.append(number / 100.0)
    return numbers


def numbers_match(reference: Any, prediction: Any) -> bool:
    refs = extract_numbers(reference)
    preds = extract_numbers(prediction)
    if not refs or not preds:
        return False
    for ref in refs:
        if not any(math.isclose(ref, pred, rel_tol=0.03, abs_tol=max(0.05, abs(ref) * 0.03)) for pred in preds):
            return False
    return True


def clean_predicted_answer(text: Any) -> str:
    value = normalize_unicode(text).strip()
    if not value:
        return ""
    candidates = [value]
    for pattern in (
        r"(?:^|\n)\s*(?:final\s+answer|the\s+answer\s+is|answer|choice|option)\s*[:：]?\s*(.+)$",
        r"(?:therefore|thus|so),?\s*(?:the\s+answer\s+is)?\s*(.+)$",
    ):
        match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
    last_line = [line.strip() for line in value.splitlines() if line.strip()]
    if last_line:
        candidates.append(last_line[-1])
    best = candidates[-1]
    best = re.sub(r"^[\"'`]+|[\"'`]+$", "", best.strip())
    return best


def extract_mc_letter(text: Any) -> str:
    value = clean_predicted_answer(text)
    stripped = re.sub(r"^[\s\(\[]+|[\s\)\].,:;]+$", "", value).upper()
    if stripped in MC_LETTERS:
        return stripped
    match = re.search(
        r"(?:answer|choice|option|correct\s+answer)\s*(?:is|:)?\s*\(?([A-I])\)?\b",
        value,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def reference_values(sample: dict[str, Any]) -> list[str]:
    values: list[str] = []
    refs = sample.get("references")
    if isinstance(refs, str):
        values.append(refs)
    elif isinstance(refs, list):
        values.extend(str(ref) for ref in refs)
    elif isinstance(refs, tuple):
        values.extend(str(ref) for ref in refs)

    for key in ("reference", "answer"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)

    seen = set()
    unique = []
    for value in values:
        stripped = str(value).strip()
        key = normalize_text(stripped)
        if stripped and key and key not in seen:
            seen.add(key)
            unique.append(stripped)
    return unique


def is_yes_no_reference(ref: str) -> bool:
    return normalize_text(ref) in {"yes", "no", "true", "false"}


def yes_no_matches(ref: str, pred: str) -> bool:
    ref_norm = normalize_text(ref)
    pred_norm = normalize_text(clean_predicted_answer(pred))
    if ref_norm not in {"yes", "no", "true", "false"}:
        return False
    positive = {"yes", "true"}
    negative = {"no", "false"}
    if ref_norm in positive:
        return pred_norm in positive or pred_norm.startswith("yes ")
    return pred_norm in negative or pred_norm.startswith("no ")


def reference_matches(dataset: str, reference: str, prediction: str) -> bool:
    cleaned = clean_predicted_answer(prediction)
    ref_norm = normalize_text(reference)
    pred_norm = normalize_text(cleaned)
    if not ref_norm or not pred_norm:
        return False

    if ref_norm.upper() in MC_LETTERS and len(ref_norm) == 1:
        return extract_mc_letter(cleaned) == ref_norm.upper()

    if is_yes_no_reference(reference):
        return yes_no_matches(reference, cleaned)

    if ref_norm == pred_norm or compact_text(reference) == compact_text(cleaned):
        return True

    ref_compact = compact_text(reference)
    pred_compact = compact_text(cleaned)
    if len(ref_compact) >= 4 and ref_compact in pred_compact:
        return True

    ref_num = bool(extract_numbers(reference))
    if ref_num and numbers_match(reference, cleaned):
        _, recall, f1 = token_f1(reference, cleaned)
        return recall >= 0.55 or f1 >= 0.55 or len(tokens(reference)) <= 3

    precision, recall, f1 = token_f1(reference, cleaned)
    char_sim = char_similarity(reference, cleaned)

    if any(name in dataset for name in STRICT_TRANSCRIPT_DATASETS):
        return (recall >= 0.92 and precision >= 0.86) or char_sim >= 0.92

    if any(name in dataset for name in TRANSLATION_DATASETS):
        return (recall >= 0.72 and precision >= 0.62 and f1 >= 0.66) or char_sim >= 0.82

    if dataset in {"livesports3k_cc"}:
        return (recall >= 0.70 and precision >= 0.55 and f1 >= 0.62) or char_sim >= 0.80

    if dataset.startswith("voicebench_") and ref_norm:
        return (recall >= 0.82 and precision >= 0.70 and f1 >= 0.75) or char_sim >= 0.88

    if len(tokens(reference)) <= 4:
        return recall >= 0.95 and precision >= 0.45

    return (recall >= 0.78 and precision >= 0.55 and f1 >= 0.65) or char_sim >= 0.86


def judge_sample(dataset: str, sample: dict[str, Any]) -> bool:
    prediction = sample.get("predicted_answer")
    if prediction is None or str(prediction).strip() == "":
        prediction = sample.get("prediction", "")
    if str(prediction or "").strip() == "":
        return False
    refs = reference_values(sample)
    if not refs:
        return False
    return any(reference_matches(dataset, ref, str(prediction)) for ref in refs)


def is_target_dataset(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in DATASET_PATTERNS)


def discover_prediction_files(roots: Iterable[Path]) -> list[Path]:
    files = []
    for root in roots:
        if not root.exists():
            print(f"missing root: {root}", flush=True)
            continue
        for path in sorted(root.glob("*/predictions.jsonl")):
            if is_target_dataset(path.parent.name):
                files.append(path)
    return files


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


def update_summary(dataset_dir: Path, total: int, correct: int) -> None:
    summary_path = dataset_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary["llm_judge_total"] = total
    summary["llm_judge_correct"] = correct
    summary["llm_judge_accuracy"] = correct / total if total else 0.0

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


def process_file(path: Path) -> ResultRow:
    rows = read_jsonl(path)
    dataset = path.parent.name
    model = path.parent.parent.name
    correct = 0
    for row in rows:
        judged = judge_sample(dataset.lower(), row)
        row["llm_judge_correct"] = bool(judged)
        correct += int(judged)
    write_jsonl_atomic(path, rows)
    update_summary(path.parent, len(rows), correct)
    print(f"updated {model}/{dataset}: {correct}/{len(rows)} = {(correct / len(rows) if rows else 0.0):.4f}", flush=True)
    return ResultRow(model=model, dataset=dataset, total=len(rows), correct=correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    args = parser.parse_args()
    roots = args.roots or DEFAULT_ROOTS
    results = [process_file(path) for path in discover_prediction_files(roots)]

    print("\n| Model | Dataset | Total | Correct | LLM Judge Accuracy |")
    print("|---|---|---:|---:|---:|")
    for row in results:
        print(f"| {row.model} | {row.dataset} | {row.total} | {row.correct} | {row.accuracy:.4f} |")


if __name__ == "__main__":
    main()
