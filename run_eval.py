import argparse
import csv
import contextlib
import importlib
import inspect
import io
import json
import os
import time
from pathlib import Path

from utils.evaluate import aggregate_scores, compute_score
from utils.llm_judge import LLMJudge
from utils.logger import EvalLogger


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_dataset_module(dataset_name):
    return importlib.import_module(f"datasets.{dataset_name}.dataloader")


def load_prompt_module(dataset_name):
    return importlib.import_module(f"datasets.{dataset_name}.prompt")


def load_model(model_name, model_path=None):
    module = importlib.import_module(f"models.{model_name}.model")
    model_cls = getattr(module, "Model")
    return model_cls(model_path=model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--llm_judge", default="False")
    parser.add_argument("--llm_judge_model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=str_to_bool(os.getenv("RESUME", "False")))
    return parser.parse_args()


def read_existing_results(output_dir, dataset):
    path = Path(output_dir) / dataset / "predictions.jsonl"
    if not path.exists():
        return [], set()

    results = []
    processed_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            results.append(result)
            sample_id = result.get("id")
            if sample_id is not None:
                processed_ids.add(str(sample_id))
    return results, processed_ids


class AppendEvalLogger:
    def __init__(self, output_dir, dataset, model):
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.model = model
        self.dataset_dir = self.output_dir / dataset
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.dataset_dir / "predictions.jsonl"
        self.summary_path = self.dataset_dir / "summary.json"
        self.all_results_path = self.output_dir / "all_results.csv"
        self.samples_file = self.samples_path.open("a", encoding="utf-8")

    def write_sample(self, result):
        self.samples_file.write(json.dumps(result, ensure_ascii=False) + "\n")


        self.samples_file.flush()

    def write_summary(self, summary):
        self.samples_file.close()
        payload = {"dataset": self.dataset, "model": self.model, **summary}
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update_all_results(self, summary):
        rows = []
        if self.all_results_path.exists():
            with self.all_results_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        row = {
            "dataset": self.dataset,
            "model": self.model,
            "accuracy": summary.get("accuracy", 0.0),
            "avg_bleu1": summary.get("avg_bleu1", 0.0),
            "avg_rouge_l": summary.get("avg_rouge_l", 0.0),
            "n_samples": summary.get("n_samples", 0),
        }
        rows = [old for old in rows if not (old.get("dataset") == self.dataset and old.get("model") == self.model)]
        rows.append(row)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.all_results_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    llm_judge_enabled = str_to_bool(args.llm_judge)

    dataset_module = load_dataset_module(args.dataset)
    prompt_module = load_prompt_module(args.dataset)
    if not hasattr(prompt_module, "SYSTEM_PROMPT") or not hasattr(prompt_module, "build_prompt"):
        raise AttributeError(f"datasets/{args.dataset}/prompt.py must define SYSTEM_PROMPT and build_prompt(...)")

    eval_type = getattr(dataset_module, "eval_type", None)
    if eval_type not in {"closed", "open"}:
        raise ValueError(f"datasets/{args.dataset}/dataloader.py must declare eval_type as 'closed' or 'open'")
    score_sample = getattr(dataset_module, "compute_score", compute_score)
    summarize = getattr(dataset_module, "aggregate_scores", aggregate_scores)

    dataset_root = Path(args.data_dir) if args.data_dir else None
    samples = dataset_module.load_data(dataset_root)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    existing_results = []
    processed_ids = set()
    if args.resume:
        existing_results, processed_ids = read_existing_results(args.output_dir, args.dataset)
        samples = [sample for sample in samples if str(sample.get("id")) not in processed_ids]
        print(f"Resume enabled: skipping {len(processed_ids)} existing samples, running {len(samples)} remaining samples")

    model = load_model(args.model, args.model_path)
    judge = None
    if llm_judge_enabled and eval_type == "open":
        judge = LLMJudge(args.llm_judge_model)

    logger_cls = AppendEvalLogger if args.resume else EvalLogger
    logger = logger_cls(args.output_dir, args.dataset, args.model)
    results = list(existing_results)
    suppress_model_stdout = bool(getattr(prompt_module, "SUPPRESS_MODEL_STDOUT", False))
    show_model_messages = bool(getattr(prompt_module, "SHOW_MODEL_MESSAGES", False))
    model_accepts_sample = "sample" in inspect.signature(model.run_inference).parameters
    prompt_accepts_model_name = "model_name" in inspect.signature(prompt_module.build_prompt).parameters
    for sample in samples:
        prompt_kwargs = {"model_name": args.model} if prompt_accepts_model_name else {}
        query = prompt_module.build_prompt(sample.get("question", ""), sample.get("options", []), **prompt_kwargs)
        stream = io.StringIO() if suppress_model_stdout else None
        inference_kwargs = {
            "audio_path": sample.get("audio_path"),
            "video_path": sample.get("video_path"),
            "image_path": sample.get("image_path"),
            "query": query,
            "system_prompt": prompt_module.SYSTEM_PROMPT,
        }
        if model_accepts_sample:
            inference_kwargs["sample"] = sample
        inference_start = time.perf_counter()
        with contextlib.redirect_stdout(stream) if stream is not None else contextlib.nullcontext():
            prediction = model.run_inference(**inference_kwargs)
        inference_time_sec = time.perf_counter() - inference_start
        whisper_inference_time_sec = getattr(model, "last_whisper_inference_time_sec", None)
        vlm_inference_time_sec = getattr(model, "last_vlm_inference_time_sec", None)
        if stream is not None and show_model_messages:
            for line in stream.getvalue().splitlines():
                if line.startswith("messages =>"):
                    print(line)
        if hasattr(prompt_module, "postprocess_prediction"):
            prediction = prompt_module.postprocess_prediction(prediction)
        if suppress_model_stdout:
            print(f"Final Ans => {prediction}")

        judged_correct = None
        if judge is not None:
            judged_correct = judge.judge(
                question=sample.get("question", ""),
                reference=sample.get("answer", ""),
                prediction=prediction,
            )

        logged_sample = dict(sample)
        logged_sample["model_name"] = args.model
        logged_sample["system_prompt"] = prompt_module.SYSTEM_PROMPT
        logged_sample["input_prompt"] = query
        result = score_sample(
            sample=logged_sample,
            prediction=prediction,
            eval_type=eval_type,
            llm_judge_correct=judged_correct,
        )
        result["inference_time_sec"] = inference_time_sec
        if whisper_inference_time_sec is not None:
            result["whisper_inference_time_sec"] = float(whisper_inference_time_sec)
        if vlm_inference_time_sec is not None:
            result["vlm_inference_time_sec"] = float(vlm_inference_time_sec)
        logger.write_sample(result)
        results.append(result)

    summary = summarize(results, eval_type)
    inference_times = [result["inference_time_sec"] for result in results if "inference_time_sec" in result]
    if inference_times:
        summary["avg_inference_time_sec"] = sum(inference_times) / len(inference_times)
    whisper_inference_times = [
        result["whisper_inference_time_sec"] for result in results if "whisper_inference_time_sec" in result
    ]
    if whisper_inference_times:
        summary["avg_whisper_inference_time_sec"] = sum(whisper_inference_times) / len(whisper_inference_times)
    vlm_inference_times = [result["vlm_inference_time_sec"] for result in results if "vlm_inference_time_sec" in result]
    if vlm_inference_times:
        summary["avg_vlm_inference_time_sec"] = sum(vlm_inference_times) / len(vlm_inference_times)
    logger.write_summary(summary)
    logger.update_all_results(summary)
    print(summary)


if __name__ == "__main__":
    main()
