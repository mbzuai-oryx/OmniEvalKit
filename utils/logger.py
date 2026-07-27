import csv
import json
from pathlib import Path


class EvalLogger:
    def __init__(self, output_dir, dataset, model):
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.model = model
        self.dataset_dir = self.output_dir / dataset
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.dataset_dir / "predictions.jsonl"
        self.summary_path = self.dataset_dir / "summary.json"
        self.all_results_path = self.output_dir / "all_results.csv"
        self.samples_file = self.samples_path.open("w", encoding="utf-8")

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


ResultLogger = EvalLogger
