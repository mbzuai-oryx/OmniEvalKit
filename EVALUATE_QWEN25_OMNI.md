# Evaluate Qwen/Qwen2.5-Omni-3B

Run from the repository root after activating your Python environment:

```bash
cd /path/to/OmniEvalKit
```

Evaluate `Qwen/Qwen2.5-Omni-3B` on `daily_omni`:

```bash
MODEL=qwen25omni \
MODEL_PATH=Qwen/Qwen2.5-Omni-3B \
EVAL_DATASETS=daily_omni \
DATA_DIR=/path/to/daily_omni \
OUTPUT_DIR=results \
LLM_JUDGE=False \
./eval.sh
```

Equivalent direct command:

```bash
python run_eval.py \
  --model qwen25omni \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --dataset daily_omni \
  --data_dir /path/to/daily_omni \
  --output_dir results \
  --llm_judge False
```

Outputs:

```text
results/daily_omni/predictions.jsonl
results/daily_omni/summary.json
results/all_results.csv
```

For a quick smoke test:

```bash
python run_eval.py \
  --model qwen25omni \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --dataset daily_omni \
  --data_dir /path/to/daily_omni \
  --output_dir results_smoke \
  --llm_judge False \
  --max_samples 3
```
