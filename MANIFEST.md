# Repository Manifest

OmniEvalKit is distributed here as a code-only evaluation harness. This
manifest summarizes included source and the external resources needed at
runtime. It is not a license; see `README.md` and `THIRD_PARTY_NOTICES.md`.

## Included top-level components

| Path | Purpose |
| --- | --- |
| `run_eval.py` | Main evaluation CLI and dynamic model/dataset loader. |
| `eval.sh` | Environment-variable-driven multi-dataset wrapper. |
| `datasets/` | Dataset dataloaders, prompts, parsers, and benchmark-specific scoring code. No dataset payloads are included. |
| `models/` | Model adapters. Required lightweight AutoGaze, VILA, and s2wrapper source is retained; weights are excluded. |
| `utils/` | Common scoring, logging, and local LLM-judge support. |
| `scripts/` | Dataset preparation, conversion, and result post-processing utilities. |
| `data/download/` | Download launchers only. Downloaded data remains ignored. |
| `tests/` | Synthetic-image and mask unit tests that do not require model checkpoints or benchmark datasets. |
| `requirements*.txt` | Common, test, data-preparation, and optional adapter dependencies. |
| `README.md` | Installation, usage, exclusions, and license-status documentation. |
| `THIRD_PARTY_NOTICES.md` | Vendored-source provenance, licenses, and redistribution obligations. |

## System and Python requirements

- Python 3.10 or newer is recommended.
- Install a platform-appropriate PyTorch build for CPU, CUDA, or ROCm.
- Audio/video adapters can require the `ffmpeg` and `ffprobe` executables.
- `requirements.txt` contains common runtime packages.
- `requirements-test.txt` adds the unit-test dependency.
- `requirements-data.txt` adds dataset preparation packages.
- `requirements-optional.txt` is a catalog, not a universally compatible
  environment. Install only the packages required by the selected adapter.
- VILA and AutoGaze retain their own upstream manifests because their pinned
  PyTorch/Transformers stacks can conflict with other adapters.

## External model requirements

No model checkpoint, model shard, tokenizer payload, or weight file is
included. Every model adapter requires either a remote model identifier or an
external local checkpoint.

- Qwen2.5-Omni, Qwen2.5-VL, Qwen3.5, Gemma, and MiniCPM adapters accept model
  IDs or local paths through `--model_path`/`MODEL_PATH`.
- `qwen3vl30b_whisper` requires `--model_path` or
  `QWEN3VL30B_MODEL_PATH`.
- `qwen3omni30b` requires `--model_path` or `QWEN3OMNI_MODEL_PATH`.
- `omnivinci` requires `--model_path` or `OMNIVINCI_MODEL_PATH`; its external
  checkpoint must also provide the custom Python files expected by the model.
- `VILA_whisper` and `VILA_whisper_1` include supporting source but require
  external VILA and AutoGaze checkpoints.
- Whisper-enabled adapters require an external ASR checkpoint or model ID;
  the common default is `openai/whisper-large-v3-turbo`.
- Advanced Qwen2.5-VL modes can require external SenseVoice, Mellow source and
  configuration, and SmolLM weights. Configure them with the adapter-specific
  environment variables documented in `README.md` and the model source.
- A Hugging Face model ID can trigger a download. Use pre-populated local paths
  for offline execution.

Model licenses are separate from the code licenses documented here. Review the
terms for every model or checkpoint before use or redistribution.

## External dataset requirements

No benchmark data, annotation payload, image, audio, or video is included.

- Pass one selected dataset directory with `--data_dir` or `DATA_DIR`.
- Without an explicit path, most adapters expect `data/<dataset-name>`.
- Relative media references are resolved from the selected dataset directory.
- Download/preparation utilities can create ignored data under `data/` and may
  require network access, access approval, or manual media acquisition.
- Dataset licenses and terms are independent of this repository and must be
  reviewed before use or redistribution.

## Generated output

Evaluation output is written beneath `results*`/`result*` paths and is ignored.
Typical files are `predictions.jsonl`, `summary.json`, and `all_results.csv`.

## Explicit exclusions

- Checkpoints, weights, model shards, and generated model artifacts
- Datasets, cached downloads, media, and archives
- Evaluation results, logs, judge outputs, and experiment directories
- Python caches, virtual environments, compiled binaries, and IDE metadata
- Git history from source or vendored repositories
- Secrets, tokens, API keys, credentials, and machine-specific configuration
