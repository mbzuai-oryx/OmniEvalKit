#!/bin/bash
set -euo pipefail

# OmniEvalKit evaluation entrypoint.
#
# EVAL_DATASETS: Comma- or space-separated dataset names to run. Keep one active
#   assignment and comment/uncomment the blocks below, following MedEvalKit style.
# MODEL: Model folder name under models/ used by run_eval.py, e.g. qwen35omni.
# MODEL_PATH: Hugging Face model id or local checkpoint path. Leave unset to use
#   the default checkpoint for MODEL.
# OUTPUT_DIR: Root directory where per-dataset results and all_results.csv are written.
# DATA_DIR: Optional path to the selected dataset directory.
# LLM_JUDGE: True/False; only open-ended datasets use LLM-as-judge.
# LLM_JUDGE_MODEL: Hugging Face causal LM used by the local LLM judge.
# USE_VLLM: True/False runtime toggle reserved for model wrappers that support vLLM.
# TEMPERATURE: Generation temperature reserved for model wrappers that expose sampling.
# MAX_NEW_TOKENS: Maximum generated tokens reserved for model wrappers that expose it.
# BATCH_SIZE: Batch size reserved for future batched evaluation support.
# MAX_SAMPLES: Optional sample limit for smoke tests.
# PYTHON_BIN: Python executable to use.

# Dataset setting
# Available local datasets with determined eval_type:
EVAL_DATASETS="${EVAL_DATASETS:-daily_omni}"
# EVAL_DATASETS="daily_omni"
# EVAL_DATASETS="voicebench_alpacaeval"
# EVAL_DATASETS="voicebench_commoneval"
# EVAL_DATASETS="voicebench_mmsu"
# EVAL_DATASETS="voicebench_sdqa"
# EVAL_DATASETS="daily_omni,voicebench_mmsu,voicebench_sdqa"
# EVAL_DATASETS="omnibench"
# EVAL_DATASETS="videomme"
# EVAL_DATASETS="videomme_short"
# EVAL_DATASETS="unobench"
# EVAL_DATASETS="unobench_mc"
# EVAL_DATASETS="worldsense"
# EVAL_DATASETS="av_odyssey"
# EVAL_DATASETS="video_holmes"
# EVAL_DATASETS="avut_benchmark_human"
# EVAL_DATASETS="avut_benchmark_gemini"
# EVAL_DATASETS="ovobench"
# EVAL_DATASETS="streamingbench_real"
# EVAL_DATASETS="streamingbench_omni_fix"
# EVAL_DATASETS="streamingbench_sqa"
# EVAL_DATASETS="futureomni"
# EVAL_DATASETS="jointavbench"
# EVAL_DATASETS="ovavel"
# EVAL_DATASETS="avmeme_full"
# EVAL_DATASETS="avmeme_main"
# EVAL_DATASETS="avhbench"
# EVAL_DATASETS="avhbench_caption"
# EVAL_DATASETS="videomathqa_mcq"
# EVAL_DATASETS="videomathqa_multi_binary"
# EVAL_DATASETS="omnibench,videomme_short,worldsense,av_odyssey"

# Model setting
MODEL="${MODEL:-qwen35omni}"
# MODEL=gemma4e2b
# MODEL_PATH=google/gemma-4-E2B-it
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "$MODEL" in
        gemma4e2b)
            MODEL_PATH="google/gemma-4-E2B-it"
            ;;
        gemma4e2b_whisper)
            MODEL_PATH="google/gemma-4-E2B-it"
            ;;
        gemma4e2b_qat)
            MODEL_PATH="google/gemma-4-E2B-it-qat-q4_0-unquantized"
            ;;
        qwen25vlomni)
            MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
            ;;
        qwen35omni)
            MODEL_PATH="Qwen/Qwen3.5-2B"
            ;;
        qwen25omni)
            MODEL_PATH="Qwen/Qwen2.5-Omni-3B"
            ;;
        omnivinci)
            MODEL_PATH="${OMNIVINCI_MODEL_PATH:-}"
            ;;
        qwen3vl30b_whisper)
            MODEL_PATH="${QWEN3VL30B_MODEL_PATH:-}"
            ;;
        qwen3omni30b)
            MODEL_PATH="${QWEN3OMNI_MODEL_PATH:-}"
            ;;
        *)
            MODEL_PATH=""
            ;;
    esac
fi

# Output setting
OUTPUT_DIR="${OUTPUT_DIR:-results}"
DATA_DIR="${DATA_DIR:-}"

# vLLM/runtime setting
USE_VLLM="${USE_VLLM:-False}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Generation setting
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# LLM judge setting
LLM_JUDGE="${LLM_JUDGE:-False}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3.5-27B}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

IFS=', ' read -r -a DATASET_LIST <<< "$EVAL_DATASETS"

for DATASET in "${DATASET_LIST[@]}"; do
    if [[ -z "$DATASET" ]]; then
        continue
    fi

    echo "Evaluating dataset: $DATASET"

    MODEL_PATH_ARGS=()
    if [[ -n "$MODEL_PATH" ]]; then
        MODEL_PATH_ARGS=(--model_path "$MODEL_PATH")
    fi

    MAX_SAMPLES_ARGS=()
    if [[ -n "$MAX_SAMPLES" ]]; then
        MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
    fi

    DATA_DIR_ARGS=()
    if [[ -n "$DATA_DIR" ]]; then
        DATA_DIR_ARGS=(--data_dir "$DATA_DIR")
    fi

    USE_VLLM="$USE_VLLM" \
    TEMPERATURE="$TEMPERATURE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    BATCH_SIZE="$BATCH_SIZE" \
    "$PYTHON_BIN" run_eval.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --llm_judge "$LLM_JUDGE" \
        --llm_judge_model "$LLM_JUDGE_MODEL" \
        "${MODEL_PATH_ARGS[@]}" \
        "${DATA_DIR_ARGS[@]}" \
        "${MAX_SAMPLES_ARGS[@]}"
done
