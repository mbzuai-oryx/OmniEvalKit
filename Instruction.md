# OmniEvalKit

OmniEvalKit evaluates text, image, video, audio, and audio-visual models through
a common runner. Model adapters are in `models/`; dataset loaders, prompts, and
scorers are in `datasets/`.

This repository is code-only. Model weights, datasets, media, and results are
not included.

## Install

Python 3.10 or newer is recommended. Audio/video evaluation also requires
`ffmpeg` and `ffprobe`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install the correct CPU, CUDA, or ROCm PyTorch build first.
python3 -m pip install -r requirements.txt
```

Optional dependencies:

```bash
python3 -m pip install -r requirements-data.txt  # dataset preparation
python3 -m pip install -r requirements-test.txt  # tests and SciPy
```

Some adapters require their own PyTorch/Transformers environment. See the
README inside the selected model directory when one is provided.

## Run an evaluation

Run commands from the repository root:

```bash
MODEL=qwen25omni \
MODEL_PATH=Qwen/Qwen2.5-Omni-3B \
EVAL_DATASETS=daily_omni \
DATA_DIR=/path/to/daily_omni \
OUTPUT_DIR=results/qwen25omni \
MAX_SAMPLES=3 \
LLM_JUDGE=False \
bash eval.sh
```

- `MODEL` must match a directory under `models/`.
- `EVAL_DATASETS` must match a directory under `datasets/`.
- `DATA_DIR` is the selected dataset directory. Omit it when data is already
  under `data/<dataset-name>`.
- Remove `MAX_SAMPLES` for the full evaluation.
- Use a separate `OUTPUT_DIR` for every model.

Multiple datasets can be evaluated together when they are stored at their
default locations:

```bash
MODEL=qwen25omni \
MODEL_PATH=Qwen/Qwen2.5-Omni-3B \
EVAL_DATASETS="daily_omni,omnibench,av_odyssey" \
OUTPUT_DIR=results/qwen25omni \
LLM_JUDGE=False \
bash eval.sh
```

Direct Python usage:

```bash
python3 run_eval.py \
  --model qwen25omni \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --dataset daily_omni \
  --data_dir /path/to/daily_omni \
  --output_dir results/qwen25omni \
  --llm_judge False
```

Outputs:

```text
<OUTPUT_DIR>/<dataset>/predictions.jsonl
<OUTPUT_DIR>/<dataset>/summary.json
<OUTPUT_DIR>/all_results.csv
```

`summary.json` contains the complete dataset-specific metrics.
`all_results.csv` contains dataset, model, accuracy, BLEU-1, ROUGE-L, and
sample count.

Resume an interrupted run with `RESUME=True`. Do not reuse a resumed output
directory across models. Remove `MAX_SAMPLES` when resuming a full run.

## Models

Use the common evaluation command above and replace `MODEL`, `MODEL_PATH`, and
the optional settings using this table:

| `MODEL` | `MODEL_PATH` | Important settings |
| --- | --- | --- |
| `qwen25omni` | `Qwen/Qwen2.5-Omni-3B` | Native audio/image/video. Embedded video audio defaults on. |
| `qwen25vlomni_advance` | `Qwen/Qwen2.5-VL-3B-Instruct` | Install `qwen-vl-utils`. For audio/AV use `QWEN25VL_USE_ASR=True QWEN25VL_ASR_DEVICE=cuda`. ASR and SenseVoice default off. |
| `qwen35omni` | `Qwen/Qwen3.5-2B` | For a separate audio path use `QWEN35OMNI_USE_ASR=True QWEN35OMNI_ASR_DEVICE=cuda`. Embedded video audio is not transcribed. |
| `qwen3omni30b` | Qwen3-Omni-30B-A3B checkpoint/ID | Explicit path required. Native omni input; `QWEN3OMNI_USE_AUDIO_IN_VIDEO=True`. |
| `qwen3vl30b_whisper` | Qwen3-VL-30B-A3B checkpoint/ID | Explicit path required. Whisper defaults on; normally set `QWEN3VL30B_ASR_DEVICE=cuda`. |
| `minicpmo45` | `openbmb/MiniCPM-o-4_5` | Native omni. Use `MINICPMO45_INIT_TTS=False`; follow `models/minicpmo45/README.md` for its pinned environment. |
| `minicpmv45_whisper` | `openbmb/MiniCPM-V-4_5` | Currently unavailable because its base `models.minicpmo45_whisper` module is missing. |
| `VILA_whisper` | `nvidia/NVILA-8B-HD-Video` | Requires AutoGaze/VILA-specific dependencies and external weights. See `models/VILA_whisper/README.md`. |
| `VILA_whisper_1` | `Efficient-Large-Model/NVILA-8B-Video` | Requires the retained VILA environment and external weights. See `models/VILA_whisper_1/README.md`. |
| `omnivinci` | `/absolute/path/to/omnivinci-checkpoint` | Must be an existing local checkpoint containing the required custom Python files. |

Example with Qwen2.5-VL and Whisper:

```bash
QWEN25VL_USE_ASR=True \
QWEN25VL_ASR_DEVICE=cuda \
MODEL=qwen25vlomni_advance \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
EVAL_DATASETS=daily_omni \
DATA_DIR=/path/to/daily_omni \
OUTPUT_DIR=results/qwen25vlomni_advance \
LLM_JUDGE=False \
bash eval.sh
```

## Available datasets

There are 118 dataset adapters: 78 open-answer and 40 closed-answer evaluators.
“Available” means evaluation code exists; the dataset payload must still be
downloaded or prepared.

A video may contain embedded audio even when no separate `audio_path` is
provided.

### Audio-only — 64

`aishell1_test`, `aishell2_test`, `audio_trivia_qa`,
`audio_web_questions`, `audiocaps_test`, `clothocaption_test`,
`commonvoice_en`, `commonvoice_fr`, `commonvoice_yue`, `commonvoice_zh`,
`covost2_ar_en`, `covost2_ca_en`, `covost2_cy_en`, `covost2_de_en`,
`covost2_en_zh`, `covost2_es_en`, `covost2_et_en`, `covost2_fa_en`,
`covost2_fr_en`, `covost2_id_en`, `covost2_it_en`, `covost2_ja_en`,
`covost2_lv_en`, `covost2_mn_en`, `covost2_nl_en`, `covost2_pt_en`,
`covost2_ru_en`, `covost2_sl_en`, `covost2_sv_se_en`, `covost2_ta_en`,
`covost2_tr_en`, `covost2_zh_en`, `fleurs_en`, `fleurs_zh`,
`gigaspeech_test`, `kespeech_test`, `librispeech_dev_clean`,
`librispeech_dev_other`, `librispeech_test_clean`,
`librispeech_test_other`, `meld`, `mmar_bench`, `mmau_test_mini`,
`mmsu_bench`, `peoples_speech_test`, `tedlium3_test`, `vocalsound`,
`voice_cmmlu`, `voicebench_advbench`, `voicebench_alpacaeval`,
`voicebench_alpacaeval_full`, `voicebench_bbh`, `voicebench_commoneval`,
`voicebench_ifeval`, `voicebench_mmsu`, `voicebench_openbookqa`,
`voicebench_sdqa`, `voicebench_wildvoice`, `voxpopuli_en`,
`wavcaps_audioset_sl`, `wavcaps_freesound`, `wavcaps_soundbible`,
`wenetspeech_test_meeting`, `wenetspeech_test_net`.

### Image-only — 14

`chartqa`, `docvqa`, `infographicvqa`, `mathverse_mini`, `mathvista`,
`mmbench`, `mme`, `mmstar`, `ocrbench`, `pixmo_count`, `pixmo_pointing`,
`pointarena_counting`, `refcoco`, `textvqa`.

### Video — 13

`avhbench`, `avhbench_caption`, `egoschema`, `livesports3k_cc`,
`longvideobench_val`, `lvbench`, `motionbench`, `mvbench`,
`videomathqa_mcq`, `videomathqa_multi_binary`,
`videomathqa_muti_binary`, `videomme`, `videomme_short`.

`videomathqa_muti_binary` is a compatibility alias for
`videomathqa_multi_binary`. `avhbench_caption` shares AVHBench data.
`livesports3k_cc` rows can expose both video and separate audio.

### Omni audio-visual and mixed multimodal — 17

`av_odyssey`, `avmeme_full`, `avmeme_main`, `avut_benchmark_gemini`,
`avut_benchmark_human`, `daily_omni`, `futureomni`, `jointavbench`,
`omnibench`, `ovobench`, `streamingbench_omni_fix`,
`streamingbench_real`, `streamingbench_sqa`, `unobench`, `unobench_mc`,
`video_holmes`, `worldsense`.

### Video/image mixed — 1

`ovavel`.

### Text, code, and math — 8

`gsm8k`, `gsm8k_socratic`, `math500`, `mbpp`, `mbpp_sanitized`,
`mbppplus`, `multiple`, `openai_humaneval`, `theoremqa`.


## Prepare datasets

```bash
# Audio datasets registered by the audio preparation script
python3 scripts/prepare_audio_datasets.py all

# Image/video datasets registered by the VLM preparation script
python3 scripts/prepare_vlm_datasets.py all

# Math datasets
python3 scripts/prepare_math_datasets.py \
  --datasets gsm8k,math500,theoremqa \
  --configs main,socratic

# Code datasets
python3 scripts/prepare_code_datasets.py \
  --datasets mbpp,mbppplus,humaneval,multiple

# List packaged Omni/AV downloads
python3 scripts/hf_download.py --list
```

Dataset-specific download scripts are under `data/download/`.

Current preparation limitations:

- PixMo Count and PixMo Point loaders require `test_clean.jsonl`, while the VLM
  preparation script currently writes `test.jsonl`.
- PointArena has no preparation/download script.
- PixMo Point requires SciPy.
- Prefer `data/download/videomme_short.sh` for Video-MME Short.

## Scoring

Most datasets use the shared open-answer or closed-answer scorer. Complete
metrics are written to each `summary.json`.

| Dataset | Custom metrics |
| --- | --- |
| `pixmo_count` | Exact integer accuracy and valid-output rate. |
| `pixmo_pointing` | 3px/5px point precision, recall, F1 and accuracy; mask accuracy/F1; pixel and normalized Euclidean distance. |
| `refcoco` | Bounding-box IoU, average IoU, and accuracy at IoU >= 0.5. |
| `pointarena_counting` | Count accuracy/MAE and pointing precision, recall, F1 and exact accuracy. |

Code datasets use text/ROUGE scoring; generated code is not executed.

## Known limitations
- Qwen3-Omni grounding output need coordinate postprocessing before
  RefCOCO, PixMo Point, or PointArena scoring, [0-1000].
- The central runner evaluates samples sequentially.


## GitHub and license

Datasets, media, checkpoints, weights, results, caches, and common credential
files are ignored by Git. Always review changes before pushing:

```bash
git status --short
git diff
git diff --cached
```

This repository currently has no project-level license. Public availability
does not automatically grant permission to use or redistribute the
project-authored code. Third-party components remain under their own licenses;
see `THIRD_PARTY_NOTICES.md`.

## Add a new dataset

No central dataset registry is required. The runner dynamically imports
`datasets/<dataset-name>/dataloader.py` and `prompt.py`.

### 1. Create the dataset package

```text
datasets/my_dataset/
├── __init__.py
├── dataloader.py
└── prompt.py
```

### 2. Implement the loader

For a standard JSONL image/video dataset:

```python
# datasets/my_dataset/dataloader.py
from datasets._vision_video_common import load_jsonl_dataset

eval_type = "open"  # or "closed"


def load_data(data_dir=None):
    return load_jsonl_dataset("my_dataset", data_dir)


load_dataset = load_data
```

The default manifest location is `data/my_dataset/test.jsonl`. Each normalized
sample should provide:

```json
{
  "id": "sample-1",
  "question": "What happens?",
  "options": [],
  "answer": "reference answer",
  "audio_path": "",
  "video_path": "",
  "image_path": ""
}
```

Use only the relevant media path. Relative paths are resolved from the dataset
directory. Existing shared loaders are:

- `datasets._audio_common` for audio datasets.
- `datasets._vision_video_common` for image/video datasets.
- `datasets._math_common` for math datasets.
- `datasets._code_common` for code datasets.

### 3. Implement the prompt

```python
# datasets/my_dataset/prompt.py
SYSTEM_PROMPT = "Answer the question using the provided media."


def build_prompt(question, options=None):
    return str(question).strip()
```

Optionally define `postprocess_prediction(prediction)` when model output needs
deterministic cleanup.

### 4. Add custom scoring only when needed

Without custom functions, the shared scorer uses the declared `eval_type`.
For benchmark-specific metrics, add both functions to `dataloader.py`:

```python
def compute_score(sample, prediction, eval_type="open", llm_judge_correct=None):
    correct = str(prediction).strip() == str(sample["answer"]).strip()
    return {**sample, "prediction": prediction, "correct": correct}


def aggregate_scores(results, eval_type="open"):
    count = len(results)
    accuracy = sum(bool(row.get("correct")) for row in results) / count if count else 0.0
    return {"n_samples": count, "accuracy": accuracy}
```

Returned values must be JSON serializable.

### 5. Test the new adapter

```bash
python3 run_eval.py \
  --model qwen25omni \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --dataset my_dataset \
  --data_dir /path/to/my_dataset \
  --output_dir results/my_dataset_smoke \
  --llm_judge False \
  --max_samples 3
```

Add focused unit tests under `tests/` for custom parsing, normalization, or
scoring. If the dataset needs automatic preparation, add it to the appropriate
script under `scripts/` or create a dedicated downloader under
`data/download/`.
