# Repository Manifest

OmniEvalKit is an evaluation harness for text, image, video, audio, and
audio-visual models. This manifest summarizes the included source and project
media, plus the external resources required at runtime. Model weights and
benchmark payloads are not included.

## Included top-level components

| Path | Purpose |
| --- | --- |
| `run_eval.py` | Main evaluation CLI and dynamic model/dataset loader. |
| `eval.sh` | Environment-variable-driven multi-dataset wrapper. |
| `datasets/` | Dataset dataloaders, prompts, parsers, and benchmark-specific scoring code. No dataset payloads are included. |
| `models/` | Model adapters. Required lightweight AutoGaze, VILA, and s2wrapper source is retained; weights are excluded. |
| `utils/` | Common scoring, logging, and local LLM-judge support. |
| `scripts/` | Dataset preparation, conversion, and result post-processing utilities. |
| `data/download/` | Benchmark download launchers. |
| `tests/` | Synthetic-image and mask unit tests that do not require model checkpoints or benchmark datasets. |
| `End_to_End_Model/` | Standalone Qwen2.5-VL, speech analysis, and optional CosyVoice3 output pipeline. |
| `assets/` | Paper logos, teaser poster, and teaser video used by `README.md`. |
| `requirements.txt` | Consolidated Python dependency list. |
| `README.md` | Paper overview, installation, model configurations, evaluation commands, and citation. |

## System and Python requirements

- Python 3.10 or newer is recommended.
- Install a platform-appropriate PyTorch build for CPU, CUDA, or ROCm.
- Audio/video adapters can require the `ffmpeg` and `ffprobe` executables.
- `requirements.txt` is the only root dependency file and contains the shared
  runtime, data-processing, and adapter packages used by this release.
- Some vendored adapters retain their own environment or package manifests.
  Their pinned PyTorch/Transformers stacks can conflict with the root
  environment. Treat the root file as a consolidated dependency list and
  follow the selected adapter's README when a separate environment is needed.

## External model requirements

No pretrained model checkpoint, model shard, or weight file is included. A
small tokenizer resource required by the vendored CosyVoice runtime is
retained. Every model adapter requires either a remote model identifier or an
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
  configuration, and SmolLM weights. Configure them with the environment
  variables defined in the adapter source.
- A Hugging Face model ID can trigger a download. Use pre-populated local paths
  for offline execution.

Model and checkpoint licenses are separate from this repository. Retained
third-party source includes its upstream license files. Review the applicable
terms before using or redistributing any external model.

## Available datasets

OmniEvalKit includes **118 dataset adapters** evaluators. An available adapter provides evaluation code;
the corresponding dataset payload must still be downloaded or prepared.

| Modality | Adapters |
| --- | ---: |
| Audio-only | 64 |
| Image-only | 14 |
| Video | 13 |
| Omni audio-visual and mixed multimodal | 17 |
| Video/image mixed | 1 |
| Text, code, and math | 9 |
| **Total** | **118** |

> [!NOTE]
> A video may contain embedded audio even when no separate `audio_path` is
> provided.

<details>
<summary><strong>🎧 Audio-only — 64 adapters</strong></summary>

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

</details>

<details>
<summary><strong>🖼️ Image-only — 14 adapters</strong></summary>

`chartqa`, `docvqa`, `infographicvqa`, `mathverse_mini`, `mathvista`,
`mmbench`, `mme`, `mmstar`, `ocrbench`, `pixmo_count`, `pixmo_pointing`,
`pointarena_counting`, `refcoco`, `textvqa`.

</details>

<details>
<summary><strong>🎬 Video — 13 adapters</strong></summary>

`avhbench`, `avhbench_caption`, `egoschema`, `livesports3k_cc`,
`longvideobench_val`, `lvbench`, `motionbench`, `mvbench`,
`videomathqa_mcq`, `videomathqa_multi_binary`,
`videomathqa_muti_binary`, `videomme`, `videomme_short`.

Notes:

- `videomathqa_muti_binary` is a compatibility alias for
  `videomathqa_multi_binary`.
- `avhbench_caption` shares AVHBench data.
- `livesports3k_cc` samples may expose both video and separate audio.

</details>

<details>
<summary><strong>🌐 Omni audio-visual and mixed multimodal — 17 adapters</strong></summary>

`av_odyssey`, `avmeme_full`, `avmeme_main`, `avut_benchmark_gemini`,
`avut_benchmark_human`, `daily_omni`, `futureomni`, `jointavbench`,
`omnibench`, `ovobench`, `streamingbench_omni_fix`,
`streamingbench_real`, `streamingbench_sqa`, `unobench`, `unobench_mc`,
`video_holmes`, `worldsense`.

</details>

<details>
<summary><strong>🎞️ Video/image mixed — 1 adapter</strong></summary>

`ovavel`.

</details>

<details>
<summary><strong>📝 Text, code, and math — 9 adapters</strong></summary>

`gsm8k`, `gsm8k_socratic`, `math500`, `mbpp`, `mbpp_sanitized`,
`mbppplus`, `multiple`, `openai_humaneval`, `theoremqa`.

</details>

## External dataset requirements

No benchmark dataset payload is included. Files under `assets/` are project
media for the repository page, not evaluation samples.

- Pass one selected dataset directory with `--data_dir` or `DATA_DIR`.
- Without an explicit path, most adapters expect `data/<dataset-name>`.
- Relative media references are resolved from the selected dataset directory.
- Download/preparation utilities can create data under `data/` and may require
  network access, access approval, or manual media acquisition. Downloaded
  payloads are not automatically ignored, so keep them out of the submitted
  source package.
- Dataset licenses and terms are independent of this repository and must be
  reviewed before use or redistribution.

## Generated output

Evaluation output is written beneath the directory selected by `OUTPUT_DIR` or
`--output_dir` (default: `results`). Typical files are `predictions.jsonl`,
`summary.json`, and `all_results.csv`. Generated results are not automatically
ignored by the current repository, so exclude them from source submissions.
