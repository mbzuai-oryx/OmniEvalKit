#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python3 scripts/prepare_audio_datasets.py \
  gigaspeech_test \
  wenetspeech_test_net \
  wenetspeech_test_meeting \
  librispeech_test_clean \
  librispeech_test_other \
  librispeech_dev_clean \
  librispeech_dev_other \
  commonvoice_en \
  commonvoice_zh \
  commonvoice_yue \
  commonvoice_fr \
  aishell1_test \
  aishell2_test \
  kespeech_test \
  voxpopuli_en \
  fleurs_zh \
  fleurs_en \
  peoples_speech_test \
  tedlium3_test \
  voicebench_alpacaeval \
  voicebench_alpacaeval_full \
  voicebench_bbh \
  voicebench_mmsu \
  voicebench_openbookqa \
  voicebench_advbench \
  voicebench_commoneval \
  voicebench_ifeval \
  voicebench_sdqa \
  voicebench_wildvoice \
  voice_cmmlu \
  mmau_test_mini \
  mmsu_bench \
  mmar_bench \
  audio_web_questions \
  audio_trivia_qa \
  audiocaps_test \
  clothocaption_test \
  wavcaps_audioset_sl \
  wavcaps_freesound \
  wavcaps_soundbible \
  covost2_zh_en \
  covost2_en_zh \
  vocalsound \
  meld \
  livesports3k_cc
