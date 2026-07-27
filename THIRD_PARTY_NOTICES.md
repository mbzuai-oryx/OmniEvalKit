# Third-Party Notices

This notice identifies third-party source retained in this repository. It is
informational and does **not** provide a project-level license for OmniEvalKit.
The project-authored portions of this repository currently have no identified
license and no project-level reuse rights are granted.

## Directly retained components

| Component | Original source or project | License | Included files or directories | Attribution and redistribution requirements |
| --- | --- | --- | --- | --- |
| NVIDIA AutoGaze | [NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze); exact source revision was not recorded in the original copy | Apache License 2.0 | `models/VILA_whisper/AutoGaze/autogaze/`, `pyproject.toml`, selected documentation, and `LICENSE` | Include the Apache-2.0 license; retain copyright, patent, trademark, and attribution notices; mark modified files; reproduce any applicable upstream NOTICE text. No upstream NOTICE file was present in the retained source. |
| NVIDIA VILA / NVILA | [NVlabs/VILA](https://github.com/NVlabs/VILA), source commit `0f1426e8da9181e6e6653e10bc15f62d515fa2f6` | Apache License 2.0 for code | `models/VILA_whisper_1/VILA/llava/` inference/runtime subset plus VILA `README.md`, `pyproject.toml`, `environment_setup.sh`, and `LICENSE` | Include the Apache-2.0 license; retain copyright and attribution notices; mark modified files. `llava/utils/merge_lora_weights_and_save_hf_model.py` carries a modification notice. Model weights are not included and have separate upstream terms. |
| S² Wrapper (`s2wrapper`) | [bfshi/scaling_on_scales](https://github.com/bfshi/scaling_on_scales), source commit `9c008a37540e761f53574b488979db6e49a64312` | MIT License | `models/VILA_whisper_1/scaling_on_scales/s2wrapper/`, `setup.py`, `README.md`, and `LICENSE.md` | Preserve the copyright notice and MIT permission notice in all copies or substantial portions. |
| OpenGVLab InternVL-derived vision encoder | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) | MIT License | `models/VILA_whisper_1/VILA/llava/model/multimodal_encoder/intern/configuration_intern_vit.py` and `modeling_intern_vit.py` | Preserve the OpenGVLab copyright and MIT permission notice. The applicable text is reproduced below because the retained VILA root license is Apache-2.0. |
| LLaVA-derived files within VILA | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | Apache License 2.0 | Files under `models/VILA_whisper_1/VILA/llava/` whose headers state that they were modified from LLaVA | Preserve the existing Haotian Liu/LLaVA copyright and Apache-2.0 notices; mark further modifications. The retained VILA Apache license supplies the license text. |
| Hugging Face Transformers and model-implementation derivatives | [huggingface/transformers](https://github.com/huggingface/transformers), with file-level attribution also naming Google AI, Facebook AI, EleutherAI, and the Qwen team | Apache License 2.0 | Attributed implementation files under `models/VILA_whisper/AutoGaze/autogaze/` and `models/VILA_whisper_1/VILA/llava/model/` | Preserve all file-level copyright and Apache-2.0 notices and mark further modifications. |
| LongLoRA-derived merge utility | [dvlab-research/LongLoRA](https://github.com/dvlab-research/LongLoRA) | Apache License 2.0 | `models/VILA_whisper_1/VILA/llava/utils/merge_lora_weights_and_save_hf_model.py` | Preserve copyright and Apache-2.0 notices. The file is explicitly marked as modified for this repository. |
| Long-context-attention and DeepSpeed-derived sequence-parallel code | [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention) and [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Apache License 2.0 | Attributed files under `models/VILA_whisper_1/VILA/llava/train/sequence_parallel/`, especially `ulysses_attn.py`, `hybrid_attn.py`, and `globals.py` | Preserve Microsoft/NVIDIA and upstream copyright, SPDX, and Apache-2.0 notices; mark further modifications. |
| Ring Flash Attention | [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) | MIT License | `models/VILA_whisper_1/VILA/llava/train/sequence_parallel/ring/` | Preserve the Zilin Zhu copyright and MIT permission notice. The applicable text is reproduced below because it was not present in the retained VILA license file. |
| Liger Kernel and Unsloth-derived utilities | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and [unslothai/unsloth](https://github.com/unslothai/unsloth) | BSD 2-Clause for Liger Kernel; Apache License 2.0 for the attributed Unsloth code | `models/VILA_whisper_1/VILA/llava/model/liger/cross_entropy.py` and `utils.py` | Preserve the LinkedIn BSD notice and conditions reproduced below. Preserve the existing Unsloth origin and modification notice and supply the Apache-2.0 text retained in the VILA `LICENSE`; mark further modifications. |
| Meta DINO and DINOv2-derived interpolation code | [facebookresearch/dino](https://github.com/facebookresearch/dino) and [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache License 2.0 | Attributed interpolation functions in `models/VILA_whisper/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py` and `tasks/video_mae_reconstruction/modeling_video_mae.py` | Preserve the source references and Apache-2.0 notices and mark further modifications. The AutoGaze `LICENSE` supplies the license text. |
| Google Big Vision-derived SigLIP loss code | [google-research/big_vision](https://github.com/google-research/big_vision) | Apache License 2.0 | Attributed loss calculation in `models/VILA_whisper_1/VILA/llava/model/multimodal_encoder/siglip/modeling_siglip.py` | Preserve the Google/Hugging Face header, source reference, and Apache-2.0 notices; mark further modifications. The VILA `LICENSE` supplies the license text. |
| MosaicML Examples and minGPT-derived OLMo implementation | [mosaicml/examples](https://github.com/mosaicml/examples) and [karpathy/minGPT](https://github.com/karpathy/minGPT) | Apache License 2.0 for MosaicML Examples; MIT License for minGPT | `models/VILA_whisper_1/VILA/llava/model/coat/activation/models/coat_olmo.py` | Preserve the existing source attribution and Apache-2.0 header. Preserve Andrej Karpathy's copyright and MIT permission notice reproduced below. |

## Locally sourced helpers with unresolved licensing

The following files were copied from sibling working directories because the
original evaluator imported them at runtime. No license header or accompanying
license was found for these files. Their inclusion here records provenance but
does not establish permission for public redistribution:

| Helper | Source location at preparation time | Included file | Required action before publication |
| --- | --- | --- | --- |
| Qwen3.5 multimodal runner | Sibling project `Qwen-omni3.5/run_qwen35_multimodal.py` | `models/qwen35omni/run_qwen35_multimodal.py` | Confirm that the repository owner holds the copyright or obtain redistribution permission; add an accurate notice if authorized. |
| Hugging Face dataset downloader | Sibling project `OmniEvalKit1/scripts/hf_download.py` | `scripts/hf_download.py` | Confirm ownership or redistribution permission; add an accurate notice if authorized. |
| Parquet-to-JSONL converter | Sibling project `OmniEvalKit1/scripts/parquet_to_jsonl.py` | `scripts/parquet_to_jsonl.py` | Confirm ownership or redistribution permission; add an accurate notice if authorized. |

Python packages named in the requirements files are installed dependencies and
are not copied into this repository. Their licenses apply when they are
installed or redistributed separately.

## OpenGVLab InternVL MIT license text

MIT License

Copyright (c) 2023 OpenGVLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Ring Flash Attention MIT license text

Copyright 2024 Zilin Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## minGPT MIT license text

The MIT License (MIT)

Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Liger Kernel BSD 2-Clause license text

BSD 2-CLAUSE LICENSE

Copyright 2024 LinkedIn Corporation

All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
