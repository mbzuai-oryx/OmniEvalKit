# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""AutoGaze model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from omegaconf import OmegaConf
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

logger = logging.get_logger(__name__)



class GazeDecoderConfig(PretrainedConfig):
    r"""
    Based on LLamaConfig from transformers.
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        attn_mode="sdpa",
        num_multi_token_pred=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.num_multi_token_pred = num_multi_token_pred
        self._attn_implementation = attn_mode
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class VisionModelConfig(PretrainedConfig):
    r"""
    Configuration for the vision model component of AutoGaze.

    Args:
        hidden_dim (`int`, *optional*, defaults to `192`):
            Hidden dimension of the vision model.
        out_dim (`int`, *optional*, defaults to `192`):
            Output dimension of the vision model.
        depth (`int`, *optional*, defaults to `1`):
            Depth of the vision model.
        kernel_size (`int`, *optional*, defaults to `16`):
            Kernel size for spatial convolution.
        temporal_patch_size (`int`, *optional*, defaults to `1`):
            Temporal patch size for video processing.
        trunk_temporal_kernel_size (`int`, *optional*, defaults to `3`):
            Temporal kernel size for trunk blocks.
        trunk_spatial_kernel_size (`int`, *optional*, defaults to `3`):
            Spatial kernel size for trunk blocks.
    """
    def __init__(
        self,
        hidden_dim=192,
        out_dim=192,
        depth=1,
        kernel_size=16,
        temporal_patch_size=1,
        trunk_temporal_kernel_size=3,
        trunk_spatial_kernel_size=3,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.temporal_patch_size = temporal_patch_size
        self.trunk_temporal_kernel_size = trunk_temporal_kernel_size
        self.trunk_spatial_kernel_size = trunk_spatial_kernel_size

        super().__init__(**kwargs)


class ConnectorConfig(PretrainedConfig):
    r"""
    Configuration for the connector component between vision encoder and gaze model.

    Args:
        hidden_dim (`int`, *optional*, defaults to `192`):
            Hidden dimension of the connector.
    """

    def __init__(
        self,
        hidden_dim=192,
        num_tokens=196,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        super().__init__(**kwargs)


class GazeModelConfig(PretrainedConfig):
    r"""
    Configuration for the gaze model, containing vision model, connector, and decoder configs.

    Args:
        num_multi_token_pred (`int`, *optional*, defaults to `1`):
            Number of tokens to predict in parallel.
        input_img_size (`int`, *optional*, defaults to `224`):
            Input image size.
        vision_model_config (`VisionModelConfig` or `dict`, *optional*):
            Configuration for the vision model.
        connector_config (`ConnectorConfig` or `dict`, *optional*):
            Configuration for the connector.
        gaze_decoder_config (`GazeDecoderConfig` or `dict`, *optional*):
            Configuration for the gaze decoder (LLaMA-based).
    """

    def __init__(
        self,
        input_img_size=224,
        vision_model_config={},
        connector_config={},
        gaze_decoder_config={},
        num_vision_tokens_each_frame=196,
        attn_mode="sdpa",
        **kwargs,
    ):
        self.input_img_size = input_img_size
        self.vision_model_config = VisionModelConfig(**vision_model_config)

        connector_config.update({
            "num_tokens": (input_img_size // self.vision_model_config.kernel_size)**2,
        })
        self.connector_config = ConnectorConfig(**connector_config)

        gaze_decoder_config.update({
            "vocab_size": num_vision_tokens_each_frame + 1,
            "eos_token_id": num_vision_tokens_each_frame,
            "attn_mode": attn_mode,
        })
        self.gaze_decoder_config = GazeDecoderConfig(**gaze_decoder_config)

        self.num_vision_tokens_each_frame = num_vision_tokens_each_frame

        super().__init__(**kwargs)


class AutoGazeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AutoGaze`] model. It is used to instantiate an
    AutoGaze model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        gazing_ratio_config (`dict`, *optional*):
            Configuration for sampling gazing ratio during training and inference.
        scales (`str` or `int`, *optional*, defaults to `"224"`):
            Scales for the vision model. Can be a single scale or multiple scales separated by '+'.
        num_vision_tokens_each_frame (`int`, *optional*, defaults to `196`):
            Number of vision tokens per frame.
        gaze_model_config (`GazeModelConfig` or `dict`, *optional*):
            Configuration for the gaze model, including vision_model_config, connector_config, and gaze_decoder_config.
        gazing_ratio_each_frame_config (`dict`, *optional*):
            Configuration for sampling gazing ratio for each frame.
        has_task_loss_requirement_during_training (`bool`, *optional*, defaults to `False`):
            Whether to use task loss requirement during training.
        has_task_loss_requirement_during_inference (`bool`, *optional*, defaults to `False`):
            Whether to use task loss requirement during inference.
        task_loss_requirement_config (`dict`, *optional*):
            Configuration for task loss requirement sampling.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention.
        max_batch_size (`int`, *optional*):
            Maximum batch size.

    ```python
    >>> from autogaze.models.autogaze import AutoGaze, AutoGazeConfig

    >>> # Initializing an AutoGaze configuration
    >>> configuration = AutoGazeConfig()

    >>> # Initializing a model from the configuration
    >>> model = AutoGaze(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "autogaze"

    def __init__(
        self,
        gazing_ratio_config=None,
        scales="224",
        num_vision_tokens_each_frame=196,
        gaze_model_config={},
        gazing_ratio_each_frame_config=None,
        has_task_loss_requirement_during_training=False,
        has_task_loss_requirement_during_inference=False,
        task_loss_requirement_config=None,
        use_flash_attn=True,
        max_batch_size=None,
        **kwargs,
    ):
        self.gazing_ratio_config = gazing_ratio_config or {
            "sample_strategy_during_training": "fixed",
            "sample_strategy_during_inference": "fixed",
            "fixed": {"gazing_ratio": 0.5},
            "uniform": {"gazing_ratio_min": 0, "gazing_ratio_max": 1},
            "exponential": {"gazing_ratio_min": 0, "gazing_ratio_max": 1, "lambda": 10},
        }
        self.scales = scales
        self.num_vision_tokens_each_frame = num_vision_tokens_each_frame
        self.attn_mode = "flash_attention_2" if use_flash_attn else "sdpa"

        gaze_model_config.update({
            "num_vision_tokens_each_frame": num_vision_tokens_each_frame,
            "attn_mode": self.attn_mode,
        })
        self.gaze_model_config = GazeModelConfig(**gaze_model_config)

        self.gazing_ratio_each_frame_config = gazing_ratio_each_frame_config or {
            "sample_strategy_during_training": "uniform",
            "sample_strategy_during_inference": "uniform",
            "uniform": {},
            "dirichlet": {"alpha": 0.5},
            "self": {},
        }
        self.has_task_loss_requirement_during_training = has_task_loss_requirement_during_training
        self.has_task_loss_requirement_during_inference = has_task_loss_requirement_during_inference
        self.task_loss_requirement_config = task_loss_requirement_config or {
            "sample_strategy_during_training": "fixed",
            "sample_strategy_during_inference": "fixed",
            "fixed": {"task_loss_requirement": 0.7},
            "uniform": {"task_loss_requirement_min": 0.6, "task_loss_requirement_max": 0.9},
        }
        self.use_flash_attn = use_flash_attn
        self.max_batch_size = max_batch_size
        
        super().__init__(**kwargs)


__all__ = [
    "AutoGazeConfig",
    "GazeModelConfig",
    "VisionModelConfig",
    "ConnectorConfig",
    "GazeDecoderConfig",
]

