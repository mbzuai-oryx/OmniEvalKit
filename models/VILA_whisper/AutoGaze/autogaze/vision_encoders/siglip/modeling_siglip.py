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
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Siglip model."""

from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
    torch_int,
)
from .configuration_siglip import SiglipVisionConfig, SiglipConfig
from transformers.models.siglip.modeling_siglip import _trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_, default_flax_embed_init


logger = logging.get_logger(__name__)



@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->Siglip
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # multi-scale setting
        self.scales = sorted([int(scale) for scale in config.scales.split('+')])
        self.num_patch_each_scale = [(scale // config.patch_size)**2 for scale in self.scales]

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = patch_pos_embed.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def mask_with_gazing(self, sequence, gazing_info):
        """
        Mask the sequence with the gazing information.
        For the padded gazing, we select a dummy token to fill in the positions (the dummy token is currently the first token in each sequence).
        
        Args:
            sequence: The sequence to mask.
            gazing_info:
                gazing_pos: The gazing positions of each whole sequence. (B, N)
                num_gazing_each_frame: The number of gazing positions for each frame, including the padded gazing. (T, )
                if_padded_gazing: Whether the gazing is padded. (B, N)
        """
        gazing_pos = gazing_info['gazing_pos'].clone()
        if_padded_gazing = gazing_info['if_padded_gazing'].clone()

        B = sequence.shape[0]

        # Map padded gazing positions to a dummy token
        gazing_pos[if_padded_gazing] = 0

        # Gather only the gazed tokens
        sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
        return sequence_gazed
    
    def get_gazed_patches_and_pos_embeddings(self, pixel_values: torch.FloatTensor, gazing_info=None, interpolate_pos_encoding: bool = False):
        """
        pixel_values: (B, T, C, H, W)
        """
        B, T = pixel_values.shape[:2]
        pixel_values = rearrange(pixel_values, 'b t c h w -> (b t) c h w')
        out_channels, _, kh, kw = self.patch_embedding.weight.shape

        # Get the raw patches and position embeddings at each scale
        patches = []
        pos_embeddings = []
        for i, scale in enumerate(self.scales):
            # interpolate the pixel values to the current scale
            patches_cur_scale = F.interpolate(pixel_values, size=(scale, scale), mode="bicubic", align_corners=False)

            # Get the current patches
            patches_cur_scale = F.unfold(patches_cur_scale, kernel_size=(kh, kw), stride=(kh, kw)).transpose(-1, -2)   # (B * T) * N * (in_channels * kh * kw)
            patches.append(patches_cur_scale)

            # Get the current position embeddings
            pos_embeddings_cur_scale = self.interpolate_pos_encoding(patches_cur_scale, scale, scale).repeat(patches_cur_scale.shape[0], 1, 1)  # (B * T) * N * dim
            pos_embeddings.append(pos_embeddings_cur_scale)

        # Free memory for unused variables (useful when video is really large)
        del(pixel_values)
        del(patches_cur_scale)
        del(pos_embeddings_cur_scale)
            
        patches = torch.cat(patches, dim=1)  # (B * T) * N * (in_channels * kh * kw)
        pos_embeddings = torch.cat(pos_embeddings, dim=1)  # (B * T) * N * dim
        
        # Reshape everything
        patches = rearrange(patches, '(b t) n c -> b (t n) c', b=B, t=T)
        pos_embeddings = rearrange(pos_embeddings, '(b t) n c -> b (t n) c', b=B, t=T)

        # Select the patches and position embeddings that are gazed
        patches = self.mask_with_gazing(patches, gazing_info)
        pos_embeddings = self.mask_with_gazing(pos_embeddings, gazing_info)
        
        to_return = {
            'patches': patches,
            'pos_embeddings': pos_embeddings,
        }
            
        return to_return

    def forward(self, pixel_values: torch.FloatTensor, gazing_info=None, interpolate_pos_encoding: bool = False):
        B, T = pixel_values.shape[:2]

        # Get the gazed patches and other embeddings
        if self.config.max_embed_batch_size is None:
            all_returns = self.get_gazed_patches_and_pos_embeddings(pixel_values, gazing_info, interpolate_pos_encoding)
        else:
            # Embed videos in mini-batches to save memory
            all_returns = []
            for i in range(0, B, self.config.max_embed_batch_size):
                cur_pixel_values = pixel_values[i:min(i+self.config.max_embed_batch_size, B)]
                cur_gazing_info = deepcopy(gazing_info)
                cur_gazing_info['gazing_pos'] = cur_gazing_info['gazing_pos'][i:min(i+self.config.max_embed_batch_size, B)]
                cur_gazing_info['if_padded_gazing'] = cur_gazing_info['if_padded_gazing'][i:min(i+self.config.max_embed_batch_size, B)]
                cur_return = self.get_gazed_patches_and_pos_embeddings(cur_pixel_values, cur_gazing_info, interpolate_pos_encoding)
                all_returns.append(cur_return)
            all_returns = {k: torch.cat([cur_return[k] for cur_return in all_returns], dim=0) for k in all_returns[0].keys()}
        patches = all_returns['patches']
        pos_embeddings = all_returns['pos_embeddings']

        # Embed the patches
        out_channels, _, kh, kw = self.patch_embedding.weight.shape
        weight = self.patch_embedding.weight   # out_channels * in_channels * kh * kw
        bias = self.patch_embedding.bias if self.patch_embedding.bias is not None else None   # out_channels
        reshaped_weight = weight.view(out_channels, -1)   # out_channels * (in_channels * kh * kw)
        embeddings = F.linear(patches, reshaped_weight, bias)

        # Add position embeddings
        embeddings = embeddings + pos_embeddings

        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = self.config._attn_implementation == 'flash_attention_2' and self.config.attn_type == 'causal'

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SiglipConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "SiglipTextEmbeddings",
        "SiglipEncoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SiglipVisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, SiglipConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoder with AltCLIP->Siglip
class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)
    
    def get_causal_mask(self, num_tokens_each_frame, num_layers, batch_size, num_heads, token_mask=None, cls_token=True, frame_independent_encoding=False, dtype=torch.float32):
        """
        Assume a input of shape B * N * C, where N contains tokens from several frames.
        Each frame has num_tokens_each_frame[t] tokens.
        Create a block-causal attention mask such that each token can only attend to tokens from either previous frames or the same frame.
        Additionally, mask any tokens indicated by token_mask (e.g., the tokens at padded gazing positions)

        Inputs:
            num_tokens_each_frame: (T)
            token_mask: (B, N)
            cls_token: whether to include the cls token in the mask
        Return:
            mask: batch x num_heads x seq_length x seq_length
        """
        T = len(num_tokens_each_frame)
        N = num_tokens_each_frame.sum()
        device = num_tokens_each_frame.device

        # If we are using flash-attention 2, then directly create a mask of size `(batch_size, seq_len)` where 0 stands for the position of padding tokens and 1 for the position of non-padding tokens.
        # We assume all padded tokens will be put to the end of each sequence.
        if self.config._attn_implementation == 'flash_attention_2':
            assert not frame_independent_encoding, "Frame-independent encoding is not supported for flash-attention 2."
            assert self.config.attn_type in ['causal', 'bidirectional'], "Flash-attention 2 only supports causal or bidirectional attention."
            mask = token_mask.to(torch.int).sort(dim=-1)[0]
            mask = 1 - mask
            if self.config.attn_type == 'causal':
                return None
            else:
                return mask
                # raise NotImplementedError("Currently if we pass the mask to flash-attention, it will raise \"illegal memory access\" error.")

        assert self.config.attn_type == 'block_causal', "Currently only block-causal attention is supported for other attention implementations except for flash-attention 2."

        # Optionally create a causal mask
        if not frame_independent_encoding:
            mask = torch.tril(torch.ones(batch_size, N, N, device=device, dtype=dtype))
        else:
            mask = torch.zeros(batch_size, N, N, device=device, dtype=dtype)

        # Make the tokens inside each frame attend to each other
        for t in range(T):
            mask[:, sum(num_tokens_each_frame[:t]):sum(num_tokens_each_frame[:t+1]), sum(num_tokens_each_frame[:t]):sum(num_tokens_each_frame[:t+1])] = 1

        # Mask out tokens indicated by token_mask
        if token_mask is not None:
            token_mask = token_mask.unsqueeze(1).repeat(1, N, 1)
            mask = mask * (~token_mask).to(dtype)

        # Add mask for cls token
        if cls_token:
            mask_ = mask.clone()
            mask = torch.tril(torch.ones(batch_size, N + 1, N + 1, device=device, dtype=dtype))
            mask[:, 1:, 1:] = mask_

        # Each token must be able to attend to itself
        mask[:, torch.arange(N), torch.arange(N)] = 1
        
        # According to different attention implementations, the mask values are different.
        if self.config._attn_implementation == "flex_attention" or self.config._attn_implementation == "sdpa" or self.config._attn_implementation == "eager":
            # mask is a float tensor that will be added to the attention scores. This means the tokens to be attended should have mask value of 0, and the rest should have mask value of -inf.
            mask = torch.where(mask == 1, 0, -torch.inf).to(dtype)
        else:
            raise NotImplementedError(f"Attention type {self.config.attn_type} is not supported.")
        
        mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        return mask.to(num_tokens_each_frame.device)

    def forward(
        self,
        pixel_values,
        gazing_info: Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        """
        pixel_values: (B, T, C, H, W)
        gazing_info:
            gazing_pos: The gazing positions of each whole sequence. (B, N)
            num_gazing_each_frame: The number of gazing positions for each frame, including the padded gazing. (T, )
            if_padded_gazing: Whether the gazing is padded. (B, N)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        B, T = pixel_values.shape[:2]
        hidden_states = self.embeddings(pixel_values, gazing_info=gazing_info, interpolate_pos_encoding=interpolate_pos_encoding)
        
        # Get the encoder attention mask
        encoder_attn_mask = self.get_causal_mask(
            gazing_info['num_gazing_each_frame'], 
            self.config.num_hidden_layers, 
            B, 
            self.config.num_attention_heads, 
            token_mask=gazing_info['if_padded_gazing'], 
            cls_token=False, 
            frame_independent_encoding=self.config.frame_independent_encoding,
            dtype=pixel_values.dtype,
        )
        
        # If we are using flash-attention 2, then we need to put all the padded tokens to the end of each sequence.
        if self.config._attn_implementation == 'flash_attention_2':
            _, indices = gazing_info['if_padded_gazing'].to(torch.int).sort(dim=1, stable=True)
            hidden_states = hidden_states[torch.arange(B)[..., None], indices]

        # Encoding
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attn_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # If we are using flash-attention 2, then we need to put all the padded tokens back to the original positions.
        if self.config._attn_implementation == 'flash_attention_2':
            inv_indices = indices.argsort(dim=1)
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[torch.arange(B)[..., None], inv_indices]
            if encoder_outputs.hidden_states is not None:
                encoder_outputs.hidden_states = [encoder_outputs.hidden_states[layer][torch.arange(B)[..., None], inv_indices] for layer in range(len(encoder_outputs.hidden_states))]

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.scales = sorted([int(scale) for scale in config.scales.split('+')])
        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        gazing_info: Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            gazing_info=gazing_info,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


__all__ = [
    "SiglipPreTrainedModel",
    "SiglipVisionModel",
]