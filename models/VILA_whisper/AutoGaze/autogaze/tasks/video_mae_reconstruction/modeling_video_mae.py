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
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT MAE (masked autoencoder) model."""

import collections.abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from .configuration_video_mae import ViTMAEConfig


logger = logging.get_logger(__name__)


@dataclass
class ViTMAEModelOutput(ModelOutput):
    """
    Class for ViTMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """
    Class for ViTMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    num_decoded_tokens_each_frame: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss_each_reconstruction_frame (`torch.FloatTensor` of shape `(batch_size, num_selected_frames)`):
            Pixel reconstruction loss for each reconstruction frame.
        loss_mean (`torch.FloatTensor` of shape `(1,)`):
            Mean of the pixel reconstruction loss for each reconstruction frame.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss_each_reconstruction_frame: Optional[torch.FloatTensor] = None
    loss_mean: Optional[torch.FloatTensor] = None
    reconstruction: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.patch_size = config.patch_size
        self.config = config

        # multi-scale setting
        self.scales = sorted([int(scale) for scale in config.scales.split('+')])
        self.num_patch_each_scale = [(scale // config.patch_size)**2 for scale in self.scales]
        if config.scale_embed:
            self.scale_embed = nn.Parameter(torch.randn(len(self.scales), config.hidden_size) * 0)

        # time embedding
        if config.time_embed:
            self.time_embed = nn.Parameter(torch.randn(config.max_num_frames, config.hidden_size) * 0)

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

        # initialize scale embed
        if self.config.scale_embed:
            torch.nn.init.normal_(self.scale_embed, std=self.config.initializer_range)
        
        # initialize time embed
        if self.config.time_embed:
            torch.nn.init.normal_(self.time_embed, std=self.config.initializer_range)

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

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

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

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
        num_gazing_each_frame = gazing_info['num_gazing_each_frame'].clone()
        if_padded_gazing = gazing_info['if_padded_gazing'].clone()

        B, seq_length, dim = sequence.shape
        gaze_length = gazing_pos.shape[1]
        assert gaze_length == num_gazing_each_frame.sum()

        # Record the original sequence length into gazing_info
        gazing_info['original_seq_length'] = seq_length

        # Pad the sequence with an additional token for padded gazing to select
        sequence = torch.cat([sequence, sequence[:, :1]], dim=1)

        # Change all the padded gazing id to the last token id
        gazing_pos = gazing_pos.flatten()
        gazing_pos[if_padded_gazing.flatten()] = seq_length
        gazing_pos = gazing_pos.view(B, -1)

        # Get the unmasked part of the sequence for MAE encoding
        sequence_unmasked = sequence[torch.arange(B)[..., None], gazing_pos]

        return sequence_unmasked

    def forward(self, pixel_values, gazing_info=None, noise=None, interpolate_pos_encoding: bool = False):
        """
        pixel_values: (B, T, C, H, W)
        """
        B, T = pixel_values.shape[:2]
        pixel_values = rearrange(pixel_values, 'b t c h w -> (b t) c h w')
        
        embeddings = []
        for i, scale in enumerate(self.scales):
            pixel_values_cur_scale = F.interpolate(pixel_values, size=(scale, scale), mode="bicubic", align_corners=False)
            embeddings_cur_scale = self.patch_embeddings(pixel_values_cur_scale, interpolate_pos_encoding=interpolate_pos_encoding)
            if interpolate_pos_encoding:
                position_embeddings_cur_scale = self.interpolate_pos_encoding(embeddings_cur_scale, scale, scale)
            else:
                position_embeddings_cur_scale = self.position_embeddings

            # add position embeddings w/o cls token
            embeddings_cur_scale = embeddings_cur_scale + position_embeddings_cur_scale[:, 1:, :]

            # add scale embedding
            if self.config.scale_embed:
                scale_embeddings_cur_scale = self.scale_embed[i][None, None]
                embeddings_cur_scale = embeddings_cur_scale + scale_embeddings_cur_scale
                    
            embeddings.append(embeddings_cur_scale)
        embeddings = torch.cat(embeddings, dim=1)  # (B * T) * N * C

        # add time embedding
        embeddings = rearrange(embeddings, '(b t) n c -> b t n c', b=B, t=T)  # B * T * N * C
        if self.config.time_embed:
            time_embeddings = self.time_embed[None, :T, None, :]  # 1 * T * 1 * C
            embeddings = embeddings + time_embeddings
        
        embeddings = rearrange(embeddings, 'b t n c -> b (t n) c')  # B * (T * N) * C

        # masking: length -> length * config.mask_ratio
        embeddings = self.mask_with_gazing(embeddings, gazing_info)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
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
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTMAE
class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                assert False, "SDPA doesn't support output_attentions=True. If falling back to eager, please change the attention mask implementation."
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTMAE
class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTMAE
class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTMAE
class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->ViTMAE
class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTMAE,VIT->VITMAE
class ViTMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ViTMAE
class ViTMAEEncoder(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTMAEEmbeddings):
            module.initialize_weights()
        elif isinstance(module, ViTMAEDecoder):
            module.mask_token.data.zero_()
            module.decoder_pos_embed.data.zero_()
            if self.config.scale_embed:
                torch.nn.init.normal_(module.decoder_scale_embed, std=self.config.initializer_range)
            if self.config.time_embed:
                torch.nn.init.normal_(module.time_embed, std=self.config.initializer_range)


class ViTMAEModel(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        gazing_info: Optional[dict] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = head_mask.to(self.dtype)

        embedding_output = self.embeddings(
            pixel_values, gazing_info=gazing_info, noise=noise, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return sequence_output + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTMAEDecoder(ViTMAEPreTrainedModel):
    def __init__(self, config, num_patches):
        super().__init__(config)
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config

        # multi-scale setting
        self.scales = sorted([int(scale) for scale in config.scales.split('+')])
        self.num_patch_each_frame_each_scale = [(scale // config.patch_size)**2 for scale in self.scales]
        if self.config.scale_embed:
            self.decoder_scale_embed = nn.Parameter(torch.randn(len(self.scales), config.decoder_hidden_size) * 0)
        
        # time embed
        if self.config.time_embed:
            self.time_embed = nn.Parameter(torch.randn(config.max_num_frames, config.decoder_hidden_size) * 0)
        
        self.num_token_each_frame = sum(self.num_patch_each_frame_each_scale)

        self.initialize_weights(num_patches)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        This method is a modified version of the interpolation function for ViT-mae model at the decoder, that
        allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
        resolution images.

        Adapted from:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # -1 removes the class dimension since we later append it without interpolation
        embeddings_positions = embeddings.shape[1] - 1

        # Separation of class token and patch tokens
        class_pos_embed = self.decoder_pos_embed[:, :1]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]

        # To retain the final 3d tensor with the required dimensions
        dim = self.decoder_pos_embed.shape[-1]

        # Increasing a dimension to enable bicubic interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)

        # permute to bring the dimension to be interpolated, to the last
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolating the decoder position embeddings shape wrt embeddings shape i.e (x).
        # we keep the second last dimension constant
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(patch_pos_embed.shape[-2], embeddings_positions),
            mode="bicubic",
            align_corners=False,
        )

        # Converting back to the original shape
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # Adding the class token back
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        gazing_info=None,
        frame_idx_to_reconstruct=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
    ):
        gazing_pos = gazing_info['gazing_pos']
        num_gazing_each_frame = gazing_info['num_gazing_each_frame']
        if_padded_gazing = gazing_info['if_padded_gazing']
        original_seq_length = gazing_info['original_seq_length']

        B = hidden_states.shape[0]
        gaze_length = gazing_pos.shape[1]
        assert gaze_length == num_gazing_each_frame.sum()
        T = len(num_gazing_each_frame)
        original_seq_length_each_frame = original_seq_length // T

        # embed tokens
        x = self.decoder_embed(hidden_states)

        # Take out cls token
        x_ = x[:, 1:, :]
        cls_token = x[:, :1, :]

        # Change all the padded gazing id to the last token id
        gazing_pos = gazing_pos.flatten()
        gazing_pos[if_padded_gazing.flatten()] = original_seq_length
        gazing_pos = gazing_pos.view(B, -1)

        # add mask tokens back to the sequence (temporarily append an additional token for padded gazing to select)
        full_seq = self.mask_token.repeat(x.shape[0], original_seq_length + 1, 1).to(x.dtype)
        full_seq[torch.arange(B)[..., None], gazing_pos] = x_
        full_seq = full_seq[:, :-1, :]

        # add pos embed and scale embed
        full_seq = rearrange(full_seq, 'b (t n) c -> (b t) n c', t=T)
        decoder_pos_embed = []
        decoder_scale_embed = []
        for i, scale in enumerate(self.scales):
            x_cur_scale = full_seq[:, sum(self.num_patch_each_frame_each_scale[:i]):sum(self.num_patch_each_frame_each_scale[:i+1])]
            if interpolate_pos_encoding:
                decoder_pos_embed_cur_scale = self.interpolate_pos_encoding(F.pad(x_cur_scale, (0, 0, 1, 0)))[:, 1:]
            else:
                decoder_pos_embed_cur_scale = self.decoder_pos_embed
            decoder_pos_embed.append(decoder_pos_embed_cur_scale)
            if self.config.scale_embed:
                decoder_scale_embed.append(self.decoder_scale_embed[i][None, None].repeat(1, decoder_pos_embed_cur_scale.shape[1], 1))
        decoder_pos_embed = torch.cat(decoder_pos_embed, dim=1)
        decoder_scale_embed = torch.cat(decoder_scale_embed, dim=1) if self.config.scale_embed else 0
        full_seq = full_seq + decoder_pos_embed + decoder_scale_embed
        full_seq = rearrange(full_seq, '(b t) n c -> b (t n) c', t=T)

        # add time embed
        if self.config.time_embed:
            time_embed = self.time_embed[None, :T, None, :]
            full_seq = rearrange(full_seq, 'b (t n) c -> b t n c', t=T)
            full_seq = full_seq + time_embed
            full_seq = rearrange(full_seq, 'b t n c -> b (t n) c', t=T)
        
        # Get the index of tokens to feed into decoder (encoded tokens + mask tokens for selected frames)
        idx_to_decode = gazing_pos.clone()
        idx_to_decode = list(idx_to_decode.split(num_gazing_each_frame.tolist(), dim=-1))
        for frame_idx in frame_idx_to_reconstruct:
            idx_to_decode[frame_idx] = torch.arange(original_seq_length_each_frame, device=gazing_pos.device)[None].repeat(B, 1) + original_seq_length_each_frame * frame_idx
        idx_to_decode = torch.cat(idx_to_decode, dim=-1)

        # Get the tokens to decode
        full_seq = torch.cat([full_seq, full_seq[:, :1]], dim=1)
        hidden_states = full_seq[torch.arange(B)[..., None], idx_to_decode]

        # add cls token
        cls_token = cls_token + self.decoder_pos_embed[:, :1]
        hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        # apply Transformer layers (blocks)
        head_mask = head_mask.to(self.dtype)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=head_mask, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTMAEForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = ViTMAEModel(config)
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # multi-scale setting
        self.scales = sorted([int(scale) for scale in config.scales.split('+')])
        self.num_patch_each_scale = [(scale // config.patch_size)**2 for scale in self.scales]
        self.num_token_each_frame = sum(self.num_patch_each_scale)

        # loss setting
        self.loss_type = [str(loss) for loss in config.loss_type.split('+')]
        self.loss_weights = [float(weight) for weight in config.loss_weights.split('+')]
        self.transform = None  # will be initialized in the outer
        self.loss_fns = []
        for loss in self.loss_type:
            if loss == 'l1':
                self.loss_fns.append(self.l1_loss)
            elif loss == 'dinov2_reg':
                self.dinov2_reg = None  # will be initialized in the outer
                self.dinov2_reg_transform = None  # will be initialized in the outer
                self.loss_fns.append(self.dinov2_reg_loss)
            elif loss == 'siglip2':
                self.siglip2 = None  # will be initialized in the outer
                self.siglip2_transform = None  # will be initialized in the outer
                self.loss_fns.append(self.siglip2_loss)
            else:
                raise ValueError(f"Loss type {loss} not supported")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if not interpolate_pos_encoding and (
            pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0
        ):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_h = pixel_values.shape[2] // patch_size
        num_patches_w = pixel_values.shape[3] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`Tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values
    
    def retransform(self, image, source_transform, target_transform):
        # Revert the source transform
        image = rearrange(image, 'b c h w -> b h w c')
        if source_transform.do_normalize:
            image = image * torch.tensor(source_transform.image_std, device=image.device, dtype=image.dtype) + torch.tensor(source_transform.image_mean, device=image.device, dtype=image.dtype)
        if source_transform.do_rescale:
            if hasattr(source_transform, 'offset') and source_transform.offset:
                image = image + 1
            image = image / source_transform.rescale_factor
        image = rearrange(image, 'b h w c -> b c h w')
        
        # Apply the target transform
        image = rearrange(image, 'b c h w -> b h w c')
        if target_transform.do_rescale:
            image = image * target_transform.rescale_factor
            if hasattr(target_transform, 'offset') and target_transform.offset:
                image = image - 1
        if target_transform.do_normalize:
            image = (image - torch.tensor(target_transform.image_mean, device=image.device, dtype=image.dtype)) / torch.tensor(target_transform.image_std, device=image.device, dtype=image.dtype)
        image = rearrange(image, 'b h w c -> b c h w')

        return image

    def l1_loss(self, pred, target):
        """
        pred, target: (B, C, H, W)
        """
        return (pred - target).abs().mean(dim=(-1, -2, -3))

    def dinov2_reg_loss(self, pred, target):
        """
        pred, target: (B, C, H, W)
        """
        def get_dinov2_reg_features(image):
            image = self.retransform(image, self.transform, self.dinov2_reg_transform)
            features = self.dinov2_reg(image, output_hidden_states=True).hidden_states
            features = torch.cat([feature[:, self.dinov2_reg.config.num_register_tokens + 1:] for feature in features[-4:]], dim=-1)
            return features

        pred_features = get_dinov2_reg_features(pred)
        target_features = get_dinov2_reg_features(target)

        # Get average l2 loss over last k layers' features
        loss = (pred_features - target_features).pow(2).mean(dim=(-1, -2))

        return loss

    def siglip2_loss(self, pred, target):
        """
        pred, target: (B, C, H, W)
        """
        def get_siglip2_features(image):
            image = self.retransform(image, self.transform, self.siglip2_transform)
            features = self.siglip2(image, output_hidden_states=True).hidden_states
            features = torch.cat(features[-4:], dim=-1)
            return features
        
        pred_features = get_siglip2_features(pred)
        target_features = get_siglip2_features(target)

        # Get average l2 loss over last k layers' features
        loss = (pred_features - target_features).pow(2).mean(dim=(-1, -2))

        return loss

    def forward_loss(self, pixel_values, pred, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, T, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, T, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        B, T = pixel_values.shape[:2]
        pixel_values = pixel_values.flatten(0, 1)  # (B * T), C, H, W
        pred = pred.flatten(0, 1)  # (B * T), N, C
        
        pred = self.unpatchify(pred, original_image_size=(pixel_values.shape[2], pixel_values.shape[3]))

        loss = 0
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss += loss_weight * loss_fn(pred, pixel_values)
        
        loss = rearrange(loss, '(b t) -> b t', b=B, t=T)
        mean_loss = loss.mean(dim=-1)

        return loss, mean_loss

    def get_reconstructed_image(self, pixel_values, pred, interpolate_pos_encoding: bool = False):
        """
        pixel_values: (B, T, C, H, W)
        pred: (B, T, N, C)
        """
        B, T = pixel_values.shape[:2]
        pixel_values = pixel_values.flatten(0, 1)  # (B * T), C, H, W
        pred = pred.flatten(0, 1)  # (B * T), N, C

        pred = self.unpatchify(pred, original_image_size=(pixel_values.shape[2], pixel_values.shape[3]))

        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=B, t=T)

        return pred

    def get_causal_mask(self, num_tokens_each_frame, num_layers, batch_size, num_heads, token_mask=None, cls_token=True):
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

        # Create a causal mask
        mask = torch.tril(torch.ones(batch_size, N, N, device=device))

        # Make the tokens inside each frame attend to each other
        for t in range(T):
            mask[:, sum(num_tokens_each_frame[:t]):sum(num_tokens_each_frame[:t+1]), sum(num_tokens_each_frame[:t]):sum(num_tokens_each_frame[:t+1])] = 1

        # Mask out tokens indicated by token_mask
        if token_mask is not None:
            token_mask = token_mask.unsqueeze(1).repeat(1, N, 1)
            mask = mask * (~token_mask).float()

        # Add mask for cls token
        if cls_token:
            mask_ = mask.clone()
            mask = torch.tril(torch.ones(batch_size, N + 1, N + 1, device=device))
            mask[:, 1:, 1:] = mask_

        # Each token must be able to attend to itself
        mask[:, torch.arange(N), torch.arange(N)] = 1
        
        # According to different attention implementations, the mask values are different.
        if self.config._attn_implementation == "flex_attention" or self.config._attn_implementation == "sdpa":
            # mask is a float tensor that will be added to the attention scores. This means the tokens to be attended should have mask value of 0, and the rest should have mask value of -inf.
            mask = torch.where(mask == 1, 0, -torch.inf)
        elif self.config._attn_implementation == "flash_attention_2":
            raise NotImplementedError("Flash attention 2 doesn't support custom attention mask. Please use attention_implementation='flex_attention'.")
        elif self.config._attn_implementation == "eager":
            # mask is a float tensor that will be multiplied to the attn prob after softmax. This means the tokens to be attended should have mask value of 1, and the rest should have mask value of 0.
            pass
        
        mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        return mask.to(num_tokens_each_frame.device)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        gazing_info: Optional[dict] = None,
        frame_idx_to_reconstruct: Optional[torch.LongTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        """
        pixel_values: (B, T, C, H, W)
        gazing_info:
            gazing_pos: The gazing positions of each whole sequence. (B, N)
            num_gazing_each_frame: The number of gazing positions for each frame, including the padded gazing. (T, )
            if_padded_gazing: Whether the gazing is padded. (B, N)
        frame_idx_to_reconstruct: (num_selected_frames, )
        """
        B, T = pixel_values.shape[:2]

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the encoder attention mask
        encoder_attn_mask = self.get_causal_mask(gazing_info['num_gazing_each_frame'], self.config.num_hidden_layers, B, self.config.num_attention_heads, token_mask=gazing_info['if_padded_gazing'], cls_token=True) if self.config.causal else None

        # Get the encoder outputs
        outputs = self.vit(
            pixel_values,
            gazing_info=gazing_info,
            noise=noise,
            head_mask=encoder_attn_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        latent = outputs.last_hidden_state  # B * N * C

        # Get the number of tokens to decode for each frame
        num_decoded_tokens_each_frame = gazing_info['num_gazing_each_frame'].clone()
        num_decoded_tokens_each_frame[frame_idx_to_reconstruct] = self.num_token_each_frame

        # Get the gazing padding mask for decoder
        if_padded_gazing_decoder = gazing_info['if_padded_gazing'].clone()
        if_padded_gazing_decoder = list(if_padded_gazing_decoder.split(gazing_info['num_gazing_each_frame'].tolist(), dim=-1))
        for frame_idx in frame_idx_to_reconstruct:
            if_padded_gazing_decoder[frame_idx] = torch.zeros(B, self.num_token_each_frame).to(gazing_info['if_padded_gazing'].device).to(torch.bool)
        if_padded_gazing_decoder = torch.cat(if_padded_gazing_decoder, dim=-1)

        # Get the decoder attention mask
        decoder_attn_mask = self.get_causal_mask(num_decoded_tokens_each_frame, self.config.decoder_num_hidden_layers, B, self.config.decoder_num_attention_heads, token_mask=if_padded_gazing_decoder, cls_token=True) if self.config.causal else None

        # Get the decoder outputs
        decoder_outputs = self.decoder(
            latent, 
            gazing_info=gazing_info,
            frame_idx_to_reconstruct=frame_idx_to_reconstruct, 
            head_mask=decoder_attn_mask, 
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        # Only keep the predictions for the selected frames
        decoded_token_idx_to_keep = []
        for frame_idx in frame_idx_to_reconstruct:
            decoded_token_idx_to_keep.append(torch.arange(sum(num_decoded_tokens_each_frame[:frame_idx]), sum(num_decoded_tokens_each_frame[:frame_idx+1])))
        decoded_token_idx_to_keep = torch.cat(decoded_token_idx_to_keep, dim=0)
        logits = logits[:, decoded_token_idx_to_keep]
        logits = rearrange(logits, 'b (t n) c -> b t n c', t=len(frame_idx_to_reconstruct))  # B * num_selected_frames * N * C

        # throw away the reconstruction and masks for smaller scales
        logits = logits[:, :, sum(self.num_patch_each_scale[:-1]):, :]

        loss_each_reconstruction_frame, loss_mean = self.forward_loss(pixel_values[:, frame_idx_to_reconstruct], logits, interpolate_pos_encoding=interpolate_pos_encoding)
        reconstruction = self.get_reconstructed_image(pixel_values[:, frame_idx_to_reconstruct], logits, interpolate_pos_encoding=interpolate_pos_encoding)  # B * num_selected_frames * C * H * W

        if not return_dict:
            output = (logits, reconstruction) + outputs[2:]
            return ((loss_each_reconstruction_frame, loss_mean) + output) if loss_each_reconstruction_frame is not None else output

        return ViTMAEForPreTrainingOutput(
            loss_each_reconstruction_frame=loss_each_reconstruction_frame,
            loss_mean=loss_mean,
            reconstruction=reconstruction,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ViTMAEForPreTraining", "ViTMAELayer", "ViTMAEModel", "ViTMAEPreTrainedModel"]