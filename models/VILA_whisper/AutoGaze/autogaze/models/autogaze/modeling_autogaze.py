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

from copy import deepcopy
from typing import Optional, Tuple
from dataclasses import dataclass
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import ConvNeXtBlock
from timm.layers import LayerNorm2d

from transformers.modeling_outputs import ModelOutput
from transformers import LogitsProcessor, LogitsProcessorList

from .configuration_autogaze import GazeModelConfig, VisionModelConfig, ConnectorConfig
from .modeling_llama_multi_token_pred import LlamaForCausalLM_MultiTokenPred


@dataclass
class AutoGazeOutput(ModelOutput):
    gaze_logits: Optional[torch.FloatTensor] = None
    gaze_probs: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    task_loss_prediction: Optional[torch.FloatTensor] = None


class NoRepeatTokensLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch_size, sequence_length)
        # scores: (batch_size, vocab_size) or (batch_size, num_multi_token_pred, vocab_size)
        if scores.ndim == 3:
            scores[torch.arange(scores.shape[0])[..., None], :, input_ids] = -float("inf")
        else:
            scores[torch.arange(scores.shape[0])[..., None], input_ids] = -float("inf")
        return scores


class NoEosTokenLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch_size, sequence_length)
        # scores: (batch_size, vocab_size) or (batch_size, num_multi_token_pred, vocab_size)
        scores[..., -1] = -float("inf")
        return scores


class AutoGazeModel(nn.Module):
    def __init__(self, gaze_model_config: GazeModelConfig):
        super().__init__()

        self.num_vision_tokens_each_frame = gaze_model_config.num_vision_tokens_each_frame
        self.input_img_size = gaze_model_config.input_img_size
        self.frame_sampling_rate = gaze_model_config.vision_model_config.temporal_patch_size
        self.num_multi_token_pred = gaze_model_config.gaze_decoder_config.num_multi_token_pred
        self.gaze_decoder_config = gaze_model_config.gaze_decoder_config  # Store for reference

        # Create the vision model, connector, and gaze decoder
        self.vision_model = ShallowVideoConvNet(gaze_model_config.vision_model_config)
        self.connector = Connector(gaze_model_config.connector_config)
        self.gaze_decoder = LlamaForCausalLM_MultiTokenPred(gaze_model_config.gaze_decoder_config)

        # Add logits processors to prevent the model from repeating the same token and generating eos token during gazing.
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(NoRepeatTokensLogitsProcessor())  # don't allow repeated gazing
        self.logits_processor.append(NoEosTokenLogitsProcessor())  # don't allow generating eos token duing gazing

    def embed(self, video=None, gaze_pos_ids=None, use_cache=False, past_conv_values=None):
        """
        inputs:
            video: (B x T x C x H x W).
            gaze_pos_ids: list of (B, N), N is the number of gazing positions in each frame. The length of the list is T // frame_sampling_rate.
        returns:
            embeds: a list of interleaved vision and gaze embeddings. list of (B, N, C)
            gaze_token_mask: a list of masks that indicate if the current embedding is a gaze embedding. (1 is gaze embedding, 0 is vision embedding). list of (N, )
            gaze_pred_source_relative: a list of (relative) source index of where the gaze prediction is coming from. For example, if the gaze prediction is coming from two tokens before it, the source index is -2. list of (N, ).
                For vision embeddings, there's no source prediction, so the source index is -1.
            attention_mask: a list of (B, N) that indicates if the current embedding should be masked out (for EOS token). 1 is not masked, 0 is masked.
        """
        B, T = video.shape[:2]
        assert (video is None or gaze_pos_ids is None) or video.shape[1] // self.frame_sampling_rate == len(gaze_pos_ids), \
            "The number of frames in the video (after subsampling) and in gaze position IDs must be the same, but got {} and {}".format(video.shape[1] // self.frame_sampling_rate, len(gaze_pos_ids))

        if video is not None:
            vision_features, new_past_conv_values = self.vision_model(video, use_cache=use_cache, past_conv_values=past_conv_values)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = self.connector(vision_features)
            vision_attention_mask = [torch.ones(B, vision_features.shape[2], device=vision_features.device).long() for _ in range(vision_features.shape[1])]

        if gaze_pos_ids is not None:
            num_gazing_each_frame = [gaze_pos_ids[t].shape[1] for t in range(len(gaze_pos_ids))]
            gaze_pos_ids = torch.cat(gaze_pos_ids, dim=1)
            gaze_attention_mask = (gaze_pos_ids != self.gaze_decoder_config.eos_token_id).to(torch.long)
            gaze_embeds = self.gaze_decoder.model.embed_tokens(gaze_pos_ids)
            gaze_embeds = list(gaze_embeds.split(num_gazing_each_frame, dim=1))
            gaze_attention_mask = list(gaze_attention_mask.split(num_gazing_each_frame, dim=1))

        embeds = []
        gaze_token_mask = []
        gaze_pred_source_relative = []
        attention_mask = []
        for t in range(T // self.frame_sampling_rate):
            if video is not None:
                embeds.append(vision_features[:, t, :, :])
                gaze_token_mask.append(torch.zeros(vision_features.shape[2], device=vision_features.device).long())
                gaze_pred_source_relative.append(torch.zeros(vision_features.shape[2], device=vision_features.device).long() - 1)
                attention_mask.append(vision_attention_mask[t])
            if gaze_pos_ids is not None:
                embeds.append(gaze_embeds[t])
                gaze_token_mask.append(torch.ones(gaze_embeds[t].shape[1], device=gaze_embeds[t].device).long())
                gaze_pred_source_relative.append(-(torch.arange(gaze_embeds[t].shape[1], device=gaze_embeds[t].device) % self.num_multi_token_pred + 1))
                attention_mask.append(gaze_attention_mask[t])
        return embeds, gaze_token_mask, gaze_pred_source_relative, attention_mask, new_past_conv_values if video is not None else None

    @torch.no_grad()
    def generate(
        self, 
        video, 
        max_gaze_tokens_each_frame=100, 
        task_loss_requirement=None, 
        use_cache=False,
        past_key_values=None, 
        past_inputs_embeds=None,
        past_attention_mask=None,
        past_conv_values=None,
        **generation_kwargs,
    ):
        """
        Inputs:
            video: (B, T, C, H, W)
            max_gaze_tokens_each_frame: int or (T, ). Indicating the max gazing length for each frame. If is int, then all frames have the same max gazing length.
            task_loss_requirement (optional): (B, T). Indicating the task loss requirement for each frame.
            past_key_values (optional): The past key values for the gaze model. Can be used for streaming generation.
            past_inputs_embeds (optional): The past inputs embeds for the gaze model. Can be used for streaming generation.
            past_attention_mask (optional): The past attention mask for the gaze model. Can be used for streaming generation.
        """
        if past_key_values is not None or past_inputs_embeds is not None or past_attention_mask is not None or past_conv_values is not None:
            assert past_key_values is not None and past_inputs_embeds is not None and past_attention_mask is not None and past_conv_values is not None, \
                "If past_key_values, past_inputs_embeds, past_attention_mask, or past_conv_values is provided, then all four must be provided!"

        # Subsample frames and resize
        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size, self.input_img_size), mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        # Embed all the frames
        video_embeds, _, __, ___, past_conv_values = self.embed(video=video, use_cache=use_cache, past_conv_values=past_conv_values)

        # Generate gaze position IDs for each frame
        gaze_pos_ids_list = []
        inputs_embeds = [] if past_inputs_embeds is None else past_inputs_embeds
        attention_mask = [] if past_attention_mask is None else past_attention_mask
        past_key_values = None if past_key_values is None else past_key_values
        num_gazing_each_frame = []
        if_padded_gazing = []
        for t in range(len(video_embeds)):

            # Update inputs_embeds and attention mask for the new frame
            inputs_embeds.append(video_embeds[t])
            attention_mask.append(torch.ones(video_embeds[t].shape[0], video_embeds[t].shape[1], device=video_embeds[t].device).long())

            # Put task loss requirement into generation config
            generation_config = self.gaze_decoder.generation_config
            generation_config.task_loss_requirement = task_loss_requirement[:, t] if task_loss_requirement is not None else None

            # Get the max gazing length for the current frame
            assert isinstance(max_gaze_tokens_each_frame, int) or len(max_gaze_tokens_each_frame) == len(video_embeds), \
                "max_gaze_tokens_each_frame must be an int or a tensor of the same length as the video embeddings, but got {} and {}".format(max_gaze_tokens_each_frame, len(video_embeds))
            max_gaze_tokens = max_gaze_tokens_each_frame if isinstance(max_gaze_tokens_each_frame, int) else max_gaze_tokens_each_frame[t]

            # Generate gaze position IDs for the current frame
            is_gradient_checkpointing = self.gaze_decoder.is_gradient_checkpointing
            if is_gradient_checkpointing:
                self.gaze_decoder.gradient_checkpointing_disable()
            gaze_outputs = self.gaze_decoder.generate(
                inputs_embeds=torch.cat(inputs_embeds, dim=1),  # We need to pass the whole sequence of inputs_embeds (both current and past) to the model even when we use use_cache=True!!!
                attention_mask=torch.cat(attention_mask, dim=1),
                position_ids=torch.cat(attention_mask, dim=1).cumsum(dim=-1) - 1,
                max_new_tokens=max_gaze_tokens,
                logits_processor=self.logits_processor,
                pad_token_id=self.gaze_decoder_config.eos_token_id,
                eos_token_id=self.gaze_decoder_config.eos_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict_in_generate=True,
                generation_config=generation_config,
                **generation_kwargs,
            )
            if is_gradient_checkpointing:
                self.gaze_decoder.gradient_checkpointing_enable()

            # Get the predicted gaze ids
            gaze_pos_ids = gaze_outputs.sequences  # B * N
            gaze_pos_ids_list.append(gaze_pos_ids + self.num_vision_tokens_each_frame * t)

            # Update inputs_embeds for the next frame
            inputs_embeds.append(self.gaze_decoder.model.embed_tokens(gaze_pos_ids))

            # Update past_key_values for the next frame
            past_key_values = gaze_outputs.past_key_values

            # Update auxiliary information
            num_gazing_each_frame.append(gaze_pos_ids.shape[1])
            if_padded_gazing.append(gaze_pos_ids == self.gaze_decoder_config.eos_token_id)

            # Update attention mask
            attention_mask.append((gaze_pos_ids != self.gaze_decoder_config.eos_token_id).to(torch.long))

        # Concatenate gaze position IDs from all frames
        gaze_pos_ids = torch.cat(gaze_pos_ids_list, dim=1)

        # Get auxiliary information
        num_gazing_each_frame = torch.tensor(num_gazing_each_frame, device=gaze_pos_ids.device).to(torch.long)
        if_padded_gazing = torch.cat(if_padded_gazing, dim=1)

        to_return = {
            "gazing_pos": gaze_pos_ids,  # In gaze_pos_ids, the padded gazing positions are not necessarily eos_token_id, so one needs to use if_padded_gazing to determine if the gazing position is padded!!!
            "num_gazing_each_frame": num_gazing_each_frame,
            "if_padded_gazing": if_padded_gazing,
            "task_loss_requirement": task_loss_requirement,
            "past_input_embeds": inputs_embeds if use_cache else None,
            "past_attention_mask": attention_mask if use_cache else None,
            "past_key_values": past_key_values if use_cache else None,
            "past_conv_values": past_conv_values if use_cache else None,
        }
        return to_return

    def forward(self, video, gazing_info, **kwargs):
        # Unpack gazing_info
        gaze_pos_ids = gazing_info["gazing_pos"]
        num_gazing_each_frame = gazing_info["num_gazing_each_frame"]
        if_padded_gazing = gazing_info["if_padded_gazing"]
        
        # Subsample frames and resize
        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size, self.input_img_size), mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        # Split the gaze frame-wise
        gaze_pos_ids_split = list(gaze_pos_ids.split(num_gazing_each_frame.tolist(), dim=1))
        gaze_pos_ids_split = [gaze_pos_ids_split[t] - self.num_vision_tokens_each_frame * t for t in range(len(gaze_pos_ids_split))]
        if_padded_gazing_split = list(if_padded_gazing.split(num_gazing_each_frame.tolist(), dim=1))

        # Fill the padded gazing positions with eos_token_id
        gaze_pos_ids_split = [gaze_pos * (~padded) + self.gaze_decoder_config.eos_token_id * padded for gaze_pos, padded in zip(gaze_pos_ids_split, if_padded_gazing_split)]

        # Embed the video and gaze position IDs
        inputs_embeds, gaze_token_mask, gaze_pred_source_relative, attention_mask, _ = self.embed(video=video, gaze_pos_ids=gaze_pos_ids_split)
        inputs_embeds = torch.cat(inputs_embeds, dim=1)  # B * N * C
        gaze_token_mask = torch.cat(gaze_token_mask, dim=0)  # N
        gaze_pred_source_relative = torch.cat(gaze_pred_source_relative, dim=0)  # N
        attention_mask = torch.cat(attention_mask, dim=1)  # B * N

        # Run model forward
        outputs = self.gaze_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=attention_mask.cumsum(dim=-1) - 1,
            **kwargs,
        )

        # Get gaze logits and probs
        logits_multi_token_pred = outputs.logits
        task_loss_prediction_multi_token_pred = outputs.task_loss_prediction  # B * N * num_multi_token_pred
        logits_multi_token_pred = rearrange(logits_multi_token_pred, 'b n (k c) -> b n k c', k=self.num_multi_token_pred)
        gaze_probs_all_multi_token_pred = F.softmax(logits_multi_token_pred, dim=-1)

        shifted_probs = []
        shifted_task_loss_prediction = []
        for i in range(self.num_multi_token_pred):
            shifted_probs.append(F.pad(gaze_probs_all_multi_token_pred[:, :-(i + 1), i, :], (0, 0, i + 1, 0), value=0))
            shifted_task_loss_prediction.append(F.pad(task_loss_prediction_multi_token_pred[:, :task_loss_prediction_multi_token_pred.shape[1] - i, i], (i, 0), value=0))
        shifted_probs = torch.stack(shifted_probs, dim=2)  # B, N, K, C
        shifted_task_loss_prediction = torch.stack(shifted_task_loss_prediction, dim=2)  # B, N, K

        gaze_probs_all = shifted_probs[:, torch.arange(logits_multi_token_pred.shape[1]), -gaze_pred_source_relative - 1]
        task_loss_prediction = shifted_task_loss_prediction[:, torch.arange(logits_multi_token_pred.shape[1]), (-gaze_pred_source_relative) % self.num_multi_token_pred]  # B, N

        gaze_input_token_pos = torch.nonzero(gaze_token_mask, as_tuple=True)[0]
        gaze_probs_all = gaze_probs_all[:, gaze_input_token_pos, :]
        task_loss_prediction = task_loss_prediction[:, gaze_input_token_pos]
        B, N = gaze_probs_all.shape[:2]
        gaze_probs = gaze_probs_all.reshape(B * N, -1)[torch.arange(B * N), torch.cat(gaze_pos_ids_split, dim=1).flatten()].reshape(B, N)  # [B, T]


        outputs = AutoGazeOutput(
            gaze_probs=gaze_probs,
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            task_loss_prediction=task_loss_prediction,
        )
        return outputs


################ Shallow Vision Encoder #################

class Conv3dBlockForStreaming(nn.Module):
    def __init__(self, hidden_dim, temporal_patch_size, spatial_kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_kernel_size = spatial_kernel_size

        self.conv3d = nn.Conv3d(
            hidden_dim, hidden_dim, 
            kernel_size=(temporal_patch_size, spatial_kernel_size, spatial_kernel_size), 
            padding=(0, (spatial_kernel_size - 1) // 2, (spatial_kernel_size - 1) // 2),  # We manually pad the temporal dimension in forward, to support streaming
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, use_cache=False, past_conv_values=None):
        if not (use_cache and past_conv_values is not None):
            x = F.pad(x, (0, 0, 0, 0, self.temporal_patch_size - 1, 0), value=0)
        else:
            x = torch.cat([past_conv_values, x], dim=2)
        new_past_conv_values = x[:, :, -(self.temporal_patch_size - 1):]

        x = self.conv3d(x)

        x = self.relu(x)

        return x, new_past_conv_values


class ShallowVideoConvNet(nn.Module):
    """
    A shallow video convolutional network for video gaze modeling, inspired by ViViT's patch embedding approach.
    Expects input of shape (B, T, C, H, W) or (B*T, C, H, W).
    """
    def __init__(self, config: VisionModelConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        out_dim = config.out_dim
        depth = config.depth
        kernel_size = config.kernel_size
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 1)

        # For video, first merge temporal and batch if needed, then apply 3D conv for temporal patching
        self.temporal_conv = nn.Conv3d(
            in_channels=3,  # RGB
            out_channels=hidden_dim,
            kernel_size=(self.temporal_patch_size, kernel_size, kernel_size),
            stride=(self.temporal_patch_size, kernel_size, kernel_size),
            bias=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.trunk_temporal_kernel_size = config.trunk_temporal_kernel_size
        self.trunk_spatial_kernel_size = config.trunk_spatial_kernel_size
        blocks = []
        for i in range(depth):
            blocks.append(
                Conv3dBlockForStreaming(
                    hidden_dim=hidden_dim,
                    temporal_patch_size=self.trunk_temporal_kernel_size,
                    spatial_kernel_size=self.trunk_spatial_kernel_size,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv3d(
            hidden_dim, out_dim, kernel_size=1, stride=1, bias=True
        )

    def forward(self, x, use_cache=False, past_conv_values=None):
        # x: (B, T, C, H, W) or (B*T, C, H, W)
        if x.dim() == 5:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        elif x.dim() == 4:
            # (B*T, C, H, W) -> (B*T, C, 1, H, W)
            x = x.unsqueeze(2)
        else:
            raise ValueError("Input must be 4D or 5D tensor")
        x = self.temporal_conv(x)  # (B, hidden_dim, T', H', W')
        # Collapse temporal dimension into batch for normalization and blocks
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # (B*T, C, H, W)
        # Flatten spatial dims for norm: (B*T, C, H*W)
        x = x.view(B * T, C, -1).permute(0, 2, 1)  # (B*T, H*W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous().view(B * T, C, H, W)  # (B*T, C, H, W)
        # Reshape back to (B, C, T, H, W)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        # Main trunk
        new_past_conv_values = []
        for i, block in enumerate(self.blocks):
            x, new_past_conv_values_i = block(
                x, 
                use_cache=use_cache, 
                past_conv_values=past_conv_values[i] if use_cache and past_conv_values is not None else None
            )
            new_past_conv_values.append(new_past_conv_values_i)
        x = self.out_proj(x)
        # Output shape: (B, out_dim, T', H', W')
        return x, new_past_conv_values


################ Connector Between Vision Encoder and Gaze Model #################

class Connector(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_tokens = config.num_tokens

        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, self.hidden_dim))

    def forward(self, x):
        """
        x: (B, T, N, C)
        """
        x = x + self.pos_embed[None, None]
        return x