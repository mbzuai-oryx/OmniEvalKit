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

import random
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf

from transformers.modeling_utils import PreTrainedModel
from autogaze.utils import get_gazing_pos_from_gazing_mask
from .modeling_autogaze import AutoGazeModel
from .configuration_autogaze import AutoGazeConfig

class AutoGaze(PreTrainedModel):
    config_class = AutoGazeConfig
    
    def __init__(self, config: AutoGazeConfig):
        super().__init__(config)
        
        self.config = config
        self.gazing_ratio_config = config.gazing_ratio_config
        self.gazing_ratio_each_frame_config = config.gazing_ratio_each_frame_config
        self.scales = sorted([int(scale) for scale in str(config.scales).split('+')])
        self.num_vision_tokens_each_frame = config.num_vision_tokens_each_frame
        self.num_vision_tokens_each_scale_each_frame = [int(scale**2 / sum([scale**2 for scale in self.scales]) * self.num_vision_tokens_each_frame) for scale in self.scales]
        self.frame_sampling_rate = config.gaze_model_config.vision_model_config.temporal_patch_size
        self.attn_mode = config.attn_mode
        
        # Create the gazing model
        self.gazing_model = AutoGazeModel(config.gaze_model_config)

        # Task loss requirement
        self.has_task_loss_requirement_during_training = config.has_task_loss_requirement_during_training
        self.has_task_loss_requirement_during_inference = config.has_task_loss_requirement_during_inference
        self.task_loss_requirement_config = config.task_loss_requirement_config
    
    def get_gazing_ratio(self, sync_across_ranks=True):
        """
        Sample the gazing ratio for the whole video according to the config.
        """
        sample_strategy = self.gazing_ratio_config['sample_strategy_during_training'] if self.training else self.gazing_ratio_config['sample_strategy_during_inference']
        if sample_strategy == 'fixed':
            ratio = self.gazing_ratio_config['fixed']['gazing_ratio']
        elif sample_strategy == 'uniform':
            ratio = random.uniform(self.gazing_ratio_config['uniform']['gazing_ratio_min'], self.gazing_ratio_config['uniform']['gazing_ratio_max'])
        elif sample_strategy == 'exponential':
            ratio = random.expovariate(self.gazing_ratio_config['exponential']['lambda'])
            while ratio < self.gazing_ratio_config['exponential']['gazing_ratio_min'] or ratio > self.gazing_ratio_config['exponential']['gazing_ratio_max']:
                ratio = random.expovariate(self.gazing_ratio_config['exponential']['lambda'])
        
        if sync_across_ranks:
            ratio = torch.tensor(ratio).cuda()
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(ratio, src=0)  # Make every rank use the same gazing ratio. Otherwise, each rank will have different gazing ratio, and the train/inference time is bounded by the slowest rank (with highest gazing ratio).
            ratio = ratio.item()

        return ratio
    
    def get_gazing_ratio_each_frame(self, inputs, video, gazing_ratio_mean, num_frames, temperature, use_cache):
        """
        Sample the gazing ratio for each frame according to the config.
        """
        sample_strategy = self.gazing_ratio_each_frame_config['sample_strategy_during_training'] if self.training else self.gazing_ratio_each_frame_config['sample_strategy_during_inference']
        if sample_strategy == 'uniform':
            gazing_ratio_each_frame = torch.ones(num_frames) * gazing_ratio_mean
        elif sample_strategy == 'dirichlet':
            gazing_ratio_agg = gazing_ratio_mean * num_frames
            alpha = self.gazing_ratio_each_frame_config['dirichlet']['alpha']
            if isinstance(alpha, str):
                alpha = torch.tensor([float(a) for a in alpha.split(',')])
                assert len(alpha) == num_frames, "The number of alpha values must be equal to the number of frames"
            gazing_ratio_each_frame = torch.distributions.dirichlet.Dirichlet(torch.ones(num_frames) * alpha).sample() * gazing_ratio_agg
            gazing_ratio_each_frame = gazing_ratio_each_frame.clamp(min=0, max=1)
        elif sample_strategy == 'self':
            assert use_cache == False, "using cache is not supported for self-predicted gazing ratio"

            # Only preserve one sample for each group
            if "group_size" in inputs:
                video = rearrange(video, '(g b) t c h w -> g b t c h w', g=inputs["group_size"])[0]

            assert video.shape[0] == 1, "Currently only batch_size=1 is supported because otherwise we need to support different gazing ratio constraints in the same batch in model.generate()"

            # Max gazing ratio for each frame
            max_gazing_ratio_each_frame = torch.ones(num_frames) * gazing_ratio_mean
            max_num_gaze_tokens_each_frame = (max_gazing_ratio_each_frame * self.num_vision_tokens_each_frame).to(torch.long).clamp(min=1)

            # Sample task loss requirement
            task_loss_requirement = self.get_task_loss_requirement(video, force_sampling=True)

            # Sample the gazing
            with torch.no_grad():
                if self.training:
                    gazing_info = self.gazing_model.generate(
                        video, 
                        max_gaze_tokens_each_frame=max_num_gaze_tokens_each_frame,
                        task_loss_requirement=task_loss_requirement,
                        do_sample=True, 
                        temperature=temperature,
                    )
                else:
                    gazing_info = self.gazing_model.generate(
                        video, 
                        max_gaze_tokens_each_frame=max_num_gaze_tokens_each_frame, 
                        task_loss_requirement=task_loss_requirement,
                        do_sample=False,
                    )
            
            if_padded_gazing = gazing_info["if_padded_gazing"]
            num_gazing_each_frame = gazing_info["num_gazing_each_frame"]
            if_padded_gazing = if_padded_gazing.split(num_gazing_each_frame.tolist(), dim=1)
            num_non_padded_gazing_each_frame = torch.stack([(~if_padded_gazing[i]).sum(dim=-1) for i in range(len(if_padded_gazing))], dim=1)  # (B, num_frames)

            gazing_ratio_each_frame = num_non_padded_gazing_each_frame[0] / self.num_vision_tokens_each_frame
        else:
            raise NotImplementedError(f"Sample strategy {sample_strategy} not implemented.")
    
        return gazing_ratio_each_frame
    
    def get_task_loss_requirement(self, video, sync_across_ranks=True, force_sampling=False):
        """
        Sample the task loss requirement for each frame according to the config.

        inputs:
            video: tensor of shape (B, T, C, H, W)
        returns:
            task_loss_requirement: tensor of shape (B, T // frame_sampling_rate), representing the task loss requirement for each frame of each video. None if no task loss requirement is used.
        """
        has_task_loss_requirement = self.has_task_loss_requirement_during_training if self.training else self.has_task_loss_requirement_during_inference
        if not has_task_loss_requirement and not force_sampling:
            return None

        B, T = video.shape[:2]
        sample_strategy = self.task_loss_requirement_config['sample_strategy_during_training'] if self.training else self.task_loss_requirement_config['sample_strategy_during_inference']
        if sample_strategy == 'fixed':
            task_loss_requirement = self.task_loss_requirement_config['fixed']['task_loss_requirement']
            task_loss_requirement = torch.ones(B, T // self.frame_sampling_rate, device=video.device) * task_loss_requirement
        elif sample_strategy == 'uniform':
            task_loss_requirement_min = self.task_loss_requirement_config['uniform']['task_loss_requirement_min']
            task_loss_requirement_max = self.task_loss_requirement_config['uniform']['task_loss_requirement_max']
            task_loss_requirement = random.uniform(task_loss_requirement_min, task_loss_requirement_max)
            task_loss_requirement = torch.ones(B, T // self.frame_sampling_rate, device=video.device) * task_loss_requirement
        else:
            raise NotImplementedError(f"Task loss requirement sample strategy {self.task_loss_requirement_config['sample_strategy']} not implemented")
        
        if sync_across_ranks:
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(task_loss_requirement, src=0)  # Make every rank use the same gazing ratio. Otherwise, each rank will have different gazing ratio, and the train/inference time is bounded by the slowest rank (with highest gazing ratio).
        
        return task_loss_requirement

    def get_mask_from_gazing_pos(self, video, gazing_pos, if_padded_gazing):
        """
        Create the video gazing mask from the gazing positions.

        inputs:
            video: B, T, C, H, W
            gazing_pos: B, N
            if_padded_gazing: B, N
        returns:
            mask: list of B * T * N_each_scale
        """
        B, T = video.shape[:2]
        mask = torch.zeros(B, self.num_vision_tokens_each_frame * (T // self.frame_sampling_rate) + 1, device=video.device)  # +1 for the padded gazing positions
        tmp_gazing_pos = gazing_pos.clone()
        tmp_gazing_pos[if_padded_gazing] = mask.shape[1] - 1  # Set the padded gazing positions to the last position
        mask[torch.arange(B)[:, None], tmp_gazing_pos] = 1
        mask = mask[:, :-1]  # Remove the last position (padded gazing positions)
        mask = mask.reshape(B, T // self.frame_sampling_rate, self.num_vision_tokens_each_frame)
        mask = [mask[:, :, sum(self.num_vision_tokens_each_scale_each_frame[:i]):sum(self.num_vision_tokens_each_scale_each_frame[:i+1])] for i in range(len(self.scales))]  # list of B * T * N_each_scale

        return mask
    
    def input_res_adapt(self, pixel_values, target_scales, target_patch_size):
        """
        Preprocess the input to adapt to the target scales and patch size.

        inputs:
            pixel_values: B, T, C, H, W
        returns:
            pixel_values: B, T, C, H, W
            res_adapt_info: dict, the information of resolution adaptation, for future recovery.
        """
        B, T, C, H, W = pixel_values.shape
        assert H == W == target_scales[-1], "Now we need the input video to be the same size as the largest scale of the vision model"  # FIXME: in the future we should use relative resize ratio as the scales, e.g., 0.125+0.25+0.5+1. In this way we can also support naflex ViT.
        assert len(self.scales) == len(target_scales), "The scales of the gaze model and the vision model must be the same"
        tile_feature_map_size_each_scale = [int(self.num_vision_tokens_each_scale_each_frame[i] ** 0.5) for i in range(len(self.scales))]
        original_feature_map_height_each_scale = [target_scales[i] // target_patch_size for i in range(len(target_scales))]
        original_feature_map_width_each_scale = [target_scales[i] // target_patch_size for i in range(len(target_scales))]
        num_tiles_height = math.ceil(original_feature_map_height_each_scale[-1] / tile_feature_map_size_each_scale[-1])
        num_tiles_width = math.ceil(original_feature_map_width_each_scale[-1] / tile_feature_map_size_each_scale[-1])
        pad_H = num_tiles_height * tile_feature_map_size_each_scale[-1] * target_patch_size - H
        pad_W = num_tiles_width * tile_feature_map_size_each_scale[-1] * target_patch_size - W
        pixel_values = F.pad(pixel_values, (0, pad_W, 0, pad_H))
        pixel_values = rearrange(pixel_values, 'b t c (nh sh) (nw sw) -> (b nh nw) t c sh sw', nh=num_tiles_height, nw=num_tiles_width)
        res_adapt_info = {
            'tile_feature_map_size_each_scale': tile_feature_map_size_each_scale,
            'original_feature_map_height_each_scale': original_feature_map_height_each_scale,
            'original_feature_map_width_each_scale': original_feature_map_width_each_scale,
            'num_tiles_height': num_tiles_height,
            'num_tiles_width': num_tiles_width,
            'pad_H': pad_H,
            'pad_W': pad_W,
        }

        return pixel_values, res_adapt_info
    
    def recover_output_from_res_adapt(self, gaze_outputs, res_adapt_info):
        """
        Postprocess the output to recover from resolution adaptation.

        inputs:
            gaze_outputs: dict, the outputs of the gazing model.
            res_adapt_info: dict, the information of resolution adaptation.
        returns:
            gaze_outputs: dict, the outputs of the gazing model.
        """
        num_tiles_height = res_adapt_info['num_tiles_height']
        num_tiles_width = res_adapt_info['num_tiles_width']
        tile_feature_map_size_each_scale = res_adapt_info['tile_feature_map_size_each_scale']
        original_feature_map_height_each_scale = res_adapt_info['original_feature_map_height_each_scale']
        original_feature_map_width_each_scale = res_adapt_info['original_feature_map_width_each_scale']

        # Recover the gazing mask. Remove the gazing for the padded regions.
        new_gazing_mask = []
        for scale_idx in range(len(gaze_outputs['scales'])):
            cur_gazing_mask = gaze_outputs['gazing_mask'][scale_idx]
            cur_gazing_mask = rearrange(cur_gazing_mask, '(b nh nw) t (sh sw) -> b t (nh sh) (nw sw)', nh=num_tiles_height, nw=num_tiles_width, sh=tile_feature_map_size_each_scale[scale_idx], sw=tile_feature_map_size_each_scale[scale_idx])
            cur_gazing_mask = cur_gazing_mask[:, :, :original_feature_map_height_each_scale[scale_idx], :original_feature_map_width_each_scale[scale_idx]]
            cur_gazing_mask = cur_gazing_mask.flatten(-2, -1)  # (b t (nh sh) (nw sw)) -> (b t (nh sh * nw sw))
            new_gazing_mask.append(cur_gazing_mask)

        # Recover the num_gazing_each_frame and num_vision_tokens_each_frame
        new_num_vision_tokens_each_frame = sum([mask.shape[-1] for mask in new_gazing_mask])

        # Recover the gazing pos, if_padded_gazing, and num_gazing_each_frame, by inderring from the gazing mask. Note this will lose the original order of the gazing!
        new_gazing_mask_all_scales = torch.cat(new_gazing_mask, dim=-1)  # B, T, N
        B, T = new_gazing_mask_all_scales.shape[:2]
        new_gazing_pos, new_if_padded_gazing = get_gazing_pos_from_gazing_mask(new_gazing_mask_all_scales.flatten(0, 1))
        new_gazing_pos, new_if_padded_gazing = rearrange(new_gazing_pos, '(b t) n -> b t n', b=B, t=T), rearrange(new_if_padded_gazing, '(b t) n -> b t n', b=B, t=T)
        max_num_gazing_each_frame = (~new_if_padded_gazing).sum(dim=-1).max(dim=0)[0]
        assert all([torch.all(new_if_padded_gazing[:, t, num:] == True) for t, num in enumerate(max_num_gazing_each_frame)]), "The removed gazing should all be padded."
        new_gazing_pos = [new_gazing_pos[:, t, :num] for t, num in enumerate(max_num_gazing_each_frame)]
        new_if_padded_gazing = [new_if_padded_gazing[:, t, :num] for t, num in enumerate(max_num_gazing_each_frame)]
        new_gazing_pos = [gazing_pos + new_num_vision_tokens_each_frame * t for t, gazing_pos in enumerate(new_gazing_pos)]
        new_gazing_pos, new_if_padded_gazing = torch.cat(new_gazing_pos, dim=1), torch.cat(new_if_padded_gazing, dim=1)
        new_num_gazing_each_frame = max_num_gazing_each_frame

        # Update the outputs
        gaze_outputs['gazing_pos'] = new_gazing_pos
        gaze_outputs['gazing_mask'] = new_gazing_mask
        gaze_outputs['frame_sampling_rate'] = gaze_outputs['frame_sampling_rate']
        gaze_outputs['num_vision_tokens_each_frame'] = new_num_vision_tokens_each_frame
        gaze_outputs['num_gazing_each_frame'] = new_num_gazing_each_frame
        gaze_outputs['if_padded_gazing'] = new_if_padded_gazing

        # Currently we haven't reordered actions probs and task loss prediction based on the new gazing pos, so delete it for now for safety.
        del(gaze_outputs['log_action_probs'])
        del(gaze_outputs['task_loss_prediction'])

        return gaze_outputs

    #FIXME: separate forward and generate functions
    def forward(
        self,
        inputs,
        target_scales=None,
        target_patch_size=None,
        gazing_info=None,
        temperature=1,
        gazing_ratio=None,
        task_loss_requirement=None,
        generate_only=False,
        use_cache=False,
        past_key_values=None,
        past_inputs_embeds=None,
        past_attention_mask=None,
        past_conv_values=None,
    ):
        """
        inputs:
            video: B, T, C, H, W (preprocessed with AutoGaze's own transform)
            target_scales: list of scales for downstream vision model. If None, then use the scales in the gaze model.
            target_patch_size: patch size for downstream vision model. If None, then use the patch size in the gaze model.
            gazing_info: dict, the ground truth gazing information for NTP pre-training. If None, then run the gazing model to predict gazing positions.
            temperature: temperature for generating gazing.
            gazing_ratio: gazing ratio for the gazing model. If None, then sample the gazing ratio according to the config.
            task_loss_requirement: task loss requirement for the gazing model. If None, then sample the task loss requirement according to the config.
            generate_only: whether to only generate the gazing positions, or to also calculate the probability of taking such gaze.
            use_cache: whether to use the cache for the gazing model.
            past_key_values: the past key values for the gazing model.
            past_inputs_embeds: the past inputs embeds for the gazing model.
            past_attention_mask: the past attention mask for the gazing model.
            past_conv_values: the past conv values for the gazing model.
        returns:
            to_return: dict, the outputs of the gazing model.
        """
        if not generate_only:
            assert past_key_values is None and past_inputs_embeds is None and past_attention_mask is None and past_conv_values is None, \
                "If not in generate-only mode, we don't support past_key_values, past_inputs_embeds, past_attention_mask, and past_conv_values yet."

        video = inputs['video']

        # Preprocess the input for resolution adaptation
        if target_scales is not None and target_patch_size is not None:
            if not (target_scales == self.scales and [(scale // target_patch_size) ** 2 for scale in target_scales] == self.num_vision_tokens_each_scale_each_frame):
                video, res_adapt_info = self.input_res_adapt(video, target_scales, target_patch_size)

        B, T = video.shape[:2]

        # If gazing_pos is already provided, then directly calculate the probability of taking such gaze. Usually in the cases of calculating pi(a|s) in PPO/GRPO/etc.
        # Otherwise, run the gazing model first to predict gazing positions.
        if gazing_info is None or len(gazing_info) == 0:
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.attn_mode == "flash_attention_2" else nullcontext():

                if gazing_ratio is not None and task_loss_requirement is not None:
                    # If the user specifies the gazing ratio and task loss requirement, then use gazing ratio as the max gazing ratio and use task loss requirement to control when to stop
                    if isinstance(gazing_ratio, list):
                        assert len(gazing_ratio) == T // self.frame_sampling_rate, "The number of gazing ratios must be equal to the number of frames"
                        gazing_ratio = torch.tensor(gazing_ratio)
                    gazing_ratio_each_frame = torch.ones(T // self.frame_sampling_rate) * gazing_ratio
                    num_gaze_tokens_each_frame = (gazing_ratio_each_frame * self.num_vision_tokens_each_frame).to(torch.long).clamp(min=1)
                    task_loss_requirement = torch.ones(B, T // self.frame_sampling_rate, device=video.device) * task_loss_requirement
                elif gazing_ratio is not None:
                    # If the user specifies the gazing ratio, then turn off the task loss requirement
                    if isinstance(gazing_ratio, list):
                        assert len(gazing_ratio) == T // self.frame_sampling_rate, "The number of gazing ratios must be equal to the number of frames"
                        gazing_ratio = torch.tensor(gazing_ratio)
                    gazing_ratio_each_frame = torch.ones(T // self.frame_sampling_rate) * gazing_ratio
                    num_gaze_tokens_each_frame = (gazing_ratio_each_frame * self.num_vision_tokens_each_frame).to(torch.long).clamp(min=1)
                    task_loss_requirement = None
                elif task_loss_requirement is not None:
                    # If the user specifies the task loss requirement, then turn off the gazing ratio limit
                    gazing_ratio = 1
                    gazing_ratio_each_frame = torch.ones(T // self.frame_sampling_rate) * gazing_ratio
                    num_gaze_tokens_each_frame = (gazing_ratio_each_frame * self.num_vision_tokens_each_frame).to(torch.long).clamp(min=1)
                    task_loss_requirement = torch.ones(B, T // self.frame_sampling_rate, device=video.device) * task_loss_requirement
                else:
                    gazing_ratio = self.get_gazing_ratio()
                    gazing_ratio_each_frame = self.get_gazing_ratio_each_frame(inputs, video, gazing_ratio, T // self.frame_sampling_rate, temperature, use_cache)
                    num_gaze_tokens_each_frame = (gazing_ratio_each_frame * self.num_vision_tokens_each_frame).to(torch.long).clamp(min=1)
                    task_loss_requirement = self.get_task_loss_requirement(video)

                if self.training:
                    gazing_info = self.gazing_model.generate(
                        video, 
                        max_gaze_tokens_each_frame=num_gaze_tokens_each_frame,
                        task_loss_requirement=task_loss_requirement,
                        do_sample=True, 
                        temperature=temperature,
                        use_cache=use_cache,
                        past_key_values=past_key_values,
                        past_inputs_embeds=past_inputs_embeds,
                        past_attention_mask=past_attention_mask,
                        past_conv_values=past_conv_values,
                    )
                else:
                    gazing_info = self.gazing_model.generate(
                        video, 
                        max_gaze_tokens_each_frame=num_gaze_tokens_each_frame, 
                        task_loss_requirement=task_loss_requirement,
                        do_sample=False,
                        use_cache=use_cache,
                        past_key_values=past_key_values,
                        past_inputs_embeds=past_inputs_embeds,
                        past_attention_mask=past_attention_mask,
                        past_conv_values=past_conv_values,
                    )

        # Unpack gazing_info
        gazing_pos = gazing_info["gazing_pos"]
        num_gazing_each_frame = gazing_info["num_gazing_each_frame"]
        if_padded_gazing = gazing_info["if_padded_gazing"]
        task_loss_requirement = gazing_info.get("task_loss_requirement", None)
        new_past_key_values = gazing_info.get("past_key_values", None)
        new_past_inputs_embeds = gazing_info.get("past_input_embeds", None)
        new_past_attention_mask = gazing_info.get("past_attention_mask", None)
        new_past_conv_values = gazing_info.get("past_conv_values", None)

        # Get the log probablity of taking such gaze (log_action_probs)
        if not generate_only:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward_outputs = self.gazing_model(video, gazing_info)  # B * N
                action_probs = forward_outputs.gaze_probs
                task_loss_prediction = forward_outputs.task_loss_prediction
            log_action_probs = torch.log(action_probs + 1e-8)  # B * N
        else:
            log_action_probs = None
            task_loss_prediction = None

        # Generate (multi-scale) gazing masks for ease of visualization
        mask = self.get_mask_from_gazing_pos(video, gazing_pos, if_padded_gazing)

        to_return = {
            'gazing_pos': gazing_pos,
            'log_action_probs': log_action_probs,
            'gazing_mask': mask,
            "scales": self.scales,
            "frame_sampling_rate": self.frame_sampling_rate,
            "num_vision_tokens_each_frame": self.num_vision_tokens_each_frame,
            "num_gazing_each_frame": num_gazing_each_frame,
            "if_padded_gazing": if_padded_gazing,
            "task_loss_prediction": task_loss_prediction,
            "has_task_loss_requirement": task_loss_requirement is not None,
            "task_loss_requirement": task_loss_requirement,
            "past_key_values": new_past_key_values if use_cache else None,
            "past_input_embeds": new_past_inputs_embeds if use_cache else None,
            "past_attention_mask": new_past_attention_mask if use_cache else None,
            "past_conv_values": new_past_conv_values if use_cache else None,
        }

        # Postprocess the output to recover from resolution adaptation
        if target_scales is not None and target_patch_size is not None:
            if not (target_scales == self.scales and [(scale // target_patch_size) ** 2 for scale in target_scales] == self.num_vision_tokens_each_scale_each_frame):
                to_return.update(self.recover_output_from_res_adapt(to_return, res_adapt_info))

        return to_return
