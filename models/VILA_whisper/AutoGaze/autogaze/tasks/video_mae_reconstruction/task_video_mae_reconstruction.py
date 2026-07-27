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

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoImageProcessor, VivitImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

from .modeling_video_mae import ViTMAEForPreTraining
from .visualize_video_mae_reconstruction import VisualizeReconstruction


class VideoMAEReconstruction(nn.Module):
    def __init__(self, recon_model, recon_model_config, scales, recon_sample_rate, attn_mode):
        super().__init__()

        # Create model
        self.scales = sorted([int(scale) for scale in str(scales).split("+")])
        self.transform = VivitImageProcessor.from_pretrained(recon_model, size=self.scales[-1])  # use mae image preprocessor config to intialize video preprocessor
        self.mae = ViTMAEForPreTraining.from_pretrained(recon_model, attn_implementation="sdpa", scales=str(scales), **OmegaConf.to_container(recon_model_config))
        self.mae.transform = self.transform
        if "dinov2_reg" in self.mae.loss_type:
            self.mae.dinov2_reg = AutoModel.from_pretrained(recon_model_config.dinov2_reg_loss_config.model, attn_implementation=attn_mode)
            self.mae.dinov2_reg_transform = AutoImageProcessor.from_pretrained(recon_model_config.dinov2_reg_loss_config.model)
            for param in self.mae.dinov2_reg.parameters():
                param.requires_grad = False
            self.mae.dinov2_reg.eval()
        if "siglip2" in self.mae.loss_type:
            if "naflex" in recon_model_config.siglip2_loss_config.model:
                self.mae.siglip2 = Siglip2VisionModel.from_pretrained(recon_model_config.siglip2_loss_config.model, attn_implementation=attn_mode)
            else:
                self.mae.siglip2 = SiglipVisionModel.from_pretrained(recon_model_config.siglip2_loss_config.model, attn_implementation=attn_mode)
            self.mae.siglip2_transform = AutoImageProcessor.from_pretrained(recon_model_config.siglip2_loss_config.model)
            for param in self.mae.siglip2.parameters():
                param.requires_grad = False
            self.mae.siglip2.eval()

        # Sampling strategy for reconstruction
        self.recon_sample_rate = recon_sample_rate

        # Create visualization methods
        self.visualize_methods = [VisualizeReconstruction()]

        # kwargs for the gaze model input. Will be passed to the gaze model during training.
        self.gaze_model_kwargs = {
            "target_scales": self.scales,
            "target_patch_size": self.mae.config.patch_size,
        }
    
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward_output(self, inputs, gaze_outputs, frame_idx_to_reconstruct=None):
        """
        Get all the outputs from the inputs
        """
        video = inputs['video_for_task']
        gazing_pos = gaze_outputs['gazing_pos']
        num_gazing_each_frame = gaze_outputs['num_gazing_each_frame']
        if_padded_gazing = gaze_outputs['if_padded_gazing']
        frame_sampling_rate = gaze_outputs['frame_sampling_rate']
        num_vision_tokens_each_frame = gaze_outputs['num_vision_tokens_each_frame']

        assert frame_sampling_rate == 1, "If frame_sampling_rate > 1, we can downsample the video here but ideally we don't want to do that"
        assert num_vision_tokens_each_frame == sum([(scale // self.mae.config.patch_size) ** 2 for scale in self.scales]), "The number of vision tokens in each frame is not consistent between gaze model and MAE model"

        # Frame sampling strategy for reconstruction
        B, T = video.shape[:2]
        if frame_idx_to_reconstruct is None:
            frame_idx_to_reconstruct = torch.randperm(T)[:int(T * self.recon_sample_rate)].to(video.device)

        # Reconstruct the video
        gazing_info = {
            'gazing_pos': gazing_pos,
            'num_gazing_each_frame': num_gazing_each_frame,
            'if_padded_gazing': if_padded_gazing,
        }
        recon_output = self.mae(video, gazing_info=gazing_info, frame_idx_to_reconstruct=frame_idx_to_reconstruct, interpolate_pos_encoding=True)
        recon_loss_mean = recon_output.loss_mean
        recon_loss_each_reconstruction_frame = recon_output.loss_each_reconstruction_frame
        num_gazing_before_each_reconstruction_frame = torch.stack([num_gazing_each_frame[:frame_idx+1].sum(dim=-1) for frame_idx in frame_idx_to_reconstruct], dim=0)
        num_non_padded_gazing_at_each_reconstruction_frame = [(~if_padded_gazing)[:, num_gazing_each_frame[:frame_idx].sum():num_gazing_each_frame[:frame_idx+1].sum()].sum(dim=-1) for frame_idx in frame_idx_to_reconstruct]
        num_non_padded_gazing_at_each_reconstruction_frame = torch.stack(num_non_padded_gazing_at_each_reconstruction_frame, dim=-1)  # B * num_reconstruction_frames

        # Organize the recon loss at each gazing token
        if_padded_gazing_each_frame = list(if_padded_gazing.split(num_gazing_each_frame.tolist(), dim=-1))
        reconstruction_loss_each_gazing_token = [torch.zeros(*if_padded_gazing_each_frame[t].shape, dtype=gazing_pos.dtype, device=gazing_pos.device) for t in range(len(num_gazing_each_frame))]
        reconstruction_loss_each_gazing_token_mask = [torch.zeros(*if_padded_gazing_each_frame[t].shape, dtype=gazing_pos.dtype, device=gazing_pos.device) for t in range(len(num_gazing_each_frame))]
        for i, frame_idx in enumerate(frame_idx_to_reconstruct):
            cur_mask = F.pad(if_padded_gazing_each_frame[frame_idx][:, 1:], (0, 1), value=True).to(torch.float)
            reconstruction_loss_each_gazing_token[frame_idx] = recon_loss_each_reconstruction_frame[:, i:i+1] * cur_mask
            reconstruction_loss_each_gazing_token_mask[frame_idx] = cur_mask
        reconstruction_loss_each_gazing_token = torch.cat(reconstruction_loss_each_gazing_token, dim=-1)  # B * N
        reconstruction_loss_each_gazing_token_mask = torch.cat(reconstruction_loss_each_gazing_token_mask, dim=-1)  # B * N

        outputs = {
            "reconstruction": recon_output.reconstruction,
            "reconstruction_loss": recon_loss_mean,
            "reconstruction_loss_each_reconstruction_frame": recon_loss_each_reconstruction_frame,
            "reconstruction_loss_each_gazing_token": reconstruction_loss_each_gazing_token,
            "reconstruction_loss_each_gazing_token_mask": reconstruction_loss_each_gazing_token_mask,
            "num_gazing_before_each_reconstruction_frame": num_gazing_before_each_reconstruction_frame,
            "num_non_padded_gazing_at_each_reconstruction_frame": num_non_padded_gazing_at_each_reconstruction_frame,
            "frame_idx_to_reconstruct": frame_idx_to_reconstruct,
            "image_mean": self.transform.image_mean,
            "image_std": self.transform.image_std,
            "rescale_factor": self.transform.rescale_factor,
            "scales": self.scales,
        }
        return outputs
    
    def loss(self, inputs, gaze_outputs, outputs):
        """
        Compute the loss of the outputs. Used for training the task itself.
        """
        reconstruction_loss = outputs['reconstruction_loss']
        reconstruction_loss_each_gazing_token = outputs['reconstruction_loss_each_gazing_token']
        reconstruction_loss_each_gazing_token_mask = outputs['reconstruction_loss_each_gazing_token_mask']
        return reconstruction_loss, reconstruction_loss_each_gazing_token, reconstruction_loss_each_gazing_token_mask

    def reward(self, inputs, gaze_outputs, outputs):
        """
        Compute the reward of the outputs. Used for training the gazing model.
        """
        reconstruction_loss_each_reconstruction_frame = outputs['reconstruction_loss_each_reconstruction_frame']
        rewards = -reconstruction_loss_each_reconstruction_frame.detach()

        # Gazing length before each reward
        traj_len_each_reward = outputs['num_gazing_before_each_reconstruction_frame']
        
        return rewards, traj_len_each_reward

    def metric(self, inputs, gaze_outputs, outputs):
        """
        Compute the metric used for recording during validation.
        """
        # Reconstruction loss
        reconstruction_loss, _, __ = self.loss(inputs, gaze_outputs, outputs)
        reconstruction_loss = reconstruction_loss.mean()

        # Average gazing ratio per frame
        bs, num_frames = inputs['video_for_task'].shape[:2]
        num_vision_tokens_each_frame = gaze_outputs['num_vision_tokens_each_frame']
        num_gazing_total = (~gaze_outputs['if_padded_gazing']).sum()
        avg_gazing_ratio = num_gazing_total / (bs *num_frames * num_vision_tokens_each_frame)

        metrics = {
            'reconstruction_loss': reconstruction_loss,
            'avg_gazing_ratio_per_frame': avg_gazing_ratio,
        }
        return metrics
    
    def visualize(self, inputs, gaze_outputs, task_outputs, rl_outputs=None):
        """
        Visualize the outputs.
        """
        for method in self.visualize_methods:
            method(inputs, gaze_outputs, task_outputs, rl_outputs)

    def forward(self, inputs, gaze_outputs):
        """
        Compute the outputs and the loss, reward, and metric of the outputs.
        inputs:
            image: B, C, H, W
        gaze_outputs:
            gazing_pos: B, N
        """
        outputs = self.forward_output(inputs, gaze_outputs)
        loss, reconstruction_loss_each_gazing_token, reconstruction_loss_each_gazing_token_mask = self.loss(inputs, gaze_outputs, outputs)
        reward, traj_len_each_reward = self.reward(inputs, gaze_outputs, outputs)
        metric = self.metric(inputs, gaze_outputs, outputs)

        to_return = {
            'outputs': outputs,
            'loss': loss,
            'reward': reward,
            'traj_len_each_reward': traj_len_each_reward,
            'task_losses': reconstruction_loss_each_gazing_token,
            'task_losses_mask': reconstruction_loss_each_gazing_token_mask,
            'metrics': metric,
        }
        return to_return