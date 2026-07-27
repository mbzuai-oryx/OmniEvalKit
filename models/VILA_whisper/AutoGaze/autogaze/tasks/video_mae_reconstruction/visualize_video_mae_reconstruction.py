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

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from autogaze.utils import UnNormalize

class VisualizeReconstruction:
    def __init__(self, **kwargs):
        
        self.visualize_step = 0
        if wandb.run is not None:
            # define our custom x axis metric
            wandb.define_metric("visualize_gaze/visualize_step")
            # define which metrics will be plotted against it
            wandb.define_metric("visualize_gaze/*", step_metric="visualize_gaze/visualize_step")

    @torch.no_grad()
    def __call__(self, inputs, gaze_outputs, task_outputs, rl_outputs):
        # Get all information for visualization
        videos = inputs['video_for_task']
        gazing_mask = gaze_outputs['gazing_mask'] # containing multi-scale masks; list of B * T * N_each_scale
        frame_sampling_rate = gaze_outputs['frame_sampling_rate']
        scales = task_outputs['outputs']['scales']
        reconstruction = task_outputs['outputs']['reconstruction']
        frame_idx_to_reconstruct = task_outputs['outputs']['frame_idx_to_reconstruct']
        image_mean = task_outputs['outputs']['image_mean']
        image_std = task_outputs['outputs']['image_std']
        rescale_factor = task_outputs['outputs']['rescale_factor']
        num_scales = len(scales)

        # sample the frames to visualize
        videos = videos[:, ::frame_sampling_rate]
        assert videos.shape[1] == gazing_mask[0].shape[1]

        # only visualize the first instance
        video = videos[0]
        gazing_mask = [m[0] for m in gazing_mask]
        reconstruction = reconstruction[0]

        unnormalize = UnNormalize(image_mean, image_std, rescale_factor)

        video = unnormalize(video)
        reconstruction = unnormalize(reconstruction)
        video = video.cpu().float().numpy()
        reconstruction = reconstruction.cpu().float().numpy()

        # complete the reconstruction by filling the unselected frames
        reconstruction_full = np.zeros_like(video)
        reconstruction_full[frame_idx_to_reconstruct.cpu().numpy()] = reconstruction
        reconstruction = reconstruction_full

        # Create a figure with subplots: original video frames and one row for each scale's masked video frames
        T = video.shape[0]  # Number of frames
        fig, axes = plt.subplots(num_scales + 2, T, figsize=(3 * T, 3 * (num_scales + 2)))
        
        # Plot original video frames
        for t in range(T):
            frame = video[t].transpose(1, 2, 0)  # C * H * W -> H * W * C
            axes[0, t].imshow(frame)
            axes[0, t].set_title(f'Original Frame {t+1}')
            axes[0, t].axis('off')
        
        # Visualize masked video for each scale
        for scale_idx in range(num_scales):
            scale_mask = gazing_mask[scale_idx]  # T * N
            
            for t in range(T):
                frame_mask = scale_mask[t]  # N
                
                # Reshape if it's flattened
                if frame_mask.dim() == 1:
                    h = w = int(frame_mask.shape[0] ** 0.5)
                    frame_mask = frame_mask.reshape(h, w)
                
                # Resize mask to match current scale
                frame_mask = F.interpolate(
                    frame_mask.unsqueeze(0).unsqueeze(0), 
                    size=(scales[scale_idx], scales[scale_idx]), 
                    mode='nearest'
                ).squeeze()
                
                frame_mask = frame_mask.cpu().float().numpy()

                # Resize frame to match mask dimensions
                frame = video[t]  # C * H * W
                scale_frame = F.interpolate(
                    torch.from_numpy(frame).unsqueeze(0), 
                    size=(scales[scale_idx], scales[scale_idx]), 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze().clamp(0, 1).numpy()

                masked_frame = scale_frame * (0.8 * frame_mask[None, :, :] + 0.2)
                
                # Plot this frame's masked image
                axes[scale_idx + 1, t].imshow(masked_frame.transpose(1, 2, 0))
                
                # Add red borders around gazed patches
                original_mask = gazing_mask[scale_idx][t]
                if original_mask.dim() == 1:
                    patch_grid_size = int(original_mask.shape[0] ** 0.5)
                    original_mask = original_mask.reshape(patch_grid_size, patch_grid_size)
                
                patch_size = scales[scale_idx] // patch_grid_size
                for i in range(patch_grid_size):
                    for j in range(patch_grid_size):
                        if original_mask[i, j] > 0.5:  # If this patch is gazed at
                            rect = plt.Rectangle((j * patch_size - 0.5, i * patch_size - 0.5), 
                                               patch_size, patch_size, 
                                               linewidth=1, edgecolor='red', facecolor='none')
                            axes[scale_idx + 1, t].add_patch(rect)
                
                axes[scale_idx + 1, t].set_title(f'Scale {scales[scale_idx]} Frame {t+1}')
                axes[scale_idx + 1, t].axis('off')
        
        # plot the reconstruction
        for t in range(T):
            frame = reconstruction[t].transpose(1, 2, 0)  # C * H * W -> H * W * C
            axes[num_scales + 1, t].imshow(frame)
            axes[num_scales + 1, t].set_title(f'Reconstructed Frame {t+1}')
            axes[num_scales + 1, t].axis('off')
        
        # Adjust layout and log to wandb
        plt.tight_layout()
        wandb.log({
            "visualize_gaze/visualize_step": self.visualize_step,
            "visualize_gaze/visualize_gaze": wandb.Image(plt)
        })
        
        # Close the figure to free memory
        plt.close(fig)

        self.visualize_step += 1
