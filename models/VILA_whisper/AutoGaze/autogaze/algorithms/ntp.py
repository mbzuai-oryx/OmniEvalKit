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

import torch
import torch.nn.functional as F

class NTP:
    def __init__(self, optimize_task_loss_prediction=False):
        self.optimize_task_loss_prediction = optimize_task_loss_prediction

    def preprocess_inputs(self, inputs):
        return inputs
    
    def get_loss_mask(self, inputs, gaze_outputs, task_outputs):
        # Get NTP loss mask -- True means the loss should be calculated, False means the loss should be ignored
        if_padded_gazing = gaze_outputs['if_padded_gazing']
        ntp_loss_mask = ~if_padded_gazing

        return ntp_loss_mask

    def ntp_loss(self, inputs, gaze_outputs, task_outputs):
        """
        inputs:
            image: B, C, H, W
            gt_gazing_pos: B, N
        gaze_outputs:
            log_action_probs: B, N
            num_gazing_each_frame: B
            if_padded_gazing: B, N
        task_outputs:
            task_losses: B, N (optional, used for task loss prediction)
            task_losses_mask: B, N (optional, used for task loss prediction)
        """
        log_action_probs = gaze_outputs['log_action_probs']

        # Get NTP loss mask -- True means the loss should be calculated, False means the loss should be ignored
        ntp_loss_mask = self.get_loss_mask(inputs, gaze_outputs, task_outputs)

        # calculate the NTP loss
        ntp_loss = -(log_action_probs * ntp_loss_mask).sum(dim=-1)
        average_length = ntp_loss_mask.sum() / ntp_loss_mask.shape[0] + 1e-6
        ntp_loss = ntp_loss / average_length

        return ntp_loss

    def task_loss_prediction_loss(self, inputs, gaze_outputs, task_outputs):
        """
        inputs:
            image: B, C, H, W
            gt_gazing_pos: B, N
        gaze_outputs:
            log_action_probs: B, N
            num_gazing_each_frame: B
            if_padded_gazing: B, N
        task_outputs:
            task_losses: B, N (optional, used for task loss prediction)
            task_losses_mask: B, N (optional, used for task loss prediction)
        """
        B = gaze_outputs['log_action_probs'].shape[0]

        # Get loss mask -- True means the loss should be calculated, False means the loss should be ignored
        loss_mask = self.get_loss_mask(inputs, gaze_outputs, task_outputs)

        # Calculate the task loss prediction loss
        if self.optimize_task_loss_prediction:
            task_loss_prediction = gaze_outputs['task_loss_prediction']
            gt_task_losses = task_outputs['task_losses']
            gt_task_losses_mask = task_outputs['task_losses_mask']
            gt_task_losses_mask = gt_task_losses_mask * loss_mask
            task_loss_prediction_loss = ((task_loss_prediction - gt_task_losses).pow(2) * gt_task_losses_mask).sum(dim=-1)
            average_length = gt_task_losses_mask.sum() / gt_task_losses_mask.shape[0] + 1e-6
            task_loss_prediction_loss = task_loss_prediction_loss / average_length
        else:
            if "task_loss_prediction" in gaze_outputs:
                task_loss_prediction_loss = torch.zeros(B, dtype=gaze_outputs['log_action_probs'].dtype, device=gaze_outputs['log_action_probs'].device) * gaze_outputs['task_loss_prediction'].mean()  # to let the task loss prediction head still have gradients
            else:
                task_loss_prediction_loss = torch.zeros(B, dtype=gaze_outputs['log_action_probs'].dtype, device=gaze_outputs['log_action_probs'].device)

        return task_loss_prediction_loss
    
    def loss(self, inputs, gaze_outputs, task_outputs):
        """
        inputs:
            image: B, C, H, W
            gt_gazing_pos: B, N
        gaze_outputs:
            log_action_probs: B, N
            num_gazing_each_frame: B
            if_padded_gazing: B, N
        task_outputs:
            task_losses: B, N (optional, used for task loss prediction)
            task_losses_mask: B, N (optional, used for task loss prediction)
        """
        ntp_loss = self.ntp_loss(inputs, gaze_outputs, task_outputs)
        task_loss_prediction_loss = self.task_loss_prediction_loss(inputs, gaze_outputs, task_outputs)

        losses = {
            "ntp_loss": ntp_loss,
            "task_loss_prediction_loss": task_loss_prediction_loss,
        }

        return losses
    
    def __call__(self, inputs, gaze_outputs, task_outputs):
        losses = self.loss(inputs, gaze_outputs, task_outputs)
        loss = sum(losses.values())
        metrics = {k: v.mean() for k, v in losses.items()}

        to_return = {
            'loss': loss,
            'metrics': metrics,
        }
        return to_return