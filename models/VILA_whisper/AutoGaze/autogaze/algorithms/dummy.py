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

class Dummy:
    def __init__(self):
        pass

    def preprocess_inputs(self, inputs):
        return inputs
    
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
            outputs: dict of various outputs of the task
            loss: G*B
            reward: G*B, num_reward_each_traj  (There can be multiple rewards taken at different step index for each trajectory)
            traj_len_each_reward: list with length of num_reward_each_traj  (The length of the trajectory before each reward is taken)
            metrics: dict of metrics of the task
            task_losses: B, N (optional, used for task loss prediction)
            task_losses_mask: B, N (optional, used for task loss prediction)
        """
        return torch.zeros(gaze_outputs['gazing_pos'].shape[0], device=gaze_outputs['gazing_pos'].device)

    def __call__(self, inputs, gaze_outputs, task_outputs):
        loss = self.loss(inputs, gaze_outputs, task_outputs)
        metrics = {"dummy_loss": loss.mean()}

        to_return = {
            'loss': loss,
            'metrics': metrics,
        }
        return to_return