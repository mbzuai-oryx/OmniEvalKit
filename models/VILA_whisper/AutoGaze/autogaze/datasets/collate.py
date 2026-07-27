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


def process_gazing_info(gazing_pos, task_losses):
    """
    Process the ground truth gazing information.
    gazing_pos: List of lists of lists, each sub-list contains the gazing positions for an instance, and each sub-sub-list contains the gazing positions for a frame.
    task_losses: List of lists of lists, each sub-list contains the reconstruction losses for an instance, and each sub-sub-list contains the reconstruction losses for a frame.
    """
    num_frame = len(gazing_pos[0])

    max_gaze_length_each_frame = []
    for frame_idx in range(num_frame):
        max_gaze_length_each_frame.append(max([len(gazing_pos[i][frame_idx]) for i in range(len(gazing_pos))]))

    for i in range(len(gazing_pos)):
        for frame_idx in range(num_frame):
            gazing_pos[i][frame_idx].extend([-1] * (max_gaze_length_each_frame[frame_idx] - len(gazing_pos[i][frame_idx]) + 1))  # +1 for the eos token
            task_losses[i][frame_idx].extend([task_losses[i][frame_idx][-1]] * (max_gaze_length_each_frame[frame_idx] - len(task_losses[i][frame_idx]) + 1))  # +1 for the eos token
    
    gazing_pos = [sum(gazing_pos[i], []) for i in range(len(gazing_pos))]
    task_losses = [sum(task_losses[i], []) for i in range(len(task_losses))]
    
    gazing_pos = torch.tensor(gazing_pos).to(torch.long)
    task_losses = torch.tensor(task_losses).to(torch.float)
    if_padded_gazing = (gazing_pos == -1)
    num_gazing_each_frame = torch.tensor(max_gaze_length_each_frame).to(torch.long) + 1  # +1 for the eos token

    return {
        "gazing_pos": gazing_pos,
        "task_losses": task_losses,
        "if_padded_gazing": if_padded_gazing,
        "num_gazing_each_frame": num_gazing_each_frame,
    }

def collate_fn(batch):
    """
    Custom collate function that collates the batch.
    """

    # Filter out invalid samples
    filtered = [item for item in batch if item.get("is_valid", True)]
    if len(filtered) == 0:
        # If all samples are invalid, raise an error to avoid empty batch
        raise ValueError("All samples in batch are invalid/corrupt.")

    # Collect every key 
    collated_batch = {}
    for key in filtered[0].keys():
        if key == "gt_gazing_info":
            if filtered[0]["gt_gazing_info"] == []:
                collated_batch["gt_gazing_info"] = []
            else:
                gazing_pos = [item["gt_gazing_info"]["gazing_pos"] for item in filtered]
                task_losses = [item["gt_gazing_info"]["task_losses"] for item in filtered]
                gt_gazing_info = process_gazing_info(gazing_pos, task_losses)
                collated_batch["gt_gazing_info"] = gt_gazing_info
        else:
            collated_batch[key] = torch.utils.data.default_collate([item[key] for item in filtered])

    return collated_batch