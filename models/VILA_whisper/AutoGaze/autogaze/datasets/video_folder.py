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

import os

import av
import numpy as np
import json
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, Compose

from .video_utils import read_video_pyav, sample_frame_indices, process_video_frames, transform_video_for_pytorch, get_relative_video_path

_SPLITS = {
    "train": "train",
    "val": "val",
}


class VideoFolder(Dataset):
    def __init__(
        self,
        root,
        split,
        gt_gazing_pos_paths,
        random_sample_frame=False,
        train_data_aug=None,
        gaze_transform=None,
        task_transform=None,
        clip_len=64,
        frame_sample_rate=1,
        video_ext=".mp4",
    ):
        self.name = "VideoFolder"

        if split not in _SPLITS:
            raise ValueError(f"Split '{split}' not in {_SPLITS.keys()}")
        self.split = split

        # Handle multiple root directories
        self.roots = [r.strip() for r in root.split(",")]
        
        # Handle multiple gt_gazing_pos_paths
        gt_paths = gt_gazing_pos_paths[split]
        self.gt_gazing_pos_paths = [p.strip() for p in gt_paths.split(",")] if gt_paths else []

        self.data = []
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.video_ext = video_ext
        self.random_sample_frame = random_sample_frame
        self.gaze_transform = gaze_transform
        self.task_transform = task_transform
        self.train_data_aug = train_data_aug

        # Process each root directory to get all video files
        for root_dir in self.roots:
            root_path = os.path.join(root_dir, _SPLITS[split])
            if not os.path.exists(root_path):
                raise ValueError(f"Path '{root_path}' does not exist")
            if not os.path.isdir(root_path):
                raise ValueError(f"Path '{root_path}' is not a directory")

            # Get all video files in the directory and sort them for deterministic order
            for fname in sorted(os.listdir(root_path)):
                path = os.path.join(root_path, fname)
                if path.endswith(self.video_ext):
                    self.data.append(path)

        # Load the ground truth gazing positions from multiple paths
        self.gt_gazing_pos_dict = {}
        for gt_path in self.gt_gazing_pos_paths:
            if glob.has_magic(gt_path):
                # Handle path format with wildcard
                matching_files = glob.glob(gt_path)
                for file_path in matching_files:
                    with open(file_path, "r") as f:
                        self.gt_gazing_pos_dict.update(json.load(f))
            else:
                # Handle single file path
                with open(gt_path, "r") as f:
                    self.gt_gazing_pos_dict.update(json.load(f))

        # Only keep the relative path of each key in gt_gazing_pos_dict
        self.gt_gazing_pos_dict = {get_relative_video_path(path): value for path, value in self.gt_gazing_pos_dict.items()}
        
        # Filter out samples without gt gazing info if gt paths were provided
        if self.gt_gazing_pos_paths:
            valid_ids = [idx for idx, video_path in enumerate(self.data) if get_relative_video_path(video_path) in self.gt_gazing_pos_dict]
            self.data = [self.data[idx] for idx in valid_ids]
        
        # Initialize train data augmentation if enabled
        # Use gaze_transform to determine the output size (both transforms should produce the same spatial size)
        ref_transform = self.gaze_transform or self.task_transform
        self.data_aug = []
        if split == "train" and self.train_data_aug.aug_type is not None:
            aug_type = self.train_data_aug.aug_type.split("+")
            if "random_resized_crop" in aug_type:
                if "shortest_edge" in ref_transform.size:
                    output_size = ref_transform.size["shortest_edge"]
                elif "height" in ref_transform.size and "width" in ref_transform.size:
                    output_size = (ref_transform.size["height"], ref_transform.size["width"])
                else:
                    raise ValueError("Size is not provided in the transform")
                self.data_aug.append(RandomResizedCrop(
                    size=output_size,
                    scale=(self.train_data_aug.random_resized_crop.scale_min, self.train_data_aug.random_resized_crop.scale_max),
                    ratio=(self.train_data_aug.random_resized_crop.ratio_min, self.train_data_aug.random_resized_crop.ratio_max),
                ))
        self.data_aug = Compose(self.data_aug)
    
    def check_dataset_is_not_random(self):
        instance = self.__getitem__(0)
        for i in range(5):
            new_instance = self.__getitem__(0)
            if (instance['video'] != new_instance['video']).any():
                return False
        return True

    def __getitem__(self, idx):
        video_path = self.data[idx]
        gt_gazing_info = self.gt_gazing_pos_dict.get(get_relative_video_path(video_path), [])

        try:
            container = av.open(video_path)
            # sample frames using hyperparameters
            indices = sample_frame_indices(
                clip_len=self.clip_len,
                frame_sample_rate=self.frame_sample_rate,
                seg_len=container.streams.video[0].frames,
                random_sample_frame=self.random_sample_frame,
            )
            video = read_video_pyav(container=container, indices=indices)
            container.close()
        except Exception as e:
            # If the video is corrupt, sample another video
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # Process video frames
        video = process_video_frames(video, self.clip_len)

        # Apply both transforms to produce video (for gaze model) and video_for_task (for task)
        video_gaze = transform_video_for_pytorch(video, self.gaze_transform)
        video_task = transform_video_for_pytorch(video, self.task_transform)

        # Apply the same train data augmentation to both videos by concatenating along the
        # frame dimension so that RandomResizedCrop uses the same random crop for both.
        combined = torch.cat([video_gaze, video_task], dim=0)
        combined = self.data_aug(combined)
        video_gaze = combined[:self.clip_len]
        video_task = combined[self.clip_len:]

        return {
            "video": video_gaze,
            "video_for_task": video_task,
            "is_valid": True,
            "video_path": video_path,
            "gt_gazing_info": gt_gazing_info,
        }

    def __len__(self):
        return len(self.data)
