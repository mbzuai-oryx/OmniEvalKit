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

"""Common utilities for video loading and processing."""

import av
import numpy as np
import torch


def get_relative_video_path(path):
    """
    Get the last three levels of the path as the relative path to the video.
    Args:
        path (str): Path to get the last three levels of.
    Returns:
        last_three (str): Last three levels of the path.
    """
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-3:]) if len(parts) >= 3 else path


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len, random_sample_frame=False):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= converted_len:
        # Not enough frames, just return the first clip_len frames (or as many as possible)
        indices = np.arange(min(clip_len, seg_len))
        indices = np.pad(indices, (0, max(0, clip_len - len(indices))), mode="edge")
        return indices.astype(np.int64)
    if random_sample_frame:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
    else:
        start_idx = 0
        end_idx = converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def process_video_frames(video, clip_len):
    """
    Process video frames to ensure correct shape and length.
    Args:
        video (np.ndarray): Video frames of shape (num_frames, H, W, 3)
        clip_len (int): Target number of frames
    Returns:
        video (np.ndarray): Processed video of shape (clip_len, H, W, 3)
    """
    # Ensure video has shape (clip_len, H, W, 3)
    if video.shape[0] != clip_len:
        # Pad or repeat last frame if needed
        if video.shape[0] < clip_len:
            pad_frames = clip_len - video.shape[0]
            last_frame = video[-1:]
            video = np.concatenate(
                [video, np.repeat(last_frame, pad_frames, axis=0)], axis=0
            )
        else:
            video = video[:clip_len]

    assert video.shape[0] == clip_len, (
        f"Video has {video.shape[0]} frames, expected {clip_len}"
    )
    assert video.ndim == 4 and video.shape[-1] == 3, (
        f"Video shape is {video.shape}, expected (clip_len, H, W, 3)"
    )
    
    return video


def transform_video_for_pytorch(video, transform=None):
    """
    Transform video frames and convert to PyTorch format.
    Args:
        video (np.ndarray): Video frames of shape (clip_len, H, W, 3)
        transform: Optional transform to apply
    Returns:
        img (np.ndarray): Transformed video of shape (clip_len, C, H, W)
    """
    if transform is not None:
        imgs = transform(list(video)).pixel_values
        if isinstance(imgs[0], list):  # frames are wrapped in a python list
            img = imgs[0]
        else:
            img = imgs  # frames are not wrapped in a python list
        img = np.stack(img)
    else:
        img = video  # fallback: return raw video

    # Ensure output is (clip_len, C, H, W) for pytorch
    if img.shape[1] == 3 and img.shape[-1] != 3:
        # Already (clip_len, C, H, W)
        pass
    elif img.shape[-1] == 3:
        # (clip_len, H, W, 3) -> (clip_len, 3, H, W)
        img = np.transpose(img, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unexpected image shape after transform: {img.shape}")

    clip_len = img.shape[0]
    assert img.shape[0] == clip_len and img.shape[1] == 3, (
        f"Output img shape: {img.shape}"
    )
    
    return torch.tensor(img)
