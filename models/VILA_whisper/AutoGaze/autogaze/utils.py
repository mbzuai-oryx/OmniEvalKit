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

import builtins

from omegaconf import OmegaConf
from loguru import logger
import sys
import os
import torch
import numpy as np
import wandb
import random
from torch.nn.parallel import DistributedDataParallel as DDP


class UnNormalize(object):
    def __init__(self, mean, std, rescale_factor=None):
        self.mean = mean
        self.std = std
        self.rescale_factor = rescale_factor

    def __call__(self, image):
        image2 = torch.clone(image)
        dims = len(image2.shape)
        if dims == 3:
            image2 = image2.unsqueeze(0)
        image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        image2 = image2.permute(1, 0, 2, 3)
        if dims == 3:
            image2 = image2.squeeze(0)

        if self.rescale_factor is not None:
            standard_rescale = 1.0 / 255.0
            if abs(self.rescale_factor - standard_rescale) > 1e-6:
                # if the processor uses 1/127.5, needs /2.0 + 0.5 correction
                image2 = image2 / 2.0 + 0.5
        
        return torch.clamp(image2, 0, 1)


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean


def plot_grad_norms(named_parameters, name_prefix=''):
    for name, param in named_parameters:
        if param.grad is not None:
            norm = torch.linalg.vector_norm(param.grad, 2.0).item()
            wandb.log({f'{name_prefix}{name}': norm})


def suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def suppress_wandb():
    """Suppresses wandb logging from the current_process."""
    # Store original functions
    original_functions = {}
    for attr_name in dir(wandb):
        attr = getattr(wandb, attr_name)
        if callable(attr) and not attr_name.startswith('__'):
            original_functions[attr_name] = attr

            # Replace with no-op function
            def make_noop(name):
                def noop(*args, **kwargs):
                    pass
                return noop

            setattr(wandb, attr_name, make_noop(attr_name))


def suppress_logging():
    """Suppresses loguru logging from the current process."""
    logger.remove()  # Remove all handlers
    logger.add(lambda _: None)  # Add a no-op handler


def dump_cfg(cfg, logdir):
    out_f = os.path.join(logdir, "config.yaml")
    with open(out_f, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print("Wrote config to: {}".format(out_f))


def get_scheduled_temperature(step, total_steps, temp_schedule_args):
    if temp_schedule_args['mode'] == 'exp':
        t_start = temp_schedule_args['exp']['temp_start']
        t_end = temp_schedule_args['exp']['temp_end']
        return t_start * (t_end / t_start) ** (step / total_steps)
    else:
        raise ValueError(f"Unknown temp_schedule_args: {temp_schedule_args}")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed + worker_id)  # Add worker_id to make it different


def format_kwargs(cfg, optional_args):
    return {
        arg_name: getattr(getattr(cfg, section), attr)
        for arg_name, section, attr in optional_args
        if hasattr(cfg, section) and hasattr(getattr(cfg, section), attr)
    }


def move_inputs_to_cuda(inputs):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.cuda()
        elif isinstance(v, dict):
            inputs[k] = move_inputs_to_cuda(v)
    return inputs


def unwrap_model(model):
    """Unwrap DDP model if needed."""
    if isinstance(model, DDP):
        return model.module
    return model


def get_gazing_pos_from_gazing_mask(gazing_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the gazing positions from the gazing mask.
    inputs:
        gazing_mask: (B, N). 1 means gazed, 0 means not gazed.
    outputs:
        gazing_pos: (B, K). K is the maximum number of gazed tokens per instance. If the instance has less than K gazed tokens, the remaining positions are padded with -1.
        if_padded_gazing: (B, K). 1 means padded, 0 means not padded.
    """
    # x: (B, N) with 0/1 values (float/bool/int all fine)
    gazing_mask = gazing_mask.to(torch.long)
    B, N = gazing_mask.shape

    # Indices per row
    idx = torch.arange(N, device=gazing_mask.device).expand(B, N)

    # Sort key: put ones first, keep original order among ones/zeros
    #  - ones get key = idx (0..N-1)
    #  - zeros get key = N + idx (pushed after all ones)
    key = (1 - gazing_mask) * N + idx
    order = key.argsort(dim=1, stable=True)        # (B, N)
    sorted_idx = idx.gather(1, order)              # ones first, then zeros

    # Max number of ones (K) and per-row counts
    counts = gazing_mask.sum(dim=1)                           # (B,)
    K = int(counts.max().item())

    if K == 0:
        return sorted_idx[:, :0]  # (B, 0) empty result

    topk = sorted_idx[:, :K]                          # (B, K)
    pos = torch.arange(K, device=gazing_mask.device).expand(B, K)
    mask = pos < counts.unsqueeze(1)                  # True where a real "1" exists

    # Pad with -1 where the row has fewer than K ones
    gazing_pos = topk.masked_fill(~mask, -1)
    if_padded_gazing = (gazing_pos == -1)

    return gazing_pos, if_padded_gazing