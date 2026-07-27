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
from datetime import datetime
import itertools
import hydra
import torch
import wandb
import logging
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from autogaze.utils import (
    seed_everything,
    seed_worker,
    dump_cfg, suppress_print, suppress_wandb, suppress_logging
)
from autogaze.datasets.collate import collate_fn
from autogaze.models.autogaze import AutoGaze, AutoGazeConfig
from autogaze.models.autogaze.processing_autogaze import AutoGazeImageProcessor


def _determine_batch_size(global_batch_size, per_gpu_max_size, world_size, global_rank):

    if global_rank == 0:
        logging.info(f'Requested global batch size: {global_batch_size} with per GPU max size: {per_gpu_max_size} and world size: {world_size}')

    local_batch_size = global_batch_size // world_size
    if global_batch_size % world_size != 0:
        if global_rank == 0:
            logger.warning(f'Global (train) batch size {global_batch_size} is not divisible by world size {world_size}. Adjusting global batch size to {local_batch_size * world_size}.')
        global_batch_size = local_batch_size * world_size

    # If the local batch size is larger than the per GPU max size, adjust the gradient accumulation steps
    grad_acc_steps = 1
    if local_batch_size > per_gpu_max_size:
        grad_acc_steps = local_batch_size // per_gpu_max_size
        if local_batch_size % per_gpu_max_size != 0:
            if global_rank == 0:
                logger.warning(f'Local batch size {local_batch_size} is not divisible by per GPU max size {per_gpu_max_size}. Adjusting local batch size to {per_gpu_max_size}.')

        local_batch_size = min(local_batch_size, per_gpu_max_size)
        global_batch_size = local_batch_size * grad_acc_steps * world_size

    if global_rank == 0:
        logger.info(f'Global (train) batch size: {global_batch_size} ({local_batch_size} x {grad_acc_steps} x {world_size})')

    # Determine the val batch size
    val_batch_size = min(local_batch_size, per_gpu_max_size)
    if global_rank == 0:
        logger.info(f'Global (val) batch size: {val_batch_size * world_size} ({val_batch_size} x {world_size})')
    return local_batch_size, val_batch_size, grad_acc_steps


def setup_dist():
    if 'LOCAL_RANK' in os.environ and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        logging.warning('No distributed training detected. Running in single process mode.')
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)
        local_rank = 0
        global_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


@hydra.main(config_name='in1k_mae_reconstruction_ar_gaze_grpo', config_path='./configs', version_base=None)
def main(cfg: DictConfig):

    local_rank, global_rank, world_size = setup_dist()

    seed_everything(cfg.trainer.seed * (global_rank + 1))

    if cfg.trainer.exp_name is None:
        timestamp = datetime.now().strftime('%d%m%Y-%H%M')
        train_info = [
            f'{HydraConfig.get().job.config_name}',
            f'lr{str(cfg.trainer.lr)}',
            f'bs{cfg.trainer.batch_size}',
        ]

        exp_name = '_'.join(train_info + [timestamp])
    else:
        exp_name = cfg.trainer.exp_name

    exp_path = os.path.join(cfg.trainer.logdir, exp_name)

    # Ensure all processes wait for directory creation
    if global_rank == 0:
        os.makedirs(exp_path, exist_ok=True)
    torch.distributed.barrier()  # Wait for rank 0 to create directory

    # Wandb initialization
    if global_rank == 0:
        dump_cfg(cfg, exp_path)
        logger.add(os.path.join(exp_path, 'training.log'))
        logger.info(f'Logging to {exp_path}')
        logger.info(f'Config: \n {OmegaConf.to_yaml(cfg)}')

        wandb.init(
            project='autogaze',
            name=exp_name,
            id=exp_name,
            config=OmegaConf.to_container(cfg),
            dir=exp_path,
            resume='allow',
        )
    else:
        suppress_print()
        suppress_wandb()
        suppress_logging()
    
    # Create algorithm
    algorithm = instantiate(cfg.algorithm)

    # Create model
    cfg.model.max_num_frames = cfg.dataset.clip_len
    model_cfg = AutoGazeConfig(**OmegaConf.to_container(cfg.model))
    model = AutoGaze(model_cfg)
    cur_device = torch.cuda.current_device()
    ddp_model = DDP(model.cuda(), find_unused_parameters=True, device_ids=[cur_device], output_device=cur_device)

    # Create task
    task = instantiate(cfg.task)
    cur_device = torch.cuda.current_device()
    ddp_task = DDP(task.cuda(), find_unused_parameters=True, device_ids=[cur_device], output_device=cur_device)

    # Create transforms: AutoGaze uses its own preprocessing config but overrides size with task's scales
    task_scales = sorted([int(s) for s in str(cfg.task.scales).split('+')])
    preprocessing_cfg = OmegaConf.to_container(cfg.model.preprocessing, resolve=True)
    preprocessing_cfg['size'] = {"shortest_edge": task_scales[-1]}
    gaze_transform = AutoGazeImageProcessor(**preprocessing_cfg)

    # Create datasets with separate transforms for gaze model and task
    train_dataset = instantiate(cfg.dataset, split='train', gaze_transform=gaze_transform, task_transform=task.transform)
    val_dataset = instantiate(cfg.dataset, split='val', gaze_transform=gaze_transform, task_transform=task.transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Determine the batch size
    local_train_batch_size, local_val_batch_size, grad_acc_steps = _determine_batch_size(cfg.trainer.batch_size, cfg.trainer.per_gpu_max_batch_size, world_size, global_rank)
    train_loader = DataLoader(train_dataset, local_train_batch_size,  num_workers=4, drop_last=True, shuffle=False, sampler=train_sampler, worker_init_fn=seed_worker, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, local_val_batch_size, num_workers=4, drop_last=False, shuffle=False, sampler=val_sampler, worker_init_fn=seed_worker, collate_fn=collate_fn)

    # Create optimizer
    if cfg.trainer.optimizer == 'adam':
        optimizer = torch.optim.Adam(itertools.chain(ddp_model.parameters(), ddp_task.parameters()), cfg.trainer.lr)
    elif cfg.trainer.optimizer == 'sgd':
        optimizer = torch.optim.SGD(itertools.chain(ddp_model.parameters(), ddp_task.parameters()), cfg.trainer.lr)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.trainer.optimizer}")

    # Create trainer
    trainer = instantiate(
        cfg.trainer,
        gaze_model=ddp_model,
        task=ddp_task,
        algorithm=algorithm,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        save_dir=exp_path,
        grad_acc_steps=grad_acc_steps,
        gaze_processor=gaze_transform,
    )

    # Start training
    trainer.trainval()

    # Clean up
    destroy_process_group()

if __name__ == '__main__':
    main()
