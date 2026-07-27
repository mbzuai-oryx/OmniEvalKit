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
import torch
import wandb
import shutil
from loguru import logger
from torch.optim.lr_scheduler import LinearLR, ConstantLR
from tqdm import tqdm
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup


from autogaze.utils import get_scheduled_temperature, move_inputs_to_cuda, unwrap_model
from autogaze.train import seed_everything


class Trainer:
    def __init__(self, gaze_model, task, algorithm, train_loader, val_loader, optimizer, n_epochs, temp_schedule_args, 
                 train_gaze=True, train_task=True, detach_task=False, val_nsteps=100, save_nsteps=300, save_dir=None, grad_acc_steps=1, resume=False, gaze_weights=None, task_weights=None, 
                 val_only=False, gaze_processor=None, **config):

        # Core modules
        self.gaze_model = gaze_model
        self.gaze_processor = gaze_processor
        self.task = task
        self.algorithm = algorithm
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.train_gaze = train_gaze
        self.train_task = train_task
        self.train_w_ntp = 'ntp' in type(self.algorithm).__name__.lower()
        self.detach_task = detach_task
        self.val_nsteps = val_nsteps
        self.save_nsteps = save_nsteps
        self.save_dir = save_dir
        self.grad_acc_steps = grad_acc_steps
        self.temperature = -1
        self.temp_schedule_args = temp_schedule_args
        self.total_steps = (len(self.train_loader) / grad_acc_steps) * self.n_epochs
        self.config = config
        self.val_only = val_only

        # Optimization related
        lr_schedule = config['lr_schedule']
        start_lr = config['lr']
        total_iters = len(self.train_loader) * self.n_epochs // self.grad_acc_steps
        if lr_schedule == 'linear':
            min_lr = config['min_lr']
            end_factor = min_lr / start_lr
            self.scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=total_iters
            )
        elif lr_schedule == 'linear_w_warmup':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_iters // 10,
                num_training_steps=total_iters
            )
        elif lr_schedule == 'constant':
            self.scheduler = ConstantLR(optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

        self.truncate_grads = self.config.get('truncate_grads', False)
        self.grad_norm = self.config.get('grad_norm', 1.0)

        if not self.train_gaze:
            for param in self.gaze_model.parameters():
                param.requires_grad = False
        if not self.train_task:
            for param in self.task.parameters():
                param.requires_grad = False
                
        # Calculate the number of trainable parameters
        num_trainable_params = sum(p.numel() for p in self.gaze_model.parameters() if p.requires_grad) + sum(p.numel() for p in self.task.parameters() if p.requires_grad)
        num_params = sum(p.numel() for p in self.gaze_model.parameters()) + sum(p.numel() for p in self.task.parameters())
        logger.info(f'Number of trainable params: {num_trainable_params} / {num_params} ({num_trainable_params / num_params:.2%})')
        
        # W&B logging
        self.train_step = 0
        wandb.define_metric("train/train_step")
        wandb.define_metric("train/*", step_metric="train/train_step")
        self.val_step = 0
        wandb.define_metric("val/val_step")
        wandb.define_metric("val/*", step_metric="val/val_step")

        # Checkpointing
        self.start_epoch = 0
        self.start_iteration = 0
        if gaze_weights is not None or task_weights is not None:
            self.load_checkpoint(gaze_model_path=gaze_weights, task_path=task_weights, resume=False)
        if resume == 'auto':
            self.load_checkpoint(resume=True)
        elif resume:
            self.load_checkpoint(resume_path=resume, resume=True)

    def save_checkpoint(self, epoch, iteration):
        task_ckpt = self.task.state_dict()
        train_ckpt = {
            'epoch': epoch,
            'iteration': iteration,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_step': self.train_step,
            'val_step': self.val_step,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_gaze_dir = os.path.join(self.save_dir, 'checkpoint_latest_gaze')
        unwrap_model(self.gaze_model).save_pretrained(latest_gaze_dir)
        if self.gaze_processor is not None:
            self.gaze_processor.save_pretrained(latest_gaze_dir)
        torch.save(task_ckpt, os.path.join(self.save_dir, 'checkpoint_latest_task.pt'))
        torch.save(train_ckpt, os.path.join(self.save_dir, 'checkpoint_latest_train.pt'))

        # Save periodic checkpoint
        checkpoint_dir = os.path.join(self.save_dir, f'checkpoint_ep{epoch}_iter{iteration}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        periodic_gaze_dir = os.path.join(checkpoint_dir, 'checkpoint_gaze')
        unwrap_model(self.gaze_model).save_pretrained(periodic_gaze_dir)
        if self.gaze_processor is not None:
            self.gaze_processor.save_pretrained(periodic_gaze_dir)
        torch.save(task_ckpt, os.path.join(checkpoint_dir, 'checkpoint_task.pt'))
        torch.save(train_ckpt, os.path.join(checkpoint_dir, 'checkpoint_train.pt'))

        # Keep only latest 2 checkpoint folders
        checkpoint_folders = [d for d in os.listdir(self.save_dir) 
                            if os.path.isdir(os.path.join(self.save_dir, d)) 
                            and d.startswith('checkpoint_ep')]
        checkpoint_folders.sort(key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))
        while len(checkpoint_folders) > 2:
            oldest_folder = checkpoint_folders.pop(0)
            shutil.rmtree(os.path.join(self.save_dir, oldest_folder))

    def load_checkpoint(self, gaze_model_path=None, task_path=None, resume_path=None, resume=False):
        if resume:
            if resume_path is None:
                resume_path = self.save_dir
            gaze_ckpt_path = os.path.join(resume_path, 'checkpoint_latest_gaze')
            train_ckpt_path = os.path.join(resume_path, 'checkpoint_latest_train.pt')
            task_ckpt_path = os.path.join(resume_path, 'checkpoint_latest_task.pt')

            if not os.path.exists(gaze_ckpt_path) or not os.path.exists(task_ckpt_path) or not os.path.exists(train_ckpt_path):
                logger.warning("Resuming checkpoint not found. Starting fresh.")
                return

            logger.info(f"Resuming from {resume_path}")
            
            task_ckpt = torch.load(task_ckpt_path, map_location='cpu')
            train_ckpt = torch.load(train_ckpt_path, map_location='cpu')

            unwrap_model(self.gaze_model).from_pretrained(gaze_ckpt_path)
            self.task.load_state_dict(task_ckpt)
            self.optimizer.load_state_dict(train_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(train_ckpt['scheduler_state_dict'])
            self.train_step = train_ckpt['train_step']
            self.val_step = train_ckpt['val_step']
            self.start_epoch = train_ckpt['epoch']
            self.start_iteration = train_ckpt['iteration']
        else:
            if gaze_model_path is not None:
                logger.info(f"Loading gaze model from {gaze_model_path}")
                gaze_ckpt = unwrap_model(self.gaze_model).from_pretrained(gaze_model_path)
                missing_keys, unexpected_keys = unwrap_model(self.gaze_model).load_state_dict(gaze_ckpt.state_dict(), strict=False)
                logger.info(f"Missing keys: {missing_keys}")
                logger.info(f"Unexpected keys: {unexpected_keys}")
            if task_path is not None:
                logger.info(f"Loading task model from {task_path}")
                task_ckpt = torch.load(task_path, map_location='cpu')
                missing_keys, unexpected_keys = self.task.load_state_dict(task_ckpt, strict=False)
                logger.info(f"Missing keys: {missing_keys}")
                logger.info(f"Unexpected keys: {unexpected_keys}")
    
    def _one_step(self, inputs):
        # temperature annealing
        self.temperature = get_scheduled_temperature(self.train_step, self.total_steps, self.temp_schedule_args)
        # Predict the gaze
        gaze_outputs = self.gaze_model(inputs, temperature=self.temperature, **getattr(self.task.module, 'gaze_model_kwargs', {}))

        # Run through the task
        if self.detach_task:
            with torch.no_grad():
                task_outputs = self.task(inputs, gaze_outputs)
        else:
            task_outputs = self.task(inputs, gaze_outputs)

        # Run through the RL algorithm
        alg_outputs = self.algorithm(inputs, gaze_outputs, task_outputs)

        return gaze_outputs, task_outputs, alg_outputs

    def _one_step_ntp(self, inputs):
        # Get the ground truth gaze
        gt_gazing_info = inputs['gt_gazing_info']

        # Get the probability of gazing
        gaze_outputs = self.gaze_model(inputs, gazing_info=gt_gazing_info, **getattr(self.task.module, 'gaze_model_kwargs', {}))

        # Get the task losses from the GT gazing info
        task_outputs = {}
        if "task_losses" in gt_gazing_info:
            assert gt_gazing_info['task_losses'].shape == gaze_outputs['gazing_pos'].shape, f'{gt_gazing_info["task_losses"].shape} != {gaze_outputs["gazing_pos"].shape}'
            task_outputs['task_losses'] = gt_gazing_info['task_losses']
            task_outputs['task_losses_mask'] = torch.ones_like(task_outputs['task_losses'])  # 1 means task loss is calculated for that gaze position. 0 means otherwise.

        # Get the NTP loss (to be consistent with previous naming, we still call it rl outputs here)
        alg_outputs = self.algorithm(inputs, gaze_outputs, task_outputs)

        return gaze_outputs, None, alg_outputs

    def extract_metrics(self, gaze_outputs=None, task_outputs=None, alg_outputs=None):
        metrics = dict()
        if task_outputs is not None:
            metrics.update(task_outputs['metrics'])
        if alg_outputs is not None:
            metrics.update(alg_outputs['metrics'])
        return metrics

    def train_epoch(self, ep, start_iter):
        pbar = tqdm(total=(len(self.train_loader) // self.grad_acc_steps)) if torch.distributed.get_rank() == 0 else None
        self.gaze_model.train()
        self.task.train()
        has_unapplied_grads = False
        accum_metrics = defaultdict(float)
        for i, inputs in enumerate(self.train_loader):
            # Skip the first start_iter iterations
            if i < start_iter:
                if (i % self.grad_acc_steps) == (self.grad_acc_steps - 1) and pbar is not None:
                    pbar.update(1)
                continue

            # Check for validation
            if self.train_step % self.val_nsteps == 0 and not has_unapplied_grads:
                logger.info(f'Validation at step {self.train_step}')
                self.validate()
            
            # Check for saving checkpoint
            if self.train_step % self.save_nsteps == 0 and not has_unapplied_grads:
                if torch.distributed.get_rank() == 0:
                    logger.info(f'Saving checkpoint at step {self.train_step}')
                    self.save_checkpoint(ep, i)
            
            inputs = move_inputs_to_cuda(inputs)
            inputs = self.algorithm.preprocess_inputs(inputs)

            # Do the forward pass
            if (i % self.grad_acc_steps) == (self.grad_acc_steps - 1):
                # Forward gaze, task, RL algorithm, and get metrics
                if self.train_w_ntp:
                    gaze_outputs, task_outputs, alg_outputs = self._one_step_ntp(inputs)
                else:
                    gaze_outputs, task_outputs, alg_outputs = self._one_step(inputs)
                metrics = self.extract_metrics(gaze_outputs, task_outputs, alg_outputs)
                metrics = {k: v / self.grad_acc_steps for k, v in metrics.items()}

                # Aggregate the losses
                loss = 0
                if self.train_gaze:
                    loss += alg_outputs['loss'].mean()
                if self.train_task:
                    loss += task_outputs['loss'].mean()

                # Backward pass
                loss /= self.grad_acc_steps
                loss.backward()

                # Gradient clipping
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.gaze_model.parameters(), self.grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.task.parameters(), self.grad_norm)

                # Step the optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()
                has_unapplied_grads = False

                # Update the learning rate
                self.scheduler.step()
                self.last_lr = self.optimizer.param_groups[0]['lr']

                for metric_name in metrics:
                    accum_metrics[metric_name] += metrics[metric_name]

                # Logging
                to_log = {
                    'train/train_step': self.train_step,
                    'train/lr': self.last_lr,
                    'train/temperature': self.temperature,
                }
                for k in accum_metrics:
                    to_log[f'train/{k}'] = accum_metrics[k].item()

                wandb.log(to_log)

                accum_metrics = defaultdict(float)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description(f'{ep} | loss: {loss * self.grad_acc_steps:.4f} probe_lr: {self.last_lr:.2e}')
                self.train_step += 1

            else:
                with self.gaze_model.no_sync():
                    with self.task.no_sync():
                        # Forward gaze, task, RL algorithm, and get metrics                
                        if self.train_w_ntp:
                            gaze_outputs, task_outputs, alg_outputs = self._one_step_ntp(inputs)
                        else:
                            gaze_outputs, task_outputs, alg_outputs = self._one_step(inputs)
                        metrics = self.extract_metrics(gaze_outputs, task_outputs, alg_outputs)
                        metrics = {k: v / self.grad_acc_steps for k, v in metrics.items()}

                        # Aggregate the losses
                        loss = 0
                        if self.train_gaze:
                            loss += alg_outputs['loss'].mean()
                        if self.train_task:
                            loss += task_outputs['loss'].mean()

                        # Backward pass
                        loss /= self.grad_acc_steps
                        loss.backward()

                        has_unapplied_grads = True

                        for metric_name in metrics:
                            accum_metrics[metric_name] += metrics[metric_name]

        if has_unapplied_grads:
            logger.warning('Batch size did not divide dataset size: applying partial-batch gradients.')
            # Apply the gradients to the model
            self.optimizer.step()
            self.optimizer.zero_grad()
            has_unapplied_grads = False
            self.scheduler.step()
            self.last_lr = self.optimizer.param_groups[0]['lr']

    @torch.no_grad()
    def validate(self):
        seed_everything(torch.distributed.get_rank())
        self.gaze_model.eval()
        self.task.eval()
        pbar = tqdm(total=len(self.val_loader)) if torch.distributed.get_rank() == 0 else None
        total = 0
        accum_metrics = defaultdict(float)
        for inputs in self.val_loader:
            inputs = move_inputs_to_cuda(inputs)

            # Forward gaze, task, and get metrics
            gaze_outputs = self.gaze_model(inputs, **getattr(self.task.module, 'gaze_model_kwargs', {}))
            task_outputs = self.task(inputs, gaze_outputs)
            metrics = self.extract_metrics(gaze_outputs, task_outputs)

            # Keep track of the metrics
            B = gaze_outputs['gazing_pos'].shape[0]
            for k in metrics:
                accum_metrics[k] += metrics[k] * B
            total += B

            # Visualizations
            self.task.module.visualize(inputs, gaze_outputs, task_outputs)

            if pbar is not None:
                pbar.update(1)
                # Create description string from accumulated metrics
                desc_parts = []
                for k, v in accum_metrics.items():
                    desc_parts.append(f'{k}: {v / total:.4f}')
                pbar.set_description_str(f'val: {" ".join(desc_parts)}')

        # Gather from all processes
        total = torch.tensor(total).cuda()
        torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM, async_op=False)
        for k in accum_metrics:
            accum_metrics[k] = torch.tensor(accum_metrics[k]).cuda()
            torch.distributed.all_reduce(accum_metrics[k], torch.distributed.ReduceOp.SUM, async_op=False)
            accum_metrics[k] = accum_metrics[k].item() / total

        # Command line logging
        for k in accum_metrics:
            logger.info(f'Validation {k}: {accum_metrics[k]:.4f}')

        # WandB logging
        to_log = {
            'val/val_step': self.val_step,
        }
        for k in accum_metrics:
            to_log[f'val/{k}'] = accum_metrics[k]
        wandb.log(to_log)

        self.val_step += 1
        self.gaze_model.train()
        self.task.train()

        seed_everything(self.train_step + torch.distributed.get_rank())
    
    def trainval(self):
        if self.val_only:
            self.validate()
            return
        
        for ep in range(self.start_epoch, self.n_epochs):
            logger.info(f"Epoch {ep}")
            self.train_epoch(ep, start_iter=self.start_iteration if ep == self.start_epoch else 0)

        logger.info("Final validation")
        self.validate()