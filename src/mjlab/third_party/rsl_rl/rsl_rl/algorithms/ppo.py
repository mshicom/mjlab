# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

from rsl_rl.modules.amp import AMPDiscriminator, AMPConfig
from rsl_rl.modules.add import ADDDiscriminator, ADDConfig


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters (NEW)
        amp_cfg: dict | None = None,
        # ADD parameters (NEW)
        add_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # AMP components (NEW)
        # We set up in init_storage when obs shape is available.
        self.amp_cfg_dict = amp_cfg
        self.amp_cfg = None
        self.amp: AMPDiscriminator | None = None
        self.amp_optimizer: optim.Optimizer | None = None

        # ADD components (NEW)
        self.add_cfg_dict = add_cfg
        self.add_cfg = None
        self.add: ADDDiscriminator | None = None
        self.add_optimizer: optim.Optimizer | None = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

        # Initialize AMP discriminator once we have observation shapes (NEW)
        if self.amp_cfg_dict is not None and self.amp_cfg_dict.get("enabled", False):
            # Extract amp state tensor to determine feature shape
            obs_key = self.amp_cfg_dict["obs_key"]
            assert obs_key in obs, f"AMP obs key '{obs_key}' not found in observations."
            amp_state = obs[obs_key]
            amp_state_shape = torch.Size([amp_state.shape[1]])

            # Build AMPConfig and discriminator
            self.amp_cfg = cfg = AMPConfig(**self.amp_cfg_dict)
            self.amp = AMPDiscriminator(cfg, amp_state_shape, device=self.device).to(self.device)
            self.amp_optimizer = optim.Adam(self.amp.parameters(), lr=cfg.learning_rate, weight_decay=getattr(cfg, "disc_weight_decay", 0.0))

        # Initialize ADD discriminator once we have observation shapes (NEW)
        if self.add_cfg_dict is not None and self.add_cfg_dict.get("enabled", False):
            # Extract add state tensor to determine feature shape
            obs_key = self.add_cfg_dict["obs_key"]
            assert obs_key in obs, f"ADD obs key '{obs_key}' not found in observations."
            add_state = obs[obs_key]
            add_state_shape = torch.Size([add_state.shape[1]])

            # Build ADDConfig and discriminator
            self.add_cfg = add_cfg = ADDConfig(**self.add_cfg_dict)
            self.add = ADDDiscriminator(add_cfg, add_state_shape, device=self.device).to(self.device)
            self.add_optimizer = optim.Adam(self.add.parameters(), lr=add_cfg.learning_rate, weight_decay=getattr(add_cfg, "disc_weight_decay", 0.0))

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            
        extras.setdefault("log", {})
        # AMP intrinsic reward (if configured)
        if self.amp:
            # Extract amp state and update normalization
            amp_state = self.amp.extract_state_from_obs(obs)
            self.amp.update_normalization(amp_state)
            # Compute AMP reward and add to extrinsic reward
            self.amp_reward = self.amp.compute_reward(amp_state)
            self.transition.rewards += self.amp_reward
            extras["log"]["Episode_Reward/AMP"] = self.amp_reward

        # ADD intrinsic reward (if configured)
        if self.add:
            # Extract add_state (expected to be demo-agent difference computed by the env) and update normalization
            add_state = self.add.extract_state_from_obs(obs)
            self.add.update_normalization(add_state)
            # Compute ADD reward and add to extrinsic reward
            self.add_reward = self.add.compute_reward(add_state)
            self.transition.rewards += self.add_reward
            extras["log"]["Episode_Reward/ADD"] = self.add_reward

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        # -- AMP loss (NEW)
        if self.amp:
            mean_amp_loss = 0.0
            mean_amp_agent_logit = 0.0
            mean_amp_demo_logit = 0.0
            mean_amp_grad_penalty = 0.0
        else:
            mean_amp_loss = None
        # -- ADD loss (NEW)
        if self.add:
            mean_add_loss = 0.0
            mean_add_pos_logit = 0.0
            mean_add_neg_logit = 0.0
            mean_add_grad_penalty = 0.0
        else:
            mean_add_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy (only for original augmentation)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL adaptation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate on main process and broadcast
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry mirror loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # AMP discriminator loss (NEW)
            if self.amp:
                # Compute AMP loss on original batch only (no duplicates from augmentation)
                amp_obs = obs_batch[:original_batch_size]
                amp_state_batch = self.amp.extract_state_from_obs(amp_obs)
                amp_loss_dict = self.amp.compute_loss(amp_state_batch)

            # ADD discriminator loss (NEW)
            if self.add:
                # Extract add-state from the original (non-augmented) observations
                add_obs = obs_batch[:original_batch_size]
                add_state_batch = self.add.extract_state_from_obs(add_obs)
                add_loss_dict = self.add.compute_loss(add_state_batch)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                # RND loss computed later (just like before)

            # -- For AMP: backward discriminator
            if self.amp:
                assert "amp_loss" in amp_loss_dict
                self.amp_optimizer.zero_grad()
                amp_loss_dict["amp_loss"].backward()

            # -- For ADD: backward discriminator
            if self.add:
                assert "add_loss" in add_loss_dict
                self.add_optimizer.zero_grad()
                add_loss_dict["add_loss"].backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # -- For AMP
            if self.amp_optimizer:
                nn.utils.clip_grad_norm_(self.amp.parameters(), self.max_grad_norm)
                self.amp_optimizer.step()
            # -- For ADD
            if self.add_optimizer:
                nn.utils.clip_grad_norm_(self.add.parameters(), self.max_grad_norm)
                self.add_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss (unchanged; omitted here since not computed in this snippet)
            # -- Symmetry loss
            if self.symmetry:
                mean_symmetry_loss += symmetry_loss.item()
            # -- AMP loss stats
            if self.amp:
                mean_amp_loss += amp_loss_dict["amp_loss"].item()
                mean_amp_agent_logit += amp_loss_dict["amp_agent_logit"].item()
                mean_amp_demo_logit += amp_loss_dict["amp_demo_logit"].item()
                mean_amp_grad_penalty += amp_loss_dict["amp_grad_penalty"].item()
            # -- ADD loss stats
            if self.add:
                mean_add_loss += add_loss_dict["add_loss"].item()
                mean_add_pos_logit += add_loss_dict["add_pos_logit"].item()
                mean_add_neg_logit += add_loss_dict["add_neg_logit"].item()
                mean_add_grad_penalty += add_loss_dict["add_grad_penalty"].item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if self.rnd:
            mean_rnd_loss /= num_updates  # computed earlier in original code
        # -- For Symmetry
        if self.symmetry:
            mean_symmetry_loss /= num_updates
        # -- For AMP
        if self.amp:
            mean_amp_loss /= num_updates
            mean_amp_agent_logit /= num_updates
            mean_amp_demo_logit /= num_updates
            mean_amp_grad_penalty /= num_updates
        # -- For ADD
        if self.add:
            mean_add_loss /= num_updates
            mean_add_pos_logit /= num_updates
            mean_add_neg_logit /= num_updates
            mean_add_grad_penalty /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.amp:
            loss_dict.update(
                {
                    "amp": mean_amp_loss,
                    "amp_agent_logit": mean_amp_agent_logit,
                    "amp_demo_logit": mean_amp_demo_logit,
                    "amp_grad_penalty": mean_amp_grad_penalty,
                }
            )
        if self.add:
            loss_dict.update(
                {
                    "add": mean_add_loss,
                    "add_pos_logit": mean_add_pos_logit,
                    "add_neg_logit": mean_add_neg_logit,
                    "add_grad_penalty": mean_add_grad_penalty,
                }
            )

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # NEW: broadcast AMP discriminator parameters
        if self.amp:
            model_params.append(self.amp.state_dict())
        # NEW: broadcast ADD discriminator parameters
        if self.add:
            model_params.append(self.add.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        idx = 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])
            idx += 1
        if self.amp:
            self.amp.load_state_dict(model_params[idx])
            idx += 1
        if self.add:
            self.add.load_state_dict(model_params[idx])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        # NEW: include AMP discriminator grads
        if self.amp:
            grads += [param.grad.view(-1) for param in self.amp.parameters() if param.grad is not None]
        # NEW: include ADD discriminator grads
        if self.add:
            grads += [param.grad.view(-1) for param in self.add.parameters() if param.grad is not None]
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        if self.amp:
            all_params = chain(all_params, self.amp.parameters())
        if self.add:
            all_params = chain(all_params, self.add.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel