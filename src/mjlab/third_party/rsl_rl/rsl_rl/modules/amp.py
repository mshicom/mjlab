# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import autograd

from mjlab.amp.config import AmpCfg, AmpDatasetCfg, AmpFeatureSetCfg
from mjlab.amp.loader import AmpMotionLoader
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.storage import ReplayBuffer


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, amp_reward_coef: float, hidden_layer_sizes: list[int], device: str, task_reward_lerp: float = 0.0):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        amp_layers: list[nn.Module] = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)
        self.trunk.train()
        self.amp_linear.train()
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: EmpiricalNormalization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, state.device)
                next_state = normalizer.normalize_torch(next_state, next_state.device)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(), d


class AdversarialMotionPrior(nn.Module):
    """
    New AMP that:
      - consumes per-step env features from extras["amp_observations"]
      - uses Trajectory npz + FeatureManager via AmpMotionLoader for expert batches
    """

    def __init__(
        self,
        reward_coef: float,
        discr_hidden_dims: list[int],
        task_reward_lerp: float,
        feature_set: AmpFeatureSetCfg,
        dataset: AmpDatasetCfg,
        replay_buffer_size: int = 10000,
        # Optimization / regularization
        state_normalization: bool = False,
        grad_penalty_lambda: float = 10.0,
        # Device
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # Expert loader
        self.loader = AmpMotionLoader(dataset, feature_set, device=device)
        observation_dim = int(self.loader.observation_dim)

        # Normalizer
        self.normalizer = EmpiricalNormalization(observation_dim) if state_normalization else None

        # Discriminator
        self.discriminator = Discriminator(
            input_dim=observation_dim * 2,
            amp_reward_coef=reward_coef,
            hidden_layer_sizes=discr_hidden_dims,
            device=device,
            task_reward_lerp=task_reward_lerp,
        ).to(device)

        # Replay buffer for policy transitions
        self.replay = ReplayBuffer(observation_dim, replay_buffer_size, device)

        # Coefficients
        self.reward_coef = reward_coef
        self.loss_coef = reward_coef
        self.grad_penalty_lambda = grad_penalty_lambda

        # Previous env feature to build (s, s_next)
        self._env_prev_feat: torch.Tensor | None = None
        self._env_prev_has: torch.BoolTensor | None = None

    def train(self, mode: bool = True):
        self.discriminator.train(mode)
        return self

    def eval(self):
        self.discriminator.eval()
        return self

    @torch.no_grad()
    def predict_reward(self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor) -> torch.Tensor:
        reward, _ = self.discriminator.predict_amp_reward(
            state, next_state, task_reward, normalizer=self.normalizer
        )
        return reward

    def add_transition(self, state: torch.Tensor, next_state: torch.Tensor):
        self.replay.insert(state, next_state)

    def policy_generator(self, num_batches: int, batch_size: int):
        return self.replay.feed_forward_generator(num_batches, batch_size)

    def expert_generator(self, num_batches: int, batch_size: int):
        return self.loader.feed_forward_generator(num_batches, batch_size)

    def update_from_env_extras(self, extras: dict, dones: torch.Tensor | None = None):
        curr_feat = extras["amp_observations"]
        if curr_feat is None:
            return
        if curr_feat.ndim == 1:
            curr_feat = curr_feat.unsqueeze(0)
        n_envs = curr_feat.shape[0]

        if self._env_prev_feat is None or self._env_prev_feat.shape[0] != n_envs:
            self._env_prev_feat = torch.zeros_like(curr_feat)
            self._env_prev_has = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

        done_mask = torch.zeros(n_envs, dtype=torch.bool, device=self.device) if dones is None else dones.reshape(-1).bool()
        prev_has = self._env_prev_has  # type: ignore

        if prev_has.any():
            add_mask = prev_has & (~done_mask)
            if add_mask.any():
                s = self._env_prev_feat[add_mask]  # type: ignore
                s_next = curr_feat[add_mask]
                self.add_transition(s, s_next)

        self._env_prev_feat = curr_feat
        self._env_prev_has.fill_(True)  # type: ignore
        self._env_prev_has[done_mask] = False  # type: ignore

    def compute_batch_losses(
            self,
            policy_state: torch.Tensor,
            policy_next_state: torch.Tensor,
            expert_state: torch.Tensor,
            expert_next_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
            """Compute discriminator/adversarial losses and stats."""
            # Optional normalization
            if self.normalizer is not None:
                with torch.no_grad():
                    policy_state = self.normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)

            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=self.grad_penalty_lambda
            )

            return amp_loss, grad_pen_loss, policy_d.mean().item(), expert_d.mean().item()
    
    def reset_env_buffer(self):
        self._env_prev_feat = None
        self._env_prev_has = None

    def state_dict(self, include_replay: bool = False) -> dict:
        """
        Returns a serializable AMP state dict.

        Args:
            include_replay: If True, include the replay buffer contents (can be large).
        """
        state = {
            "discriminator": self.discriminator.state_dict(),
            "reward_coef": self.reward_coef,
            "loss_coef": self.loss_coef,
            "grad_penalty_lambda": self.grad_penalty_lambda,
            "observation_dim": self.replay.obs_shape[0] if hasattr(self, "replay") else None,
        }
        # Normalizer stats (if enabled)
        if self.normalizer is not None:
            state["normalizer"] = {
                "count": self.normalizer.count.clone().cpu(),
                "mean": self.normalizer.mean.clone().cpu(),
                "var": self.normalizer.var.clone().cpu(),
            }
        else:
            state["normalizer"] = None

        # Replay buffer (optional)
        if include_replay and hasattr(self, "replay") and self.replay is not None:
            rb = self.replay
            state["replay"] = {
                "size": int(rb.size),
                "ptr": int(rb.ptr),
                "obs": rb.obs.clone().cpu(),
                "next_obs": rb.next_obs.clone().cpu(),
            }
        else:
            state["replay"] = None

        # No need to persist expert loader contents; re-hydrated from dataset paths.
        return state

    def load_state_dict(self, state: dict, strict: bool = True):
        """
        Loads AMP state from a dict created by state_dict().
        """
        # Discriminator
        self.discriminator.load_state_dict(state["discriminator"], strict=strict)

        # Hyperparameters (optional)
        self.reward_coef = state.get("reward_coef", self.reward_coef)
        self.loss_coef = state.get("loss_coef", self.loss_coef)
        self.grad_penalty_lambda = state.get("grad_penalty_lambda", self.grad_penalty_lambda)

        # Normalizer
        norm = state.get("normalizer", None)
        if norm is not None and self.normalizer is not None:
            with torch.no_grad():
                self.normalizer.count.copy_(norm["count"].to(self.device))
                self.normalizer.mean.copy_(norm["mean"].to(self.device))
                self.normalizer.var.copy_(norm["var"].to(self.device))

        # Replay buffer (optional)
        rb_state = state.get("replay", None)
        if rb_state is not None and hasattr(self, "replay") and self.replay is not None:
            rb = self.replay
            with torch.no_grad():
                rb.obs.copy_(rb_state["obs"].to(self.device))
                rb.next_obs.copy_(rb_state["next_obs"].to(self.device))
                rb.size = rb_state["size"]
                rb.ptr = rb_state["ptr"]

        # Clear env prev buffers (always recomputed on-the-fly)
        self.reset_env_buffer()
        return self

def resolve_amp_config(alg_cfg: dict, env) -> dict:
    """
    Resolve AMP config defaults:
      - If amp_cfg is present, ensure time_between_frames exists (from env.step_dt).
      - replay_buffer_size defaults to num_preload_transitions.
    """
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        amp_cfg:AmpCfg = alg_cfg["amp_cfg"]
       
    return alg_cfg