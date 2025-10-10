from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from .config import AmpCfg
from .discriminator import Discriminator
from .motion_loader import MotionDataset
from .normalizer import Normalizer
from .replay_buffer import ReplayBuffer
from .dist import (
  is_dist_avail_and_initialized,
  broadcast_module_params,
  all_reduce_mean_,
  sync_running_mean_std,
  reduce_gradients,
)


class AmpPPO:
  """Joint PPO + AMP trainer with optional DDP wrapping for multi-GPU.

  - If DDP is enabled (ddp_enable=True and torch.distributed initialized), policy and
    discriminator are wrapped with DistributedDataParallel. Gradients for these modules
    are synchronized automatically; manual gradient all-reduce is skipped.
  - Otherwise, synchronous gradient averaging (manual all-reduce) is used.
  - AMP normalizer running statistics (mean/var/count) are always synchronized via all-reduce.
  """

  def __init__(
    self,
    actor_critic: ActorCritic,
    amp_cfg: AmpCfg,
    amp_dataset: MotionDataset,
    device: str = "cpu",
    amp_normalizer: Optional[Normalizer] = None,
    # DDP options
    ddp_enable: bool = False,
    ddp_find_unused_parameters: bool = False,
    ddp_broadcast_buffers: bool = True,
    ddp_static_graph: bool = False,
  ) -> None:
    self.device = torch.device(device)
    self._ddp_enabled = bool(ddp_enable) and is_dist_avail_and_initialized()

    # Base modules
    actor_critic = actor_critic.to(self.device)
    in_dim = amp_cfg.discriminator.input_dim
    discriminator = Discriminator(
      input_dim=in_dim,
      hidden_dims=tuple(amp_cfg.discriminator.hidden_dims),
      reward_scale=amp_cfg.discriminator.reward_scale,
      reward_clamp_epsilon=amp_cfg.discriminator.reward_clamp_epsilon,
      loss_type=amp_cfg.discriminator.loss_type,
      eta_wgan=amp_cfg.discriminator.eta_wgan,
      device=self.device,
    )

    # DDP wrap policy + discriminator when enabled
    if self._ddp_enabled:
      # DDP auto-broadcasts parameters on first forward; no need for manual broadcast
      self.actor_critic = DDP(
        actor_critic,
        device_ids=[self.device.index] if self.device.type == "cuda" else None,
        output_device=self.device if self.device.type == "cuda" else None,
        find_unused_parameters=ddp_find_unused_parameters,
        broadcast_buffers=ddp_broadcast_buffers,
        static_graph=ddp_static_graph,
      )
      self.discriminator = DDP(
        discriminator,
        device_ids=[self.device.index] if self.device.type == "cuda" else None,
        output_device=self.device if self.device.type == "cuda" else None,
        find_unused_parameters=False,  # discriminator graph is simple
        broadcast_buffers=ddp_broadcast_buffers,
        static_graph=ddp_static_graph,
      )
    else:
      self.actor_critic = actor_critic
      self.discriminator = discriminator
      # One-time parameter broadcast to ensure identical init
      if is_dist_avail_and_initialized():
        broadcast_module_params(self.actor_critic, src=0)
        broadcast_module_params(self.discriminator, src=0)

    self._storage_uses_tensordict = False

    # AMP dataset/storage/normalizer
    obs_dim = in_dim // 2
    self.amp_dataset = amp_dataset
    self.amp_storage = ReplayBuffer(obs_dim=obs_dim, buffer_size=amp_cfg.replay.size, device=self.device)
    self.amp_normalizer = amp_normalizer

    # PPO hyperparameters
    L = amp_cfg.learn
    self.clip_param = L.clip_param
    self.num_learning_epochs = L.num_learning_epochs
    self.num_mini_batches = L.num_mini_batches
    self.value_loss_coef = L.value_loss_coef
    self.entropy_coef = L.entropy_coef
    self.gamma = L.gamma
    self.lam = L.lam
    self.max_grad_norm = L.max_grad_norm
    self.use_clipped_value_loss = L.use_clipped_value_loss
    self.schedule = L.schedule
    self.desired_kl = L.desired_kl
    self.use_smooth_ratio_clipping = L.use_smooth_ratio_clipping
    self.grad_penalty_lambda = amp_cfg.discriminator.grad_penalty_lambda
    self.amp_batch_size_override = amp_cfg.amp_batch_size_override

    # Single optimizer for both models (DDP exposes .parameters() of wrapped module)
    self.optimizer = torch.optim.Adam(
      [
        {"params": self.actor_critic.parameters(), "name": "policy"},
        {"params": self.discriminator.parameters(), "name": "disc_all"},
      ],
      lr=L.learning_rate,
    )

    self.storage: Optional[RolloutStorage] = None
    self.transition = RolloutStorage.Transition()
    self.amp_transition = RolloutStorage.Transition()

  # Helpers to access the underlying modules (works for DDP and non-DDP)
  @property
  def policy(self):
    return self.actor_critic.module if isinstance(self.actor_critic, DDP) else self.actor_critic

  @property
  def _disc(self):
    return self.discriminator.module if isinstance(self.discriminator, DDP) else self.discriminator

  # Storage orchestration

  def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    actor_obs_shape: Tuple[int, ...],
    critic_obs_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
  ) -> None:
    obs_kwargs = {
      "training_type": "rl",
      "num_envs": num_envs,
      "num_transitions_per_env": num_transitions_per_env,
      "actions_shape": action_shape,
      "device": self.device,
    }

    storage_created = False
    policy_obs = torch.zeros(num_envs, *actor_obs_shape, device=self.device)
    critic_shape = critic_obs_shape if critic_obs_shape is not None else actor_obs_shape
    critic_obs = torch.zeros(num_envs, *critic_shape, device=self.device)
    try:
      self.storage = RolloutStorage(
        obs={"policy": policy_obs, "critic": critic_obs},
        **obs_kwargs,
      )
      self._storage_uses_tensordict = isinstance(getattr(self.storage, "observations", None), TensorDict)
      storage_created = True
    except TypeError:
      self._storage_uses_tensordict = False

    if not storage_created:
      self.storage = RolloutStorage(
        obs_shape=actor_obs_shape,
        privileged_obs_shape=critic_obs_shape,
        rnd_state_shape=None,
        **obs_kwargs,
      )
      self._storage_uses_tensordict = False

  # Acting

  def act(
    self,
    obs: torch.Tensor | TensorDict,
    critic_obs: torch.Tensor | TensorDict,
  ) -> torch.Tensor:
    ac = self.policy
    if ac.is_recurrent:
      self.transition.hidden_states = ac.get_hidden_states()
    self.transition.actions = ac.act(obs).detach()
    self.transition.values = ac.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = ac.get_actions_log_prob(self.transition.actions).detach()
    self.transition.action_mean = ac.action_mean.detach()
    self.transition.action_sigma = ac.action_std.detach()
    self.transition.observations = self._prepare_storage_observation(obs, critic_obs)
    if not self._storage_uses_tensordict:
      self.transition.privileged_observations = critic_obs
    return self.transition.actions

  def act_amp(self, amp_obs: torch.Tensor) -> None:
    self.amp_transition.observations = amp_obs

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict) -> None:
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in infos:
      time_outs = infos["time_outs"].to(self.device)
      if time_outs.ndim == 1:
        time_outs = time_outs.unsqueeze(-1)
      self.transition.rewards += self.gamma * (self.transition.values * time_outs)
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.policy.reset(dones)

  def process_amp_step(self, amp_next_obs: torch.Tensor) -> None:
    self.amp_storage.insert(self.amp_transition.observations, amp_next_obs)
    self.amp_transition.clear()

  def compute_returns(self, last_critic_obs: torch.Tensor | TensorDict) -> None:
    last_values = self.policy.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  # Learning

  def update(self) -> tuple[float, float, float, float, float, float, float, float, float]:
    assert self.storage is not None
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_amp_loss = 0.0
    mean_gp = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    mean_acc_pol = 0.0
    mean_acc_exp = 0.0
    mean_acc_pol_n = 0.0
    mean_acc_exp_n = 0.0
    mean_kl = 0.0

    # RL mini-batch generator
    if self.policy.is_recurrent:
      generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    # AMP generators
    mb_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
    mb_amp_size = int(self.amp_batch_size_override) if self.amp_batch_size_override is not None else mb_size
    policy_amp_gen = self.amp_storage.feed_forward_generator(
      num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
      mini_batch_size=mb_amp_size,
      allow_replacement=True,
    )
    expert_amp_gen = self.amp_dataset.feed_forward_generator(
      num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
      mini_batch_size=mb_amp_size,
    )

    for sample, pol_amp, exp_amp in zip(generator, policy_amp_gen, expert_amp_gen):
      if self._storage_uses_tensordict:
        obs_td = sample[0]
        actions_b = sample[1]
        target_values_b = sample[2]
        advantages_b = sample[3]
        returns_b = sample[4]
        old_log_prob_b = sample[5]
        old_mu_b = sample[6]
        old_sigma_b = sample[7]
        hid_states_b = sample[8] if len(sample) > 8 else (None, None)
        masks_b = sample[9] if len(sample) > 9 else None
        rnd_state_b = sample[10] if len(sample) > 10 else None
        # Keep TensorDict as-is for the policy (it supports dict-like indexing)
        if isinstance(obs_td, TensorDict):
          obs_b = obs_td
          critic_obs_b = obs_td
        elif isinstance(obs_td, dict):
          # obs_td is a regular dict (not TensorDict), likely with "policy"/"critic" keys
          obs_b = obs_td
          critic_obs_b = obs_td
        else:
          # obs_td is a plain tensor, wrap it
          if hasattr(self.policy, 'obs_groups') and isinstance(self.policy.obs_groups, dict):
            obs_b = {"policy": obs_td}
            critic_obs_b = {"policy": obs_td}
          else:
            critic_obs_b = obs_td
            obs_b = obs_td
      else:
        (
          obs_b,
          critic_obs_b,
          actions_b,
          target_values_b,
          advantages_b,
          returns_b,
          old_log_prob_b,
          old_mu_b,
          old_sigma_b,
          hid_states_b,
          masks_b,
          rnd_state_b,
        ) = sample
        # Wrap observations for policy that expects dict (not in legacy mode)
        if hasattr(self.policy, 'obs_groups') and isinstance(self.policy.obs_groups, dict):
          legacy = getattr(self.policy, '_legacy_mode', False)
          if not legacy:
            obs_b = {"policy": obs_b}
            if critic_obs_b is not None and critic_obs_b is not obs_b:
              critic_obs_b = {"policy": obs_b["policy"], "critic": critic_obs_b}
            else:
              critic_obs_b = {"policy": obs_b["policy"]}

      # PPO forward
      ac = self.policy
      ac.act(obs_b, masks=masks_b, hidden_states=hid_states_b[0])
      actions_log_prob_b = ac.get_actions_log_prob(actions_b)
      value_b = ac.evaluate(critic_obs_b, masks=masks_b, hidden_states=hid_states_b[1])
      mu_b = ac.action_mean
      sigma_b = ac.action_std
      entropy_b = ac.entropy

      # Adaptive LR via global KL if requested
      if self.schedule == "adaptive" and self.desired_kl is not None:
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_b / (old_sigma_b + 1e-8) + 1.0e-5)
            + (old_sigma_b.pow(2) + (old_mu_b - mu_b).pow(2)) / (2.0 * sigma_b.pow(2))
            - 0.5,
            dim=-1,
          )
          kl_mean = kl.mean()
          if is_dist_avail_and_initialized():
            all_reduce_mean_(kl_mean)  # global KL for consistent LR
          mean_kl += kl_mean.item()
          lr = self.optimizer.param_groups[0]["lr"]
          if kl_mean > self.desired_kl * 2.0:
            lr = max(1e-5, lr / 1.5)
          elif kl_mean > 0.0 and kl_mean < self.desired_kl / 2.0:
            lr = min(1e-2, lr * 1.5)
          for g in self.optimizer.param_groups:
            g["lr"] = lr

      # PPO losses
      ratio = torch.exp(actions_log_prob_b - torch.squeeze(old_log_prob_b))
      if self.use_smooth_ratio_clipping:
        min_, max_ = 1.0 - self.clip_param, 1.0 + self.clip_param
        clipped_ratio = 1 / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4)) * (max_ - min_) + min_
      else:
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate = -torch.squeeze(advantages_b) * ratio
      surrogate_clipped = -torch.squeeze(advantages_b) * clipped_ratio
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_b + (value_b - target_values_b).clamp(-self.clip_param, self.clip_param)
        v_loss = torch.max((value_b - returns_b).pow(2), (value_clipped - returns_b).pow(2)).mean()
      else:
        v_loss = (returns_b - value_b).pow(2).mean()

      ppo_loss = surrogate_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_b.mean()

      # AMP losses
      pol_s, pol_ns = pol_amp
      exp_s, exp_ns = exp_amp
      if self.amp_normalizer is not None:
        with torch.no_grad():
          pol_s = self.amp_normalizer.normalize(pol_s)
          pol_ns = self.amp_normalizer.normalize(pol_ns)
          exp_s = self.amp_normalizer.normalize(exp_s)
          exp_ns = self.amp_normalizer.normalize(exp_ns)

      Bpol = pol_s.size(0)
      disc_in = torch.cat((torch.cat([pol_s, pol_ns], dim=-1), torch.cat([exp_s, exp_ns], dim=-1)), dim=0)
      logits = self._disc(disc_in)
      pol_logits, exp_logits = logits[:Bpol], logits[Bpol:]

      gp = self._disc.gradient_penalty(
        expert_pair=(exp_s, exp_ns),
        policy_pair=(pol_s, pol_ns),
        lambda_gp=self.grad_penalty_lambda,
      )
      amp_loss, gp_loss = self._disc.compute_losses(pol_logits, exp_logits, gp)

      # Normalizer update BEFORE backward pass to ensure consistent normalization
      # This prevents divergence from different normalizer states affecting gradients
      if self.amp_normalizer is not None:
        with torch.no_grad():
          self.amp_normalizer.update(pol_s)
          self.amp_normalizer.update(exp_s)
          # Sync normalizer stats immediately to ensure all ranks have same normalization
          sync_running_mean_std(self.amp_normalizer)

      # Combine and step
      loss = ppo_loss + (amp_loss + gp_loss)
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()

      # If not using DDP, manually all-reduce grads for both modules
      if not self._ddp_enabled:
        reduce_gradients([self.policy, self._disc])

      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()

      # Stats
      pol_prob = torch.sigmoid(pol_logits)
      exp_prob = torch.sigmoid(exp_logits)
      mean_value_loss += v_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_amp_loss += amp_loss.item()
      mean_gp += gp_loss.item()
      mean_policy_pred += pol_prob.mean().item()
      mean_expert_pred += exp_prob.mean().item()
      mean_acc_pol += torch.sum(torch.round(pol_prob) == torch.zeros_like(pol_prob)).item()
      mean_acc_exp += torch.sum(torch.round(exp_prob) == torch.ones_like(exp_prob)).item()
      mean_acc_pol_n += pol_prob.numel()
      mean_acc_exp_n += exp_prob.numel()

    nupd = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= nupd
    mean_surrogate_loss /= nupd
    mean_amp_loss /= nupd
    mean_gp /= nupd
    mean_policy_pred /= nupd
    mean_expert_pred /= nupd
    mean_acc_pol = mean_acc_pol / max(1.0, mean_acc_pol_n)
    mean_acc_exp = mean_acc_exp / max(1.0, mean_acc_exp_n)
    mean_kl /= nupd
    self.storage.clear()



    return (
      mean_value_loss,
      mean_surrogate_loss,
      mean_amp_loss,
      mean_gp,
      mean_policy_pred,
      mean_expert_pred,
      mean_acc_pol,
      mean_acc_exp,
      mean_kl,
    )

  def _prepare_storage_observation(
    self,
    actor_obs: torch.Tensor | TensorDict,
    critic_obs: Optional[torch.Tensor | TensorDict],
  ) -> torch.Tensor | TensorDict:
    if not self._storage_uses_tensordict:
      # Extract tensor from dict if needed
      if isinstance(actor_obs, dict):
        actor_obs = actor_obs.get("policy", actor_obs.get("observation", next(iter(actor_obs.values()))))
      return actor_obs
    if isinstance(actor_obs, TensorDict):
      return actor_obs.clone()
    critic_tensor = critic_obs if critic_obs is not None else actor_obs
    if isinstance(critic_tensor, TensorDict):
      critic_tensor = critic_tensor.get("critic", critic_tensor.get("policy"))
    if isinstance(critic_tensor, dict):
      critic_tensor = critic_tensor.get("critic", critic_tensor.get("policy", next(iter(critic_tensor.values()))))
    actor_tensor = actor_obs.detach().clone() if isinstance(actor_obs, torch.Tensor) else actor_obs
    if isinstance(actor_tensor, dict):
      actor_tensor = actor_tensor.get("policy", next(iter(actor_tensor.values()))).detach().clone()
    critic_tensor = critic_tensor.detach().to(actor_tensor.device).clone() if isinstance(critic_tensor, torch.Tensor) else critic_tensor
    if isinstance(critic_tensor, dict):
      critic_tensor = critic_tensor.get("critic", next(iter(critic_tensor.values()))).detach().clone()
    return TensorDict(
      {
        "policy": actor_tensor,
        "critic": critic_tensor,
      },
      batch_size=[actor_tensor.shape[0]],
      device=actor_tensor.device,
    )