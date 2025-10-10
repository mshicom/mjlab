from __future__ import annotations

import os
import statistics
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import torch
from collections import deque

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from mjlab.tasks.velocity.rl.amp.algorithm import AmpPPO
from mjlab.tasks.velocity.rl.amp.config import AmpCfg, AmpDatasetCfg, AmpDiscriminatorCfg
from mjlab.tasks.velocity.rl.amp.motion_loader import MotionDataset
from mjlab.tasks.velocity.rl.amp.normalizer import Normalizer
from mjlab.tasks.velocity.rl.amp.observation import build_amp_obs_from_env
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def _ampcfg_from_dict(d: dict) -> AmpCfg:
  """Build AmpCfg from dicts inside train_cfg."""
  ds = d.get("dataset", {})
  disc = d.get("discriminator", {})
  dataset = AmpDatasetCfg(
    root=ds["root"],
    names=list(ds["names"]),
    weights=list(ds["weights"]),
    slow_down_factor=int(ds.get("slow_down_factor", 1)),
  )
  discriminator = AmpDiscriminatorCfg(
    input_dim=int(disc["input_dim"]),
    hidden_dims=tuple(disc.get("hidden_dims", (256, 256))),
    reward_scale=float(disc.get("reward_scale", 0.1)),
    reward_clamp_epsilon=float(disc.get("reward_clamp_epsilon", 1e-4)),
    loss_type=disc.get("loss_type", "BCEWithLogits"),
    eta_wgan=float(disc.get("eta_wgan", 0.3)),
    grad_penalty_lambda=float(disc.get("grad_penalty_lambda", 10.0)),
  )
  cfg_dict = dict(d)
  cfg_dict["dataset"] = dataset
  cfg_dict["discriminator"] = discriminator
  # Let AmpCfg defaults fill the rest if not provided
  return AmpCfg(**{k: v for k, v in cfg_dict.items() if k in AmpCfg.__annotations__})


class VelocityAmpOnPolicyRunner(VelocityOnPolicyRunner):
  """AMP-enabled runner aligned with rsl_rl OnPolicyRunner.

  Differences from the base class:
  - _construct_algorithm builds AmpPPO instead of PPO.
  - learn() augments rollout with AMP observation pairs and blends style reward with task reward.
  - Logging uses a loss_dict similar to rsl_rl.PPO for compatibility.
  """

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    # Calls into _construct_algorithm() which we override below.
    super().__init__(env, train_cfg, log_dir, device)

  # Algorithm construction aligned with base class, but creating AmpPPO.
  def _construct_algorithm(self, obs) -> AmpPPO:
    # Resolve policy class and construct actor-critic just like base runner.
    policy_cfg = dict(self.policy_cfg)  # copy to avoid in-place edits
    actor_cls = eval(policy_cfg.pop("class_name"))
    actor_critic: ActorCritic | ActorCriticRecurrent = actor_cls(
      obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
    ).to(self.device)

    # Build the AMP dataset and normalizer.
    amp_section = self.cfg.get("amp", None)
    if amp_section is None:
      raise ValueError("VelocityAmpOnPolicyRunner expects 'amp' section in train_cfg.")

    amp_cfg: AmpCfg
    if is_dataclass(amp_section):
      amp_cfg = amp_section  # AmpCfg already
    elif isinstance(amp_section, dict):
      amp_cfg = _ampcfg_from_dict(amp_section)
    else:
      raise TypeError("amp section must be AmpCfg dataclass or dict.")

    # Step dt heuristic: use env.unwrapped.step_dt when present, else default to 1/50.
    step_dt = getattr(getattr(self.env, "unwrapped", self.env), "step_dt", 1.0 / 50.0)
    self._amp_dataset = MotionDataset(
      root=amp_cfg.dataset.root,
      names=amp_cfg.dataset.names,
      weights=amp_cfg.dataset.weights,
      simulation_dt=float(step_dt),
      slow_down_factor=amp_cfg.dataset.slow_down_factor,
      device=self.device,
    )
    self._amp_normalizer = Normalizer(
      input_dim=self._amp_dataset.amp_dim(), epsilon=1e-4, clip_obs=10.0, device=self.device
    )

    # DDP flags (optional) stored at top-level of config (kept optional).
    ddp_flags = {
      "ddp_enable": bool(self.cfg.get("ddp_enable", False)),
      "ddp_find_unused_parameters": bool(self.cfg.get("ddp_find_unused_parameters", False)),
      "ddp_broadcast_buffers": bool(self.cfg.get("ddp_broadcast_buffers", True)),
      "ddp_static_graph": bool(self.cfg.get("ddp_static_graph", False)),
    }

    # Initialize AmpPPO.
    alg = AmpPPO(
      actor_critic=actor_critic,
      amp_cfg=amp_cfg,
      amp_dataset=self._amp_dataset,
      device=self.device,
      amp_normalizer=self._amp_normalizer,
      **ddp_flags,
    )

    # Initialize storage (aligned to base).
    # Derive actor/critic shapes from obs (tensor or dict-like).
    if isinstance(obs, torch.Tensor):
      actor_obs_shape = (obs.shape[1],)
      critic_obs_shape = actor_obs_shape
    elif isinstance(obs, dict) or hasattr(obs, '__getitem__'):
      # Handle dict or dict-like mapping objects
      try:
        pol = obs["policy"] if "policy" in obs else None
        cri = obs["critic"] if "critic" in obs else pol
      except (KeyError, TypeError):
        # Fallback: get first available observation
        try:
          first_key = next(iter(obs.keys() if hasattr(obs, 'keys') else obs))
          first = obs[first_key]
          actor_obs_shape = (first.shape[1],)
          critic_obs_shape = actor_obs_shape
        except Exception:
          raise TypeError(f"Cannot extract observation shapes from type: {type(obs)}, keys: {list(obs.keys()) if hasattr(obs, 'keys') else 'N/A'}")
      else:
        if pol is None:
          # Fallback: get first available observation
          first_key = next(iter(obs.keys() if hasattr(obs, 'keys') else obs))
          first = obs[first_key]
          actor_obs_shape = (first.shape[1],)
          critic_obs_shape = actor_obs_shape
        else:
          actor_obs_shape = (pol.shape[1],)
          critic_obs_shape = (cri.shape[1],) if cri is not None else actor_obs_shape
    else:
      # last resort: raise with helpful error message
      raise TypeError(f"Unsupported observation type for AmpPPO storage initialization: {type(obs)}")

    alg.init_storage(
      num_envs=self.env.num_envs,
      num_transitions_per_env=self.num_steps_per_env,
      actor_obs_shape=actor_obs_shape,
      critic_obs_shape=critic_obs_shape,
      action_shape=(self.env.num_actions,),
    )
    return alg

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
    # initialize writer (same as base)
    self._prepare_logging_writer()

    # randomize initial episode lengths (for exploration) — same as base
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    # start learning
    # In rsl_rl's OnPolicyRunner, env.get_observations() returns a Tensor (or mapping); here we fetch and move to device.
    obs = self.env.get_observations()
    obs = obs.to(self.device) if isinstance(obs, torch.Tensor) else obs
    self.train_mode()

    # Book keeping (aligned with base)
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    # Ensure all parameters are in sync (optional; AmpPPO may no-op under DDP)
    if getattr(self, "is_distributed", False) and hasattr(self.alg, "broadcast_parameters"):
      try:
        self.alg.broadcast_parameters()
      except Exception:
        pass

    # Prepare AMP observation for current step
    amp_obs = build_amp_obs_from_env(getattr(self.env, "unwrapped", self.env))

    # Start training loop (aligned variable names for logging)
    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()
      # Rollout
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          # Sample actions
          actions = self.alg.act(obs, obs) if hasattr(self.alg, "act") else self.alg.policy.act(obs)
          # Log AMP current obs before stepping env; step the environment
          next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          # Build next AMP obs and style reward
          next_amp_obs = build_amp_obs_from_env(getattr(self.env, "unwrapped", self.env))
          style_rewards = self.alg._disc.predict_reward(
            amp_obs, next_amp_obs, normalizer=self._amp_normalizer
          )
          # Blend task and style rewards (50/50 by default), ensure matching shapes
          rewards = rewards.to(self.device)
          style_rewards = style_rewards.to(self.device)
          # Ensure both have shape [B, 1]
          if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
          if style_rewards.ndim == 1:
            style_rewards = style_rewards.unsqueeze(-1)
          blended_rewards = 0.5 * rewards + 0.5 * style_rewards

          # Move obs-like to device
          if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.to(self.device)
          dones = dones.to(self.device)
          # Process the step in AmpPPO and AMP replay pair
          self.alg.process_env_step(blended_rewards, dones, extras)
          if hasattr(self.alg, "process_amp_step"):
            self.alg.act_amp(amp_obs)
            self.alg.process_amp_step(next_amp_obs)
          amp_obs = next_amp_obs  # shift window

          # Book keeping (aligned with base)
          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += blended_rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            if len(new_ids) > 0:
              rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
              lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
              cur_reward_sum[new_ids] = 0
              cur_episode_length[new_ids] = 0

          obs = next_obs  # advance loop variable

        stop = time.time()
        collection_time = stop - start
        start = stop

        # compute returns (use obs as critic obs by default to align to base)
        self.alg.compute_returns(obs)

      # update policy (AmpPPO returns tuple; convert to dict for base logger)
      (
        mean_value_loss,
        mean_surrogate_loss,
        mean_amp_loss,
        mean_grad_pen_loss,
        mean_policy_pred,
        mean_expert_pred,
        mean_accuracy_policy,
        mean_accuracy_expert,
        mean_kl_divergence,
      ) = self.alg.update()

      loss_dict = {
        "value_function": mean_value_loss,
        "surrogate": mean_surrogate_loss,
        "amp": mean_amp_loss,
        "amp_grad_pen": mean_grad_pen_loss,
        "kl": mean_kl_divergence,
        # Extra AMP classifier stats (logged as losses for convenience)
        "amp_policy_pred": mean_policy_pred,
        "amp_expert_pred": mean_expert_pred,
        "amp_acc_policy": mean_accuracy_policy,
        "amp_acc_expert": mean_accuracy_expert,
      }

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      # logging (reuse base class' logic)
      if self.log_dir is not None and not getattr(self, "disable_logs", False):
        # Build the locals the base .log expects.
        log_locals = dict(
          it=it,
          tot_iter=tot_iter,
          start_iter=start_iter,
          num_learning_iterations=num_learning_iterations,
          collection_time=collection_time,
          learn_time=learn_time,
          ep_infos=ep_infos,
          rewbuffer=rewbuffer,
          lenbuffer=lenbuffer,
          loss_dict=loss_dict,
        )
        self.log(log_locals)  # base OnPolicyRunner.log

        # Save model
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      # Clear episode infos
      ep_infos.clear()
      # Save code state on first iter (same pattern as base)
      if it == start_iter and self.log_dir is not None and not getattr(self, "disable_logs", False):
        from rsl_rl.utils import store_code_state, __file__ as rsl_file
        git_file_paths = store_code_state(self.log_dir, [rsl_file])
        if self.logger_type in ["wandb", "neptune"] and git_file_paths:
          for path in git_file_paths:
            self.writer.save_file(path)

    # Save the final model after training (same as base)
    if self.log_dir is not None and not getattr(self, "disable_logs", False):
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))