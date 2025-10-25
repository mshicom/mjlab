# Copyright (c) 2021-2025, ETH Zurich
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import torch
from torch import nn
from torch import autograd

from rsl_rl.networks.normalization import EmpiricalNormalization


@dataclass
class AMPConfig:
    """Configuration for the Adversarial Motion Prior (AMP) discriminator.

    This configuration governs both the network architecture and all loss and
    reward-shaping terms. It is modeled after MimicKit's AMP settings:
      - BCE classification of agent vs. demo states.
      - Optional Wasserstein-like loss with gradient penalty.
      - L2 regularization on the final logit layer.
      - Gradient penalty on the demo inputs (BCE) or mixed inputs (Wasserstein).
      - Optional manual weight decay on all discriminator weights.
      - Reward shaping:
          * BCE: r = -log(max(1 - sigmoid(logit), eps)) * scale,
          * Wasserstein: r = exp(tanh(eta * logit)) * scale.

    Critical math/technical details matched from the amp-rsl-rl example discriminator:
      - Wasserstein loss uses tanh(eta * D(x)) both for reward and loss stabilization.
      - Wasserstein gradient penalty uses interpolation between expert and policy with ||∇D||_2 penalty to target 1.
    """
    enabled: bool = False
    '''Whether to enable AMP training. Default is False.'''
    
    obs_key: str = "amp_state"
    '''Name of the observation group (key) in the environment's observation dict that contains AMP state.
    This must point to a flattened feature vector per environment step (shape [B, F]).
    Asserts will fail if this key is missing or the shape is inconsistent.'''

    hidden_dims: Tuple[int, ...] = (1024, 512)
    '''Hidden layer dimensions of the discriminator MLP (backbone).'''

    activation: str = "relu"
    '''Activation function name: one of {"elu", "relu", "tanh", "gelu"}.
    Asserts on unsupported names.'''

    init_output_scale: float = 1.0
    '''Uniform initialization scale for the final logit layer weights.
    If > 0, logits.weight is initialized uniformly in [-scale, scale] and bias=0.
    This mirrors MimicKit's amp_model.AMPModel disc head initialization.'''

    learning_rate: float = 5e-5
    '''Discriminator optimizer learning rate. Uses Adam optimizer.'''
    
    disc_loss_coef: float = 5.0
    '''Global scaling coefficient applied to the discriminator loss (for stability/weighting).'''

    logit_reg: float = 0.01
    '''Coefficient for L2 penalty on the final logit layer weights ||W_logit||^2.'''

    grad_penalty: float = 5.0
    '''Coefficient for gradient penalty:
       - BCE: E[ ||∂D/∂x_demo||^2 ].
       - Wasserstein: E[ (||∇D(α x_demo + (1-α) x_agent)||_2 - 1)^2 ].'''

    disc_weight_decay: float = 0.0001
    '''Manual L2 penalty coefficient on all discriminator weights Σ||W||^2 (if applied in loss).
    Typically optimizer weight_decay is preferred; keep 0 here if you set optimizer weight decay.'''

    reward_scale: float = 1.0
    '''Scale for the AMP reward.'''

    reward_clamp_epsilon: float = 1e-4
    '''Epsilon used to clamp arguments to log in BCE-style rewards: -log(max(1 - p, eps)).'''

    eval_batch_size: int = 0
    '''Optional chunk size for reward evaluation to control memory.
    If 0, no chunking is performed. If > 0, AMP reward is computed in chunks of this size.'''

    norm_until: Optional[int] = None
    '''Maximum number of samples for updating the running normalization statistics.
    Mirrors EmpiricalNormalization.until behavior; set None to update indefinitely.'''

    demo_provider: Optional[Callable[[int, torch.device], torch.Tensor]] = None
    '''Callable used to sample demo AMP states: demo_provider(n: int, device: torch.device) -> Tensor [n, F].
    This must return normalized-shape-compatible demos (same feature dimension as amp_state).
    Asserts in runtime if AMP is enabled and this is None.'''

    demo_batch_ratio: float = 0.2
    '''Ratio determining how many demo samples to draw relative to current agent batch size:
    n_demo = max(1, int(n_agent * demo_batch_ratio)).'''

    loss_type: str = "Wasserstein"
    '''Discriminator loss type: "BCEWithLogits" (default) or "Wasserstein".'''

    eta_wgan: float = 0.3
    '''Scaling factor applied inside tanh for Wasserstein-like stabilization per amp-rsl-rl example.'''


def _activation(name: str) -> nn.Module:
    lname = name.lower()
    if lname == "elu":
        return nn.ELU()
    if lname == "relu":
        return nn.ReLU()
    if lname == "tanh":
        return nn.Tanh()
    if lname == "gelu":
        return nn.GELU()
    assert False, f"Unsupported activation: {name}"


class AMPDiscriminator(nn.Module):
    """Adversarial Motion Prior discriminator with optional Wasserstein-like loss.

    This module contains:
      - An empirical normalizer over AMP state features (EmpiricalNormalization).
      - A small MLP backbone and a linear logit head (scalar output).
      - Methods to compute:
          * Reward:
              - BCE: r = -log(max(1 - sigmoid(logit), eps)) * reward_scale.
              - Wasserstein: r = exp(tanh(eta * logit)) * reward_scale.
          * Loss:
              - BCE: 0.5*(BCE(agent,0) + BCE(demo,1)) + logit L2 + demo grad penalty (+ optional manual WD).
              - Wasserstein: mean(tanh(eta)*D(agent)) - mean(tanh(eta)*D(demo)) + WGAN grad penalty
                using interpolated inputs and target 1-norm, plus optional logit L2 and manual WD.
      - Utilities for chunked reward evaluation and extracting AMP state from obs.

    The Wasserstein formulation follows the reference amp-rsl-rl discriminator implementation.
    """

    def __init__(self, amp_cfg: AMPConfig, amp_state_shape: torch.Size, device: str = "cpu"):
        """
        Args:
            amp_cfg: AMPConfig configuration (asserts will validate invariants).
            amp_state_shape: Shape of a single AMP state feature vector (F,) excluding batch dim.
                             Must be 1-D (flattened). Mirrors MimicKit AMP env output via flatten.
            device: Device for module and buffers.
        """
        super().__init__()
        # Validate shape
        assert len(amp_state_shape) == 1 and amp_state_shape.numel() > 0, (
            f"AMP state shape must be 1D flattened, got {amp_state_shape}."
        )
        # Store config and device
        self.cfg = amp_cfg
        self.device = torch.device(device)
        # Normalizer (matches MimicKit semantics: record both agent and demo samples over time)
        self.state_norm = EmpiricalNormalization(shape=amp_state_shape, until=amp_cfg.norm_until).to(self.device)
        # Discriminator MLP: backbone + final logit
        dims = [amp_state_shape.numel(), *amp_cfg.hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_activation(amp_cfg.activation))
        self.backbone = nn.Sequential(*layers)
        self.logits = nn.Linear(dims[-1], 1)
        # Initialize logit head optionally (uniform in [-s, s], bias=0) as in MimicKit
        if amp_cfg.init_output_scale != 0.0:
            nn.init.uniform_(self.logits.weight, -amp_cfg.init_output_scale, amp_cfg.init_output_scale)
            nn.init.zeros_(self.logits.bias)
        # BCE loss if needed
        self.bce = nn.BCEWithLogitsLoss() if self.cfg.loss_type == "BCEWithLogits" else None

    def get_logit_weights(self) -> torch.Tensor:
        """Return flattened weights of the final logit layer for L2 regularization."""
        return self.logits.weight.view(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits.

        Args:
            x: Tensor [B, F] normalized AMP states.

        Returns:
            logits: Tensor [B] (squeezed scalar logit per sample).
        """
        assert x.dim() == 2, f"AMP forward expects [B, F] normalized features, got shape {tuple(x.shape)}"
        h = self.backbone(x)
        return self.logits(h).squeeze(-1)

    def compute_reward(self, amp_state: torch.Tensor,*, detach: bool = True) -> torch.Tensor:
        """Compute AMP reward for given raw amp_state batch.

        - BCE: r = -log(max(1 - sigmoid(logit), eps)) * reward_scale
        - Wasserstein: r = exp(tanh(eta * logit)) * reward_scale

        Notes:
          - Uses EmpiricalNormalization for inputs (normalizer parameters are updated elsewhere).
          - Mirrors amp-rsl-rl example for Wasserstein reward path.
        """
        with torch.set_grad_enabled(not detach):
            x = self.state_norm(amp_state)
            if self.cfg.eval_batch_size and self.cfg.eval_batch_size > 0 and x.shape[0] > self.cfg.eval_batch_size:
                rewards = []
                bs = self.cfg.eval_batch_size
                for i in range(0, x.shape[0], bs):
                    logits = self.forward(x[i : i + bs])
                    rewards.append(self._reward_from_logits(logits))
                return torch.cat(rewards, dim=0)
            
            else:
                logits = self.forward(x)
                return self._reward_from_logits(logits)

    def _reward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Internal helper to map logits to rewards depending on loss_type."""
        if self.cfg.loss_type == "Wasserstein":
            # Per amp-rsl-rl example: tanh(eta * D) then exp() then scale
            d = torch.tanh(self.cfg.eta_wgan * logits)
            r = torch.exp(d)
        else:
            prob = torch.sigmoid(logits)
            r = -torch.log(torch.clamp(1.0 - prob, min=self.cfg.reward_clamp_epsilon))
        return r * self.cfg.reward_scale

    @torch.no_grad()
    def update_normalization(self, amp_state: torch.Tensor):
        """Update empirical normalization with a batch of raw amp_state."""
        self.state_norm.update(amp_state)

    def compute_loss(self, agent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses using agent vs. demo batches.

        BCE path mirrors MimicKit (AMPAgent._compute_disc_loss):
          - BCEWithLogitsLoss targeting 0 for agent logits, 1 for demo logits.
          - Average the two losses.
          - Add L2 penalty on final logit weights (logit_reg).
          - Add gradient penalty on demo inputs grad_penalty * E[||∂D/∂x_demo||^2].

        Wasserstein path mirrors amp-rsl-rl design:
          - amp_loss = mean(tanh(eta)*D(agent)) - mean(tanh(eta)*D(demo))
          - Gradient penalty on interpolated inputs: (||∇D(α demo + (1-α) agent)||_2 - 1)^2.
          - Optional logit L2 regularization.
          - check the underlying math at https://zhuanlan.zhihu.com/p/388486502

        Additionally:
          - Updates normalization statistics for both agent and demo inputs before evaluating loss.
        """
        device = agent_state.device
        B = agent_state.shape[0]
        
        # Sample demo batch
        demo_state = self.cfg.demo_provider(B, device=device)
        
        # Update normalization (recording step)
        with torch.no_grad():
            self.state_norm.update(agent_state)
            self.state_norm.update(demo_state)

        # Normalize inputs
        agent_norm = self.state_norm(agent_state)
        demo_norm = self.state_norm(demo_state)

        if self.cfg.loss_type == "Wasserstein":
            # Forward raw logits
            agent_logit = self.forward(agent_norm)
            demo_logit  = self.forward(demo_norm)

            # WGAN-like loss with tanh stabilization
            agent_w = torch.tanh(self.cfg.eta_wgan * agent_logit)
            demo_w  = torch.tanh(self.cfg.eta_wgan * demo_logit)
            amp_loss = agent_w.mean() - demo_w.mean()

            # Gradient penalty on interpolated samples (target unit norm), WGAN-GP
            alpha = torch.rand(demo_norm.size(0), 1, device=device)
            # Expand alpha across feature dimension
            alpha = alpha.expand_as(demo_norm)
            mixed = alpha * demo_norm + (1.0 - alpha) * agent_norm
            mixed.requires_grad_(True)
            scores = self.forward(mixed)
            grad = autograd.grad(
                outputs=scores,
                inputs=mixed,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            # GP term: (||grad||_2 - 1)^2
            gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
            loss = amp_loss + self.cfg.grad_penalty * gp

            # Logit L2 regularization
            if self.cfg.logit_reg != 0.0:
                loss = loss + self.cfg.logit_reg * torch.sum(self.get_logit_weights() ** 2)

            # Scaling
            loss = loss * self.cfg.disc_loss_coef

            return {
                "amp_loss": loss,
                "amp_grad_penalty": gp.detach(),
                "amp_agent_logit": agent_w.mean().detach(),
                "amp_demo_logit": demo_w.mean().detach(),
            }
            
        else:
            # ----------------- BCE PATH -----------------
            demo_norm.requires_grad_(True)
            agent_logit = self.forward(agent_norm)
            demo_logit  = self.forward(demo_norm)

            # BCEWithLogits losses
            assert self.bce is not None, "BCE loss requested but BCE criterion is None"
            loss_agent = self.bce(agent_logit, torch.zeros_like(agent_logit, device=device))
            loss_demo  = self.bce(demo_logit,  torch.ones_like(demo_logit,  device=device))
            loss = 0.5 * (loss_agent + loss_demo)

            # Logit weight L2 regularization
            loss = loss + self.cfg.logit_reg * torch.sum(self.get_logit_weights() ** 2)

            # Gradient penalty on demo inputs: E[||∂D/∂x_demo||^2]
            grad = autograd.grad(
                demo_logit,
                demo_norm,
                grad_outputs=torch.ones_like(demo_logit, device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_sq = torch.sum(grad * grad, dim=-1)
            grad_penalty = torch.mean(grad_sq)
            loss = loss + self.cfg.grad_penalty * grad_penalty

            # overall scaling
            loss = loss * self.cfg.disc_loss_coef
            
            return {
                "amp_loss": loss,
                "amp_grad_penalty": grad_penalty.detach(),
                "amp_agent_logit": torch.sigmoid(agent_logit).mean().detach(),
                "amp_demo_logit":  torch.sigmoid(demo_logit).mean().detach(),
            }


    def extract_state_from_obs(self, obs: object) -> torch.Tensor:
        """Extract the AMP state from an observations container by key.

        Expectations:
          - obs is a mapping-like object (e.g., dict, TensorDict) supporting obs[self.cfg.obs_key].
          - Returns tensor of shape [B, F].
        """
        x = obs[self.cfg.obs_key]
        return x


def resolve_amp_config(alg_cfg: Dict[str, Any], obs: Any, obs_groups: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """Resolve and validate AMP configuration.

    - If alg_cfg contains "amp_cfg" and amp_cfg["enabled"] is True:
        * Validate that obs contains the AMP obs_key (default "amp_state") at top-level (obs[obs_key]).
        * If env exposes a callable "sample_amp_demos(n, device) -> Tensor [n, F]" and amp_cfg lacks a demo_provider,
          assign it.
    - Otherwise, no changes.

    Args:
        alg_cfg: Algorithm configuration dict (will be modified in-place).
        obs: Observations returned by env.get_observations() at runner start.
        obs_groups: Resolved obs_groups (unused here but kept for symmetry with other resolvers).
        env: VecEnv (used to resolve demo provider if available).

    Returns:
        Updated alg_cfg (same object as input).
    """
    if "amp_cfg" not in alg_cfg:
        return alg_cfg
    amp_cfg = alg_cfg["amp_cfg"]
    if amp_cfg is None:
        return alg_cfg
    if not amp_cfg["enabled"]:
        return alg_cfg

    # Validate obs_key presence
    obs_key = amp_cfg["obs_key"]
    assert amp_cfg["obs_key"] in obs, f"AMP enabled but obs does not contain key '{obs_key}'. "

    # Resolve demo provider from environment if not provided
    if amp_cfg.get("demo_provider", None) is None:
        assert hasattr(env.unwrapped, "sample_amp_demos"), "AMP enabled but no demo_provider provided and env lacks sample_amp_demos(n, device)."
        amp_cfg["demo_provider"] = lambda n, device: env.unwrapped.sample_amp_demos(n, device=device)

    return alg_cfg