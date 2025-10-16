# Copyright (c) 2021-2025, ETH Zurich
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import torch
from torch import nn

from rsl_rl.networks.normalization import EmpiricalNormalization


@dataclass
class AMPConfig:
    """Configuration for the Adversarial Motion Prior (AMP) discriminator.

    This configuration governs both the network architecture and all loss and
    reward-shaping terms. It is modeled after MimicKit's AMP settings:
      - BCE classification of agent vs. demo states.
      - L2 regularization on the final logit layer.
      - Gradient penalty on the demo inputs.
      - Optional manual weight decay on all discriminator weights.
      - Reward shaping: r = -log(max(1 - sigmoid(logit), 1e-4)) * scale.

    Critical math/technical details matched from MimicKit:
      - Loss = 0.5*(BCE(agent, 0) + BCE(demo, 1)) + λ_logit||W_logit||^2
               + λ_gp E[||∂D/∂x_demo||^2] + λ_wd Σ||W_l||^2.
      - Reward uses probability p = sigmoid(logit): r = -log(max(1 - p, 1e-4)) * reward_scale.
      - Normalization: empirical running mean/std computed over agent and demo states.
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

    opt_weight_decay: float = 0.0
    '''Weight decay for the optimizer (Adam). This is separate from "disc_weight_decay" below.
    The latter adds an explicit L2 penalty term to the loss, matching MimicKit.'''

    logit_reg: float = 0.01
    '''Coefficient for L2 penalty on the final logit layer weights ||W_logit||^2.'''

    grad_penalty: float = 5.0
    '''Coefficient for gradient penalty on demo inputs: E[ ||∂D/∂x_demo||^2 ].
    Implemented exactly as in MimicKit (_compute_disc_loss).'''

    disc_weight_decay: float = 0.0001
    '''Manual L2 penalty coefficient on all discriminator weights Σ||W||^2.
    This matches MimicKit's "disc_weight_decay" which is separate from optimizer weight_decay.'''

    reward_scale: float = 1.0
    '''Scale for the AMP reward r_amp = -log(max(1 - sigmoid(logit), 1e-4)) * reward_scale.'''

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

    demo_batch_ratio: float = 1.0
    '''Ratio determining how many demo samples to draw relative to current agent batch size:
    n_demo = max(1, int(n_agent * demo_batch_ratio)).'''


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
    """Adversarial Motion Prior discriminator.

    This module contains:
      - An empirical normalizer over AMP state features (EmpiricalNormalization).
      - A small MLP backbone and a linear logit head (scalar output).
      - Methods to compute:
          * Reward: r = -log(max(1 - sigmoid(logit), 1e-4)) * reward_scale.
          * Loss: BCE agent vs. demo, logit L2 reg, demo grad penalty, manual weight decay.
      - Utilities for chunked reward evaluation and extracting AMP state from obs.

    All math and operations mirror MimicKit's AMPAgent/AMPModel design:
      - BCEWithLogitsLoss with targets 0/1.
      - Logit weight L2 regularization over the final linear layer only (logit_reg).
      - Gradient penalty computed on demo inputs with torch.autograd.grad.
      - Manual weight decay over all Linear layer weights (disc_weight_decay).
      - Reward computed from sigmoid(logits) and using an epsilon clamp of 1e-4.
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

    # -------------------------------
    # Utilities: weight collections
    # -------------------------------
    def get_logit_weights(self) -> torch.Tensor:
        """Return flattened weights of the final logit layer for L2 regularization."""
        return self.logits.weight.view(-1)

    def get_all_weights(self) -> torch.Tensor:
        """Concatenate flattened weights across all linear layers (including logits)."""
        flat_ws = []
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                flat_ws.append(m.weight.view(-1))
        flat_ws.append(self.logits.weight.view(-1))
        return torch.cat(flat_ws) if flat_ws else torch.zeros(1, device=self.device)

    # -------------------------------
    # Forward and reward
    # -------------------------------
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

    @torch.no_grad()
    def compute_reward(self, amp_state: torch.Tensor) -> torch.Tensor:
        """Compute AMP reward for given raw amp_state batch.

        reward = -log(max(1 - sigmoid(logit), 1e-4)) * reward_scale.

        Notes:
          - Uses EmpiricalNormalization for inputs (normalizer parameters are updated elsewhere).
          - Mirrors MimicKit's _calc_disc_rewards exactly (including clamp 1e-4 epsilon).
        """
        # Normalize inputs
        x = self.state_norm(amp_state)
        # Optionally chunk to reduce memory
        if self.cfg.eval_batch_size and self.cfg.eval_batch_size > 0 and x.shape[0] > self.cfg.eval_batch_size:
            rewards = []
            bs = self.cfg.eval_batch_size
            for i in range(0, x.shape[0], bs):
                logits = self.forward(x[i : i + bs])
                prob = torch.sigmoid(logits)
                r = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
                rewards.append(r * self.cfg.reward_scale)
            return torch.cat(rewards, dim=0)
        else:
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            r = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
            return r * self.cfg.reward_scale

    @torch.no_grad()
    def update_normalization(self, amp_state: torch.Tensor):
        """Update empirical normalization with a batch of raw amp_state."""
        # EmpiricalNormalization.update() safely no-ops in eval mode or if until reached.
        self.state_norm.update(amp_state)

    # -------------------------------
    # Loss
    # -------------------------------
    def compute_loss(self, agent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses using agent vs. demo batches.

        Mirrors MimicKit (AMPAgent._compute_disc_loss) with:
          - BCEWithLogitsLoss targeting 0 for agent logits, 1 for demo logits.
          - Average the two losses.
          - Add L2 penalty on final logit weights (logit_reg).
          - Add gradient penalty on demo inputs grad_penalty * E[||∂D/∂x_demo||^2].
          - Add manual weight decay disc_weight_decay * Σ||W||^2 across all Linear weights.

        Additionally:
          - Updates normalization statistics for both agent and demo inputs before evaluating loss, as in MimicKit
            (they record both agent and demo observations before updating normalizer).
        """
        device = agent_state.device
        B = agent_state.shape[0]
        n_demo = max(1, int(B * float(self.cfg.demo_batch_ratio)))

        # Sample demo batch
        demo_state = self.cfg.demo_provider(n_demo, device=device)
        
        # Update normalization (recording step, mirrors MimicKit behavior)
        with torch.no_grad():
            self.state_norm.update(agent_state)
            self.state_norm.update(demo_state)

        # Normalize inputs
        agent_norm = self.state_norm(agent_state)
        demo_norm = self.state_norm(demo_state)
        # Enable grad on demo inputs for gradient penalty
        if self.cfg.grad_penalty != 0.0:
            demo_norm.requires_grad_(True)

        # Logits
        agent_logit = self.forward(agent_norm)
        demo_logit = self.forward(demo_norm)

        # BCEWithLogits losses
        bce = nn.BCEWithLogitsLoss()
        loss_agent = bce(agent_logit, torch.zeros_like(agent_logit))
        loss_demo = bce(demo_logit, torch.ones_like(demo_logit))
        loss = 0.5 * (loss_agent + loss_demo)

        # Logit weight L2 regularization
        if self.cfg.logit_reg != 0.0:
            loss += self.cfg.logit_reg * torch.sum(self.get_logit_weights() ** 2)

        # Gradient penalty on demo inputs: E[||∂D/∂x_demo||^2]
        grad_penalty = torch.tensor(0.0, device=device)
        if self.cfg.grad_penalty != 0.0:
            grad = torch.autograd.grad(
                demo_logit,
                demo_norm,
                grad_outputs=torch.ones_like(demo_logit),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            # Sum of squared grads over feature dimension, mean over batch
            grad_sq = torch.sum(grad * grad, dim=-1)
            grad_penalty = torch.mean(grad_sq)
            loss += self.cfg.grad_penalty * grad_penalty

        # Manual discriminator weight decay over all layer weights (separate from optimizer)
        if self.cfg.disc_weight_decay != 0.0:
            all_w = self.get_all_weights()
            loss += self.cfg.disc_weight_decay * torch.sum(all_w * all_w)

        # Accuracy metrics (fraction correct)
        agent_acc = (agent_logit < 0.0).float().mean()
        demo_acc = (demo_logit > 0.0).float().mean()

        # Means of logits (for diagnostics)
        agent_logit_mean = agent_logit.mean()
        demo_logit_mean = demo_logit.mean()

        return {
            "amp_loss": loss,
            "amp_loss_agent": loss_agent.detach(),
            "amp_loss_demo": loss_demo.detach(),
            "amp_grad_penalty": grad_penalty.detach(),
            "amp_agent_acc": agent_acc.detach(),
            "amp_demo_acc": demo_acc.detach(),
            "amp_agent_logit": agent_logit_mean.detach(),
            "amp_demo_logit": demo_logit_mean.detach(),
        }

    # -------------------------------
    # Observation extraction
    # -------------------------------
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