# Copyright (c) 2021-2025, ETH Zurich
# SPDX-License-Identifier: BSD-3-Clause
# ADD module: independent config + discriminator using MimicKit DiffNormalizer semantics

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Sequence

import functools
import torch
from torch import nn
import torch.distributed as dist


@dataclass
class ADDConfig:
    """Configuration for the ADDAgent-style discriminator (ADD).

    This class is intentionally independent (does not subclass AMPConfig) so the
    ADD hyperparameters are explicit and discoverable. Each field below is
    documented immediately after its declaration to improve discoverability.
    """

    enabled: bool = False
    """Whether ADD is enabled. If False the discriminator is not constructed."""

    obs_key: str = "add_state"
    """Observation dict key for ADD inputs. The environment must provide obs[obs_key]
    as the difference vectors (demo_state - agent_state) with shape [B, F]."""

    hidden_dims: Tuple[int, ...] = (1024, 512)
    """MLP backbone hidden layer sizes. Example: (256, 256)."""

    activation: str = "relu"
    """Activation function for backbone. One of {'elu','relu','tanh','gelu'}."""

    init_output_scale: float = 1.0
    """Uniform initialization scale for final linear (logit) layer weights. If 0.0, default init is used."""

    learning_rate: float = 1e-4
    """Learning rate for the ADD optimizer (Adam)."""

    logit_reg: float = 0.01
    """Coefficient for L2 penalty on final logit weights ||W_logit||^2 applied in loss."""

    grad_penalty: float = 1.0
    """Coefficient for gradient penalty on negative (diff) inputs (E[||∂D/∂x||^2])."""

    disc_weight_decay: float = 0.0001
    """Optimizer weight decay (L2) passed to Adam."""

    disc_loss_weight: float = 5.0
    """Global scalar to multiply ADD loss with (keeps parity with AMP naming)."""

    reward_scale: float = 2.0
    """Scale applied to intrinsic reward computed from discriminator logits."""

    min_diff: float = 1e-4
    """Minimum denominator used by DiffNormalizer to avoid division by very small means."""

    clip: float = float("inf")
    """Optional clipping value applied by DiffNormalizer to normalized diffs. Use inf to disable clipping."""


class DiffNormalizer(nn.Module):
    """MimicKit-style normalizer for difference inputs.

    Tracks per-feature running mean absolute values and normalizes diffs by dividing
    by clamp_min(mean_abs, min_diff). Supports distributed reduction via torch.distributed
    in update().
    """

    def __init__(
        self,
        shape: Sequence[int],
        device: torch.device | str = "cpu",
        init_mean: Optional[torch.Tensor] = None,
        min_diff: float = 1e-4,
        clip: float = float("inf"),
        dtype=torch.float,
    ):
        super().__init__()
        self._min_diff = float(min_diff)
        self._clip = float(clip)
        self.dtype = dtype
        self._build_params(shape, device, init_mean)

    def _build_params(self, shape, device, init_mean):
        device = torch.device(device)
        # long counter and mean-abs stored as non-grad parameters
        self._count = torch.nn.Parameter(torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), requires_grad=False)
        self._mean_abs = torch.nn.Parameter(torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)
        if init_mean is not None:
            assert tuple(init_mean.shape) == tuple(shape)
            self._mean_abs[:] = init_mean
        # accumulators for incoming records (kept as regular tensors)
        self._new_count = 0
        self._new_sum_abs = torch.zeros_like(self._mean_abs)

    def record(self, x: torch.Tensor) -> None:
        """Accumulate |x| sums and counts from a batch x (expects last dim == feature dim)."""
        assert x.dim() >= 2, "DiffNormalizer.record expects tensor with batch dim and feature dim"
        # flatten leading dims to collapse time/batch into a single axis if needed
        if x.dim() > 2:
            x_flat = x.flatten(start_dim=0, end_dim=x.dim() - 2)
        else:
            x_flat = x
        abs_sum = torch.sum(torch.abs(x_flat), dim=0)  # [F]
        self._new_count += x_flat.shape[0]
        self._new_sum_abs = self._new_sum_abs + abs_sum

    def update(self) -> None:
        """Aggregate accumulated statistics and update running mean_abs.

        If torch.distributed is initialized, reduce accumulators across processes
        using all_reduce(SUM) to match MimicKit's mp_util behavior.
        """
        new_count = torch.tensor(int(self._new_count), dtype=torch.long, device=self._mean_abs.device)
        new_sum_abs = self._new_sum_abs.to(self._mean_abs.device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(new_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_sum_abs, op=dist.ReduceOp.SUM)

        if new_count.item() == 0:
            # nothing to do; reset local accumulators
            self._new_count = 0
            self._new_sum_abs.zero_()
            return

        new_mean_abs = new_sum_abs / new_count.item()
        total = self._count + new_count
        w_old = (self._count.type(torch.float) / total.type(torch.float)).clamp(min=0.0)
        w_new = (new_count.type(torch.float) / total.type(torch.float)).clamp(min=0.0)
        # in-place updates for buffers
        self._mean_abs[:] = w_old * self._mean_abs + w_new * new_mean_abs
        self._count[:] = total

        # reset accumulators
        self._new_count = 0
        self._new_sum_abs.zero_()

    def get_shape(self):
        return tuple(self._mean_abs.shape)

    def get_abs_mean(self) -> torch.Tensor:
        return self._mean_abs

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by dividing with clamp_min(mean_abs, min_diff); optionally clip."""
        denom = torch.clamp_min(self._mean_abs, self._min_diff)
        norm_x = x / denom
        if self._clip != float("inf"):
            norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        return norm_x.type(self.dtype)

    def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp_min(self._mean_abs, self._min_diff)
        return (norm_x * denom).type(self.dtype)


class ADDDiscriminator(nn.Module):
    """Adversarial Differential Discriminators. https://ar5iv.labs.arxiv.org/html/2505.04961#A3

    This is an independent implementation (no subclassing). It contains:
      - A DiffNormalizer to record and normalize difference inputs (demo - agent).
      - An MLP backbone and a final linear logit head.
      - A fixed zero positive prototype buffer (_pos_diff).
      - Methods:
          * compute_loss(diff_batch) -> dict with losses/metrics (expects diff_batch = demo - agent).
          * compute_reward_from_diff(diff_batch) -> Tensor [B] intrinsic rewards.
          * save/load is supported via state_dict (buffers/parameters).
    """

    def __init__(self, cfg: ADDConfig, add_state_shape: torch.Size, device: str = "cpu"):
        super().__init__()
        assert len(add_state_shape) == 1 and add_state_shape.numel() > 0, "add_state_shape must be 1D (F,)"
        self.cfg = cfg
        self.device = torch.device(device)

        # Diff normalizer
        self.diff_norm = DiffNormalizer(shape=tuple(add_state_shape), device=self.device, init_mean=None, min_diff=cfg.min_diff, clip=cfg.clip)

        # Build MLP backbone
        dims = [add_state_shape.numel(), *list(cfg.hidden_dims)]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            act = cfg.activation.lower()
            if act == "elu":
                layers.append(nn.ELU())
            elif act == "relu":
                layers.append(nn.ReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {cfg.activation}")
        self.backbone = nn.Sequential(*layers)
        self.logits = nn.Linear(dims[-1], 1)
        if cfg.init_output_scale != 0.0:
            nn.init.uniform_(self.logits.weight, -cfg.init_output_scale, cfg.init_output_scale)
            nn.init.zeros_(self.logits.bias)

        # Positive prototype buffer (zero vector), saved/loaded with state_dict
        pos = torch.zeros(add_state_shape.numel(), device=self.device, dtype=torch.float32)
        self.register_buffer("_pos_diff", pos, persistent=True)

        self.bce = nn.BCEWithLogitsLoss()
        
        # Move module to device
        self.to(self.device)

    def get_logit_weights(self) -> torch.Tensor:
        return self.logits.weight.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits on already-normalized inputs or raw inputs (callers decide)."""
        h = self.backbone(x)
        logits = self.logits(h).squeeze(-1)
        return logits

    def compute_reward(self, diff_state: torch.Tensor,*, detach: bool = True) -> torch.Tensor:
        """Compute intrinsic reward for difference inputs (demo - agent).
        r = -log(max(1 - sigmoid(logit), 1e-4)) * reward_scale
        """
        with torch.set_grad_enabled(not detach):
            norm_diff = self.diff_norm.normalize(diff_state)
            logits = self.forward(norm_diff)
            prob = torch.sigmoid(logits)
            r = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
            r = r * self.cfg.reward_scale
        return r

    @torch.no_grad()
    def update_normalization(self, add_state: torch.Tensor):
        """Update normalization with a batch of raw add_state."""
        self.diff_norm.record(add_state)
        self.diff_norm.update()
        
    def compute_loss(self, diff_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss on a batch of diffs (demo - agent).

        Steps:
          - normalize diffs with diff_norm.normalize().
          - build positive prototype batch (unnormalized, to match MimicKit behavior) and evaluate logits:
              pos_logit = D(pos_batch)  # pos_batch is not normalized
              neg_logit = D(norm_diff)
          - BCELoss: 0.5*(BCE(pos,1)+BCE(neg,0)) + logit_reg + grad_penalty + manual weight decay
          - multiply by amp_loss_weight
        """
        device = diff_batch.device
        B = diff_batch.shape[0]
        pos_batch = self._pos_diff.unsqueeze(0).expand(B, -1).to(device)
            
        norm_diff = self.diff_norm.normalize(diff_batch)
        norm_diff.requires_grad_(True)

        pos_logit = self.forward(pos_batch)
        neg_logit = self.forward(norm_diff)

        loss_pos = self.bce(pos_logit, torch.ones_like(pos_logit))
        loss_neg = self.bce(neg_logit, torch.zeros_like(neg_logit))
        loss = 0.5 * (loss_pos + loss_neg)
        
        # logit weight regularization
        loss += self.cfg.logit_reg * torch.sum(self.get_logit_weights() ** 2)

        # gradient penalty on negative (diff) inputs
        grad = torch.autograd.grad(
            neg_logit,
            norm_diff,
            grad_outputs=torch.ones_like(neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_sq = torch.sum(grad * grad, dim=-1)
        grad_penalty = torch.mean(grad_sq)
        loss += self.cfg.grad_penalty * grad_penalty

        # scale total loss
        loss *= self.cfg.disc_loss_weight

        pos_acc = (pos_logit > 0.0).float().mean()
        neg_acc = (neg_logit < 0.0).float().mean()
        pos_logit_mean = pos_logit.mean()
        neg_logit_mean = neg_logit.mean()

        return {
            "add_loss": loss,
            "add_loss_pos": loss_pos.detach(),
            "add_loss_neg": loss_neg.detach(),
            "add_grad_penalty": grad_penalty.detach(),
            "add_pos_acc": pos_acc.detach(),
            "add_neg_acc": neg_acc.detach(),
            "add_pos_logit": pos_logit_mean.detach(),
            "add_neg_logit": neg_logit_mean.detach(),
        }


    def extract_state_from_obs(self, obs: object) -> torch.Tensor:
        x = obs[self.cfg.obs_key]
        return x

def resolve_add_config(alg_cfg: Dict[str, Any], obs: Any, obs_groups: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """Resolve and validate ADD configuration.
    """
    if "add_cfg" not in alg_cfg:
        return alg_cfg
    add_cfg = alg_cfg["add_cfg"]
    if add_cfg is None:
        return alg_cfg
    if not add_cfg["enabled"]:
        return alg_cfg

    # Validate obs_key presence
    obs_key = add_cfg["obs_key"]
    assert add_cfg["obs_key"] in obs, f"ADD enabled but obs does not contain key '{obs_key}'. "
    
    return alg_cfg