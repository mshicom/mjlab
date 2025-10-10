from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
  """Binary discriminator for AMP with safe numerics, optional WGAN-like mode."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    reward_scale: float,
    reward_clamp_epsilon: float = 1e-4,
    loss_type: Literal["BCEWithLogits", "Wasserstein"] = "BCEWithLogits",
    eta_wgan: float = 0.3,
    device: str | torch.device = "cpu",
  ) -> None:
    super().__init__()
    self.device = torch.device(device)
    self.input_dim = int(input_dim)
    self.reward_scale = float(reward_scale)
    self.reward_clamp_epsilon = float(reward_clamp_epsilon)
    self.loss_type = loss_type
    self.eta_wgan = float(eta_wgan)

    layers = []
    in_dim = self.input_dim
    for h in hidden_dims:
      layers += [nn.Linear(in_dim, h), nn.ReLU()]
      in_dim = h
    self.trunk = nn.Sequential(*layers).to(self.device)
    self.head = nn.Linear(in_dim, 1).to(self.device)

    if loss_type == "BCEWithLogits":
      self.loss_fun = nn.BCEWithLogitsLoss()
    elif loss_type == "Wasserstein":
      self.loss_fun = None  # use custom loss + grad penalty
    else:
      raise ValueError(f"Unsupported loss_type: {loss_type}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.trunk(x)
    return self.head(h)

  @torch.no_grad()
  def predict_reward(
    self, state: torch.Tensor, next_state: torch.Tensor, normalizer=None
  ) -> torch.Tensor:
    if normalizer is not None:
      state = normalizer.normalize(state)
      next_state = normalizer.normalize(next_state)
    logits = self.forward(torch.cat([state, next_state], dim=-1))
    if self.loss_type == "Wasserstein":
      logits = torch.tanh(self.eta_wgan * logits)
      return (self.reward_scale * torch.exp(logits)).squeeze(-1)
    prob = torch.sigmoid(logits)
    safe = torch.maximum(1 - prob, torch.tensor(self.reward_clamp_epsilon, device=prob.device))
    reward = -torch.log(safe)
    return (self.reward_scale * reward).squeeze(-1)

  def compute_losses(
    self,
    policy_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    grad_penalty: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.loss_type == "BCEWithLogits":
      expert_loss = nn.functional.binary_cross_entropy_with_logits(
        expert_logits, torch.ones_like(expert_logits)
      )
      policy_loss = nn.functional.binary_cross_entropy_with_logits(
        policy_logits, torch.zeros_like(policy_logits)
      )
      amp_loss = 0.5 * (expert_loss + policy_loss)
    else:
      policy_d = torch.tanh(self.eta_wgan * policy_logits)
      expert_d = torch.tanh(self.eta_wgan * expert_logits)
      amp_loss = policy_d.mean() - expert_d.mean()
    gp = grad_penalty if grad_penalty is not None else torch.tensor(0.0, device=amp_loss.device)
    return amp_loss, gp

  def gradient_penalty(
    self,
    expert_pair: tuple[torch.Tensor, torch.Tensor],
    policy_pair: tuple[torch.Tensor, torch.Tensor] | None,
    lambda_gp: float,
  ) -> torch.Tensor:
    expert = torch.cat(expert_pair, dim=-1)
    if self.loss_type == "Wasserstein" and policy_pair is not None:
      policy = torch.cat(policy_pair, dim=-1)
      alpha = torch.rand(expert.size(0), 1, device=expert.device)
      mix = alpha * expert + (1 - alpha) * policy
      target = 1.0  # WGAN-GP enforces norm ~ 1
    else:
      mix = expert
      target = 0.0  # BCE mode: push gradient norm toward 0 (regularization)
    mix.requires_grad_(True)
    out = self.forward(mix)
    grad = autograd.grad(
      outputs=out,
      inputs=mix,
      grad_outputs=torch.ones_like(out),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_gp * (grad.norm(2, dim=1) - target).pow(2).mean()