from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .amp.dist import is_dist_avail_and_initialized, reduce_gradients


@dataclass
class StateEstimatorCfg:
  input_dim: int
  output_dim: int  # e.g., 3 for base lin vel in body frame
  hidden_dims: Sequence[int] = (128, 128)
  activation: str = "elu"
  loss: Literal["l2", "l1", "smooth_l1"] = "l2"
  learning_rate: float = 1e-3


class ConcurrentStateEstimator(nn.Module):
  """On-device supervised estimator that predicts target states from actor obs.

  In VelocityAmpOnPolicyRunner we predict base linear velocity in body frame and
  replace the first 3 dims of the actor observation. Supports distributed grad averaging.
  """

  def __init__(self, cfg: StateEstimatorCfg, device: str | torch.device = "cpu") -> None:
    super().__init__()
    self.device = torch.device(device)
    Act = nn.ELU if cfg.activation == "elu" else nn.ReLU
    layers = []
    in_dim = cfg.input_dim
    for h in cfg.hidden_dims:
      layers += [nn.Linear(in_dim, h), Act()]
      in_dim = h
    layers += [nn.Linear(in_dim, cfg.output_dim)]
    self.net = nn.Sequential(*layers).to(self.device)
    self.cfg = cfg
    if cfg.loss == "l2":
      self._loss = nn.MSELoss()
    elif cfg.loss == "l1":
      self._loss = nn.L1Loss()
    else:
      self._loss = nn.SmoothL1Loss()
    self._opt = optim.Adam(self.parameters(), lr=cfg.learning_rate)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x.to(self.device))

  def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
    """Single-GPU fallback step for backward compatibility."""
    pred = self.forward(x)
    loss = self._loss(pred, y.to(self.device))
    self._opt.zero_grad(set_to_none=True)
    loss.backward()
    self._opt.step()
    return float(loss.item())

  def train_step_distributed(self, x: torch.Tensor, y: torch.Tensor) -> float:
    """Same as train_step but averages gradients across ranks when dist initialized."""
    pred = self.forward(x)
    loss = self._loss(pred, y.to(self.device))
    self._opt.zero_grad(set_to_none=True)
    loss.backward()
    if is_dist_avail_and_initialized():
      reduce_gradients([self])
    self._opt.step()
    return float(loss.item())