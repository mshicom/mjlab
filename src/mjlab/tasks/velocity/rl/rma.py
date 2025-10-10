from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class RmaCfg:
  input_dim: int
  hidden_dims: Sequence[int] = (128, 128)
  latent_dim: int = 16
  activation: str = "elu"


class RmaEncoder(nn.Module):
  """RMA encoder that maps actor observation into a latent conditioning vector.

  This module is used in the runner to compute a residual added to the actor obs
  via a small learnable projection head (keeps obs dimensionality unchanged).
  """

  def __init__(self, cfg: RmaCfg, device: str | torch.device = "cpu") -> None:
    super().__init__()
    self.device = torch.device(device)
    layers = []
    in_dim = cfg.input_dim
    Act = nn.ELU if cfg.activation == "elu" else nn.ReLU
    for h in cfg.hidden_dims:
      layers += [nn.Linear(in_dim, h), Act()]
      in_dim = h
    layers += [nn.Linear(in_dim, cfg.latent_dim)]
    self.net = nn.Sequential(*layers).to(self.device)
    self.latent_dim = cfg.latent_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x.to(self.device))