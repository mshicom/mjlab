from __future__ import annotations

from typing import Tuple, Union

import torch


class RunningMeanStd:
  """Running mean/std with numerically stable parallel variance update."""

  def __init__(
    self,
    epsilon: float = 1e-4,
    shape: Tuple[int, ...] = (),
    device: Union[str, torch.device] = "cpu",
  ) -> None:
    self.device = torch.device(device)
    self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
    self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
    self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

  @torch.no_grad()
  def update(self, arr: torch.Tensor) -> None:
    batch = arr.to(self.device, dtype=torch.float32)
    batch_mean = batch.mean(dim=0)
    batch_var = batch.var(dim=0, unbiased=False)
    batch_count = torch.tensor(batch.shape[0], dtype=torch.float32, device=self.device)
    self._update_from_moments(batch_mean, batch_var, batch_count)

  @torch.no_grad()
  def _update_from_moments(
    self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor
  ) -> None:
    delta = batch_mean - self.mean
    total_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
    new_var = m2 / total_count
    self.mean.copy_(new_mean)
    self.var.copy_(new_var)
    self.count.copy_(total_count)


class Normalizer(RunningMeanStd):
  """Normalizer with clipping."""

  def __init__(
    self,
    input_dim: Union[int, Tuple[int, ...]],
    epsilon: float = 1e-4,
    clip_obs: float = 10.0,
    device: Union[str, torch.device] = "cpu",
  ) -> None:
    shape = (input_dim,) if isinstance(input_dim, int) else tuple(input_dim)
    super().__init__(epsilon=epsilon, shape=shape, device=device)
    self.epsilon = epsilon
    self.clip_obs = clip_obs

  def normalize(self, input: torch.Tensor) -> torch.Tensor:
    x = input.to(self.device, dtype=torch.float32)
    std = (self.var + self.epsilon).sqrt()
    y = (x - self.mean) / std
    return torch.clamp(y, -self.clip_obs, self.clip_obs)