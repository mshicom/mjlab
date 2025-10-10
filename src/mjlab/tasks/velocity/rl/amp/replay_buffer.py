from __future__ import annotations

from typing import Generator, Tuple, Union

import torch


class ReplayBuffer:
  """Fixed-size circular buffer of (state, next_state)."""

  def __init__(
    self,
    obs_dim: int,
    buffer_size: int,
    device: Union[str, torch.device] = "cpu",
  ) -> None:
    self.device = torch.device(device)
    self.buffer_size = int(buffer_size)
    self.states = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
    self.next_states = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
    self.step = 0
    self.num_samples = 0

  @torch.no_grad()
  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    states = states.to(self.device, dtype=torch.float32)
    next_states = next_states.to(self.device, dtype=torch.float32)
    bsz = states.shape[0]
    end = self.step + bsz
    if end <= self.buffer_size:
      self.states[self.step:end] = states
      self.next_states[self.step:end] = next_states
    else:
      first = self.buffer_size - self.step
      self.states[self.step:] = states[:first]
      self.next_states[self.step:] = next_states[:first]
      rem = bsz - first
      self.states[:rem] = states[first:]
      self.next_states[:rem] = next_states[first:]
    self.step = end % self.buffer_size
    self.num_samples = min(self.buffer_size, self.num_samples + bsz)

  def feed_forward_generator(
    self,
    num_mini_batch: int,
    mini_batch_size: int,
    allow_replacement: bool = True,
  ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    total = num_mini_batch * mini_batch_size
    if total > self.num_samples:
      if not allow_replacement:
        raise ValueError(
          f"Requested {total} but buffer contains {self.num_samples} samples."
        )
      cycles = (total + self.num_samples - 1) // self.num_samples
      big_size = self.num_samples * cycles
      big_perm = torch.randperm(big_size, device=self.device)
      indices = big_perm[:total] % self.num_samples
    else:
      indices = torch.randperm(self.num_samples, device=self.device)[:total]
    for i in range(num_mini_batch):
      batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
      yield self.states[batch_idx], self.next_states[batch_idx]

  def __len__(self) -> int:
    return self.num_samples