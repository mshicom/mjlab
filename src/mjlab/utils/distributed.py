"""Utilities for coordinating multi-GPU execution launched via ``torchrun``."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DistributedContext:
  """Stores basic information about the torchrun distributed environment."""

  is_distributed: bool
  world_size: int
  global_rank: int
  local_rank: int

  @property
  def is_main_process(self) -> bool:
    """Whether the current process is rank 0 in the distributed group."""

    return self.global_rank == 0


def get_distributed_context() -> DistributedContext:
  """Derive the distributed context from environment variables.

  Returns:
    A :class:`DistributedContext` describing the launch parameters. When the
    process is launched via :mod:`torchrun` this reflects the distributed
    configuration, otherwise a single-rank context is returned.
  """

  world_size = int(os.getenv("WORLD_SIZE", "1"))
  if world_size <= 1:
    return DistributedContext(False, world_size, 0, 0)

  global_rank = int(os.getenv("RANK", "0"))
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  return DistributedContext(True, world_size, global_rank, local_rank)


def resolve_distributed_device(requested_device: str, context: DistributedContext) -> str:
  """Resolve the device string that should be used for the current process.

  When running with multiple GPUs, each rank must target ``cuda:{LOCAL_RANK}``
  for :mod:`rsl_rl` to initialise its manual distributed backend correctly.

  Args:
    requested_device: The device specified by the user/configuration.
    context: The distributed launch context produced by
      :func:`get_distributed_context`.

  Returns:
    The device string that should be used for environment and runner
    construction.
  """

  if not context.is_distributed:
    return requested_device

  if not requested_device.startswith("cuda"):
    raise ValueError(
      "Multi-GPU training requires a CUDA device, but got "
      f"'{requested_device}'."
    )

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available but multi-GPU training was requested.")

  device_count = torch.cuda.device_count()
  if context.local_rank >= device_count:
    raise RuntimeError(
      f"LOCAL_RANK {context.local_rank} is outside the available CUDA device range "
      f"(found {device_count} devices)."
    )

  torch.cuda.set_device(context.local_rank)
  return f"cuda:{context.local_rank}"


def wait_for_path_update(path: Path, start_time: float, poll_interval: float = 0.1) -> None:
  """Wait until ``path`` exists and has been modified after ``start_time``.

  ``torchrun`` launches processes concurrently, so helper files created by the
  main rank may appear slightly later for the secondary ranks. This polling
  utility offers a lightweight synchronisation point before the process group
  itself has been initialised.
  """

  while True:
    if path.exists():
      try:
        mtime = path.stat().st_mtime
      except FileNotFoundError:
        # Another process might be updating the file between the existence
        # check and ``stat`` call. Retry immediately.
        time.sleep(poll_interval)
        continue
      if mtime >= start_time:
        return
    time.sleep(poll_interval)
