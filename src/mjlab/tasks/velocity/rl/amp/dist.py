import torch
import torch.distributed as dist
from torch import nn


def is_dist_avail_and_initialized() -> bool:
  return dist.is_available() and dist.is_initialized()


def get_rank(default: int = 0) -> int:
  return dist.get_rank() if is_dist_avail_and_initialized() else default


def get_world_size(default: int = 1) -> int:
  return dist.get_world_size() if is_dist_avail_and_initialized() else default


@torch.no_grad()
def broadcast_module_params(module: nn.Module, src: int = 0) -> None:
  """Broadcasts parameters and buffers of a module from src to all ranks."""
  if not is_dist_avail_and_initialized():
    return
  # state_dict preserves tensor dtypes/devices; broadcast each tensor
  for tensor in module.state_dict().values():
    dist.broadcast(tensor, src)


@torch.no_grad()
def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
  """In-place all-reduce SUM and divide by world size. Returns tensor for convenience."""
  if not is_dist_avail_and_initialized():
    return tensor
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
  tensor /= float(get_world_size())
  return tensor


@torch.no_grad()
def sync_running_mean_std(rms) -> None:
  """Synchronize RunningMeanStd-like stats by averaging mean/var and summing count."""
  if not is_dist_avail_and_initialized():
    return
  ws = float(get_world_size())
  mean = rms.mean.clone()
  var = rms.var.clone()
  count = rms.count.clone()

  dist.all_reduce(mean, op=dist.ReduceOp.SUM)
  dist.all_reduce(var, op=dist.ReduceOp.SUM)
  dist.all_reduce(count, op=dist.ReduceOp.SUM)

  mean /= ws
  var /= ws
  rms.mean.copy_(mean)
  rms.var.copy_(var)
  rms.count.copy_(count)


def reduce_gradients(modules: list[nn.Module]) -> None:
  """All-reduce gradients in-place for a list of modules. Averages over ranks.

  Uses per-parameter all-reduce (memory-friendly, simple; DDP would be faster in practice).
  """
  if not is_dist_avail_and_initialized():
    return
  ws = float(get_world_size())
  for m in modules:
    for p in m.parameters():
      if p.grad is None:
        continue
      dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
      p.grad.data /= ws