#!/usr/bin/env python3
"""
Benchmark SlidingWindow vs alternatives on the hot path.

Scenarios:
  A) reset-only
  B) push-only
  C) push + _get_hist_data (raw window)
  D) push + endpoint SG smoothing (diff_order=0..3) via generic aggregator
  E) push + periodic reset (reset every R pushes)

Configs (defaults chosen to stress the memory bandwidth):
  - num_envs=8196
  - obs_dim=50
  - max_window_size=12
  - iters=4096
  - device=cpu (set --device cuda to run on GPU if available)

Alternatives:
  - SlidingWindow (from mjlab.utils.sliding_window) [time-last, double-width]
  - RingBufferTorch (time-first baseline)
  - RingBufferTorchTL (time-last, double-width)
  - DequeBuffer (collections.deque + torch.stack)
  - TensorDictBuffer (if tensordict is installed; otherwise skipped)

Run:
  python scripts/bench_sliding_window.py --device cpu
  python scripts/bench_sliding_window.py --device cuda --iters 8192
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import torch

# SlidingWindow + SG endpoint kernel
from mjlab.utils.sliding_window import SlidingWindow, sg_endpoint_kernel


def _sync(device: str) -> None:
  if device == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize()


def _format_num(n: float) -> str:
  if n >= 1e9:
    return f"{n/1e9:.2f} G"
  if n >= 1e6:
    return f"{n/1e6:.2f} M"
  if n >= 1e3:
    return f"{n/1e3:.2f} K"
  return f"{n:.2f}"


class RingBufferTorch:
  """Naive torch-only ring buffer (time-first)."""
  def __init__(self, num_envs: int, feat_dim: int, max_window_size: int, device: torch.device):
    self.W = int(max_window_size)
    self.N = int(num_envs)
    self.C = int(feat_dim)
    self.dev = device
    self.buf = torch.zeros(self.W, self.N, self.C, device=self.dev, dtype=torch.float32)
    self.count = 0
    self.idx = 0

  def reset(self) -> None:
    self.buf.zero_()
    self.count = 0
    self.idx = 0

  def push(self, x: torch.Tensor) -> None:
    # x: (N, C)
    self.buf[self.idx].copy_(x)
    self.idx = (self.idx + 1) % self.W
    self.count = min(self.count + 1, self.W)

  def _get_hist_data(self, max_window_size: int) -> torch.Tensor:
    # Returns (T, N, C); chronological; T <= W
    if max_window_size < 1:
      raise AssertionError("max_window_size must be >= 1")
    if self.count == 0:
      return self.buf.new_zeros((0, self.N, self.C))
    T = min(max_window_size, self.count)
    if self.count < self.W:
      window = self.buf[: self.count]
    else:
      if self.idx == 0:
        window = self.buf
      else:
        window = torch.cat((self.buf[self.idx:], self.buf[: self.idx]), dim=0)
    return window[-T:]


class RingBufferTorchTL:
  """Time-last ring buffer with double-width for contiguous tail slicing.

  Storage: (N, C, 2W), mirror writes to [i] and [i+W], so last T frames are
  _buf[:, :, i+W-T : i+W] contiguous.

  _get_hist_data returns (T, N, C) to match generic aggregator expectations.
  """
  def __init__(self, num_envs: int, feat_dim: int, max_window_size: int, device: torch.device):
    self.W = int(max_window_size)
    self.N = int(num_envs)
    self.C = int(feat_dim)
    self.dev = device
    self.buf = torch.zeros(self.N, self.C, 2 * self.W, device=self.dev, dtype=torch.float32)
    self.count = 0
    self.idx = 0  # modulo W

  def reset(self) -> None:
    self.buf.zero_()
    self.count = 0
    self.idx = 0

  def push(self, x: torch.Tensor) -> None:
    # x: (N, C)
    i = self.idx
    W = self.W
    self.buf[:, :, i].copy_(x)
    self.buf[:, :, i + W].copy_(x)
    self.idx = (i + 1) % W
    if self.count < W:
      self.count += 1

  def _get_hist_data(self, max_window_size: int) -> torch.Tensor:
    # Returns (T, N, C); chronological; T <= W
    if max_window_size < 1:
      raise AssertionError("max_window_size must be >= 1")
    if self.count == 0:
      return self.buf.new_zeros((0, self.N, self.C))
    T = min(max_window_size, self.count, self.W)
    end = self.idx + self.W  # exclusive
    start = end - T
    seq_nct = self.buf[:, :, start:end]  # (N, C, T) contiguous
    return seq_nct.permute(2, 0, 1).contiguous()  # (T, N, C)


class DequeBuffer:
  """collections.deque-based history buffer. Stores copies of inputs, stacks on read."""
  def __init__(self, num_envs: int, feat_dim: int, max_window_size: int, device: torch.device):
    self.W = int(max_window_size)
    self.N = int(num_envs)
    self.C = int(feat_dim)
    self.dev = device
    self.deq: deque[torch.Tensor] = deque(maxlen=self.W)

  def reset(self) -> None:
    self.deq.clear()

  def push(self, x: torch.Tensor) -> None:
    # Store a copy to be consistent with ring buffers (write into internal storage)
    self.deq.append(x.clone())

  def _get_hist_data(self, max_window_size: int) -> torch.Tensor:
    if max_window_size < 1:
      raise AssertionError("max_window_size must be >= 1")
    if not self.deq:
      return torch.zeros(0, self.N, self.C, device=self.dev, dtype=torch.float32)
    T = min(max_window_size, len(self.deq))
    return torch.stack(list(self.deq)[-T:], dim=0)  # (T, N, C)


class TensorDictBuffer:
  """TensorDict-based ring buffer (optional; requires tensordict)."""
  def __init__(self, num_envs: int, feat_dim: int, max_window_size: int, device: torch.device):
    from tensordict import TensorDict  # type: ignore
    self.W = int(max_window_size)
    self.N = int(num_envs)
    self.C = int(feat_dim)
    self.dev = device
    self.td = TensorDict({"buf": torch.zeros(self.W, self.N, self.C, device=self.dev, dtype=torch.float32)},
                         batch_size=())
    self.count = 0
    self.idx = 0

  def reset(self) -> None:
    self.td["buf"].zero_()
    self.count = 0
    self.idx = 0

  def push(self, x: torch.Tensor) -> None:
    self.td["buf"][self.idx].copy_(x)
    self.idx = (self.idx + 1) % self.W
    self.count = min(self.count + 1, self.W)

  def _get_hist_data(self, max_window_size: int) -> torch.Tensor:
    if max_window_size < 1:
      raise AssertionError("max_window_size must be >= 1")
    if self.count == 0:
      return self.td["buf"].new_zeros((0, self.N, self.C))
    T = min(max_window_size, self.count)
    if self.count < self.W:
      window = self.td["buf"][: self.count]
    else:
      if self.idx == 0:
        window = self.td["buf"]
      else:
        window = torch.cat((self.td["buf"][self.idx:], self.td["buf"][: self.idx]), dim=0)
    return window[-T:]


def sg_endpoint_aggregate(hist_TNC: torch.Tensor, max_window_size: int, diff_order: int, dt: float, poly_degree: int) -> torch.Tensor:
  """Generic endpoint SG aggregator applied to raw history.
  hist_TNC: (T, N, C), chronological.
  Returns: (N, C)
  """
  assert diff_order >= 0
  dev = hist_TNC.device
  dtype = hist_TNC.dtype
  T = hist_TNC.shape[0]
  N, C = hist_TNC.shape[1], hist_TNC.shape[2]

  # replicate-pad on the left if needed
  min_req = max(diff_order + 1, 1)
  target_T = max(min_req, min(max_window_size, max(T, 1)))
  if T < target_T:
    needed = target_T - T
    if T == 0:
      hist_TNC = torch.zeros(target_T, N, C, device=dev, dtype=dtype)
    else:
      pad = hist_TNC[:1].repeat(needed, 1, 1)
      hist_TNC = torch.cat([pad, hist_TNC], dim=0)
    T = target_T

  deg = max(diff_order, min(poly_degree, T - 1))
  w = sg_endpoint_kernel(T, deg, diff_order, dt, device=dev, dtype=dtype)  # (T,)
  return (hist_TNC * w.view(T, 1, 1)).sum(dim=0)  # (N, C)


def run_scenario_reset_only(name: str, impl, iters: int, device: str) -> float:
  _sync(device)
  t0 = time.perf_counter()
  for _ in range(iters):
    impl.reset()
  _sync(device)
  t1 = time.perf_counter()
  return iters / (t1 - t0)


def run_scenario_push_only(name: str, impl, inputs, iters: int, device: str) -> float:
  _sync(device)
  t0 = time.perf_counter()
  for i in range(iters):
    x = inputs[i % len(inputs)]
    impl.push(x)
  _sync(device)
  t1 = time.perf_counter()
  return iters / (t1 - t0)


def run_scenario_push_get_hist(name: str, impl, inputs, iters: int, max_window_size: int, device: str) -> float:
  _sync(device)
  t0 = time.perf_counter()
  for i in range(iters):
    x = inputs[i % len(inputs)]
    impl.push(x)
    _ = impl._get_hist_data(max_window_size)
  _sync(device)
  t1 = time.perf_counter()
  return iters / (t1 - t0)


def run_scenario_push_sg(name: str, impl, inputs, iters: int, max_window_size: int, diff_order: int, dt: float, poly_degree: int, device: str) -> float:
  _sync(device)
  t0 = time.perf_counter()
  for i in range(iters):
    x = inputs[i % len(inputs)]
    impl.push(x)
    hist = impl._get_hist_data(max_window_size)  # expected shape (T, N, C)
    _ = sg_endpoint_aggregate(hist, max_window_size=max_window_size, diff_order=diff_order, dt=dt, poly_degree=poly_degree)
  _sync(device)
  t1 = time.perf_counter()
  return iters / (t1 - t0)


def run_scenario_push_periodic_reset(name: str, impl, inputs, iters: int, max_window_size: int, reset_every: int, device: str) -> float:
  """Push with a periodic reset every `reset_every` steps. Returns ops/sec of loop iterations."""
  assert reset_every >= 1
  _sync(device)
  t0 = time.perf_counter()
  for i in range(iters):
    if (i % reset_every) == 0:
      impl.reset()
    x = inputs[i % len(inputs)]
    impl.push(x)
    _ = impl._get_hist_data(max_window_size)
  _sync(device)
  t1 = time.perf_counter()
  return iters / (t1 - t0)


def maybe_build_tensordict_buffer(num_envs: int, obs_dim: int, max_window_size: int, device: torch.device):
  try:
    import tensordict  # noqa: F401
  except Exception:
    return None
  return TensorDictBuffer(num_envs, obs_dim, max_window_size, device)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
  parser.add_argument("--num-envs", type=int, default=8196)
  parser.add_argument("--obs-dim", type=int, default=50)
  parser.add_argument("--window", type=int, default=12)
  parser.add_argument("--iters", type=int, default=4096)
  parser.add_argument("--dt", type=float, default=0.02)
  parser.add_argument("--diff-order", type=int, default=0, help="0=smooth/value, 1=vel, 2=acc, 3=jerk")
  parser.add_argument("--poly-degree", type=int, default=2)
  parser.add_argument("--warmup", type=int, default=128)
  parser.add_argument("--reset-every", type=int, default=256, help="Reset every R iterations in periodic reset scenario")
  args = parser.parse_args()

  dev = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
  device_str = "cuda" if dev.type == "cuda" else "cpu"

  N = args.num_envs
  C = args.obs_dim
  W = args.window

  print(f"Benchmark config:")
  print(f"  device         : {device_str}")
  print(f"  num_envs       : {N}")
  print(f"  obs_dim        : {C}")
  print(f"  window_size    : {W}")
  print(f"  iters          : {args.iters}")
  print(f"  dt             : {args.dt}")
  print(f"  diff_order     : {args.diff_order}")
  print(f"  poly_degree    : {args.poly_degree}")
  print(f"  reset_every    : {args.reset_every}")
  print()

  # Prepare inputs: reuse W random tensors to reduce allocation variability
  torch.manual_seed(0)
  inputs = [torch.randn(N, C, device=dev, dtype=torch.float32) for _ in range(W)]

  # Build implementations
  impls: list[tuple[str, object]] = []
  impls.append(("SlidingWindow", SlidingWindow(num_envs=N, feature_shape=torch.Size([C]), max_window_size=W, device=dev)))
  impls.append(("RingBufferTorch", RingBufferTorch(num_envs=N, feat_dim=C, max_window_size=W, device=dev)))
  impls.append(("RingBufferTorchTL", RingBufferTorchTL(num_envs=N, feat_dim=C, max_window_size=W, device=dev)))
  td_impl = maybe_build_tensordict_buffer(N, C, W, dev)
  if td_impl is not None:
    impls.append(("TensorDictBuffer", td_impl))
  impls.append(("DequeBuffer", DequeBuffer(num_envs=N, feat_dim=C, max_window_size=W, device=dev)))

  # Warm-up
  for name, impl in impls:
    # fill buffers and call reset a few times
    for i in range(max(args.warmup, W * 2)):
      impl.push(inputs[i % len(inputs)])
    _ = impl._get_hist_data(W)
    impl.reset()
    for i in range(W):
      impl.push(inputs[i % len(inputs)])
    _ = impl._get_hist_data(W)
    if device_str == "cuda":
      torch.cuda.synchronize()

  # Run scenarios
  print("Throughput (ops/sec). Higher is better.")
  header = f"{'Impl':<18} | {'Reset-only':>12} | {'Push-only':>12} | {'Push+get_hist':>14} | {'Push+SG(endpoint)':>18} | {'Push+periodic-reset(R='+str(args.reset_every)+')':>30}"
  print(header)
  print("-" * len(header))

  for name, impl in impls:
    reset_only = run_scenario_reset_only(name, impl, args.iters, device_str)
    push_only = run_scenario_push_only(name, impl, inputs, args.iters, device_str)
    push_hist = run_scenario_push_get_hist(name, impl, inputs, args.iters, args.window, device_str)
    push_sg = run_scenario_push_sg(name, impl, inputs, args.iters, args.window, args.diff_order, args.dt, args.poly_degree, device_str)
    push_periodic_reset = run_scenario_push_periodic_reset(name, impl, inputs, args.iters, args.window, args.reset_every, device_str)
    print(f"{name:<18} | {_format_num(reset_only):>12} | {_format_num(push_only):>12} | {_format_num(push_hist):>14} | {_format_num(push_sg):>18} | {_format_num(push_periodic_reset):>30}")

  print("\nNotes:")
  print("- Reset-only measures raw reset() throughput; implementations differ (full buffer zero vs clear metadata).")
  print("- Push+periodic-reset mimics env resets; tune --reset-every to your workload.")
  print("- Push+SG uses a generic endpoint Savitzky–Golay aggregator applied to the raw window for all implementations.")
  print("- On CUDA, ensure exclusive GPU usage during the run for consistent results.")


if __name__ == "__main__":
  main()