from __future__ import annotations

from typing import Optional

import math
import torch
from torch import Tensor
from functools import cache


@cache
def sg_endpoint_kernel(
  window_size: int,
  poly_degree: int,
  deriv_order: int,
  dt: float,
  device: Optional[torch.device] = None,
  dtype: torch.dtype = torch.float32,
) -> Tensor:
  """Compute endpoint Savitzky–Golay convolution weights.

  Maps the last `window_size` samples (oldest->newest) to the endpoint value (deriv_order=0)
  or endpoint derivative of order `deriv_order` (>=1) by least-squares polynomial fit.
  """
  assert window_size >= 1
  assert poly_degree >= 0
  assert deriv_order >= 0
  assert dt > 0.0

  K = int(window_size)
  p = min(poly_degree, K - 1)
  if p < deriv_order:
    p = deriv_order

  dev = device if device is not None else torch.device("cpu")
  x = torch.arange(-K + 1, 1, device=dev, dtype=dtype) * float(dt)  # (K,)
  # Vandermonde A[i,j] = x[i]^j, (K, p+1)
  A = torch.stack([x.pow(j) for j in range(p + 1)], dim=1)
  ATA = A.transpose(0, 1) @ A
  ATA_pinv = torch.linalg.pinv(ATA)
  AT = A.transpose(0, 1)
  e_m = torch.zeros(p + 1, device=dev, dtype=dtype)
  e_m[deriv_order] = 1.0
  coeff_row = (e_m @ ATA_pinv) @ AT  # (K,)
  w = coeff_row * float(math.factorial(deriv_order))
  return w


class SlidingWindow(torch.nn.Module):
  """Time-first sliding window optimized for throughput (single-width ring buffer) with partial reset.

  Storage layout:
    - _buffer: (W, N, C_flat) where W=window_size

  API:
    - push(x): x is (N, C_flat), contiguous preferred.
    - reset(env_ids: Optional[Tensor|slice]): partial or global reset.
    - get_hist_data(T): (T, N, C_flat) chronological tail with per-env replicate padding (and global zero-left-pad after global reset).
    - get_hist_data_smooth_and_diffed(T, diff_order, diff_dt, poly_degree): (N, C_flat)
      Applies per-env replicate padding and then SG endpoint.

  Notes:
    - _valid_len (N,) tracks per-env valid history length (0..W). Updated on push and reset.
    - Endpoint SG supports variable window, adaptive polynomial degree, and per-env replicate padding.
  """

  def __init__(
    self,
    num_envs: int,
    feature_shape: torch.Size,
    max_window_size: int,
    device: Optional[torch.device] = None,
  ):
    super().__init__()
    assert max_window_size >= 1
    self.max_window_size: int = int(max_window_size)
    self.num_envs: int = int(num_envs)

    nf = 1
    for d in feature_shape:
      nf *= int(d)
    self.num_features: int = nf

    dev = device if device is not None else torch.device("cpu")
    # Time-first ring buffer: (W, N, C)
    self.register_buffer("_buffer", torch.zeros(self.max_window_size, self.num_envs, self.num_features, device=dev))
    # Per-env valid length since last (partial) reset: 0..W
    self.register_buffer("_valid_len", torch.zeros(self.num_envs, dtype=torch.int32, device=dev))
    # Global counters (Python ints to avoid device sync)
    self._count: int = 0           # max valid length across all envs (for global zero-pad in get_hist_data)
    self._write_idx: int = 0       # modulo W

  def reset(self, env_ids: Optional[torch.Tensor | slice] = None) -> None:
    """Reset history. If env_ids is None, global reset; otherwise, partial reset for selected envs."""
    if env_ids is None:
      # Cheap global reset: do not zero buffer; pad zeros/replicate will handle reads.
      self._valid_len.zero_()
      self._count = 0
      self._write_idx = 0
    else:
      # Partial reset: only reset valid length for given envs.
      self._valid_len[env_ids] = 0
      # Keep global counters; writes continue happily.

  def push(self, x: Tensor) -> None:
    """Push newest observation x: (N, C_flat)."""
    if x.dim() != 2 or x.shape[0] != self.num_envs or x.shape[1] != self.num_features:
      raise AssertionError(f"Expected x of shape (N={self.num_envs}, C_flat={self.num_features}), got {tuple(x.shape)}")
    i = self._write_idx
    self._buffer[i].copy_(x)  # contiguous row write
    self._write_idx = (i + 1) % self.max_window_size
    if self._count < self.max_window_size:
      self._count += 1
    # Increment per-env valid length, clamp to W.
    self._valid_len.add_(1)
    torch.clamp_(self._valid_len, max=self.max_window_size)

  def _apply_per_env_replicate_pad(self, seq_TNC: Tensor, valid_len: torch.Tensor) -> Tensor:
    """Apply per-env replicate padding in-place-like: for each env n, replicate seq[first_idx[n]] into the first pad_len[n] rows.

    Args:
      seq_TNC: (T, N, C)
      valid_len: (N,) ints in [0..W]

    Returns:
      (T, N, C) with per-env replicate padding applied.
    """
    T = seq_TNC.shape[0]
    if T == 0:
      return seq_TNC
    # pad_len[n] = max(0, T - valid_len[n])  (number of frames to replicate at the head)
    pad_len = (T - valid_len.to(torch.int64)).clamp_min(0).clamp_max(T)  # (N,)
    if torch.any(pad_len > 0):
      N = seq_TNC.shape[1]
      dev = seq_TNC.device
      # First actual index per env within the returned window
      first_idx = torch.minimum(pad_len, torch.tensor(T - 1, device=dev, dtype=pad_len.dtype))  # (N,)
      env_idx = torch.arange(N, device=dev)
      first = seq_TNC[first_idx, env_idx, :]  # (N, C)
      # If valid_len==0 (no actual frames), use zeros for first
      first = torch.where(valid_len.view(N, 1) > 0, first, torch.zeros_like(first))
      # Build mask (T, N, 1): True where t < pad_len[n]
      t_idx = torch.arange(T, device=dev).view(T, 1)
      mask = (t_idx < pad_len.view(1, N)).unsqueeze(-1)  # (T, N, 1)
      # Apply replicate padding
      seq_TNC = torch.where(mask, first.unsqueeze(0), seq_TNC)
    return seq_TNC

  def get_hist_data(self, window_size: int) -> Tensor:
    """Return raw chronological tail as (T, N, C_flat), with per-env replicate padding applied.

    Behavior:
      - Builds chronological tail from the ring buffer (time-first).
      - If the global buffer contains fewer than T frames (immediately after a global reset),
        zero-left-pad the sequence.
      - Then, for each environment, replicate-pad the head up to its per-env pad length
        based on valid_len (handles partial resets).
    """
    if window_size < 1:
      raise AssertionError("window_size must be >= 1")
    if self._count == 0:
      return self._buffer.new_zeros((0, self.num_envs, self.num_features))

    T_req = min(window_size, self.max_window_size)
    T_avail = min(self._count, T_req)

    i = self._write_idx
    if T_avail <= i:
      seq = self._buffer[i - T_avail : i]  # (T_avail, N, C)
    else:
      left_len = T_avail - i
      left = self._buffer[self.max_window_size - left_len : self.max_window_size]  # (left_len, N, C)
      right = self._buffer[0: i]  # (i, N, C)
      seq = torch.cat((left, right), dim=0)  # (T_avail, N, C)

    if T_avail < T_req:
      pad_len = T_req - T_avail
      zeros = self._buffer.new_zeros((pad_len, self.num_envs, self.num_features))
      seq = torch.cat((zeros, seq), dim=0)  # left-pad to requested T

    # Per-env replicate padding (handles partial resets)
    seq = self._apply_per_env_replicate_pad(seq, self._valid_len)
    return seq  # (T_req, N, C)

  def get_hist_data_smooth_and_diffed(
    self,
    window_size: int,
    diff_order: int,
    diff_dt: float,
    poly_degree: int = 2,
  ) -> Tensor:
    """Endpoint SG smoothing/differentiation at the latest sample. Returns (N, C_flat)."""
    if window_size < 1:
      raise AssertionError("window_size must be >= 1")
    if not (0 <= diff_order <= 3):
      raise AssertionError("diff_order must be in [0, 3]")
    if diff_dt <= 0.0:
      raise AssertionError("diff_dt must be positive")

    T_req = min(window_size, self.max_window_size)
    # Build chronological tail with global zero-pad and per-env replicate padding
    seq = self.get_hist_data(T_req)  # (T, N, C) with per-env replicate pad applied

    # SG kernel for T_req with adaptive degree.
    deg = max(diff_order, min(poly_degree, T_req - 1))
    w = sg_endpoint_kernel(T_req, deg, diff_order, diff_dt, device=seq.device, dtype=seq.dtype)  # (T,)

    # Fused GEMV: (T, N*C) @ (T,) -> (N*C,)
    T, N, C = seq.shape
    seq2d = seq.reshape(T, N * C)  # contiguous view
    out_flat = torch.mv(seq2d.t(), w)  # (N*C,)
    out = out_flat.view(N, C)
    return out  # (N, C_flat)

  @property
  def length(self) -> int:
    """Current global valid history length (<= window_size)."""
    return self._count
  

def main() -> None:
  """Visualize smoothing and differentiation using endpoint SG kernels with replicate padding.

  Requires matplotlib to be installed. Produces a few plots comparing:
    - Raw signal vs SG-smoothed (different window sizes/degrees).
    - 1st and 2nd derivatives estimated via SG at the endpoint.
  """
  import matplotlib.pyplot as plt
  
  T = 400
  noise_std=0.1
  dt = 0.05
  theta1 = 2.0 * math.pi * 1  #  Hz
  theta2 = 2.0 * math.pi * 0.5 # 0.5 Hz
  t = torch.arange(T, dtype=torch.float32) * dt
  y = torch.sin(theta1 * t) + 0.5 * torch.sin(theta2 * t)
  dy = theta1 * torch.cos(theta1 * t) + 0.5 * theta2 * torch.cos(theta2 * t)
  ddy = - (theta1)**2 * torch.sin(theta1 * t) - 0.5 * (theta2)**2 * torch.sin(theta2 * t)
  sig = y + noise_std * torch.randn_like(y)
  
  # Prepare sliding window and outputs
  sw = SlidingWindow(num_envs=1, feature_shape=torch.Size([1]), window_size=25, device=torch.device("cpu"))

  sm_outputs = {
    "win5_deg2": [],
    "win9_deg2": [],
    "win15_deg3": [],
  }
  d1_outputs = {
    "win9_deg2": [],
    "win15_deg3": [],
  }
  d2_outputs = {
    "win15_deg3": [],
  }

  for i in range(T):
    x = sig[i].view(1, 1)  # (N=1, C=1)
    sw.push(x)

    sm_outputs["win5_deg2"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=5, diff_order=0, diff_dt=dt, poly_degree=2)[0, 0].item()
    )
    sm_outputs["win9_deg2"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=9, diff_order=0, diff_dt=dt, poly_degree=2)[0, 0].item()
    )
    sm_outputs["win15_deg3"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=15, diff_order=0, diff_dt=dt, poly_degree=3)[0, 0].item()
    )

    d1_outputs["win9_deg2"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=9, diff_order=1, diff_dt=dt, poly_degree=2)[0, 0].item()
    )
    d1_outputs["win15_deg3"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=15, diff_order=1, diff_dt=dt, poly_degree=3)[0, 0].item()
    )

    d2_outputs["win15_deg3"].append(
      sw.get_hist_data_smooth_and_diffed(window_size=15, diff_order=2, diff_dt=dt, poly_degree=3)[0, 0].item()
    )

  t = torch.arange(T, dtype=torch.float32) * dt
  fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
  axs[0].plot(t.numpy(), y.numpy(), label="true", color="red", alpha=0.5)
  axs[0].plot(t.numpy(), sig.numpy(), '-.', label="raw",color="gray", alpha=0.5)
  axs[0].plot(t.numpy(), sm_outputs["win5_deg2"], label="SG smooth (win=5, deg=2)")
  axs[0].plot(t.numpy(), sm_outputs["win9_deg2"], label="SG smooth (win=9, deg=2)")
  axs[0].plot(t.numpy(), sm_outputs["win15_deg3"], label="SG smooth (win=15, deg=3)")
  axs[0].set_title("Endpoint SG smoothing (latest sample)")
  axs[0].legend()
  axs[0].grid(True, alpha=0.2)

  axs[1].plot(t.numpy(), dy.numpy(), label="dy true", color="red", alpha=0.5)
  axs[1].plot(t.numpy(), d1_outputs["win9_deg2"], label="SG deriv1 (win=9, deg=2)")
  axs[1].plot(t.numpy(), d1_outputs["win15_deg3"], label="SG deriv1 (win=15, deg=3)")
  axs[1].set_title("Endpoint SG 1st derivative (latest sample)")
  axs[1].legend()
  axs[1].grid(True, alpha=0.2)

  axs[2].plot(t.numpy(), ddy.numpy(), label="ddy true", color="red", alpha=0.5)
  axs[2].plot(t.numpy(), d2_outputs["win15_deg3"], label="SG deriv2 (win=15, deg=3)")
  axs[2].set_title("Endpoint SG 2nd derivative (latest sample)")
  axs[2].legend()
  axs[2].grid(True, alpha=0.2)

  axs[2].set_xlabel("time (s)")
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()