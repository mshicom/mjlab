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
  """Time-first ring buffer optimized for throughput with partial resets.

  Storage:
    - _buffer: (W, N, C_flat) with W = max_window_size

  API:
    - push(x): x is (N, C_flat)
    - reset(env_ids: Optional[Tensor|slice]): partial or global reset
    - _get_hist_data(window_size: int, replicate_pad: bool = False, layout: str = "NCW")
      - layout="NCW" (default): returns (N, C, W)
      - layout="TNC": returns (T, N, C)
      replicate_pad applies per-env replicate padding when True.
    - _get_hist_data_smooth_and_diffed(window_size, diff_order, diff_dt, poly_degree=2)
      Returns (N, C_flat); applies per-env replicate padding internally.

  Notes:
    - No per-env ops on push; per-env padding is computed lazily on read when requested.
    - Partial reset tracked by per-env last_reset_step; valid_len is derived as (step - last_reset_step).
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

    # Flatten feature shape to a single C_flat for storage/perf
    nf = 1
    for d in feature_shape:
      nf *= int(d)
    self.num_features: int = nf

    dev = device if device is not None else torch.device("cpu")

    # Time-first buffer (W, N, C)
    self.register_buffer("_buffer", torch.zeros(self.max_window_size, self.num_envs, self.num_features, device=dev))
    # Per-env last reset "step" counter; used to compute valid_len lazily
    self.register_buffer("_last_reset_step", torch.zeros(self.num_envs, dtype=torch.int64, device=dev))

    # Global counters (Python ints to avoid device syncs)
    self._count: int = 0           # global number of valid frames since last global reset, capped at W
    self._write_idx: int = 0       # ring index [0..W-1]
    self._step: int = 0            # monotonically increasing "time step" since last global reset

  # ----------------
  # State management
  # ----------------

  def reset(self, env_ids: Optional[torch.Tensor | slice] = None) -> None:
    """Reset history. If env_ids is None, global reset; else, per-env partial reset."""
    if env_ids is None:
      # Global reset: do not zero buffer; leave it as scratch. Reads will zero/replicate-pad as needed.
      self._last_reset_step.zero_()
      self._count = 0
      self._write_idx = 0
      self._step = 0
    else:
      # Partial reset: mark selected envs as newly reset at current step.
      self._last_reset_step[env_ids] = int(self._step)

  # ----
  # Push
  # ----

  def push(self, x: Tensor) -> None:
    """Push newest observation x: (N, C_flat)."""
    # Minimal runtime checks for hot path
    if x.dim() != 2 or x.shape[0] != self.num_envs or x.shape[1] != self.num_features:
      raise AssertionError(f"Expected x of shape (N={self.num_envs}, C_flat={self.num_features}), got {tuple(x.shape)}")

    i = self._write_idx
    self._buffer[i].copy_(x)  # contiguous row write

    self._write_idx = (i + 1) % self.max_window_size
    if self._count < self.max_window_size:
      self._count += 1
    self._step += 1  # advance logical time

  # ------------
  # Read helpers
  # ------------

  def _slice_tail(self, T_req: int) -> Tensor:
    """Return chronological tail as (T_req, N, C), zero-left-padded after global reset as needed.
    No per-env replicate padding here.
    """
    if self._count == 0:
      return self._buffer.new_zeros((0, self.num_envs, self.num_features))

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

    return seq  # (T_req, N, C)

  def _apply_per_env_replicate_pad(self, seq_TNC: Tensor, T_req: int) -> Tensor:
    """Apply per-env replicate padding to seq_TNC (T_req, N, C) using last_reset_step and self._step."""
    if T_req == 0:
      return seq_TNC

    # valid_len[n] = clamp(step - last_reset[n], 0, T_req)
    # number of head frames to replicate per env: pad_len[n] = T_req - valid_len[n]
    # Note: We do not cap by global _count here; _slice_tail already zero-left-padded globally.
    step = torch.tensor(self._step, device=seq_TNC.device, dtype=torch.int64)
    valid_len = (step - self._last_reset_step).clamp_min(0).clamp_max(T_req)  # (N,)
    pad_len = (T_req - valid_len).clamp_min(0)  # (N,)

    if not torch.any(pad_len > 0):
      return seq_TNC  # nothing to replicate

    N = seq_TNC.shape[1]
    dev = seq_TNC.device
    # Index of first actual frame per env inside the returned window
    first_idx = torch.minimum(pad_len, torch.tensor(T_req - 1, device=dev, dtype=pad_len.dtype))  # (N,)
    env_idx = torch.arange(N, device=dev)
    first = seq_TNC[first_idx, env_idx, :]  # (N, C)
    # Envs with valid_len==0 -> no actual frames; use zeros for "first"
    first = torch.where(valid_len.view(N, 1) > 0, first, torch.zeros_like(first))

    # Mask: (T, N, 1) True for head positions t < pad_len[n]
    t_idx = torch.arange(T_req, device=dev).view(T_req, 1)
    mask = (t_idx < pad_len.view(1, N)).unsqueeze(-1)

    return torch.where(mask, first.unsqueeze(0), seq_TNC)

  # ---------------
  # Public read API
  # ---------------

  def _get_hist_data(self, window_size: int, replicate_pad: bool = False, layout: str = "NCW") -> Tensor:
    """Return history tail.

    Args:
      window_size: desired history length T (<= max_window_size).
      replicate_pad: if True, apply per-env replicate padding at the head for partial resets.
      layout: "NCW" (default) for (N, C, W) or "TNC" for (T, N, C).

    Returns:
      Tensor shaped per layout:
        - (N, C, W) if layout="NCW"
        - (T, N, C) if layout="TNC"
    """
    if window_size < 1:
      raise AssertionError("window_size must be >= 1")
    T_req = min(window_size, self.max_window_size)
    seq_tnc = self._slice_tail(T_req)
    if replicate_pad and T_req > 0:
      seq_tnc = self._apply_per_env_replicate_pad(seq_tnc, T_req)

    if layout.upper() == "TNC":
      return seq_tnc
    elif layout.upper() == "NCW":
      return seq_tnc.permute(1, 2, 0).contiguous()
    else:
      raise ValueError("layout must be 'NCW' or 'TNC'")

  def _get_hist_data_smooth_and_diffed(
    self,
    window_size: int,
    diff_order: int,
    diff_dt: float,
    poly_degree: int = 2,
  ) -> Tensor:
    """Endpoint SG smoothing/differentiation at the latest sample. Returns (N, C_flat).

    Per-env replicate padding is applied prior to SG to correctly handle partial resets.
    """
    if window_size < 1:
      raise AssertionError("window_size must be >= 1")
    if not (0 <= diff_order <= 3):
      raise AssertionError("diff_order must be in [0, 3]")
    if diff_dt <= 0.0:
      raise AssertionError("diff_dt must be positive")

    T_req = min(window_size, self.max_window_size)
    # Build chronological tail and apply per-env replicate padding on-demand; keep time-major for GEMV
    seq_tnc = self._get_hist_data(T_req, replicate_pad=True, layout="TNC")  # (T, N, C)

    # SG kernel for T_req with adaptive degree.
    deg = max(diff_order, min(poly_degree, T_req - 1))
    w = sg_endpoint_kernel(T_req, deg, diff_order, diff_dt, device=seq_tnc.device, dtype=seq_tnc.dtype)  # (T,)

    # Fused GEMV: (T, N*C) @ (T,) -> (N*C,)
    T, N, C = seq_tnc.shape
    if T == 0:
      return self._buffer.new_zeros((N, C))
    seq2d = seq_tnc.reshape(T, N * C)  # contiguous view
    out_flat = torch.mv(seq2d.t(), w)  # (N*C,)
    return out_flat.view(N, C)

  # Optional callable alias to support previous underscore API usage
  def __call__(self, length: Optional[int] = None, diff_order: Optional[int] = None, diff_dt: Optional[float] = None, poly_degree: int = 2, *args, **kwargs):
    T = length or self.max_window_size
    if diff_order is not None:
      if diff_dt is None:
        raise AssertionError("diff_dt must be provided if diff_order is provided")
      return self._get_hist_data_smooth_and_diffed(T, diff_order, float(diff_dt), poly_degree)
    # Default to user-friendly (N, C, W)
    return self._get_hist_data(T, replicate_pad=False, layout="NCW")

  @property
  def length(self) -> int:
    """Current global valid history length (<= max_window_size)."""
    return self._count