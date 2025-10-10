from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


@dataclass
class AmpDatasetCfg:
  # Root directory that contains .npy AMP motion files.
  root: Path
  # List of file basenames (without .npy) to load as expert datasets.
  names: list[str]
  # Relative sampling weights per dataset (will be normalized).
  weights: list[float]
  # Slow down factor for original FPS (integer >= 1).
  slow_down_factor: int = 1


@dataclass
class AmpDiscriminatorCfg:
  input_dim: int  # state concat next_state dimension expected by discriminator
  hidden_dims: Sequence[int] = (256, 256)
  reward_scale: float = 0.1
  reward_clamp_epsilon: float = 1e-4
  loss_type: Literal["BCEWithLogits", "Wasserstein"] = "BCEWithLogits"
  eta_wgan: float = 0.3
  grad_penalty_lambda: float = 10.0


@dataclass
class AmpReplayBufferCfg:
  size: int = 100_000


@dataclass
class AmpLearningCfg:
  # PPO part (these should match/sync with your PPO cfg or be read from it)
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  learning_rate: float = 1e-3
  max_grad_norm: float = 1.0
  use_clipped_value_loss: bool = True
  schedule: Literal["fixed", "adaptive"] = "adaptive"
  desired_kl: float = 0.01
  use_smooth_ratio_clipping: bool = False


@dataclass
class AmpVizCfg:
  # Simple toggle to help with quick debug printing or sanity checks.
  verbose: bool = False


@dataclass
class AmpCfg:
  # High-level AMP configuration
  dataset: AmpDatasetCfg
  discriminator: AmpDiscriminatorCfg
  replay: AmpReplayBufferCfg = field(default_factory=AmpReplayBufferCfg)
  learn: AmpLearningCfg = field(default_factory=AmpLearningCfg)
  viz: AmpVizCfg = field(default_factory=AmpVizCfg)
  # Whether to keep all expert buffers on GPU (recommended).
  device_resident: bool = True
  # If provided, caps per-mini-batch samples used for AMP (leave None for default).
  amp_batch_size_override: int | None = None