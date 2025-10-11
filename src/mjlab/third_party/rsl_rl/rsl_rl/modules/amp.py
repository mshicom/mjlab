# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import json
from typing import Generator, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.storage import ReplayBuffer


class AmpFeatureExtractor:
    """Configurable feature extractor for AMP supporting temporal windows.

    Responsibilities:
    - Extract per-frame base features from env extras or motion frames.
    - Optionally aggregate features over a sliding window using a transform (stack, mean, fft magnitude).
    - Maintain per-env rolling buffers for windowed env features (handles done resets).

    Notes:
    - Motion frames are assumed to be raw pose arrays with the following layout:
      [joint_pos(20), joint_vel(20), end_eff_pos(12), ...]
      which matches previous implementation. You can toggle which parts to use.
    - Env extras should already provide a per-frame AMP vector of the same base feature set as motion,
      but this class will still operate generically for any feature dimension.
    """

    DEFAULT_ENV_KEYS = ("amp_observations", "amp_obs", "amp_state", "amp")

    # Default indices for motion frame parsing (compatible with previous impl)
    JOINT_POS_SIZE = 20
    JOINT_VEL_SIZE = 20
    END_EFFECTOR_POS_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    def __init__(
        self,
        env_obs_keys: Iterable[str] | None = None,
        use_joint_pos: bool = True,
        use_joint_vel: bool = True,
        use_end_pos: bool = True,
        # Windowing
        window_size: int = 1,
        window_transform: str = "stack",  # "stack", "mean", "fft_mag"
        fft_keep: int | None = None,  # if None keep all rfft bins
        pad_initial: bool = True,  # pad startup windows by repeating first valid frame
        require_full_window: bool = False,  # if True, None is returned until window filled (env only)
    ):
        # Env keys
        self.env_obs_keys = tuple(env_obs_keys) if env_obs_keys is not None else self.DEFAULT_ENV_KEYS

        # Motion frame selection
        self.use_joint_pos = use_joint_pos
        self.use_joint_vel = use_joint_vel
        self.use_end_pos = use_end_pos
        self._slices: list[slice] = []
        if self.use_joint_pos:
            self._slices.append(slice(self.JOINT_POSE_START_IDX, self.JOINT_POSE_END_IDX))
        if self.use_joint_vel:
            self._slices.append(slice(self.JOINT_VEL_START_IDX, self.JOINT_VEL_END_IDX))
        if self.use_end_pos:
            self._slices.append(slice(self.END_POS_START_IDX, self.END_POS_END_IDX))

        # Window configuration
        self.window_size = int(max(1, window_size))
        self.window_transform = window_transform
        self.fft_keep = fft_keep
        self.pad_initial = pad_initial
        self.require_full_window = require_full_window

        # Derived
        self._base_feat_dim: int | None = None  # set after first motion/env feature seen
        self._out_feat_dim_cache: dict[int, int] = {}

        # Per-env buffers for windowed operation
        self._env_win_buf: torch.Tensor | None = None  # [N, W, F]
        self._env_win_filled: torch.Tensor | None = None  # [N] ints (0..W)

    @property
    def is_windowed(self) -> bool:
        return self.window_size > 1

    def reset_env_buffers(self):
        """Clear per-env temporal buffers."""
        self._env_win_buf = None
        self._env_win_filled = None

    # --------- Env feature extraction (per step) ---------

    def from_env_extras(self, extras: dict, device: torch.device | str) -> torch.Tensor | None:
        """Extract per-frame features directly from environment extras (no windowing)."""
        for key in self.env_obs_keys:
            if key in extras:
                val = extras[key]
                if not torch.is_tensor(val):
                    val = torch.as_tensor(val, dtype=torch.float32, device=device)
                else:
                    val = val.to(device)
                # lazily set base dim if needed
                if self._base_feat_dim is None:
                    self._base_feat_dim = int(val.shape[-1])
                return val
        return None

    def env_step(self, extras: dict, device: torch.device | str, dones: torch.Tensor | None = None) -> torch.Tensor | None:
        """Extract windowed features from environment extras, maintaining per-env buffers.

        Returns:
            Tensor [N, out_dim] if window available (or padded), otherwise None if require_full_window=True and not filled.
        """
        curr = self.from_env_extras(extras, device)
        if curr is None:
            return None

        # Normalize shape
        if curr.ndim == 1:
            curr = curr.unsqueeze(0)
        n_envs, feat_dim = curr.shape

        # Allocate buffers if needed or if env count changed
        if self._env_win_buf is None or self._env_win_buf.shape[0] != n_envs or self._env_win_buf.shape[2] != feat_dim:
            self._env_win_buf = torch.zeros(n_envs, self.window_size, feat_dim, device=device, dtype=curr.dtype)
            self._env_win_filled = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Handle resets prior to writing current frame to avoid cross-episode leakage
        if dones is not None:
            done_mask = dones.reshape(-1).to(device).bool()
            if done_mask.any():
                self._env_win_buf[done_mask].zero_()
                self._env_win_filled[done_mask] = 0  # type: ignore

        # Shift left and insert current at the end
        self._env_win_buf = torch.roll(self._env_win_buf, shifts=-1, dims=1)  # type: ignore
        self._env_win_buf[:, -1, :] = curr  # type: ignore
        # Update filled counts
        self._env_win_filled = torch.clamp(self._env_win_filled + 1, max=self.window_size)  # type: ignore

        if self.require_full_window and torch.any(self._env_win_filled < self.window_size):  # type: ignore
            return None

        # Prepare window tensor with optional padding
        win = self._env_win_buf  # [N, W, F]
        if self.pad_initial:
            # For envs not yet filled fully, pad the leading frames with the earliest available frame (replicate)
            fill_counts = self._env_win_filled  # [N]
            not_full = fill_counts < self.window_size
            if torch.any(not_full):
                # index for per-env first valid frame position within buffer
                # Current policy: replicate last (most recent) frame backwards to fill
                # To implement pad at the front: set first (W - filled) frames equal to the first valid frame
                # Here, since we keep most recent at -1, and shift each step, earlier frames contain history or zeros.
                # We'll just take the last frame and broadcast to the entire window where needed, then overwrite the last
                # 'filled' slots are already correct; the front (W-filled) will be overwritten by the broadcast below masked.
                pad_view = win[not_full]  # [M, W, F]
                last = pad_view[:, -1:, :].expand(-1, self.window_size, -1)
                # Build mask for padded positions
                m = (torch.arange(self.window_size, device=device).unsqueeze(0) < (self.window_size - fill_counts[not_full].unsqueeze(1)))
                # Assign padded positions
                pad_view[m] = last[m]

        # Transform window -> feature vector
        out = self._transform_window(win)
        return out

    # --------- Motion feature extraction ---------

    def from_motion_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract per-frame base features from a single motion frame [D]."""
        parts = [frame[s] for s in self._slices]
        feat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if self._base_feat_dim is None:
            self._base_feat_dim = int(feat.shape[-1])
        return feat

    def from_motion_frame_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract per-frame base features from a batch of motion frames [B, D]."""
        parts = [frames[:, s] for s in self._slices]
        feat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if self._base_feat_dim is None:
            self._base_feat_dim = int(feat.shape[-1])
        return feat

    def from_motion_window_batch(self, frames_win: torch.Tensor) -> torch.Tensor:
        """Extract windowed features from a batch of motion windows.

        Args:
            frames_win: Tensor [B, W, D] of raw motion frames (not yet sliced to base features).

        Returns:
            Tensor [B, out_dim] of processed window features.
        """
        B, W, D = frames_win.shape
        # Slice base features per frame, then reshape to [B, W, F]
        base = self.from_motion_frame_batch(frames_win.reshape(B * W, D)).reshape(B, W, -1)
        return self._transform_window(base)

    # --------- Output dimension helpers ---------

    def output_dim_for_feat_dim(self, feat_dim: int) -> int:
        """Return output feature dimension given per-frame base feature dimension."""
        if self.window_size <= 1:
            return feat_dim
        if feat_dim in self._out_feat_dim_cache:
            return self._out_feat_dim_cache[feat_dim]
        if self.window_transform == "stack":
            out = feat_dim * self.window_size
        elif self.window_transform == "mean":
            out = feat_dim
        elif self.window_transform == "fft_mag":
            freq_bins = self.window_size // 2 + 1
            if self.fft_keep is not None:
                freq_bins = int(min(freq_bins, self.fft_keep))
            out = feat_dim * freq_bins
        else:
            raise ValueError(f"Unknown window_transform: {self.window_transform}")
        self._out_feat_dim_cache[feat_dim] = out
        return out

    # --------- Internal transforms ---------

    def _transform_window(self, win: torch.Tensor) -> torch.Tensor:
        """Apply the configured window transform to a window tensor.

        Args:
            win: Tensor [N, W, F] for env or [B, W, F] for motion.

        Returns:
            Tensor [N, out_dim] or [B, out_dim].
        """
        if self.window_size <= 1:
            # unwrap to [N, F]
            return win[:, -1, :]

        if self.window_transform == "stack":
            return win.reshape(win.shape[0], self.window_size * win.shape[2])

        if self.window_transform == "mean":
            return win.mean(dim=1)

        if self.window_transform == "fft_mag":
            # rFFT across time dimension (W)
            fft_vals = torch.fft.rfft(win, dim=1)  # [N, Freq, F]
            mag = torch.abs(fft_vals)
            if self.fft_keep is not None:
                mag = mag[:, : self.fft_keep, :]
            return mag.reshape(mag.shape[0], -1)

        raise ValueError(f"Unknown window_transform: {self.window_transform}")


class Discriminator(nn.Module):
    """Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim: Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef: Coefficient to scale the AMP reward.
        hidden_layer_sizes: Sizes of hidden layers in the MLP trunk.
        device: Device to run the model on.
        task_reward_lerp: Interpolation factor between AMP reward and task reward (0.0 => only AMP).
    """

    def __init__(self, input_dim: int, amp_reward_coef: float, hidden_layer_sizes: list[int], device: str, task_reward_lerp: float = 0.0):
        super().__init__()
        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers: list[nn.Module] = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator network.

        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        """Compute gradient penalty (on expert data) for regularization.

        Args:
            expert_state: Batch of expert states.
            expert_next_state: Batch of expert next states.
            lambda_: Gradient penalty coefficient.

        Returns:
            Scalar gradient penalty loss.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: EmpiricalNormalization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the AMP reward given state and next state, optionally interpolated with a task reward.

        Args:
            state: Current state tensor.
            next_state: Next state tensor.
            task_reward: Task-specific reward tensor.
            normalizer: Optional normalizer for inputs.

        Returns:
            (reward, d)
            - reward: Predicted AMP reward (optionally interpolated) with shape (batch_size,).
            - d: Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, state.device)
                next_state = normalizer.normalize_torch(next_state, next_state.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(), d


class AMPLoader:
    """Expert dataset loader for AMP observations."""

    # Keep constants for compatibility (not strictly required in new design)
    JOINT_POS_SIZE = AmpFeatureExtractor.JOINT_POS_SIZE
    JOINT_VEL_SIZE = AmpFeatureExtractor.JOINT_VEL_SIZE
    END_EFFECTOR_POS_SIZE = AmpFeatureExtractor.END_EFFECTOR_POS_SIZE

    JOINT_POSE_START_IDX = AmpFeatureExtractor.JOINT_POSE_START_IDX
    JOINT_POSE_END_IDX = AmpFeatureExtractor.JOINT_POSE_END_IDX

    JOINT_VEL_START_IDX = AmpFeatureExtractor.JOINT_VEL_START_IDX
    JOINT_VEL_END_IDX = AmpFeatureExtractor.JOINT_VEL_END_IDX

    END_POS_START_IDX = AmpFeatureExtractor.END_POS_START_IDX
    END_POS_END_IDX = AmpFeatureExtractor.END_POS_END_IDX

    def __init__(
        self,
        device: str,
        time_between_frames: float,
        data_dir: str = "",
        preload_transitions: bool = False,
        num_preload_transitions: int = 1_000_000,
        motion_files: list[str] | None = None,
        feature_extractor: AmpFeatureExtractor | None = None,
    ):
        """Expert dataset provides AMP observations (transitions) from mocap dataset.

        Args:
            device: Torch device.
            time_between_frames: Amount of time in seconds between transitions.
            data_dir: Optional data directory (unused if motion_files given).
            preload_transitions: Whether to preload transitions.
            num_preload_transitions: Number of transitions to preload if enabled.
            motion_files: List of file paths to motion JSON files.
            feature_extractor: Optional feature extractor for motion frames.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.feature_extractor = feature_extractor or AmpFeatureExtractor()

        if motion_files is None:
            motion_files = glob.glob("datasets/motion_amp_expert/*")

        # Raw motion trajectories (min length up to END_POS_END_IDX)
        self.trajectories_full: list[torch.Tensor] = []
        self.trajectory_names: list[str] = []
        self.trajectory_idxs: list[int] = []
        self.trajectory_lens: list[float] = []
        self.trajectory_weights: list[float] = []
        self.trajectory_frame_durations: list[float] = []
        self.trajectory_num_frames: list[float] = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                full = torch.tensor(
                    motion_data[:, : AmpFeatureExtractor.END_POS_END_IDX],
                    dtype=torch.float32,
                    device=device,
                )
                self.trajectories_full.append(full)
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

        self.trajectory_weights = np.array(self.trajectory_weights)
        self.trajectory_weights = self.trajectory_weights / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Compute observation dimension from feature extractor
        base_feat = self.feature_extractor.from_motion_frame(self.trajectories_full[0][0])
        self._base_feat_dim = int(base_feat.shape[-1])
        self._observation_dim = self.feature_extractor.output_dim_for_feat_dim(self._base_feat_dim)

        # Preload transitions for speed (features)
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_feat_s = self.get_feature_at_time_batch(traj_idxs, times)
            self.preloaded_feat_s_next = self.get_feature_at_time_batch(traj_idxs, times + self.time_between_frames)

        # Convenience big tensor if needed elsewhere
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    # --------- Sampling utilities ---------

    def weighted_traj_idx_sample(self) -> int:
        return int(np.random.choice(self.trajectory_idxs, p=self.trajectory_weights))

    def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx: int) -> float:
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0.0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    # --------- Interpolation helpers ---------

    def slerp(self, frame1: torch.Tensor, frame2: torch.Tensor, blend: torch.Tensor | float) -> torch.Tensor:
        return (1.0 - blend) * frame1 + blend * frame2

    def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        out_dim = self.trajectories_full[0].shape[1]
        all_starts = torch.zeros(len(traj_idxs), out_dim, device=self.device)
        all_ends = torch.zeros(len(traj_idxs), out_dim, device=self.device)
        for tid in set(traj_idxs):
            trajectory = self.trajectories_full[tid]
            mask = traj_idxs == tid
            all_starts[mask] = trajectory[idx_low[mask]]
            all_ends[mask] = trajectory[idx_high[mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_starts, all_ends, blend)

    # --------- Window building for motion ---------

    def get_window_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray, window_size: int) -> torch.Tensor:
        """Build a batch of motion windows ending at times.

        Returns:
            Tensor [B, W, D_full]
        """
        B = len(traj_idxs)
        W = window_size
        D_full = self.trajectories_full[0].shape[1]
        out = torch.zeros(B, W, D_full, device=self.device)
        # Group by traj id for efficiency
        unique_ids = np.unique(traj_idxs)
        for tid in unique_ids:
            mask_np = traj_idxs == tid
            mask = torch.from_numpy(mask_np).to(self.device, dtype=torch.bool)
            num = int(mask.sum().item())
            if num == 0:
                continue
            times_sel = torch.tensor(times[mask_np], device=self.device, dtype=torch.float32)  # [M]
            dt = float(self.trajectory_frame_durations[tid])
            # oldest to newest times in window
            offsets = torch.arange(-(W - 1), 1, device=self.device, dtype=torch.float32) * dt
            times_win = (times_sel.unsqueeze(1) + offsets.unsqueeze(0)).clamp_min_(0.0)  # [M, W]
            # Flatten for batch interpolation
            trajs_flat = np.full((num * W,), tid, dtype=traj_idxs.dtype)
            frames_flat = self.get_full_frame_at_time_batch(trajs_flat, times_win.reshape(-1).cpu().numpy())
            out[mask] = frames_flat.reshape(num, W, D_full)
        return out

    # --------- Feature at time sampling (handles windowed/non-windowed) ---------

    def get_feature_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        if self.feature_extractor.is_windowed:
            win = self.get_window_at_time_batch(np.array([traj_idx]), np.array([time]), self.feature_extractor.window_size)
            feat = self.feature_extractor.from_motion_window_batch(win)  # [1, F]
            return feat[0]
        # non-windowed
        frame = self.get_full_frame_at_time(traj_idx, time)
        return self.feature_extractor.from_motion_frame(frame)

    def get_feature_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        if self.feature_extractor.is_windowed:
            win = self.get_window_at_time_batch(traj_idxs, times, self.feature_extractor.window_size)  # [B, W, D]
            return self.feature_extractor.from_motion_window_batch(win)  # [B, F]
        # non-windowed
        frames = self.get_full_frame_at_time_batch(traj_idxs, times)  # [B, D]
        return self.feature_extractor.from_motion_frame_batch(frames)

    # --------- Public sampling APIs ---------

    def get_full_frame(self) -> torch.Tensor:
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames: int) -> torch.Tensor:
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Generates a batch of AMP feature transitions (expert)."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_feat_s.shape[0], size=mini_batch_size)
                s = self.preloaded_feat_s[idxs]
                s_next = self.preloaded_feat_s_next[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                s = self.get_feature_at_time_batch(traj_idxs, times)
                s_next = self.get_feature_at_time_batch(traj_idxs, times + self.time_between_frames)
            yield s, s_next

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def num_motions(self) -> int:
        return len(self.trajectory_names)

    # Legacy helpers (kept for compatibility with potential external uses)
    @staticmethod
    def get_joint_pose(pose: torch.Tensor) -> torch.Tensor:
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses: torch.Tensor) -> torch.Tensor:
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose: torch.Tensor) -> torch.Tensor:
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses: torch.Tensor) -> torch.Tensor:
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_end_pos(pose: torch.Tensor) -> torch.Tensor:
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    @staticmethod
    def get_end_pos_batch(poses: torch.Tensor) -> torch.Tensor:
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]


class AdversarialMotionPrior(nn.Module):
    """AMP module that encapsulates discriminator, expert loader, normalization, and policy replay.

    It also manages environment feature extraction and transition buffering, to avoid any
    runner-side state such as previous AMP features.
    """

    def __init__(
        self,
        reward_coef: float,
        discr_hidden_dims: list[int],
        task_reward_lerp: float = 0.0,
        # Loader config
        time_between_frames: float = 0.05,
        preload_transitions: bool = False,
        num_preload_transitions: int = 1_000_000,
        motion_files: list[str] | None = None,
        # Replay buffer
        replay_buffer_size: int | None = None,
        # Optimization / regularization
        state_normalization: bool = True,
        grad_penalty_lambda: float = 10.0,
        # Feature extraction (env + motion)
        env_obs_keys: Iterable[str] | None = None,
        feature_extractor: AmpFeatureExtractor | None = None,
        # Device
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        # Feature extractor
        self.feature_extractor = feature_extractor or AmpFeatureExtractor(env_obs_keys=env_obs_keys)

        # Expert loader
        self.loader = AMPLoader(
            device=device,
            time_between_frames=time_between_frames,
            preload_transitions=preload_transitions,
            num_preload_transitions=num_preload_transitions,
            motion_files=motion_files,
            feature_extractor=self.feature_extractor,
        )
        observation_dim = self.loader.observation_dim

        # Normalizer
        self.normalizer = EmpiricalNormalization(observation_dim) if state_normalization else None

        # Discriminator
        self.discriminator = Discriminator(
            input_dim=observation_dim * 2,
            amp_reward_coef=reward_coef,
            hidden_layer_sizes=discr_hidden_dims,
            device=device,
            task_reward_lerp=task_reward_lerp,
        ).to(device)

        # Policy replay buffer for AMP transitions
        if replay_buffer_size is None:
            replay_buffer_size = num_preload_transitions
        self.replay = ReplayBuffer(observation_dim, replay_buffer_size, device)

        # Coefficients
        self.reward_coef = reward_coef
        self.loss_coef = reward_coef  # keep parity with reward scaling
        self.grad_penalty_lambda = grad_penalty_lambda

        # Internal buffer to build (s, s_next) transitions at env-step granularity
        self._env_prev_feat: torch.Tensor | None = None
        self._env_prev_has: torch.BoolTensor | None = None

    def train(self, mode: bool = True):
        self.discriminator.train(mode)
        return self

    def eval(self):
        self.discriminator.eval()
        return self

    @torch.no_grad()
    def predict_reward(self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor) -> torch.Tensor:
        reward, _ = self.discriminator.predict_amp_reward(
            state, next_state, task_reward, normalizer=self.normalizer
        )
        return reward

    def add_transition(self, state: torch.Tensor, next_state: torch.Tensor):
        """Insert policy AMP transition into replay buffer."""
        self.replay.insert(state, next_state)

    def policy_generator(self, num_batches: int, batch_size: int):
        return self.replay.feed_forward_generator(num_batches, batch_size)

    def expert_generator(self, num_batches: int, batch_size: int):
        return self.loader.feed_forward_generator(num_batches, batch_size)

    def update_from_env_extras(self, extras: dict, dones: torch.Tensor | None = None):
        """Extract features from env extras and append AMP transitions into replay buffer.

        - Uses AmpFeatureExtractor to perform optional window aggregation (with its own env buffers).
        - Manages (prev, curr) pairing internally; masks out cross-episode transitions using dones.
        """
        curr_feat = (
            self.feature_extractor.env_step(extras, self.device, dones)
            if self.feature_extractor.is_windowed
            else self.feature_extractor.from_env_extras(extras, self.device)
        )
        if curr_feat is None:
            return

        if curr_feat.ndim == 1:
            curr_feat = curr_feat.unsqueeze(0)
        n_envs = curr_feat.shape[0]

        if self._env_prev_feat is None or self._env_prev_feat.shape[0] != n_envs:
            self._env_prev_feat = torch.zeros_like(curr_feat)
            self._env_prev_has = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

        done_mask = torch.zeros(n_envs, dtype=torch.bool, device=self.device) if dones is None else dones.reshape(-1).bool()
        prev_has = self._env_prev_has  # type: ignore

        if prev_has.any():
            add_mask = prev_has & (~done_mask)
            if add_mask.any():
                s = self._env_prev_feat[add_mask]  # type: ignore
                s_next = curr_feat[add_mask]
                self.add_transition(s, s_next)

        # Update prev buffers
        self._env_prev_feat = curr_feat
        self._env_prev_has.fill_(True)  # type: ignore
        self._env_prev_has[done_mask] = False  # type: ignore

    def reset_env_buffer(self):
        """Clear internal environment buffers (window + prev)."""
        self._env_prev_feat = None
        self._env_prev_has = None
        self.feature_extractor.reset_env_buffers()

    def compute_batch_losses(
        self,
        policy_state: torch.Tensor,
        policy_next_state: torch.Tensor,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        # Optional normalization
        if self.normalizer is not None:
            with torch.no_grad():
                policy_state = self.normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.normalizer.normalize_torch(expert_next_state, self.device)

        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
        policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)

        grad_pen_loss = self.discriminator.compute_grad_pen(
            expert_state, expert_next_state, lambda_=self.grad_penalty_lambda
        )

        return amp_loss, grad_pen_loss, policy_d.mean().item(), expert_d.mean().item()


def resolve_amp_config(alg_cfg: dict, env) -> dict:
    """Resolve the AMP configuration and defaults.

    - Sets 'time_between_frames' from env.unwrapped.step_dt if not provided.
    - Sets 'replay_buffer_size' if not provided.
    - Accepts pass-through keys for AmpFeatureExtractor such as:
        'env_obs_keys', 'window_size', 'window_transform', 'fft_keep',
        'pad_initial', 'require_full_window', 'use_joint_pos', 'use_joint_vel', 'use_end_pos'.
    """
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        amp_cfg = alg_cfg["amp_cfg"]
        # time between frames defaults to env step dt
        if "time_between_frames" not in amp_cfg or amp_cfg["time_between_frames"] is None:
            step_dt = getattr(getattr(env, "unwrapped", env), "step_dt", None)
            if step_dt is None:
                raise ValueError("AMP requires 'time_between_frames' or env.step_dt to be set.")
            amp_cfg["time_between_frames"] = float(step_dt)
        # default replay buffer size
        if "replay_buffer_size" not in amp_cfg or amp_cfg["replay_buffer_size"] is None:
            amp_cfg["replay_buffer_size"] = amp_cfg.get("num_preload_transitions", 1_000_000)
        # no further resolution required; windowing and keys are passed to AdversarialMotionPrior
    return alg_cfg