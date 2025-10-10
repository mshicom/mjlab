# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import json
from typing import Generator, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.storage import ReplayBuffer


class AmpFeatureExtractor:
    """Configurable feature extractor for AMP.

    This extractor is used to:
    - Build AMP feature vectors from environment `extras` (vectorized over envs).
    - Build AMP feature vectors from motion frames loaded from disk.

    By default it reproduces the previous hard-coded behavior:
    - Env features are extracted from the first present key among:
      ["amp_observations", "amp_obs", "amp_state", "amp"].
    - Motion features include joint positions (20), joint velocities (20), and end-effector positions (12).
    """

    # Defaults used in original implementation
    DEFAULT_ENV_KEYS = ("amp_observations", "amp_obs", "amp_state", "amp")

    # Index map (expected for motion frames)
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
    ):
        self.env_obs_keys = tuple(env_obs_keys) if env_obs_keys is not None else self.DEFAULT_ENV_KEYS
        self.use_joint_pos = use_joint_pos
        self.use_joint_vel = use_joint_vel
        self.use_end_pos = use_end_pos

        # Precompute slices for motion frames
        self._slices: list[slice] = []
        if self.use_joint_pos:
            self._slices.append(slice(self.JOINT_POSE_START_IDX, self.JOINT_POSE_END_IDX))
        if self.use_joint_vel:
            self._slices.append(slice(self.JOINT_VEL_START_IDX, self.JOINT_VEL_END_IDX))
        if self.use_end_pos:
            self._slices.append(slice(self.END_POS_START_IDX, self.END_POS_END_IDX))

    def from_env_extras(self, extras: dict, device: torch.device | str) -> torch.Tensor | None:
        """Extract AMP features from environment extras dict.

        Returns a tensor of shape [num_envs, feat_dim] or None if not present.
        """
        for key in self.env_obs_keys:
            if key in extras:
                val = extras[key]
                if not torch.is_tensor(val):
                    return torch.as_tensor(val, dtype=torch.float32, device=device)
                return val.to(device)
        return None

    def from_motion_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract AMP features from a single motion frame [D]."""
        parts = [frame[s] for s in self._slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def from_motion_frame_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract AMP features from a batch of motion frames [B, D]."""
        parts = [frames[:, s] for s in self._slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


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

    # Retain indices for default feature extractor
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

        # Discover motion files if not provided
        if motion_files is None:
            motion_files = glob.glob("datasets/motion_amp_expert/*")

        # Values to store for each trajectory.
        self.trajectories: list[torch.Tensor] = []         # feature frames
        self.trajectories_full: list[torch.Tensor] = []    # raw frames (at least up to END_POS_END_IDX)
        self.trajectory_names: list[str] = []
        self.trajectory_idxs: list[int] = []
        self.trajectory_lens: list[float] = []  # Traj length in seconds.
        self.trajectory_weights: list[float] = []
        self.trajectory_frame_durations: list[float] = []
        self.trajectory_num_frames: list[float] = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # Use at least the range required by default extractor
                full = torch.tensor(
                    motion_data[:, : AmpFeatureExtractor.END_POS_END_IDX],
                    dtype=torch.float32,
                    device=device,
                )
                self.trajectories_full.append(full)
                # Build feature frames using extractor
                feat = self.feature_extractor.from_motion_frame_batch(full)
                self.trajectories.append(feat)

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights)
        self.trajectory_weights = self.trajectory_weights / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions (features) for speed, if desired.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            # Directly preload feature transitions
            self.preloaded_feat_s = self.get_frame_at_time_batch(traj_idxs, times)
            self.preloaded_feat_s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

        # Convenience tensor of all frames if needed elsewhere
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self) -> int:
        """Get traj idx via weighted sampling."""
        return int(np.random.choice(self.trajectory_idxs, p=self.trajectory_weights))

    def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx: int) -> float:
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0.0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1: torch.Tensor, frame2: torch.Tensor, blend: torch.Tensor | float) -> torch.Tensor:
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx: int) -> torch.Tensor:
        """Returns trajectory of AMP features."""
        return self.trajectories[traj_idx]

    def get_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Returns feature frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Returns feature frames for given trajectories at specified times."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        obs_dim = self.observation_dim
        all_starts = torch.zeros(len(traj_idxs), obs_dim, device=self.device)
        all_ends = torch.zeros(len(traj_idxs), obs_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]  # features
            traj_mask = traj_idxs == traj_idx
            all_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_starts, all_ends, blend)

    def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Returns interpolated full motion frame."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Returns blended full frames for a batch of trajectories and times."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        out_dim = self.trajectories_full[0].shape[1]
        all_starts = torch.zeros(len(traj_idxs), out_dim, device=self.device)
        all_ends = torch.zeros(len(traj_idxs), out_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_starts, all_ends, blend)

    def get_full_frame(self) -> torch.Tensor:
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames: int) -> torch.Tensor:
        """Returns a batch of full frames."""
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0: torch.Tensor, frame1: torch.Tensor, blend: float) -> torch.Tensor:
        """Linearly interpolate between two frames (legacy, not used in new pipeline)."""
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

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
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            yield s, s_next

    @property
    def observation_dim(self) -> int:
        """Size of AMP feature observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self) -> int:
        return len(self.trajectory_names)

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

    Also manages environment feature extraction and transition buffering, to avoid
    storing state in the runner.
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
        # Feature extraction
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
        # scale used for adding discriminator training loss to PPO (kept equal to reward coef for parity)
        self.loss_coef = reward_coef
        self.grad_penalty_lambda = grad_penalty_lambda

        # Internal environment feature buffers (to avoid storing in runner)
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
        """Policy transitions generator (from replay)."""
        return self.replay.feed_forward_generator(num_batches, batch_size)

    def expert_generator(self, num_batches: int, batch_size: int):
        """Expert transitions generator (from dataset)."""
        return self.loader.feed_forward_generator(num_batches, batch_size)

    def update_from_env_extras(self, extras: dict, dones: torch.Tensor | None = None):
        """Extract features from env extras and add transitions to replay buffer.

        This method manages the previous-step buffer internally to avoid storing it in the runner.
        It filters out cross-episode transitions using the dones mask.

        Args:
            extras: Environment extras dict from env.step().
            dones: Done flags tensor of shape [num_envs] or [num_envs, 1]. If None, assumes all False.
        """
        curr_feat = self.feature_extractor.from_env_extras(extras, self.device)
        if curr_feat is None:
            return

        # Normalize inputs shape: [N, D]
        if curr_feat.ndim == 1:
            curr_feat = curr_feat.unsqueeze(0)

        n_envs = curr_feat.shape[0]

        if dones is None:
            done_mask = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        else:
            done_mask = dones.reshape(-1).to(self.device).bool()

        # Allocate/resize buffers on first call or env-count change
        if self._env_prev_feat is None or self._env_prev_feat.shape[0] != n_envs:
            self._env_prev_feat = torch.zeros_like(curr_feat)
            self._env_prev_has = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

        prev_has = self._env_prev_has  # type: ignore
        if prev_has.any():
            add_mask = prev_has & (~done_mask)
            if add_mask.any():
                s = self._env_prev_feat[add_mask]  # type: ignore
                s_next = curr_feat[add_mask]
                self.add_transition(s, s_next)

        # Update buffers
        self._env_prev_feat = curr_feat
        self._env_prev_has.fill_(True)  # type: ignore
        # Clear for environments that finished this step (avoid cross-episode links)
        self._env_prev_has[done_mask] = False  # type: ignore

    def reset_env_buffer(self):
        """Clear internal environment buffers (e.g., at the start of a new rollout)."""
        self._env_prev_feat = None
        self._env_prev_has = None

    def compute_batch_losses(
        self,
        policy_state: torch.Tensor,
        policy_next_state: torch.Tensor,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Compute discriminator/adversarial losses and stats."""
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
    """Resolve the AMP configuration, similar to resolve_rnd_config.

    Modifies alg_cfg in place to include derived AMP parameters.

    - Sets 'time_between_frames' from env.unwrapped.step_dt if not provided.
    - Optionally sets default replay buffer size if not provided.
    - Allows optional env feature keys via 'env_obs_keys' in amp_cfg.

    Args:
        alg_cfg: The algorithm configuration dictionary (with optional 'amp_cfg').
        env: The environment (used for step_dt and expert obs extraction).

    Returns:
        The resolved algorithm configuration dictionary.
    """
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        amp_cfg = alg_cfg["amp_cfg"]
        # time between frames defaults to environment step dt
        if "time_between_frames" not in amp_cfg or amp_cfg["time_between_frames"] is None:
            # prefer unwrapped if available
            step_dt = getattr(getattr(env, "unwrapped", env), "step_dt", None)
            if step_dt is None:
                raise ValueError("AMP requires 'time_between_frames' or env.step_dt to be set.")
            amp_cfg["time_between_frames"] = float(step_dt)
        # default replay buffer size
        if "replay_buffer_size" not in amp_cfg or amp_cfg["replay_buffer_size"] is None:
            amp_cfg["replay_buffer_size"] = amp_cfg.get("num_preload_transitions", 1_000_000)
        # pass through env obs keys if runner config defines them (optional)
        if "env_obs_keys" in amp_cfg and amp_cfg["env_obs_keys"] is not None:
            # nothing to resolve; accepted by AdversarialMotionPrior
            pass
    return alg_cfg