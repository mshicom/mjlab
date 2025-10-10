from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class MotionData:
  joint_positions: torch.Tensor
  joint_velocities: torch.Tensor
  base_lin_vel_local: torch.Tensor
  base_ang_vel_local: torch.Tensor
  base_quat_wxyz: torch.Tensor  # (T,4) wxyz

  def __len__(self) -> int:
    return self.joint_positions.shape[0]

  def amp_obs(self, idx: torch.Tensor) -> torch.Tensor:
    return torch.cat(
      (
        self.joint_positions[idx],
        self.joint_velocities[idx],
        self.base_lin_vel_local[idx],
        self.base_ang_vel_local[idx],
      ),
      dim=-1,
    )

  def next_indices(self) -> torch.Tensor:
    T = len(self)
    idx = torch.arange(T, device=self.joint_positions.device)
    return torch.clamp(idx + 1, max=T - 1)


class MotionDataset:
  """Loads .npy motion packs and exposes device-resident AMP obs buffers."""

  def __init__(
    self,
    root: Path | str,
    names: Sequence[str],
    weights: Sequence[float],
    simulation_dt: float,
    slow_down_factor: int = 1,
    device: str | torch.device = "cpu",
    expected_joint_names: Sequence[str] | None = None,
  ) -> None:
    self.root = Path(root)
    self.device = torch.device(device)

    if expected_joint_names is None:
      expected_joint_names = self._collect_union_joint_names(names)

    self.data: list[MotionData] = []
    for nm in names:
      md = self._load_one(
        path=self.root / f"{nm}.npy",
        sim_dt=simulation_dt,
        slow=slow_down_factor,
        expected_joint_names=list(expected_joint_names),
      )
      self.data.append(md)

    w = torch.tensor(weights, dtype=torch.float32, device=self.device)
    self.dataset_weights = (w / w.sum()).tolist()

    # Precompute concatenated buffers (obs and next_obs) for fast sampling
    obs_list, next_list = [], []
    for md in self.data:
      idx = torch.arange(len(md), device=self.device)
      obs = md.amp_obs(idx)
      nxt = md.amp_obs(md.next_indices())
      obs_list.append(obs)
      next_list.append(nxt)
    self.obs = torch.cat(obs_list, dim=0).to(self.device)
    self.next_obs = torch.cat(next_list, dim=0).to(self.device)

    # Per-sample weights across datasets proportional to dataset weight / length
    lens = [len(md) for md in self.data]
    per_frame = torch.cat(
      [torch.full((L,), self.dataset_weights[i] / L, device=self.device) for i, L in enumerate(lens)]
    )
    self.sample_weights = per_frame / per_frame.sum()

  def amp_dim(self) -> int:
    return self.obs.shape[1]

  def feed_forward_generator(
    self, num_mini_batch: int, mini_batch_size: int
  ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    for _ in range(num_mini_batch):
      idx = torch.multinomial(self.sample_weights, mini_batch_size, replacement=True)
      yield self.obs[idx], self.next_obs[idx]

  # Internals

  def _collect_union_joint_names(self, names: Sequence[str]) -> list[str]:
    union: list[str] = []
    seen = set()
    for nm in names:
      info = np.load(str(self.root / f"{nm}.npy"), allow_pickle=True).item()
      for j in info["joints_list"]:
        if j not in seen:
          seen.add(j)
          union.append(j)
    return union

  def _resample_linear(self, data: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    f = interp1d(t_src, data, axis=0)
    return f(t_dst)

  def _resample_quat(self, raw_xyzw: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> Rotation:
    R = Rotation.from_quat(raw_xyzw)
    slerp = Slerp(t_src, R)
    return slerp(t_dst)

  def _compute_ang_vel(self, rots: Sequence[Rotation], dt: float, local: bool) -> np.ndarray:
    R_prev = rots[:-1]
    R_next = rots[1:]
    if local:
      rel = [Rp.inv() * Rn for Rp, Rn in zip(R_prev, R_next)]
    else:
      rel = [Rn * Rp.inv() for Rp, Rn in zip(R_prev, R_next)]
    rotvec = np.stack([r.as_rotvec() for r in rel]) / dt
    return np.vstack([rotvec, rotvec[-1]])

  def _load_one(
    self,
    path: Path,
    sim_dt: float,
    slow: int,
    expected_joint_names: list[str],
  ) -> MotionData:
    info = np.load(str(path), allow_pickle=True).item()
    dataset_joint_names = info["joints_list"]

    # Build mapping to expected order
    idx_map: list[int | None] = []
    for j in expected_joint_names:
      idx_map.append(dataset_joint_names.index(j) if j in dataset_joint_names else None)

    # Reorder joint positions, fill missing with zeros
    jp = []
    for frame in info["joint_positions"]:
      arr = np.zeros((len(idx_map),), dtype=np.float32)
      for i, src in enumerate(idx_map):
        if src is not None:
          arr[i] = frame[src]
      jp.append(arr)

    # Timing
    dt_src = 1.0 / float(info["fps"]) / float(slow)
    T = len(jp)
    t_src = np.linspace(0, T * dt_src, T)
    T_new = int(T * dt_src / sim_dt)
    t_dst = np.linspace(0, T * dt_src, T_new)

    jp_r = self._resample_linear(np.asarray(jp), t_src, t_dst)
    jv_r = np.vstack([(jp_r[1:] - jp_r[:-1]) / sim_dt, (jp_r[-1:] - jp_r[-2:-1]) / sim_dt])

    root_pos_r = self._resample_linear(np.asarray(info["root_position"]), t_src, t_dst)
    root_quat_r = self._resample_quat(np.asarray(info["root_quaternion"]), t_src, t_dst)  # xyzw
    base_lin_vel_mixed = np.vstack(
      [(root_pos_r[1:] - root_pos_r[:-1]) / sim_dt, (root_pos_r[-1:] - root_pos_r[-2:-1]) / sim_dt]
    )
    base_ang_vel_mixed = self._compute_ang_vel(root_quat_r, sim_dt, local=False)

    # Convert body-frame velocities
    base_lin_vel_local = []
    for R, v in zip(root_quat_r, base_lin_vel_mixed):
      base_lin_vel_local.append(R.as_matrix().T @ v)
    base_lin_vel_local = np.stack(base_lin_vel_local, axis=0)
    base_ang_vel_local = self._compute_ang_vel(root_quat_r, sim_dt, local=True)

    # Convert to torch on target device; convert xyzw → wxyz
    quat_xyzw = root_quat_r.as_quat()
    quat_wxyz = torch.tensor(quat_xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=self.device)

    return MotionData(
      joint_positions=torch.tensor(jp_r, dtype=torch.float32, device=self.device),
      joint_velocities=torch.tensor(jv_r, dtype=torch.float32, device=self.device),
      base_lin_vel_local=torch.tensor(base_lin_vel_local, dtype=torch.float32, device=self.device),
      base_ang_vel_local=torch.tensor(base_ang_vel_local, dtype=torch.float32, device=self.device),
      base_quat_wxyz=quat_wxyz,
    )