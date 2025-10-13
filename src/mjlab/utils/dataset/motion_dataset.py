from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mujoco
import numpy as np
import torch

from mjlab.managers.observation_manager import ObservationManager
from mjlab.utils.dataset.traj_class import Trajectory
from mjlab.utils.dataset.traj_handler import TrajectoryHandler


@dataclass
class TrajectorySpec:
  path: str
  weight: float = 1.0


class _OfflineEntityData:
  def __init__(self, device: torch.device):
    self.device = str(device)

    self.root_link_lin_vel_b: torch.Tensor | None = None
    self.root_link_ang_vel_b: torch.Tensor | None = None
    self.projected_gravity_b: torch.Tensor | None = None

    self.default_joint_pos: torch.Tensor | None = None
    self.default_joint_vel: torch.Tensor | None = None
    self.joint_pos: torch.Tensor | None = None
    self.joint_vel: torch.Tensor | None = None

    self.xpos: torch.Tensor | None = None
    self.xquat: torch.Tensor | None = None
    self.cvel: torch.Tensor | None = None
    self.subtree_com: torch.Tensor | None = None
    self.site_xpos: torch.Tensor | None = None
    self.site_xmat: torch.Tensor | None = None


class _OfflineEntity:
  def __init__(self, device: torch.device):
    self._data = _OfflineEntityData(device)

  @property
  def data(self) -> _OfflineEntityData:
    return self._data


class _OfflineEnv:
  def __init__(self, mj_model: mujoco.MjModel, device: torch.device):
    self._mj_model = mj_model
    self.device = device
    self.num_envs = 1
    self.scene = {"robot": _OfflineEntity(device)}

    empty_tensor = torch.empty(1, 0, device=device)
    self.action_manager = type(
      "_OfflineActionManager",
      (),
      {"action": empty_tensor, "get_term": lambda self, name: None},
    )()
    self.command_manager = type(
      "_OfflineCommandManager",
      (),
      {"get_command": lambda self, name: None},
    )()


class WeightedReplayBuffer:
  def __init__(self, device: torch.device = torch.device("cpu")):
    self.device = device
    self._obs: list[torch.Tensor] = []
    self._weights: list[float] = []

  def add(self, obs: torch.Tensor, weight: float) -> None:
    assert obs.dim() == 2 and obs.shape[0] == 1
    self._obs.append(obs.detach().to(self.device))
    self._weights.append(float(weight))

  def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
    if not self._obs:
      return (
        torch.empty(0, 0, device=self.device),
        torch.empty(0, device=self.device),
      )
    obs_tensor = torch.cat(self._obs, dim=0)
    weights_tensor = torch.tensor(self._weights, dtype=torch.float32, device=self.device)
    return obs_tensor, weights_tensor


class MotionDataset:
  def __init__(
    self,
    mj_model: mujoco.MjModel,
    obs_manager: ObservationManager,
    trajectories: List[TrajectorySpec],
    device: torch.device | str = "cpu",
  ):
    self._mj_model = mj_model
    self._obs_manager = obs_manager
    self._specs = trajectories
    self._device = torch.device(device)
    self._buffer = WeightedReplayBuffer(device=self._device)
    self._env = _OfflineEnv(mj_model=mj_model, device=self._device)

  def build(self) -> None:
    for spec in self._specs:
      traj = Trajectory.load(spec.path, backend=np)
      traj_data, _ = TrajectoryHandler.filter_and_extend(traj.data, traj.info, self._mj_model)

      self._obs_manager.reset()
      entity = self._env.scene["robot"].data

      T = int(traj_data.qpos.shape[0])
      for t in range(T):
        qpos = torch.from_numpy(np.asarray(traj_data.qpos[t : t + 1])).to(self._device).to(torch.float32)
        qvel = torch.from_numpy(np.asarray(traj_data.qvel[t : t + 1])).to(self._device).to(torch.float32)
        entity.joint_pos = qpos
        entity.joint_vel = qvel

        if entity.default_joint_pos is None:
          entity.default_joint_pos = torch.zeros_like(qpos)
        if entity.default_joint_vel is None:
          entity.default_joint_vel = torch.zeros_like(qvel)

        if getattr(traj_data, "cvel", None) is not None and traj_data.cvel.size > 0:
          cvel = torch.from_numpy(np.asarray(traj_data.cvel[t : t + 1])).to(self._device).to(torch.float32)
          entity.cvel = cvel
          entity.root_link_lin_vel_b = cvel[:, :, 3:6][:, 0]
          entity.root_link_ang_vel_b = cvel[:, :, 0:3][:, 0]

        if getattr(traj_data, "xpos", None) is not None and traj_data.xpos.size > 0:
          entity.xpos = torch.from_numpy(np.asarray(traj_data.xpos[t : t + 1])).to(self._device).to(torch.float32)

        if getattr(traj_data, "xquat", None) is not None and traj_data.xquat.size > 0:
          entity.xquat = torch.from_numpy(np.asarray(traj_data.xquat[t : t + 1])).to(self._device).to(torch.float32)

        if getattr(traj_data, "subtree_com", None) is not None and traj_data.subtree_com.size > 0:
          entity.subtree_com = torch.from_numpy(np.asarray(traj_data.subtree_com[t : t + 1])).to(self._device).to(torch.float32)

        if getattr(traj_data, "site_xpos", None) is not None and traj_data.site_xpos.size > 0:
          entity.site_xpos = torch.from_numpy(np.asarray(traj_data.site_xpos[t : t + 1])).to(self._device).to(torch.float32)

        if getattr(traj_data, "site_xmat", None) is not None and traj_data.site_xmat.size > 0:
          entity.site_xmat = torch.from_numpy(np.asarray(traj_data.site_xmat[t : t + 1])).to(self._device).to(torch.float32)

        if entity.projected_gravity_b is None:
          entity.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]], device=self._device, dtype=torch.float32)

        obs_buffer = self._obs_manager.compute()
        concatenated = []
        for group_name in self._obs_manager.group_obs_dim.keys():
          group_data = obs_buffer[group_name]
          if isinstance(group_data, dict):
            concatenated.append(torch.cat(list(group_data.values()), dim=-1))
          else:
            concatenated.append(group_data)
        if not concatenated:
          continue
        obs_tensor = torch.cat(concatenated, dim=-1)
        self._buffer.add(obs=obs_tensor, weight=spec.weight)

  def get_replay_buffer(self) -> WeightedReplayBuffer:
    return self._buffer