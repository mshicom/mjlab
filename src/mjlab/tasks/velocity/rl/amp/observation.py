from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv


def build_amp_obs_from_env(
  env: ManagerBasedRlEnv,
  *,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Construct AMP observation = [qpos, qvel, base_lin_vel_b, base_ang_vel_b].

  Shapes:
    - qpos: (B, nq_actuated)
    - qvel: (B, nq_actuated)
    - base_lin_vel_b: (B, 3)
    - base_ang_vel_b: (B, 3)
  """
  asset: Entity = env.scene[asset_name]
  qpos = asset.data.joint_pos  # (B, DoF)
  qvel = asset.data.joint_vel  # (B, DoF)
  blv = asset.data.root_link_lin_vel_b  # (B, 3)
  bav = asset.data.root_link_ang_vel_b  # (B, 3)
  return torch.cat((qpos, qvel, blv, bav), dim=-1)