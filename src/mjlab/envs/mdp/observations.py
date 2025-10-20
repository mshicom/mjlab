"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_box_minus
if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.utils.sliding_window import SlidingWindow
  
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b




##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]

def joint_pos_abs(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids]

def joint_state(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  qvel = asset.data.joint_vel[:, jnt_ids]
  qpos = asset.data.joint_pos[:, jnt_ids]
  qpos_error = qpos - env.action_manager.action[:, jnt_ids]
  return torch.cat([qvel, qpos_error], dim=-1)

##
# Actions.
##


def last_action(env: ManagerBasedEnv, action_name: str | None = None, **kwarg) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command



##
# Observations history aggregation.
##

def aggregate_cat(env: ManagerBasedRlEnv, hist: SlidingWindow, **kwarg) -> torch.Tensor:
  # (N, C, W) -> (N, [c_1, c_2,..., c_w])
  obs = hist().flatten(start_dim=1)
  return obs


def base_lin_pos(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w

def base_lin_vel_from_pos_diff(env: ManagerBasedRlEnv, hist: SlidingWindow, diff_dt: float =0.02, **kwarg) -> torch.Tensor:
  lin_vel = hist(diff_order=1, diff_dt=diff_dt, **kwarg)
  return lin_vel.flatten(start_dim=1)


def base_ang_pos(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwarg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w

def base_ang_vel_from_pos_diff(env: ManagerBasedRlEnv, hist: SlidingWindow, diff_dt: float =0.02, **kwarg) -> torch.Tensor:
    quats = hist(**kwarg)
    ang_vel_history = []
    for i in range(quats.shape[2]-1):
      q1 = quats[:, :, i]
      q2 = quats[:, :, i + 1]
      ang_vel = quat_box_minus(q1, q2) / diff_dt
      ang_vel_history.append(ang_vel)
    return torch.cat(ang_vel_history, dim=-1)