from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from mjlab.amp.config import AmpFeatureSetCfg
from mjlab.amp.feature_manager import FeatureManager
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity import mdp


@dataclass
class _EnvInfo:
    """
    Lightweight adapter that mimics the subset of TrajectoryInfo used by FeatureManager.
    Constructed at runtime inside the env to avoid serialization issues.

    Fields used by FeatureManager:
    - joint_names, joint_name2ind_qvel         (for qvel selection)
    - body_names, body_name2ind (optional)     (for body-based sources if you extend this)
    - site_names, site_name2ind (optional)     (for site-based sources if you extend this)

    Note: We keep a tiny _ModelShim with njnt only to align with FeatureManager’s expectations.
    """
    joint_names: List[str]
    joint_name2ind_qvel: dict[str, np.ndarray]
    body_names: Optional[List[str]] = None
    body_name2ind: Optional[dict[str, np.ndarray]] = None
    site_names: Optional[List[str]] = None
    site_name2ind: Optional[dict[str, np.ndarray]] = None

    class _ModelShim:
        def __init__(self, njnt: int):
            self.njnt = njnt

    @property
    def model(self):
        return _EnvInfo._ModelShim(njnt=len(self.joint_names))


def amp_features_obs(
    env: ManagerBasedRlEnv,
    feature_set: AmpFeatureSetCfg,
    sensor_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    AMP observation term (function form).
    - Maintains per-env rolling windows for qvel, base_lin, base_ang, contacts in env._amp_obs_state.
    - Computes features via FeatureManager and writes to env.extras['amp_observations'].

    Args:
      env: ManagerBasedRlEnv
      feature_set: AmpFeatureSetCfg (declarative descriptor set)
      sensor_names: optional list of contact sensor names ('found' channel per sensor)

    Returns:
      torch.Tensor [N, F] AMP feature vector per env.

    Notes:
      - State (FeatureManager, buffers) is created lazily and stored on 'env', not in the config,
        so configs remain picklable.
    """
    state = getattr(env, "_amp_obs_state", None)
    if state is None:
        state = _init_amp_obs_state(env, feature_set, sensor_names)
        setattr(env, "_amp_obs_state", state)

    qvel = mdp.joint_vel_rel(env)        # [N, Dq]
    base_lin = mdp.base_lin_vel(env)     # [N, 3]
    base_ang = mdp.base_ang_vel(env)     # [N, 3]
    contacts = _compute_contacts(env, state["sensor_names"])  # [N, K]

    _append_frame(state["buffers"]["qvel"], qvel)
    _append_frame(state["buffers"]["base_lin"], base_lin)
    _append_frame(state["buffers"]["base_ang"], base_ang)
    _append_frame(state["buffers"]["contacts"], contacts)
    state["filled"].clamp_(max=state["window_size_max"]).add_(1)

    windows = state["buffers"]
    feats = state["manager"].compute(windows, state["dt"])
    env.extras["amp_observations"] = feats
    return feats


def amp_features_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
) -> dict:
    """
    Reset function to clear the AMP observation rolling buffers.

    Intended usage:
      - Wire as an EventManager 'reset' term so it runs on episode resets:
          self.events.reset_amp_obs = EventTermCfg(func=amp_features_reset, mode='reset')

    Args:
      env: ManagerBasedRlEnv
      env_ids: ids (1D LongTensor) or slice selecting envs to reset. If None, resets all.

    Returns:
      dict: extras/log dictionary (empty).
    """
    state = getattr(env, "_amp_obs_state", None)
    if state is None:
        return {}

    if env_ids is None:
        # Reset all envs.
        for k in state["buffers"]:
            state["buffers"][k].zero_()
        state["filled"].zero_()
        return {}

    # Reset a subset.
    if isinstance(env_ids, slice):
        for k in state["buffers"]:
            state["buffers"][k][env_ids].zero_()
        state["filled"][env_ids] = 0
    else:
        for k in state["buffers"]:
            state["buffers"][k][env_ids] = 0
        state["filled"][env_ids] = 0
    return {}


def _init_amp_obs_state(env: ManagerBasedRlEnv, feature_set: AmpFeatureSetCfg, sensor_names: Optional[List[str]]):
    device = torch.device(env.device)
    N = env.num_envs
    dt = env.step_dt
    window_size_max = max(t.window_size for t in feature_set.terms)

    qvel_dim = _qvel_dim(env)
    joint_names = _get_joint_names(env, qvel_dim)
    joint_name2ind_qvel = {nm: np.array([i], dtype=np.int64) for i, nm in enumerate(joint_names)}
    env_info = _EnvInfo(
        joint_names=joint_names,
        joint_name2ind_qvel=joint_name2ind_qvel,
        body_names=[], body_name2ind={},
        site_names=[], site_name2ind={},
    )

    manager = FeatureManager(feature_set)
    meta = {"contacts_names": sensor_names} if sensor_names else None
    manager.resolve(env_info, device, meta=meta)

    buffers = {
        "qvel": torch.zeros(N, window_size_max, qvel_dim, device=device),
        "base_lin": torch.zeros(N, window_size_max, 3, device=device),
        "base_ang": torch.zeros(N, window_size_max, 3, device=device),
        "contacts": torch.zeros(N, window_size_max, max(1, len(sensor_names or [])), device=device),
    }
    filled = torch.zeros(N, dtype=torch.long, device=device)
    return {
        "manager": manager,
        "buffers": buffers,
        "filled": filled,
        "dt": dt,
        "window_size_max": window_size_max,
        "sensor_names": sensor_names or [],
    }


def _append_frame(buf: torch.Tensor, val: torch.Tensor):
    buf[:] = torch.roll(buf, shifts=-1, dims=1)
    if val.ndim == 1:
        val = val.unsqueeze(0)
    if val.ndim == 2:
        val = val.unsqueeze(1)  # [N,1,D]
    D = buf.shape[-1]
    if val.shape[-1] != D:
        tmp = torch.zeros(val.shape[0], 1, D, device=buf.device, dtype=buf.dtype)
        tmp[..., : min(D, val.shape[-1])] = val[..., : min(D, val.shape[-1])]
        val = tmp
    buf[:, -1, :] = val[:, 0, :]


def _compute_contacts(env: ManagerBasedRlEnv, sensor_names: List[str]) -> torch.Tensor:
    if not sensor_names:
        return torch.zeros(env.num_envs, 1, device=env.device)
    asset = env.scene["robot"]
    vals = []
    for s in sensor_names:
        sensor = asset.data.sensor_data[s]  # [N,1] 'found'
        vals.append((sensor[:, 0] > 0).float())
    return torch.stack(vals, dim=1)


def _qvel_dim(env: ManagerBasedRlEnv) -> int:
    asset = env.scene["robot"]
    return asset.num_joints


def _get_joint_names(env: ManagerBasedRlEnv, D: int) -> List[str]:
    asset = env.scene["robot"]
    for attr in ("dof_names", "joint_names"):
        n = getattr(asset, attr, None)
        if n is not None and len(n) == D:
            return list(n)
    n = getattr(asset.data, "dof_names", None)
    if n is not None and len(n) == D:
        return list(n)
    return [f"dof_{i}" for i in range(D)]