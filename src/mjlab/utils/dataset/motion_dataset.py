from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from tqdm import tqdm

import mujoco
import numpy as np
import torch

from mjlab.managers.observation_manager import ObservationManager
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    interpolate_trajectories,
    recalculate_traj_angular_velocity,
    recalculate_traj_linear_velocity,
    recalculate_traj_joint_velocity,
)
from mjlab.utils.dataset.traj_handler import TrajectoryHandler, TrajCarry
from mjlab.utils.dataset.traj_process import ExtendTrajData

@dataclass
class AmpDatasetCfg:
  """Configuration for building and serving an offline MotionDataset for AMP demos.

  Fields:
    enabled: bool
      '''Whether to build the offline dataset and expose env.sample_amp_demos().'''
    group_name: str
      '''Observation group name to precompute (must exist in ObservationCfg). Default: 'amp_state'.'''
    trajectories: List[TrajectorySpec]
      '''List of trajectories and weights to include in the dataset.'''
    subsample_stride: int
      '''Use every k-th frame to reduce dataset size and temporal redundancy.'''
    max_frames_per_traj: Optional[int]
      '''Cap the number of frames per trajectory after subsampling. None disables the cap.'''
    seed: int
      '''RNG seed for deterministic dataset sampling.'''
    device: Optional[str]
      '''Device for dataset storage/sampling. If None, use env.device.'''
  """
  enabled: bool = False
  group_name: str = "amp_state"
  trajectories: List[TrajectorySpec] = field(default_factory=list)
  subsample_stride: int = 1
  max_frames_per_traj: Optional[int] = None
  seed: int = 42
  device: Optional[str] = None

@dataclass
class TrajectorySpec:
  """Specification for a single motion trajectory to include in the dataset.

  Fields:
    path: str
      '''Filesystem path to a saved Trajectory (npz). Must be readable by Trajectory.load().'''
    name: Optional[str]
      '''Optional human-readable name. If None, defaults to the filename stem.'''
  """
  path: str
  name: Optional[str] = None
  def __post_init__(self):
    assert Path(self.path).exists(), f"Trajectory path '{self.path}' does not exist."
    # extract name from path if not provided
    if self.name is None:
      self.name = Path(self.path).stem


class _OfflineEntityData:
  """Minimal data container to mirror fields expected by ObservationManager terms.

  Only the slots actually used by your ObservationManager terms need to be populated.
  Here we include the common slots used by mjlab velocity tasks: joint_pos/vel and projected_gravity.
  """
  def __init__(self, device: torch.device):
    self.device = str(device)
    # Root state (optional, populate if Observation terms use them)
    self.root_link_lin_vel_b: torch.Tensor | None = None
    self.root_link_ang_vel_b: torch.Tensor | None = None
    self.projected_gravity_b: torch.Tensor | None = None
    # Joint state (common)
    self.default_joint_pos: torch.Tensor | None = None
    self.default_joint_vel: torch.Tensor | None = None
    self.joint_pos: torch.Tensor | None = None
    self.joint_vel: torch.Tensor | None = None
    # Optional body/site kinematics (populate if needed by your Obs terms)
    self.xpos: torch.Tensor | None = None
    self.xquat: torch.Tensor | None = None
    self.cvel: torch.Tensor | None = None
    self.subtree_com: torch.Tensor | None = None
    self.site_xpos: torch.Tensor | None = None
    self.site_xmat: torch.Tensor | None = None


class _OfflineEntity:
  def __init__(self, device: torch.device, mj_model: mujoco.MjModel):
    self._data = _OfflineEntityData(device)
    self._mj_model = mj_model
    all_joint_names = self._collect_names(mujoco.mjtObj.mjOBJ_JOINT, mj_model.njnt)
    if mj_model.njnt > 0 and mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
      self._root_free_joint_names = [all_joint_names[0]]
      self._joint_names = all_joint_names[1:]
    else:
      self._root_free_joint_names = []
      self._joint_names = all_joint_names
    self._root_qpos_dofs = 7 if self._root_free_joint_names else 0
    self._root_qvel_dofs = 6 if self._root_free_joint_names else 0
    self._body_names = self._collect_names(mujoco.mjtObj.mjOBJ_BODY, mj_model.nbody, skip_first=True)
    self._geom_names = self._collect_names(mujoco.mjtObj.mjOBJ_GEOM, mj_model.ngeom)
    self._site_names = self._collect_names(mujoco.mjtObj.mjOBJ_SITE, mj_model.nsite)
    self._tendon_names = self._collect_names(mujoco.mjtObj.mjOBJ_TENDON, mj_model.ntendon)
    self._actuator_names = self._collect_names(mujoco.mjtObj.mjOBJ_ACTUATOR, mj_model.nu)
    self._total_qpos_dofs = int(mj_model.nq)
    self._total_qvel_dofs = int(mj_model.nv)

  @property
  def data(self) -> _OfflineEntityData:
    return self._data

  @property
  def joint_names(self) -> list[str]:
    return list(self._joint_names)

  @property
  def body_names(self) -> list[str]:
    return list(self._body_names)

  @property
  def geom_names(self) -> list[str]:
    return list(self._geom_names)

  @property
  def site_names(self) -> list[str]:
    return list(self._site_names)

  @property
  def tendon_names(self) -> list[str]:
    return list(self._tendon_names)

  @property
  def actuator_names(self) -> list[str]:
    return list(self._actuator_names)

  @property
  def root_qpos_dofs(self) -> int:
    return self._root_qpos_dofs

  @property
  def root_qvel_dofs(self) -> int:
    return self._root_qvel_dofs

  @property
  def num_joints(self) -> int:
    return len(self._joint_names)

  def select_joint_positions(self, qpos: torch.Tensor) -> torch.Tensor:
    if qpos.shape[-1] == self._total_qpos_dofs:
      return qpos[..., self._root_qpos_dofs :]
    return qpos

  def select_joint_velocities(self, qvel: torch.Tensor) -> torch.Tensor:
    if qvel.shape[-1] == self._total_qvel_dofs:
      return qvel[..., self._root_qvel_dofs :]
    return qvel

  def _collect_names(
    self, obj_type: int, count: int, skip_first: bool = False
  ) -> list[str]:
    names: list[str] = []
    start = 1 if skip_first else 0
    for idx in range(start, count):
      name = mujoco.mj_id2name(self._mj_model, obj_type, idx)
      if name is None:
        name = ""
      names.append(name.split("/")[-1])
    return names

  def find_joints(
    self,
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    subset = joint_subset if joint_subset is not None else self._joint_names
    return resolve_matching_names(name_keys, subset, preserve_order)

  def find_bodies(
    self,
    name_keys: str | Sequence[str],
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self._body_names, preserve_order)

  def find_geoms(
    self,
    name_keys: str | Sequence[str],
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self._geom_names, preserve_order)

  def find_sites(
    self,
    name_keys: str | Sequence[str],
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self._site_names, preserve_order)

  def find_tendons(
    self,
    name_keys: str | Sequence[str],
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self._tendon_names, preserve_order)

  def find_actuators(
    self,
    name_keys: str | Sequence[str],
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self._actuator_names, preserve_order)


class _OfflineActionManager:
  def __init__(self, device: torch.device, source_action_mgr: Any | None):
    self._device = device
    self._source = source_action_mgr

  @property
  def action(self) -> torch.Tensor:
    if self._source is not None:
      action = getattr(self._source, "action", None)
      if isinstance(action, torch.Tensor):
        act = action
        if act.dim() == 0:
          act = act.view(1, 1)
        elif act.dim() == 1:
          act = act.unsqueeze(0)
        else:
          act = act[:1]
        return act.detach().to(self._device)
    return torch.zeros(1, 0, device=self._device)

  def get_term(self, name: str):
    if self._source is not None and hasattr(self._source, "get_term"):
      try:
        return self._source.get_term(name)
      except Exception:
        return None
    return None


class _OfflineCommandManager:
  def __init__(self, device: torch.device, source_command_mgr: Any | None):
    self._device = device
    self._source = source_command_mgr
    self._fallback: dict[str, torch.Tensor] = {}

  def _format_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
    cmd = tensor
    if cmd.dim() == 0:
      cmd = cmd.view(1, 1)
    elif cmd.dim() == 1:
      cmd = cmd.unsqueeze(0)
    else:
      cmd = cmd[:1]
    return cmd.detach().to(self._device)

  def get_command(self, name: str) -> torch.Tensor:
    if self._source is not None:
      try:
        cmd = self._source.get_command(name)
      except Exception:
        cmd = None
      if isinstance(cmd, torch.Tensor):
        return self._format_tensor(cmd)
      term = None
      if hasattr(self._source, "get_term"):
        try:
          term = self._source.get_term(name)
        except Exception:
          term = None
      if term is not None and hasattr(term, "command") and isinstance(term.command, torch.Tensor):
        return self._format_tensor(term.command)
    if name not in self._fallback:
      self._fallback[name] = torch.zeros(1, 0, device=self._device)
    return self._fallback[name]

  def get_term(self, name: str):
    if self._source is not None and hasattr(self._source, "get_term"):
      try:
        return self._source.get_term(name)
      except Exception:
        return None
    return None


class _OfflineEnv:
  """Offline scaffold that mirrors enough of the runtime env for ObservationManager.compute()."""
  def __init__(self, mj_model: mujoco.MjModel, device: torch.device, source_env: Any | None = None):
    self._mj_model = mj_model
    self.device = device
    self.num_envs = 1
    self.scene = {"robot": _OfflineEntity(device, mj_model)}
    source_action_mgr = getattr(source_env, "action_manager", None) if source_env is not None else None
    source_command_mgr = getattr(source_env, "command_manager", None) if source_env is not None else None
    self.action_manager = _OfflineActionManager(device, source_action_mgr)
    self.command_manager = _OfflineCommandManager(device, source_command_mgr)


@dataclass
class _BuiltTrajectory:
  """Internal container for a built trajectory's features and sampling attributes.

  Fields:
    name: str
      '''Identifier for this trajectory (unique across dataset).'''
    path: str
      '''Filesystem path for traceability.'''
    features: torch.Tensor
      '''Tensor of shape [T_i, F] containing the precomputed observation group (e.g., 'amp_state').'''
    enabled: bool
      '''Whether this trajectory is currently enabled for sampling.'''
  """
  name: str
  path: str
  features: torch.Tensor
  enabled: bool = True


class MotionDataset:
  """Offline motion dataset that precomputes an observation group per-frame and supports sampling.

  This dataset:
    - Mirrors runtime observation computation using ObservationManager on a Mujoco model snapshot.
    - Exposes efficient per-frame sampling uniformly across all frames of all enabled trajectories.
    - Supports dynamic trajectory enable/disable (for curriculum/events).
    - Returns features with shape [N, F], matching the online amp_state group for rsl_rl AMP.

  Critical behaviors for AMP:
    - The group_name should match the env observation key used by AMP (default: 'amp_state').
    - It’s recommended to disable observation corruption for this group to keep demos clean.
    - Features must be flattened (F) and identical to online computation to ensure normalizer consistency.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel | None = None,
    obs_manager: ObservationManager | None = None,
    trajectories: Sequence[TrajectorySpec] = (),
    group_name: str = "amp_state",
    env: Optional[Any] = None,
    device: torch.device | str = "cpu",
    subsample_stride: int = 1,
    max_frames_per_traj: Optional[int] = None,
    seed: int = 42,
    dt: float = 0.02,
    auto_extend_incomplete: bool = False,
  ):
    """
    Args:
      mj_model:
        Mujoco MjModel to interpret trajectory qpos/qvel and related kinematics when computing observations.
        If None, it must be resolved from env.sim.mj_model (env required).
      obs_manager:
        ObservationManager configured like the runtime env (must include group_name).
        If None, it must be resolved from env.observation_manager (env required).
      trajectories:
        List of input TrajectorySpec describing the .npz files.
      group_name:
        The observation group key to extract per-frame (default 'amp_state').
      env:
        Optional runtime environment. If provided, mj_model/obs_manager default to:
          mj_model = env.sim.mj_model
          obs_manager = env.observation_manager
      device:
        Torch device for storage/sampling.
      subsample_stride:
        Use every k-th frame when building the dataset to control size and temporal redundancy.
      max_frames_per_traj:
        Optional cap on frames used per trajectory after subsampling.
      seed:
        RNG seed for any stochastic choices (e.g., sampling).
      auto_extend_incomplete:
        If True, attempt to extend trajectories that are missing kinematic fields using
        TrajectoryHandler. Disabled by default to keep lightweight trajectories (e.g. unit tests)
        from invoking the heavy extension path.
    """
    # Resolve mj_model / obs_manager from env if not provided
    if mj_model is None or obs_manager is None:
      assert env is not None, "If mj_model or obs_manager is None, you must provide env."
      if mj_model is None:
        assert hasattr(env, "sim") and hasattr(env.sim, "mj_model"), "env.sim.mj_model not found."
        mj_model = env.sim.mj_model
      if obs_manager is None:
        assert hasattr(env, "observation_manager"), "env.observation_manager not found."
        obs_manager = env.observation_manager

    assert mj_model is not None and obs_manager is not None, "Failed to resolve mj_model / obs_manager."

    self._mj_model = mj_model
    self._data = mujoco.MjData(self._mj_model)
    self._obs_manager = obs_manager  # used as template for cfg
    self._specs = list(trajectories)
    self._group_name = group_name
    self._device = torch.device(device)
    self._subsample = max(1, int(subsample_stride))
    self._max_frames = max_frames_per_traj
    self._rng = np.random.default_rng(seed)
    self.dt = dt
    self._auto_extend_incomplete = bool(auto_extend_incomplete)

    # Offline env scaffold
    self._env = _OfflineEnv(mj_model=mj_model, device=self._device, source_env=env)

    # Built state
    self._trajectories: list[_BuiltTrajectory] = []
    self._feature_dim: Optional[int] = None  # set after compute

  # -----------------------
  # Build lifecycle
  # -----------------------
  def load(self) -> None:
    """Prepare internal trajectory entries without computing features.

    After load(), call compute(force_recompute=False) to compute or reuse cached features.
    """
    self._trajectories.clear()
    self._feature_dim = None
    # Create empty entries for each spec; features are filled during compute()
    for spec in self._specs:
      name = spec.name if spec.name is not None else Path(spec.path).stem
      # Placeholder features (empty); will be replaced in compute()
      placeholder = torch.empty(0, 0)
      self._trajectories.append(_BuiltTrajectory(name=name, path=spec.path, features=placeholder, enabled=True))

  def _get_obs_cfg(self):
    """Fetch the ObservationManager configuration from the provided obs_manager.

    We reconstruct a local ObservationManager bound to the offline env to ensure compute()
    reads data we set on the offline env rather than the external env passed by the user/tests.
    """
    cfg = getattr(self._obs_manager, "cfg", None)
    if cfg is None:
      cfg = getattr(self._obs_manager, "_cfg", None)
    assert cfg is not None, "ObservationManager configuration not found (expected attribute 'cfg')."
    return cfg

  def _prime_offline_env_for_probe(self):
    """Populate the offline env with placeholder tensors so ObservationManager can probe dims.

    ObservationManager._prepare_terms() calls the observation term functions immediately to
    infer shapes. Those functions must return Tensors. We therefore initialize:
      - joint_pos: zeros [1, nq]
      - default_joint_pos: zeros [1, nq]
      - joint_vel: zeros [1, nv]
      - default_joint_vel: zeros [1, nv]
      - projected_gravity_b: [[0, 0, -1]]
      - root_link_lin_vel_b: zeros [1, 3]
      - root_link_ang_vel_b: zeros [1, 3]
    """
    entity_obj = self._env.scene["robot"]
    entity = entity_obj.data
    nq = int(self._mj_model.nq) - entity_obj.root_qpos_dofs
    nv = int(self._mj_model.nv) - entity_obj.root_qvel_dofs
    if entity.joint_pos is None:
      entity.joint_pos = torch.zeros(1, nq, device=self._device, dtype=torch.float32)
    if entity.default_joint_pos is None:
      entity.default_joint_pos = torch.zeros(1, nq, device=self._device, dtype=torch.float32)
    if entity.joint_vel is None:
      entity.joint_vel = torch.zeros(1, nv, device=self._device, dtype=torch.float32)
    if entity.default_joint_vel is None:
      entity.default_joint_vel = torch.zeros(1, nv, device=self._device, dtype=torch.float32)
    if entity.projected_gravity_b is None:
      entity.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]], device=self._device, dtype=torch.float32)
    if entity.root_link_lin_vel_b is None:
      entity.root_link_lin_vel_b = torch.zeros(1, 3, device=self._device, dtype=torch.float32)
    if entity.root_link_ang_vel_b is None:
      entity.root_link_ang_vel_b = torch.zeros(1, 3, device=self._device, dtype=torch.float32)  

  def compute(self, force_recompute: bool = False) -> None:
    """Compute or reuse cached per-frame features for each trajectory.

    - If a cache file exists (same directory as the original traj with a sidecar suffix) and force_recompute=False,
      load features from disk.
    - Otherwise, compute features frame-by-frame with a local ObservationManager bound to the offline env and save the cache.
    - Calls obs_manager.reset() after processing each trajectory (as requested).

    Cache filename format:
      {traj_path}.AMP_{group_name}.pt

    Saved content (torch.save):
      {
        "version": 1,
        "group_name": group_name,
        "features": FloatTensor [T_i, F] (on CPU),
        "feature_dim": int,
        "subsample_stride": int,
        "max_frames_per_traj": Optional[int],
        "mj_model_nq": int,
        "mj_model_nv": int,
      }
    """
    assert len(self._trajectories) > 0, "Call load() before compute()."

    # Only construct a local ObservationManager when we actually need to recompute
    local_om: Optional[ObservationManager] = None

    for i, tr in enumerate(self._trajectories):
      cache_path = self._cache_path(tr.path)
      loaded_from_cache = False
      if (not force_recompute) and cache_path.exists():
        try:
          payload = torch.load(cache_path, map_location="cpu", weights_only=False)
          feats = payload["features"]
          assert isinstance(feats, torch.Tensor) and feats.dim() == 2 and feats.shape[0] > 0
          self._trajectories[i].features = feats  # keep on CPU; move on demand in sample()
          if self._feature_dim is None:
            self._feature_dim = int(feats.shape[1])
          else:
            assert int(feats.shape[1]) == self._feature_dim, "Inconsistent feature dims across trajectories."
          loaded_from_cache = True
        except Exception as e:
          print(f"[MotionDataset] Failed to load cache '{cache_path}': {e}. Recomputing...")

      if loaded_from_cache:
        # Reset original obs_manager as requested even when loading cache (ensure consistent external state)
        if hasattr(self._obs_manager, "reset"):
          self._obs_manager.reset()
        continue

      # Lazily construct a local ObservationManager bound to the offline env using the same config
      if local_om is None:
        cfg = self._get_obs_cfg()
        # Prime offline env so term funcs return Tensors during OM probing
        self._prime_offline_env_for_probe()
        local_om = ObservationManager(cfg, self._env)

      # Compute features if cache not used
      traj = Trajectory.load(tr.path, backend=np)

      if self._auto_extend_incomplete and (not traj.data.is_complete):
          self._th_params = dict(random_start=False, fixed_start_conf=(0, 0))
          traj = self.extend_motion(traj)
          if hasattr(traj.info.model, "to_numpy"):
            traj.info.model = traj.info.model.to_numpy()
          if hasattr(traj.data, "to_numpy"):
            traj.data = traj.data.to_numpy()

      # recalculate velocity
      jnt_type = getattr(getattr(traj, "info", None), "model", None)
      jnt_type = getattr(jnt_type, "jnt_type", None)
      has_free_root = False
      if jnt_type is not None:
        jnt_type_np = np.asarray(jnt_type)
        if jnt_type_np.size > 0:
          has_free_root = int(jnt_type_np[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
      if has_free_root and traj.data.qpos.shape[1] >= 7 and traj.data.qvel.shape[1] >= 6:
          traj = recalculate_traj_angular_velocity(traj, frequency=1.0 / self.dt, backend=np)
          traj = recalculate_traj_linear_velocity(traj, frequency=1.0 / self.dt, backend=np)
      if traj.data.qpos.shape[1] > 7 and traj.data.qvel.shape[1] > 6:
          traj = recalculate_traj_joint_velocity(traj, frequency=1.0 / self.dt, backend=np)
      traj_data = traj.data

      entity_wrapper = self._env.scene["robot"]
      entity = entity_wrapper.data
      frames: list[torch.Tensor] = []

      T_total = int(traj_data.qpos.shape[0])
      idxs = np.arange(0, T_total, self._subsample, dtype=np.int64)
      if self._max_frames is not None:
        idxs = idxs[: self._max_frames]

      for t in idxs:
        # Load joint states
        qpos_np = np.asarray(traj_data.qpos[t : t + 1])
        qvel_np = np.asarray(traj_data.qvel[t : t + 1])
        qpos = torch.from_numpy(qpos_np).to(self._device).to(torch.float32)
        qvel = torch.from_numpy(qvel_np).to(self._device).to(torch.float32)

        joint_pos = entity_wrapper.select_joint_positions(qpos)
        joint_vel = entity_wrapper.select_joint_velocities(qvel)
        entity.joint_pos = joint_pos
        entity.joint_vel = joint_vel

        # Initialize default refs lazily once we see dimensions
        if entity.default_joint_pos is None or entity.default_joint_pos.shape[-1] != joint_pos.shape[-1]:
          entity.default_joint_pos = torch.zeros_like(joint_pos)
        if entity.default_joint_vel is None or entity.default_joint_vel.shape[-1] != joint_vel.shape[-1]:
          entity.default_joint_vel = torch.zeros_like(joint_vel)

        # Optional kinematics (populate if present)
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

        # Projected gravity default if not provided
        if entity.projected_gravity_b is None:
          entity.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]], device=self._device, dtype=torch.float32)

        # Compute observations and extract the target group using the local OM bound to offline env
        assert local_om is not None
        obs_buffer: Dict[str, Any] = local_om.compute()
        assert self._group_name in obs_buffer, (
          f"Observation group '{self._group_name}' missing from ObservationManager outputs."
        )
        group_data = obs_buffer[self._group_name]
        # group_data may be a Tensor or a dict of Tensors; flatten accordingly
        if isinstance(group_data, dict):
          tensors = list(group_data.values())
          assert all(isinstance(x, torch.Tensor) for x in tensors), "All group entries must be Tensors."
          group_tensor = torch.cat(tensors, dim=-1)
        else:
          assert isinstance(group_data, torch.Tensor), "Group data must be a Tensor or dict of Tensors."
          group_tensor = group_data

        # Expect shape [1, F] for a single offline env
        assert group_tensor.dim() == 2 and group_tensor.shape[0] == 1, (
          f"Expected group tensor shape [1, F], got {tuple(group_tensor.shape)}"
        )
        frames.append(group_tensor)

      # Convert and store
      assert len(frames) > 0, f"No frames found after subsampling for trajectory '{tr.name}'."
      features = torch.cat(frames, dim=0).to("cpu")  # store cache on CPU
      self._trajectories[i].features = features

      # Feature dimension checks
      if self._feature_dim is None:
        self._feature_dim = int(features.shape[1])
      else:
        assert int(features.shape[1]) == self._feature_dim, "Inconsistent feature dims across trajectories."

      # Save cache
      payload = {
        "version": 1,
        "group_name": self._group_name,
        "features": features,  # CPU tensor
        "feature_dim": int(features.shape[1]),
        "subsample_stride": int(self._subsample),
        "max_frames_per_traj": None if self._max_frames is None else int(self._max_frames),
        "mj_model_nq": int(self._mj_model.nq),
        "mj_model_nv": int(self._mj_model.nv),
      }
      cache_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(payload, cache_path)

      # Reset local obs manager AFTER processing each trajectory
      if hasattr(local_om, "reset"):
        local_om.reset()

    # Final validation
    assert self._feature_dim is not None and self._feature_dim > 0, "Feature dimension unresolved after compute()."

  def build(self, force_recompute: bool = False) -> None:
    """Convenience wrapper: load() then compute(force_recompute)."""
    self.load()
    self.compute(force_recompute=force_recompute)

  def play_trajectory(
      self,
      n_episodes: int = None,
      n_steps_per_episode: int = None,
      callback_class: Callable = None,
      quiet: bool = False,
  ) -> None:
      """
      Plays a demo of the loaded trajectory by forcing the model
      positions to the ones in the trajectories at every step.

      Args:
          n_episodes (int): Number of episode to replay.
          n_steps_per_episode (int): Number of steps to replay per episode.
          callback_class: Object to be called at each step of the simulation.
          quiet (bool): If True, disable tqdm.
      """
      # import jax
      assert self.th is not None
      if not self.th.is_numpy:
          self.th.to_numpy()


      traj_info = TrajCarry(key=None, traj_state=self.th.init_state())
      traj_data_sample = self.th.get_current_traj_data(traj_info, np)
      

      highest_int = np.iinfo(np.int32).max
      if n_episodes is None:
          n_episodes = highest_int
      for i in range(n_episodes):
          if n_steps_per_episode is None:
              nspe = self.th.len_trajectory(traj_info.traj_state.traj_no) - traj_info.traj_state.subtraj_step_no
          else:
              nspe = n_steps_per_episode

          for j in tqdm(range(nspe), disable=quiet):
              self._mj_model, self._data, traj_info = callback_class(
                  self, self._mj_model, self._data, traj_data_sample, traj_info
              )

              traj_data_sample = self.th.get_current_traj_data(traj_info, np)


  def extend_motion(self, traj: Trajectory) -> Trajectory:
      assert traj.data.n_trajectories == 1

      traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / self.dt, backend=np)
      traj = Trajectory(info=traj_info, data=traj_data)

      self.th =  TrajectoryHandler(model=self._mj_model, warn=False, traj=traj, control_dt=self.dt)
      traj_data, traj_info = self.th.traj.data, self.th.traj.info

      callback = ExtendTrajData(self, model=self._mj_model, n_samples=traj_data.n_samples)
      self.play_trajectory(n_episodes=self.th.n_trajectories, callback_class=callback)
      traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
      traj = replace(traj, data=traj_data, info=traj_info)

      return traj

  def set_sim_state_from_traj_data(self, data: mujoco.MjData, traj_data: TrajectoryData, carry: TrajCarry) -> mujoco.MjData:
    """
    Sets the Mujoco datastructure to the state specified in the trajectory data.

    Args:
        data (MjData): The Mujoco data structure.
        traj_data: The trajectory data containing state information.
        carry (Carry): Additional carry information.

    Returns:
        MjData: The updated Mujoco data structure.
    """
    robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid("root", self._mj_model))[:2]
    free_jnt_qpos_id = np.concatenate(
        [
            mj_jntid2qposid(i, self._mj_model)
            for i in range(self._mj_model.njnt)
            if self._mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
    ).reshape(-1, 7)
    all_free_jnt_qpos_id_xy = free_jnt_qpos_id[:, :2].reshape(-1)
    traj_state = carry.traj_state
    # get the initial state of the current trajectory
    traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, np)
    # subtract the initial state from the current state
    traj_data.qpos[all_free_jnt_qpos_id_xy] -= traj_data_init.qpos[robot_free_jnt_qpos_id_xy]

    if traj_data.xpos.size > 0:
        data.xpos = traj_data.xpos
    if traj_data.xquat.size > 0:
        data.xquat = traj_data.xquat
    if traj_data.cvel.size > 0:
        data.cvel = traj_data.cvel
    if traj_data.qpos.size > 0:
        data.qpos = traj_data.qpos
    if traj_data.qvel.size > 0:
        data.qvel = traj_data.qvel

    return data            

  # -----------------------
  # Sampling API
  # -----------------------
  @torch.no_grad()
  def sample(self, n: int, device: Optional[torch.device | str] = None) -> torch.Tensor:
    """Sample N demo frames uniformly across all frames of enabled trajectories.

    Returns:
      Tensor of shape [N, F] on requested device (defaults to dataset device).
    """
    assert n > 0, "sample(n): n must be > 0"
    device = torch.device(device) if device is not None else self._device
    assert len(self._trajectories) > 0 and self._feature_dim is not None, "Call build()/compute() before sampling."

    # Compute per-trajectory frame counts for enabled trajectories
    counts = np.array([int(tr.features.shape[0]) if tr.enabled else 0 for tr in self._trajectories], dtype=np.int64)
    total_frames = int(np.sum(counts))
    assert total_frames > 0, "No enabled trajectories with frames to sample from."

    # Choose trajectories with probability proportional to their frame counts (uniform across frames globally)
    probs = counts.astype(np.float64) / float(total_frames)
    traj_indices = self._rng.choice(len(self._trajectories), size=n, p=probs)

    # For each selected trajectory, sample a random frame uniformly within that trajectory
    out = torch.empty(n, self._feature_dim, device=device, dtype=torch.float32)
    for i, tr_idx in enumerate(traj_indices):
      tr = self._trajectories[int(tr_idx)]
      T_i = tr.features.shape[0]
      j = int(self._rng.integers(low=0, high=T_i))
      out[i] = tr.features[j].to(device)

    return out

  # -----------------------
  # Introspection
  # -----------------------
  def num_trajs(self) -> int:
    return len(self._trajectories)

  def num_samples(self) -> int:
    return int(sum(tr.features.shape[0] for tr in self._trajectories if tr.enabled))

  def get_feature_dim(self) -> int:
    assert self._feature_dim is not None
    return self._feature_dim

  def list_trajectories(self) -> list[tuple[int, str, int, bool]]:
    """Return [(idx, name, num_frames, enabled), ...] for inspection."""
    return [(i, tr.name, int(tr.features.shape[0]), bool(tr.enabled)) for i, tr in enumerate(self._trajectories)]

  # -----------------------
  # Curriculum/Event controls
  # -----------------------
  def _find_idx(self, idx_or_name: Union[int, str]) -> int:
    if isinstance(idx_or_name, int):
      assert 0 <= idx_or_name < len(self._trajectories)
      return idx_or_name
    # name lookup
    matches = [i for i, tr in enumerate(self._trajectories) if tr.name == idx_or_name]
    assert len(matches) == 1, f"Trajectory name '{idx_or_name}' not found or ambiguous ({len(matches)} matches)."
    return matches[0]

  def enable(self, idx_or_name: Union[int, str], on: bool = True) -> None:
    """Enable/disable a trajectory by index or by its unique name."""
    i = self._find_idx(idx_or_name)
    self._trajectories[i].enabled = bool(on)

  def set_enabled_mask(self, mask: Union[Sequence[bool], torch.Tensor]) -> None:
    """Set the enabled flags for all trajectories at once."""
    if isinstance(mask, torch.Tensor):
      assert mask.dtype == torch.bool and mask.dim() == 1 and mask.shape[0] == len(self._trajectories)
      flags = mask.cpu().numpy().tolist()
    else:
      flags = list(mask)
      assert len(flags) == len(self._trajectories) and all(isinstance(x, (bool, np.bool_)) for x in flags)
    for tr, f in zip(self._trajectories, flags):
      tr.enabled = bool(f)

  # -----------------------
  # Utilities
  # -----------------------
  def _cache_path(self, traj_path: str) -> Path:
    """Return the cache sidecar file path for a given trajectory path.

    Format: {traj_path}.AMP_{group_name}.pt
    Example: "walk.npz" -> "walk.npz.AMP_amp_state.pt"
    """
    return Path(traj_path).with_suffix(Path(traj_path).suffix + f".AMP_{self._group_name}.pt")


# -----------------------
# Helper to prepare AMP demos with an env
# -----------------------
def prepare_amp_demo(
  env: Any,
  trajectories: Sequence[TrajectorySpec],
  group_name: str = "amp_state",
  subsample_stride: int = 1,
  max_frames_per_traj: Optional[int] = None,
  seed: int = 42,
  force_recompute: bool = False,
) -> MotionDataset:
  """Create a MotionDataset bound to env, build (with caching), and attach env.sample_amp_demos.

  Args:
    env:
      Runtime environment. MotionDataset will auto-resolve:
        mj_model = env.sim.mj_model
        obs_manager = env.observation_manager
    trajectories:
      List of trajectories to include.
    group_name:
      Observation group to precompute and serve (must exist in env.observation_manager).
    subsample_stride:
      Use every k-th frame when building the dataset.
    max_frames_per_traj:
      Optional cap on frames per trajectory after subsampling.
    seed:
      RNG seed for dataset sampling.
    force_recompute:
      If True, recompute features even if cache files exist.

  Returns:
    MotionDataset instance. Also sets env.sample_amp_demos(n, device) that draws from this dataset.

  Side-effects:
    - Adds env.sample_amp_demos = lambda n, device=None: dataset.sample(n, device or env.device)
  """
  # Construct dataset with auto-resolved mj_model/obs_manager from env
  device = getattr(env, "device", "cpu")
  ds = MotionDataset(
    mj_model=None,  # auto-resolve
    obs_manager=None,  # auto-resolve
    trajectories=trajectories,
    group_name=group_name,
    env=env,
    device=device,
    subsample_stride=subsample_stride,
    max_frames_per_traj=max_frames_per_traj,
    seed=seed,
  )
  ds.build(force_recompute=force_recompute)

  # Attach a sampling hook on the environment for rsl_rl AMP integration
  def _sample_amp_demos(n: int, device: Optional[torch.device | str] = None) -> torch.Tensor:
    dev = device if device is not None else getattr(env, "device", "cpu")
    return ds.sample(n, device=dev)

  setattr(env, "sample_amp_demos", _sample_amp_demos)
  return ds



def mj_jnt_name2id(name, model):
    """
    Get the joint ID (in the Mujoco datastructure) from the joint name.
    """
    for i in range(model.njnt):
        j = model.joint(i)
        if j.name == name:
            return i
    raise ValueError(f"Joint name {name} not found in model!")


def mj_jntname2qposid(j_name, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qposid(j_id, model)


def mj_jntname2qvelid(j_name, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qvelid(j_id, model)


def mj_jntid2qposid(j_id, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    start_qpos_id = model.jnt_qposadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_id = [i for i in range(start_qpos_id, start_qpos_id+7)]
    else:
        qpos_id = [start_qpos_id]

    return qpos_id


def mj_jntid2qvelid(j_id, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    start_qvel_id = model.jnt_dofadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qvel_id = [i for i in range(start_qvel_id, start_qvel_id+6)]
    else:
        qvel_id = [start_qvel_id]

    return qvel_id