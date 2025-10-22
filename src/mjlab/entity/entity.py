from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity.data import EntityData
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.utils.string import resolve_expr


class EntityType(IntEnum):
  """Type of an Entity.

  ENV: Online/live entity that participates in physics simulation.
  REC: Offline/record entity backed by pre-saved frames; no physics.
  """
  ENV = 0
  REC = 1


@dataclass(frozen=True)
class EntityIndexing:
  """Maps entity elements to global indices and addresses in the simulation.

  bodies: Ordered MjSpec bodies belonging to this entity (excluding 'world').
  joints: Ordered non-free joints for this entity (free joint, if present, is handled separately).
  geoms: Ordered MjSpec geoms belonging to this entity.
  sites: Ordered MjSpec sites belonging to this entity.
  actuators: Ordered MjSpec actuators, or None if not actuated.

  body_ids: Global body IDs in mjModel matching 'bodies'.
  geom_ids: Global geom IDs in mjModel matching 'geoms'.
  site_ids: Global site IDs in mjModel matching 'sites'.
  ctrl_ids: Global actuator IDs matching 'actuators' (empty if not actuated).
  joint_ids: Global joint IDs for non-free joints.

  joint_q_adr: Global qpos addresses of the entity's non-free joints (concatenated ranges).
  joint_v_adr: Global qvel addresses of the entity's non-free joints (concatenated ranges).
  free_joint_q_adr: Global qpos addresses of the (single) free joint (if present).
  free_joint_v_adr: Global qvel addresses of the (single) free joint (if present).

  sensor_adr: Mapping from local sensor name (without prefixes) to global sensordata indices.
  """

  # Elements.
  bodies: tuple[mujoco.MjsBody, ...]
  joints: tuple[mujoco.MjsJoint, ...]
  geoms: tuple[mujoco.MjsGeom, ...]
  sites: tuple[mujoco.MjsSite, ...]
  actuators: tuple[mujoco.MjsActuator, ...] | None

  # Indices.
  body_ids: torch.Tensor
  geom_ids: torch.Tensor
  site_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  joint_ids: torch.Tensor

  # Addresses.
  joint_q_adr: torch.Tensor
  joint_v_adr: torch.Tensor
  free_joint_q_adr: torch.Tensor
  free_joint_v_adr: torch.Tensor

  sensor_adr: dict[str, torch.Tensor]

  @property
  def root_body_id(self) -> int:
    return self.bodies[0].id


@dataclass
class EntityCfg:
  @dataclass
  class InitialStateCfg:
    # Root position and orientation.
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Root linear and angular velocity (only for floating base entities).
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Articulation (only for articulated entities).
    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  spec_fn: Callable[[], mujoco.MjSpec] = field(
    default_factory=lambda: (lambda: mujoco.MjSpec())
  )
  articulation: EntityArticulationInfoCfg | None = None

  # Editors.
  lights: tuple[spec_cfg.LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[spec_cfg.CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[spec_cfg.TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[spec_cfg.MaterialCfg, ...] = field(default_factory=tuple)
  sensors: tuple[spec_cfg.SensorCfg | spec_cfg.ContactSensorCfg, ...] = field(
    default_factory=tuple
  )
  collisions: tuple[spec_cfg.CollisionCfg, ...] = field(default_factory=tuple)

  # Misc.
  debug_vis: bool = False


@dataclass
class EntityArticulationInfoCfg:
  actuators: tuple[spec_cfg.ActuatorCfg, ...] = field(default_factory=tuple)
  soft_joint_pos_limit_factor: float = 1.0


class _RecordDataBuffer:
  """A minimal data carrier mimicking mjwarp.Data for offline (record) frames.

  Instances hold global-shaped tensors sized to the compiled model (nq, nv, nbody, nsite, nu)
  and allow writing the entity-local arrays into the corresponding global addresses used by
  EntityIndexing. All tensors live on a specified device.

  nworld is always 1 for offline computation by design.
  """

  def __init__(self, *, nq: int, nv: int, nbody: int, nsite: int, nu: int, device: str) -> None:
    self.nworld = 1  # offline stream always single-world
    self.device = device

    def t(shape, fill=0.0, dtype=torch.float32) -> torch.Tensor:
      return torch.zeros(shape, device=device, dtype=dtype) if fill == 0.0 else torch.full(shape, fill, device=device, dtype=dtype)

    # DoF-level buffers
    self.qpos = t((1, nq))
    self.qvel = t((1, nv))
    self.qacc = t((1, nv))
    self.qfrc_applied = t((1, nv))
    self.ctrl = t((1, nu))
    self.actuator_force = t((1, nu))

    # Body-level buffers
    self.xpos = t((1, nbody, 3))
    self.xquat = t((1, nbody, 4)); self.xquat[..., 0] = 1.0
    self.xipos = t((1, nbody, 3))
    self.cvel = t((1, nbody, 6))
    self.subtree_com = t((1, nbody, 3))
    self.xfrc_applied = t((1, nbody, 6))

    # Geom-level buffers (unused offline, kept for completeness)
    self.geom_xpos = t((1, 0, 3))
    self.geom_xmat = t((1, 0, 9))

    # Site-level buffers
    self.site_xpos = t((1, nsite, 3))
    self.site_xmat = t((1, nsite, 9))

    # Sensor buffer (empty by default)
    self.sensordata = t((1, 0))


class Entity:
  """An entity represents a physical object in the simulation.

  Entity Type Matrix
  ==================
  MuJoCo entities can be categorized along two dimensions:

  1. Base Type:
    - Fixed Base: Entity is welded to the world (no freejoint)
    - Floating Base: Entity has 6 DOF movement (has freejoint)

  2. Articulation:
    - Non-articulated: No joints other than freejoint
    - Articulated: Has joints in kinematic tree (may or may not be actuated)

  Supported Combinations:
  ----------------------
  | Type                      | Example                    | is_fixed_base | is_articulated | is_actuated |
  |---------------------------|----------------------------|---------------|----------------|-------------|
  | Fixed Non-articulated     | Table, wall, ground plane  | True          | False          | False       |
  | Fixed Articulated         | Robot arm, door on hinges  | True          | True           | True/False  |
  | Floating Non-articulated  | Box, ball, mug             | False         | False          | False       |
  | Floating Articulated      | Humanoid, quadruped        | False         | True           | True/False  |
  """

  def __init__(self, cfg: EntityCfg, entity_type: EntityType = EntityType.ENV, alias_spec: mujoco.MjSpec | None = None) -> None:
    """Construct an Entity.

    Args:
      cfg: Entity configuration for live entities.
      entity_type: ENV for live, REC for offline record.

    Notes:
      - alias_spec is deprecated and ignored for REC entities. Use bind_target_entity().
    """
    self.cfg = cfg
    self._type = entity_type
    self._target: Entity | None = None  # NEW: target live entity for REC
    self._spec = cfg.spec_fn()

    # Identify free joint and articulated joints (will be updated after bind for REC).
    all_joints = self._spec.joints
    self._free_joint = None
    self._non_free_joints = tuple(all_joints)
    if all_joints and all_joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
      self._free_joint = all_joints[0]
      self._non_free_joints = tuple(all_joints[1:])

    if self._type is EntityType.ENV:
      self._apply_spec_editors()
      self._add_initial_state_keyframe()

    # Record bookkeeping.
    self._rec_npz: dict | None = None
    self._rec_arrays: dict[str, np.ndarray] | None = None
    self._rec_len: int = 0
    self._rec_frame: int = 0
    self._rec_buffer: _RecordDataBuffer | None = None

    # Recording state
    self._recording: bool = False
    self._record_frames: dict[str, list[np.ndarray]] | None = None
    self._record_info: dict | None = None

  # NEW: bind record entity to a target live entity.
  def bind_target_entity(self, target: "Entity") -> None:
    """Bind this record entity to a target live entity.

    - Shares the target's spec and layout (names, order).
    - After binding, initialize() will compute indexing on that spec.
    """
    assert self._type is EntityType.REC, "bind_target_entity is only for record entities."
    self._target = target
    self._spec = target.spec  # reuse target spec
    # Refresh joint caches to reflect target spec
    all_joints = self._spec.joints
    self._free_joint = None
    self._non_free_joints = tuple(all_joints)
    if all_joints and all_joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
      self._free_joint = all_joints[0]
      self._non_free_joints = tuple(all_joints[1:])

  def _apply_spec_editors(self) -> None:
    for cfg_list in [
      self.cfg.lights,
      self.cfg.cameras,
      self.cfg.textures,
      self.cfg.materials,
      self.cfg.sensors,
      self.cfg.collisions,
    ]:
      for cfg in cfg_list:
        cfg.edit_spec(self._spec)

    if self.cfg.articulation:
      spec_cfg.ActuatorSetCfg(self.cfg.articulation.actuators).edit_spec(self._spec)

  def _add_initial_state_keyframe(self) -> None:
    qpos_components = []

    if self._free_joint is not None:
      qpos_components.extend([self.cfg.init_state.pos, self.cfg.init_state.rot])

    joint_pos = None
    if self._non_free_joints:
      joint_pos = resolve_expr(self.cfg.init_state.joint_pos, self.joint_names)
      qpos_components.append(joint_pos)

    key_qpos = np.hstack(qpos_components) if qpos_components else np.array([])
    key = self._spec.add_key(name="init_state", qpos=key_qpos)

    if self.is_actuated and joint_pos is not None:
      key.ctrl = joint_pos

  # Attributes.

  @property
  def type(self) -> EntityType:
    """Entity type: ENV (live) or REC (record)."""
    return self._type

  @property
  def is_fixed_base(self) -> bool:
    """Entity is welded to the world."""
    return self._free_joint is None

  @property
  def is_articulated(self) -> bool:
    """Entity is articulated (has fixed or actuated joints)."""
    return len(self._non_free_joints) > 0

  @property
  def is_actuated(self) -> bool:
    """Entity has actuated joints."""
    return self.num_actuators > 0

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def data(self) -> EntityData:
    return self._data

  @property
  def joint_names(self) -> list[str]:
    return [j.name.split("/")[-1] for j in self._non_free_joints]

  @property
  def tendon_names(self) -> list[str]:
    return [t.name.split("/")[-1] for t in self._spec.tendons]

  @property
  def body_names(self) -> list[str]:
    return [b.name.split("/")[-1] for b in self.spec.bodies[1:]]

  @property
  def geom_names(self) -> list[str]:
    return [g.name.split("/")[-1] for g in self.spec.geoms]

  @property
  def site_names(self) -> list[str]:
    return [s.name.split("/")[-1] for s in self.spec.sites]

  @property
  def sensor_names(self) -> list[str]:
    return [s.name.split("/")[-1] for s in self.spec.sensors]

  @property
  def actuator_names(self) -> list[str]:
    return [a.name.split("/")[-1] for a in self.spec.actuators]

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def num_tendons(self) -> int:
    return len(self.tendon_names)

  @property
  def num_bodies(self) -> int:
    return len(self.body_names)

  @property
  def num_geoms(self) -> int:
    return len(self.geom_names)

  @property
  def num_sites(self) -> int:
    return len(self.site_names)

  @property
  def num_sensors(self) -> int:
    return len(self.sensor_names)

  @property
  def num_actuators(self) -> int:
    return len(self.actuator_names)

  # Methods.

  def find_bodies(
    self, name_keys: str | Sequence[str], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self.body_names, preserve_order)

  def find_joints(
    self,
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    if joint_subset is None:
      joint_subset = self.joint_names
    return resolve_matching_names(name_keys, joint_subset, preserve_order)

  def find_tendons(
    self,
    name_keys: str | Sequence[str],
    tendon_subset: list[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    if tendon_subset is None:
      tendon_subset = self.tendon_names
    return resolve_matching_names(name_keys, tendon_subset, preserve_order)

  def find_actuators(
    self,
    name_keys: str | Sequence[str],
    actuator_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if actuator_subset is None:
      actuator_subset = self.actuator_names
    return resolve_matching_names(name_keys, actuator_subset, preserve_order)

  def find_geoms(
    self,
    name_keys: str | Sequence[str],
    geom_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if geom_subset is None:
      geom_subset = self.geom_names
    return resolve_matching_names(name_keys, geom_subset, preserve_order)

  def find_sensors(
    self,
    name_keys: str | Sequence[str],
    sensor_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if sensor_subset is None:
      sensor_subset = self.sensor_names
    return resolve_matching_names(name_keys, sensor_subset, preserve_order)

  def find_sites(
    self,
    name_keys: str | Sequence[str],
    site_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if site_subset is None:
      site_subset = self.site_names
    return resolve_matching_names(name_keys, site_subset, preserve_order)

  def compile(self) -> mujoco.MjModel:
    """Compile the underlying MjSpec into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Write the MjSpec to disk."""
    with open(xml_path, "w") as f:
      f.write(self.spec.to_xml())

  def to_zip(self, path: Path) -> None:
    """Write the MjSpec to a zip file."""
    with path.open("wb") as f:
      mujoco.MjSpec.to_zip(self.spec, f)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    # For REC, indexing/layout should reflect the bound target spec.
    indexing = self._compute_indexing(mj_model, device)
    self.indexing = indexing
    nworld = data.nworld

    # Store the compiled mj_model so offline methods can use it without compiling child spec.
    self._mj_model = mj_model
    
    # Root state - only for movable entities.
    if not self.is_fixed_base:
      default_root_state = (
        tuple(self.cfg.init_state.pos)
        + tuple(self.cfg.init_state.rot)
        + tuple(self.cfg.init_state.lin_vel)
        + tuple(self.cfg.init_state.ang_vel)
      )
      default_root_state = torch.tensor(
        default_root_state, dtype=torch.float, device=device
      )
      default_root_state = default_root_state.repeat(nworld, 1)
    else:
      # Static entities have no root state.
      default_root_state = torch.empty(nworld, 0, dtype=torch.float, device=device)

    # Joint state - only for articulated entities.
    if self.is_articulated:
      default_joint_pos = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_pos, self.joint_names), device=device
      )[None].repeat(nworld, 1)
      default_joint_vel = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_vel, self.joint_names), device=device
      )[None].repeat(nworld, 1)

      if self.is_actuated:
        default_joint_stiffness = model.actuator_gainprm[:, self.indexing.ctrl_ids, 0]
        default_joint_damping = -model.actuator_biasprm[:, self.indexing.ctrl_ids, 2]
      else:
        default_joint_stiffness = torch.empty(
          nworld, 0, dtype=torch.float, device=device
        )
        default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

      # Joint limits and control parameters.
      joint_ids_global = [j.id for j in self._non_free_joints]
      dof_limits = model.jnt_range[:, joint_ids_global]
      default_joint_pos_limits = dof_limits.clone()
      joint_pos_limits = default_joint_pos_limits.clone()
      joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
      joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

      # Get soft limit factor from config.
      if self.cfg.articulation:
        soft_limit_factor = self.cfg.articulation.soft_joint_pos_limit_factor
      else:
        soft_limit_factor = 1.0

      soft_joint_pos_limits = torch.zeros(nworld, self.num_joints, 2, device=device)
      soft_joint_pos_limits[..., 0] = (
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
      )
      soft_joint_pos_limits[..., 1] = (
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor
      )
    else:
      # Non-articulated entities - create empty tensors.
      default_joint_pos = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_vel = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_pos_limits = torch.empty(
        nworld, 0, 2, dtype=torch.float, device=device
      )
      joint_pos_limits = torch.empty(nworld, 0, 2, dtype=torch.float, device=device)
      soft_joint_pos_limits = torch.empty(
        nworld, 0, 2, dtype=torch.float, device=device
      )
      default_joint_stiffness = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

    self._data = EntityData(
      indexing=indexing,
      data=data if self._type is EntityType.ENV else self._maybe_create_offline_buffer(mj_model, device),
      model=model,
      device=device,
      default_root_state=default_root_state,
      default_joint_pos=default_joint_pos,
      default_joint_vel=default_joint_vel,
      default_joint_stiffness=default_joint_stiffness,
      default_joint_damping=default_joint_damping,
      default_joint_pos_limits=default_joint_pos_limits,
      joint_pos_limits=joint_pos_limits,
      soft_joint_pos_limits=soft_joint_pos_limits,
      gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(nworld, 1),
      forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(nworld, 1),
      is_fixed_base=self.is_fixed_base,
      is_articulated=self.is_articulated,
      is_actuated=self.is_actuated,
    )

    # If this is a record entity and we already staged a NPZ, bind first frame now.
    if self._type is EntityType.REC and self._rec_arrays is not None:
      self._bind_rec_frame(self._rec_frame)

  def _maybe_create_offline_buffer(self, mj_model: mujoco.MjModel, device: str) -> _RecordDataBuffer:
    """Instantiate the offline data buffer sized to the compiled mujoco model.

    Uses mujoco.MjModel sizes to avoid triggering warp conversions on mjwarp.Data.
    nworld is forced to 1 for offline buffer.
    """
    buf = _RecordDataBuffer(
      nq=mj_model.nq,
      nv=mj_model.nv,
      nbody=mj_model.nbody,
      nsite=mj_model.nsite,
      nu=mj_model.nu,
      device=device,
    )
    self._rec_buffer = buf
    return buf  # type: ignore[return-value]

  def update(self, dt: float) -> None:
    del dt  # Unused.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.clear_state(env_ids)

  def write_data_to_sim(self) -> None:
    pass

  def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._data.clear_state(env_ids)

  # Offline record I/O.

  def load_record(
    self,
    *,
    npz_path: Path,
    source_xml: Path | None = None,
    mj_model: mujoco.MjModel,
    record_frequency: float | None = None,
    resample_to_frequency: float | None = None,  # NEW: optional resampling target
  ) -> None:
    """Load a record npz, adapt to current entity order, and stage arrays.

    If qvel is missing, it is reconstructed from qpos using 'record_frequency' (Hz)
    or the 'frequency' value in _info. If resample_to_frequency is provided,
    the record is resampled via interpolate_record() after adaptation.

    Args:
      npz_path: Path to npz file.
      source_xml: Optional original XML path, used to reconstruct _info if missing/incomplete.
      mj_model: Compiled model (used to compute kinematics when needed).
      record_frequency: Frequency in Hz of the qpos sampling, required if qvel is absent and _info has no 'frequency'.
      resample_to_frequency: If provided, resample to this frequency after adaptation.
    """
    assert self._type is EntityType.REC, "load_record() is only valid for record entities."
    arrays = _load_record_npz(npz_path, missing_ok=False)

    # Build/validate info: prefer _info from file; otherwise derive from source_xml.
    info = arrays.get("_info", None)
    if info is None or not isinstance(info, dict) or ("joint_names" not in info or "jnt_type" not in info):
      assert source_xml is not None and source_xml.exists(), (
        "Record _info missing or incomplete. Provide a valid source_xml to reconstruct metadata."
      )
      info = _info_from_xml(source_xml)
    # loco-mujoco compatibility: try to get from arrays directly
    if "frequency" not in info:
      info["frequency"] = arrays.get("frequency", None)
    
    # Recover qvel if missing
    if "qvel" not in arrays or arrays["qvel"].size == 0:
      freq = record_frequency if record_frequency is not None else info.get("frequency", None)
      assert freq is not None and freq > 0.0, "Record frequency must be provided (record_frequency or _info['frequency']) to recover qvel from qpos."
      arrays["qvel"] = _recover_qvel_from_qpos(arrays["qpos"], info["joint_names"], np.asarray(info["jnt_type"], dtype=np.int32), float(freq))

    # Adapt entity-local ordering for qpos/qvel (and any existing kinematic arrays).
    adapted = self._filter_and_extend_to_entity(arrays, info, mj_model)

    # If kinematic arrays are missing, compute them via MuJoCo forward per frame using adapted qpos/qvel.
    need_x = any(k not in adapted for k in ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat"))
    if need_x:
      xpos, xquat, cvel, subtree_com, site_xpos, site_xmat = self._compute_rest_via_forward(
        mj_model=mj_model,
        qpos_local=adapted["qpos"],
        qvel_local=adapted["qvel"],
      )
      adapted["xpos"] = xpos
      adapted["xquat"] = xquat
      adapted["cvel"] = cvel
      adapted["subtree_com"] = subtree_com
      adapted["site_xpos"] = site_xpos
      adapted["site_xmat"] = site_xmat

    # Ensure consistency and fill any remaining missing with safe defaults.
    adapted = self.extend_motion(adapted)

    # Persist frequency for later interpolation.
    freq = info.get("frequency", record_frequency)
    if freq is not None:
      adapted["_info"]["frequency"] = float(freq)

    # Stage
    self._rec_npz = {"path": npz_path, "source_xml": source_xml}
    self._rec_arrays = adapted
    self._rec_len = adapted["qpos"].shape[0]
    self._rec_frame = 0

    # Optionally resample to requested frequency.
    if resample_to_frequency is not None:
      # Use the mj_model passed into load_record whenever available (preferred).
      self.interpolate_record(resample_to_frequency, recompute_kinematics=True, mj_model=mj_model)

    if self._rec_buffer is not None:
      self._bind_rec_frame(0)

  def frames(self, start: int = 0, end: int | None = None):
    """Generator over record frames. Each iteration updates self.data to the frame.

    Args:
      start: First frame index (inclusive).
      end: Last frame index (exclusive). If None, iterate until the last available frame.

    Yields:
      Frame index (int). The entity's data properties (e.g., data.joint_vel) reflect this frame.
    """
    assert self._rec_arrays is not None, "No record loaded."
    assert self._rec_buffer is not None, "Entity must be initialized before streaming frames."
    T = self._rec_len
    s = max(0, start)
    e = T if end is None else min(end, T)
    assert 0 <= s < e <= T, f"Invalid frame range: [{s}, {e}) with T={T}"
    for idx in range(s, e):
      self._rec_frame = idx
      self._bind_rec_frame(idx)
      yield idx

  # Recording API (split start_record + add_frame + save_record).

  def start_record(self, *, frequency_hz: float | None = None) -> None:
    """Begin recording frames from the current entity data.

    Args:
      frequency_hz: Optional sampling frequency to embed into the saved file.
    """
    assert not self._recording, "Recording already started."
    self._recording = True
    self._record_frames = {
      "qpos": [],
      "qvel": [],
      "xpos": [],
      "xquat": [],
      "cvel": [],
      "subtree_com": [],
      "site_xpos": [],
      "site_xmat": [],
    }
    self._record_info = {
      "joint_names": self.joint_names_with_free(),
      "jnt_type": _jnt_type_vector(self),
      "body_names": self.body_names,
      "site_names": self.site_names,
    }
    if frequency_hz is not None:
      self._record_info["frequency"] = float(frequency_hz)
    # Embed compiled XML for 3rd-party tools
    self._record_mjcf_xml = self.cfg.spec_fn().to_xml()
      
  def add_frame(self) -> None:
    """Append a frame from current self._data into the in-memory recording buffer."""
    assert self._recording and self._record_frames is not None
    # Prepare local slices
    qpos_local = self._data.data.qpos[:, self.indexing.free_joint_q_adr.tolist() + self.indexing.joint_q_adr.tolist()]
    qvel_local = self._data.data.qvel[:, self.indexing.free_joint_v_adr.tolist() + self.indexing.joint_v_adr.tolist()]

    def gather_b(t: torch.Tensor) -> np.ndarray:
      return t[0].detach().cpu().numpy().astype(np.float32)

    self._record_frames["qpos"].append(gather_b(qpos_local))
    self._record_frames["qvel"].append(gather_b(qvel_local))

    # Body-level arrays (safe when there is at least one body)
    if self.indexing.body_ids.numel() > 0:
      self._record_frames["xpos"].append(gather_b(self._data.data.xpos[:, self.indexing.body_ids]))
      self._record_frames["xquat"].append(gather_b(self._data.data.xquat[:, self.indexing.body_ids]))
      self._record_frames["cvel"].append(gather_b(self._data.data.cvel[:, self.indexing.body_ids]))
      self._record_frames["subtree_com"].append(gather_b(self._data.data.subtree_com[:, self.indexing.body_ids]))
    else:
      # Shouldn't happen in practice, but keep shapes consistent
      self._record_frames["xpos"].append(np.zeros((0, 3), dtype=np.float32))
      self._record_frames["xquat"].append(np.zeros((0, 4), dtype=np.float32))
      self._record_frames["cvel"].append(np.zeros((0, 6), dtype=np.float32))
      self._record_frames["subtree_com"].append(np.zeros((0, 3), dtype=np.float32))

    # Site-level arrays: avoid touching warp buffers when there are no sites
    if self.indexing.site_ids.numel() > 0:
      self._record_frames["site_xpos"].append(gather_b(self._data.data.site_xpos[:, self.indexing.site_ids]))
      self._record_frames["site_xmat"].append(gather_b(self._data.data.site_xmat[:, self.indexing.site_ids]))
    else:
      self._record_frames["site_xpos"].append(np.zeros((0, 3), dtype=np.float32))
      self._record_frames["site_xmat"].append(np.zeros((0, 9), dtype=np.float32))

  def save_record(self, rec_npz_path: Path, *, qpos_only: bool = False) -> None:
    """Finalize and save the in-memory recording to a record npz.

    Args:
      rec_npz_path: Output path.
      qpos_only: If True, only qpos is saved (qvel and kinematics omitted). Useful for compact storage.
    """
    assert self._recording and self._record_frames is not None and self._record_info is not None
    # Stack lists into arrays
    arrays: dict[str, np.ndarray] = {"_info": dict(self._record_info)}
    arrays["qpos"] = np.stack(self._record_frames["qpos"], axis=0).astype(np.float32)
    if not qpos_only:
      arrays["qvel"] = np.stack(self._record_frames["qvel"], axis=0).astype(np.float32)
      arrays["xpos"] = np.stack(self._record_frames["xpos"], axis=0).astype(np.float32)
      arrays["xquat"] = np.stack(self._record_frames["xquat"], axis=0).astype(np.float32)
      arrays["cvel"] = np.stack(self._record_frames["cvel"], axis=0).astype(np.float32)
      arrays["subtree_com"] = np.stack(self._record_frames["subtree_com"], axis=0).astype(np.float32)
      arrays["site_xpos"] = np.stack(self._record_frames["site_xpos"], axis=0).astype(np.float32)
      arrays["site_xmat"] = np.stack(self._record_frames["site_xmat"], axis=0).astype(np.float32)
    T = arrays["qpos"].shape[0]
    arrays["split_points"] = np.array([0, T], dtype=np.int64)
    arrays["mjcf_xml"] = self._record_mjcf_xml
    _save_record_npz(rec_npz_path, arrays)
    self._recording = False
    self._record_frames = None
    self._record_info = None
    self._record_mjcf_xml = None


  # Resampling / speed alignment.

  def interpolate_record(self, new_frequency: float, *, recompute_kinematics: bool = True, mj_model: mujoco.MjModel | None = None) -> None:
    """Resample currently loaded record to a new frequency and update internal buffers.

    - qpos is linearly interpolated for translational/scalar joints.
    - Free-joint quaternion is SLERP-interpolated.
    - qvel is recovered from the resampled qpos via finite differences.
    - If recompute_kinematics=True, xpos/xquat/cvel/subtree_com/site_xpos/site_xmat are recomputed via mj_forward.

    Requires a loaded record (REC entity) and initialized indexing.

    Args:
      new_frequency: Target sampling frequency in Hz.
      recompute_kinematics: Whether to recompute kinematic arrays from the model.
      mj_model: Optional compiled mujoco.MjModel to use for mj_forward. If not provided, the entity must have been initialized previously (which stores mj_model on the entity).
    """
    assert self._type is EntityType.REC and self._rec_arrays is not None, "No record loaded."
    assert new_frequency > 0

    # Prefer mj_model argument, fall back to stored _mj_model set at initialize().
    mjm = mj_model if mj_model is not None else getattr(self, "_mj_model", None)
    if recompute_kinematics and mjm is None:
      raise AssertionError("mj_model is required to recompute kinematics; provide mj_model or call initialize() before interpolate_record()")

    info = self._rec_arrays.get("_info", {})
    old_freq = info.get("frequency", None)
    assert old_freq is not None and old_freq > 0, "Original record frequency missing to resample."
    old_freq = float(old_freq)
    qpos = self._rec_arrays["qpos"]
    T_old = qpos.shape[0]
    # Build time vectors in seconds
    dur = (T_old - 1) / old_freq if T_old > 1 else 0.0
    if dur == 0.0:
      # Update meta even if no-op to keep consistency
      self._rec_arrays["_info"]["frequency"] = float(new_frequency)
      self._rec_len = self._rec_arrays["qpos"].shape[0]
      self._rec_frame = min(self._rec_frame, max(0, self._rec_len - 1))
      return
    t_old = np.linspace(0.0, dur, T_old, dtype=np.float64)
    T_new = int(round(dur * new_frequency)) + 1
    T_new = max(T_new, 2)
    t_new = np.linspace(0.0, dur, T_new, dtype=np.float64)

    # Helpers
    def lininterp(arr: np.ndarray) -> np.ndarray:
      # arr shape (T, D)
      out = np.stack([np.interp(t_new, t_old, arr[:, d]) for d in range(arr.shape[1])], axis=1)
      return out.astype(np.float32)

    # Split qpos into free and joints based on entity layout
    if self._free_joint is not None:
      pos_old = qpos[:, :3]
      quat_old = qpos[:, 3:7]  # wxyz
      joint_old = qpos[:, 7:] if qpos.shape[1] > 7 else np.zeros((T_old, 0), dtype=qpos.dtype)
      pos_new = lininterp(pos_old)
      quat_new = _slerp_sequence(quat_old, t_old, t_new)
      joint_new = lininterp(joint_old) if joint_old.shape[1] > 0 else joint_old.astype(np.float32)
      qpos_new = np.concatenate([pos_new, quat_new, joint_new], axis=1)
    else:
      qpos_new = lininterp(qpos)

    # qvel from qpos
    jnames = self.joint_names_with_free()
    jtypes = _jnt_type_vector(self)
    qvel_new = _recover_qvel_from_qpos(qpos_new, jnames, jtypes, new_frequency)

    # Update arrays
    self._rec_arrays["qpos"] = qpos_new
    self._rec_arrays["qvel"] = qvel_new
    self._rec_arrays["split_points"] = np.array([0, T_new], dtype=np.int64)
    self._rec_arrays["_info"]["frequency"] = float(new_frequency)

    # IMPORTANT: keep derived counters in sync to avoid OOB during frames()
    self._rec_len = T_new
    if self._rec_frame >= self._rec_len:
      self._rec_frame = max(0, self._rec_len - 1)

    # Recompute kinematics if requested
    if recompute_kinematics:
      xpos, xquat, cvel, subtree_com, site_xpos, site_xmat = self._compute_rest_via_forward(
        mj_model=mjm,
        qpos_local=qpos_new,
        qvel_local=qvel_new,
      )
      self._rec_arrays["xpos"] = xpos
      self._rec_arrays["xquat"] = xquat
      self._rec_arrays["cvel"] = cvel
      self._rec_arrays["subtree_com"] = subtree_com
      self._rec_arrays["site_xpos"] = site_xpos
      self._rec_arrays["site_xmat"] = site_xmat

  # Utilities for record alignment.

  def is_compatible(self, rec_info: dict) -> bool:
    """Quick name compatibility check for joints/body/sites (ignoring order)."""
    def _norm(xs): return {x.split("/")[-1] for x in xs} if xs is not None else set()
    ej = set(self.joint_names)
    eb = set(self.body_names)
    es = set(self.site_names)
    tj_all = rec_info.get("joint_names", [])
    # rec includes free joint; ignore it for comparison of articulated joints
    tj = set([n for n in tj_all if n != self._free_name_or_none()])
    tb = _norm(rec_info.get("body_names")) if rec_info.get("body_names") is not None else set()
    ts = _norm(rec_info.get("site_names")) if rec_info.get("site_names") is not None else set()
    return (ej.issubset(tj) or tj.issubset(ej)) and (not eb or eb == tb or not tb) and (not es or es == ts or not ts)

  def extend_motion(self, arrays: dict) -> dict:
    """Ensure all optional arrays exist with safe defaults matching lengths and sizes."""
    T = arrays["qpos"].shape[0]
    nb = len(self.body_names)
    ns = len(self.site_names)
    def ensure(name, shape, fill=0.0):
      if name not in arrays:
        arrays[name] = np.full(shape, fill, dtype=np.float32)
    ensure("xpos", (T, nb, 3))
    ensure("xquat", (T, nb, 4)); arrays["xquat"][..., 0] = 1.0
    ensure("cvel", (T, nb, 6))
    ensure("subtree_com", (T, nb, 3))
    ensure("site_xpos", (T, ns, 3))
    ensure("site_xmat", (T, ns, 9))
    if "split_points" not in arrays:
      arrays["split_points"] = np.array([0, T], dtype=np.int64)
    return arrays

  def _filter_and_extend_to_entity(self, arrays: dict, info: dict, mj_model: mujoco.MjModel) -> dict:
    """Adapt record arrays to this entity's ordering and sizes, with strict assertions."""
    # Build qpos/qvel index mapping per joint from rec info.
    rec_joint_names: list[str] = info.get("joint_names", [])
    rec_jnt_type: np.ndarray = np.array(info.get("jnt_type", []), dtype=np.int32)
    assert len(rec_joint_names) == len(rec_jnt_type), "joint_names and jnt_type must match length."

    # Build running indices for qpos/qvel in record
    tj_qpos_idx: dict[str, np.ndarray] = {}
    tj_qvel_idx: dict[str, np.ndarray] = {}
    qpos_i = 0
    qvel_i = 0
    traj_free_name = None
    for name, jt in zip(rec_joint_names, rec_jnt_type):
      if jt == mujoco.mjtJoint.mjJNT_FREE:
        tj_qpos_idx[name] = np.arange(qpos_i, qpos_i + 7)
        tj_qvel_idx[name] = np.arange(qvel_i, qvel_i + 6)
        qpos_i += 7
        qvel_i += 6
        traj_free_name = name
      elif jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        tj_qpos_idx[name] = np.array([qpos_i])
        tj_qvel_idx[name] = np.array([qvel_i])
        qpos_i += 1
        qvel_i += 1
      else:
        raise AssertionError(f"Unsupported joint type {jt} for record joint {name}")

    rec_qpos = arrays["qpos"]
    rec_qvel = arrays.get("qvel", np.zeros((rec_qpos.shape[0], 0), dtype=np.float32))
    T = rec_qpos.shape[0]
    out_qpos_list = []
    out_qvel_list = []

    if self._free_joint is not None:
      out_qpos_list.append(rec_qpos[:, tj_qpos_idx[traj_free_name]])
      out_qvel_list.append(rec_qvel[:, tj_qvel_idx[traj_free_name]])

    for jn in self.joint_names:
      if jn not in tj_qpos_idx:
        out_qpos_list.append(np.zeros((T, 1), dtype=rec_qpos.dtype))
        out_qvel_list.append(np.zeros((T, 1), dtype=rec_qvel.dtype))
      else:
        out_qpos_list.append(rec_qpos[:, tj_qpos_idx[jn]])
        out_qvel_list.append(rec_qvel[:, tj_qvel_idx[jn]])

    out_qpos = np.concatenate(out_qpos_list, axis=1) if out_qpos_list else np.zeros((T, 0), dtype=np.float32)
    out_qvel = np.concatenate(out_qvel_list, axis=1) if out_qvel_list else np.zeros((T, 0), dtype=np.float32)

    # Bodies/sites: if present, align by name; otherwise leave missing to be computed via forward.
    result = {
      "qpos": out_qpos.astype(np.float32),
      "qvel": out_qvel.astype(np.float32),
      "split_points": arrays.get("split_points", np.array([0, T], dtype=np.int64)).copy(),
      "_info": {
        "joint_names": self.joint_names_with_free(),
        "jnt_type": _jnt_type_vector(self),
        "body_names": self.body_names,
        "site_names": self.site_names,
      },
    }

    if "xpos" in arrays and "xquat" in arrays and "cvel" in arrays and "subtree_com" in arrays:
      # Map by body names if available in info; otherwise assume same order (rare).
      tb_map = {n.split("/")[-1]: i for i, n in enumerate(info.get("body_names", []))} if info.get("body_names", None) is not None else {}
      def gather_or_zero(mat: np.ndarray, name: str, width: int) -> np.ndarray:
        if not tb_map:
          return mat  # assume already aligned
        i = tb_map.get(name, None)
        return mat[:, i:i+1, :] if i is not None else np.zeros((T, 1, width), dtype=np.float32)
      xpos_list, xquat_list, cvel_list, subtree_list = [], [], [], []
      if tb_map:
        for bn in self.body_names:
          xpos_list.append(gather_or_zero(arrays["xpos"], bn, 3))
          xquat_list.append(gather_or_zero(arrays["xquat"], bn, 4))
          cvel_list.append(gather_or_zero(arrays["cvel"], bn, 6))
          subtree_list.append(gather_or_zero(arrays["subtree_com"], bn, 3))
        result["xpos"] = np.concatenate(xpos_list, axis=1)
        result["xquat"] = np.concatenate(xquat_list, axis=1)
        result["cvel"] = np.concatenate(cvel_list, axis=1)
        result["subtree_com"] = np.concatenate(subtree_list, axis=1)

    if "site_xpos" in arrays and "site_xmat" in arrays:
      ts_map = {n.split("/")[-1]: i for i, n in enumerate(info.get("site_names", []))} if info.get("site_names", None) is not None else {}
      if ts_map:
        spos_list, smat_list = [], []
        for sn in self.site_names:
          i = ts_map.get(sn, None)
          if i is not None:
            spos_list.append(arrays["site_xpos"][:, i:i+1, :])
            smat_list.append(arrays["site_xmat"][:, i:i+1, :])
          else:
            spos_list.append(np.zeros((T, 1, 3), dtype=np.float32))
            smat_list.append(np.tile(np.eye(3, dtype=np.float32).reshape(1, 1, 9), (T, 1, 1)))
        result["site_xpos"] = np.concatenate(spos_list, axis=1)
        result["site_xmat"] = np.concatenate(smat_list, axis=1)

    # Carry over frequency if present
    if "frequency" in info:
      result["_info"]["frequency"] = float(info["frequency"])

    # Carry over embedded MJCF XML if present
    if "mjcf_xml" in arrays:
      result["mjcf_xml"] = arrays["mjcf_xml"]

    return result

  def _bind_rec_frame(self, idx: int) -> None:
    """Copy local-arrays of frame idx into the global-shaped offline buffer (nworld=1)."""
    assert self._rec_buffer is not None and self._rec_arrays is not None
    buf = self._rec_buffer
    qpos = torch.as_tensor(self._rec_arrays["qpos"][idx], device=buf.device)
    qvel = torch.as_tensor(self._rec_arrays["qvel"][idx], device=buf.device)
    xpos = torch.as_tensor(self._rec_arrays["xpos"][idx], device=buf.device)
    xquat = torch.as_tensor(self._rec_arrays["xquat"][idx], device=buf.device)
    cvel = torch.as_tensor(self._rec_arrays["cvel"][idx], device=buf.device)
    subtree = torch.as_tensor(self._rec_arrays["subtree_com"][idx], device=buf.device)
    site_xpos = torch.as_tensor(self._rec_arrays["site_xpos"][idx], device=buf.device)
    site_xmat = torch.as_tensor(self._rec_arrays["site_xmat"][idx], device=buf.device)

    if self._data.is_fixed_base:
      if self.indexing.joint_q_adr.numel() > 0:
        buf.qpos[:, self.indexing.joint_q_adr] = qpos[None, :]
      if self.indexing.joint_v_adr.numel() > 0:
        buf.qvel[:, self.indexing.joint_v_adr] = qvel[None, :]
    else:
      buf.qpos[:, self.indexing.free_joint_q_adr] = qpos[None, :7]
      buf.qvel[:, self.indexing.free_joint_v_adr] = qvel[None, :6]
      if self.indexing.joint_q_adr.numel() > 0:
        buf.qpos[:, self.indexing.joint_q_adr] = qpos[None, 7:]
      if self.indexing.joint_v_adr.numel() > 0:
        buf.qvel[:, self.indexing.joint_v_adr] = qvel[None, 6:]

    if xpos.numel() > 0:
      buf.xpos[:, self.indexing.body_ids] = xpos[None, :, :]
    if xquat.numel() > 0:
      buf.xquat[:, self.indexing.body_ids] = xquat[None, :, :]
    if cvel.numel() > 0:
      buf.cvel[:, self.indexing.body_ids] = cvel[None, :, :]
    if subtree.numel() > 0:
      buf.subtree_com[:, self.indexing.body_ids] = subtree[None, :, :]
    if site_xpos.numel() > 0:
      buf.site_xpos[:, self.indexing.site_ids] = site_xpos[None, :, :]
    if site_xmat.numel() > 0:
      buf.site_xmat[:, self.indexing.site_ids] = site_xmat[None, :, :]

  # Convenience helpers.

  def joint_names_with_free(self) -> list[str]:
    names = []
    fn = self._free_name_or_none()
    if fn is not None:
      names.append(fn)
    names.extend(self.joint_names)
    return names

  def _free_name_or_none(self) -> str | None:
    return None if self._free_joint is None else self._free_joint.name.split("/")[-1]

  def update_frame(self, idx: int) -> None:
    """Update the bound offline buffer to frame idx."""
    assert self._rec_arrays is not None and 0 <= idx < self._rec_len
    self._rec_frame = idx
    self._bind_rec_frame(idx)

  def write_root_state_to_sim(
    self, root_state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    self._data.write_root_state(root_state, env_ids)

  def write_root_link_pose_to_sim(
    self,
    root_pose: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the root pose into the simulation. Like `write_root_state_to_sim()`
    but only sets position and orientation.

    Args:
      root_pose: Tensor of shape (N, 7) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_root_pose(root_pose, env_ids)

  def write_root_link_velocity_to_sim(
    self,
    root_velocity: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the root velocity into the simulation. Like `write_root_state_to_sim()`
    but only sets linear and angular velocity.

    Args:
      root_velocity: Tensor of shape (N, 6) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_root_velocity(root_velocity, env_ids)

  def write_joint_state_to_sim(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint state into the simulation.

    The joint state consists of joint positions and velocities. It does not include
    the root state.

    Args:
      position: Tensor of shape (N, num_joints) where N is the number of environments.
      velocity: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_state(position, velocity, joint_ids, env_ids)

  def write_joint_position_to_sim(
    self,
    position: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint positions into the simulation. Like `write_joint_state_to_sim()`
    but only sets joint positions.

    Args:
      position: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_position(position, joint_ids, env_ids)

  def write_joint_velocity_to_sim(
    self,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint velocities into the simulation. Like `write_joint_state_to_sim()`
    but only sets joint velocities.

    Args:
      velocity: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_velocity(velocity, joint_ids, env_ids)

  def write_joint_position_target_to_sim(
    self,
    position_target: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set the joint position targets for PD control.

    Args:
      position_target: Tensor of shape (N, num_joints) where N is the number of
        environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_ctrl(position_target, joint_ids, env_ids)

  def write_external_wrench_to_sim(
    self,
    forces: torch.Tensor,
    torques: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    body_ids: Sequence[int] | slice | None = None,
  ) -> None:
    """Apply external wrenches to bodies in the simulation.

    Underneath the hood, this sets the `xfrc_applied` field in the MuJoCo data
    structure. The wrenches are specified in the world frame and persist until
    the next call to this function or until the simulation is reset.

    Args:
      forces: Tensor of shape (N, num_bodies, 3) where N is the number of
        environments.
      torques: Tensor of shape (N, num_bodies, 3) where N is the number of
        environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
      body_ids: Optional list of body indices or slice specifying which bodies to
        apply the wrenches to. If None, wrenches are applied to all bodies.
    """
    self._data.write_external_wrench(forces, torques, body_ids, env_ids)

  ##
  # Private methods.
  ##

  def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
    bodies = tuple([b for b in self.spec.bodies[1:]])
    joints = self._non_free_joints
    geoms = tuple(self.spec.geoms)
    sites = tuple(self.spec.sites)

    body_ids = torch.tensor([b.id for b in bodies], dtype=torch.int, device=device)
    geom_ids = torch.tensor([g.id for g in geoms], dtype=torch.int, device=device)
    site_ids = torch.tensor([s.id for s in sites], dtype=torch.int, device=device)
    joint_ids = torch.tensor([j.id for j in joints], dtype=torch.int, device=device)

    if self.is_actuated:
      actuators = tuple(self.spec.actuators)
      ctrl_ids = torch.tensor([a.id for a in actuators], dtype=torch.int, device=device)
    else:
      actuators = None
      ctrl_ids = torch.empty(0, dtype=torch.int, device=device)

    joint_q_adr = []
    joint_v_adr = []
    free_joint_q_adr = []
    free_joint_v_adr = []
    for joint in self.spec.joints:
      jnt = model.joint(joint.name)
      jnt_type = jnt.type[0]
      vadr = jnt.dofadr[0]
      qadr = jnt.qposadr[0]
      if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_v_adr.extend(range(vadr, vadr + 6))
        free_joint_q_adr.extend(range(qadr, qadr + 7))
      else:
        joint_v_adr.extend(range(vadr, vadr + dof_width(jnt_type)))
        joint_q_adr.extend(range(qadr, qadr + qpos_width(jnt_type)))
    joint_q_adr = torch.tensor(joint_q_adr, dtype=torch.int, device=device)
    joint_v_adr = torch.tensor(joint_v_adr, dtype=torch.int, device=device)
    free_joint_v_adr = torch.tensor(free_joint_v_adr, dtype=torch.int, device=device)
    free_joint_q_adr = torch.tensor(free_joint_q_adr, dtype=torch.int, device=device)

    sensor_adr = {}
    for sensor in self.spec.sensors:
      sensor_name = sensor.name
      sns = model.sensor(sensor_name)
      dim = sns.dim[0]
      start_adr = sns.adr[0]
      sensor_adr[sensor_name.split("/")[-1]] = torch.arange(
        start_adr, start_adr + dim, dtype=torch.int, device=device
      )

    return EntityIndexing(
      bodies=bodies,
      joints=joints,
      geoms=geoms,
      sites=sites,
      actuators=actuators,
      body_ids=body_ids,
      geom_ids=geom_ids,
      site_ids=site_ids,
      ctrl_ids=ctrl_ids,
      joint_ids=joint_ids,
      joint_q_adr=joint_q_adr,
      joint_v_adr=joint_v_adr,
      free_joint_q_adr=free_joint_q_adr,
      free_joint_v_adr=free_joint_v_adr,
      sensor_adr=sensor_adr,
    )

  def _compute_rest_via_forward(
    self: Entity,
    mj_model: mujoco.MjModel,
    qpos_local: np.ndarray,
    qvel_local: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized wrapper to compute all frames' kinematics via MuJoCo forward."""
    T = qpos_local.shape[0]
    nb = len(self.body_names)
    ns = len(self.site_names)
    xpos = np.zeros((T, nb, 3), dtype=np.float32)
    xquat = np.zeros((T, nb, 4), dtype=np.float32); xquat[..., 0] = 1.0
    cvel = np.zeros((T, nb, 6), dtype=np.float32)
    subtree = np.zeros((T, nb, 3), dtype=np.float32)
    site_xpos = np.zeros((T, ns, 3), dtype=np.float32)
    site_xmat = np.tile(np.eye(3, dtype=np.float32).reshape(1, 1, 9), (T, ns, 1))
    for t in range(T):
      xb, xq, cv, st, sp, sm = _compute_rest_via_forward_per_frame(
        mj_model, self.indexing, qpos_local, qvel_local, t
      )
      xpos[t] = xb
      xquat[t] = xq
      cvel[t] = cv
      subtree[t] = st
      site_xpos[t] = sp
      site_xmat[t] = sm
    return xpos, xquat, cvel, subtree, site_xpos, site_xmat

# ======== Internal NPZ loader/writer (no external deps) & math utils ========

# ======== NPZ, XML metadata helpers ========

def _load_record_npz(path: Path, *, missing_ok: bool = False) -> dict:
  if not path.exists():
    assert missing_ok, f"Record file not found: {path}"
    return {}
  data = np.load(str(path), allow_pickle=True)
  arrays: dict = {}
  for k in data.files:
    arrays[k] = data[k]
  if "_info" in arrays and not isinstance(arrays["_info"], dict):
    arrays["_info"] = arrays["_info"].item()
  if "mjcf_xml" in arrays and not isinstance(arrays["mjcf_xml"], str):
    try:
      arrays["mjcf_xml"] = arrays["mjcf_xml"].item()
    except Exception:
      pass
  if not missing_ok:
    assert "qpos" in arrays, "Invalid record npz: missing qpos."
  return arrays

def _save_record_npz(path: Path, arrays: dict) -> None:
  out = {k: v for k, v in arrays.items() if k != "_info"}
  info = arrays.get("_info", {})
  out["_info"] = np.array(info, dtype=object)
  if "mjcf_xml" in arrays:
    out["mjcf_xml"] = np.array(arrays["mjcf_xml"], dtype=object)
  np.savez(str(path), **out)

def _info_from_xml(xml_path: Path) -> dict:
  """Parse a source MJCF to recover minimal record metadata (_info)."""
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # Joints: include potential free joint first if present
  jnames: list[str] = []
  jtypes: list[int] = []
  if len(spec.joints) > 0 and spec.joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
    jnames.append(spec.joints[0].name.split("/")[-1])
    jtypes.append(int(mujoco.mjtJoint.mjJNT_FREE))
    for j in spec.joints[1:]:
      jnames.append(j.name.split("/")[-1])
      jtypes.append(int(j.type))
  else:
    for j in spec.joints:
      jnames.append(j.name.split("/")[-1])
      jtypes.append(int(j.type))
  bnames = [b.name.split("/")[-1] for b in spec.bodies[1:]]
  snames = [s.name.split("/")[-1] for s in spec.sites]
  return {
    "joint_names": jnames,
    "jnt_type": np.array(jtypes, dtype=np.int32),
    "body_names": bnames,
    "site_names": snames,
  }


def _jnt_type_vector(ent: Entity) -> np.ndarray:
  """Build vector of joint types including free joint as first if present, then non-free joints."""
  types = []
  if ent._free_joint is not None:
    types.append(int(mujoco.mjtJoint.mjJNT_FREE))
  for j in ent._non_free_joints:
    types.append(int(j.type))
  return np.array(types, dtype=np.int32)


def _recover_qvel_from_qpos(qpos: np.ndarray, joint_names: list[str], jnt_types: np.ndarray, freq: float) -> np.ndarray:
  """Recover qvel from qpos via finite differences.

  Assumes qpos layout matches joint_names/jnt_types: [free(7)?] + 1 per hinge/slide in order.
  Free joint quaternion is (w, x, y, z) starting at index 3 within its 7-tuple.

  Returns:
    qvel array of shape (T, 6 + num_hinge/slide) if free present, else (T, num_hinge/slide).
  """
  assert qpos.ndim == 2 and qpos.shape[0] >= 1
  T = qpos.shape[0]
  # Build index vectors
  qvel_cols = []
  col = 0
  for jt in jnt_types.tolist():
    if jt == mujoco.mjtJoint.mjJNT_FREE:
      # qpos: [x,y,z, w,x,y,z], qvel: [vx,vy,vz, wx,wy,wz]
      pos = qpos[:, col:col+3]
      quat = qpos[:, col+3:col+7]  # (w,x,y,z)
      # linear
      lin = np.zeros_like(pos)
      lin[1:] = (pos[1:] - pos[:-1]) * freq
      # angular
      ang = np.zeros((T, 3), dtype=qpos.dtype)
      if T > 1:
        q_prev = quat[:-1]
        q_next = quat[1:]
        # q_delta = q_prev^{-1} * q_next
        q_prev_inv = np.concatenate([q_prev[:, :1], -q_prev[:, 1:]], axis=1)
        w = (q_prev_inv[:, 0] * q_next[:, 0] - np.sum(q_prev_inv[:, 1:] * q_next[:, 1:], axis=1))
        x = (q_prev_inv[:, 0] * q_next[:, 1] + q_prev_inv[:, 1] * q_next[:, 0] + q_prev_inv[:, 2] * q_next[:, 3] - q_prev_inv[:, 3] * q_next[:, 2])
        y = (q_prev_inv[:, 0] * q_next[:, 2] - q_prev_inv[:, 1] * q_next[:, 3] + q_prev_inv[:, 2] * q_next[:, 0] + q_prev_inv[:, 3] * q_next[:, 1])
        z = (q_prev_inv[:, 0] * q_next[:, 3] + q_prev_inv[:, 1] * q_next[:, 2] - q_prev_inv[:, 2] * q_next[:, 1] + q_prev_inv[:, 3] * q_next[:, 0])
        # shortest path
        # clamp and convert to angle-axis
        s = np.clip(2.0 * (w**2) - 1.0, -1.0, 1.0)
        angle = np.arccos(s)
        axis = np.stack([x, y, z], axis=1)
        norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = axis / np.clip(norm, 1e-9, None)
        ang[1:] = axis * angle[:, None] * freq
      qvel_cols.append(np.concatenate([lin, ang], axis=1))
      col += 7
    elif jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      v = np.zeros((T, 1), dtype=qpos.dtype)
      if T > 1:
        v[1:, 0] = (qpos[1:, col] - qpos[:-1, col]) * freq
      qvel_cols.append(v)
      col += 1
    else:
      raise AssertionError(f"Unsupported joint type {jt}")

  qvel = np.concatenate(qvel_cols, axis=1) if qvel_cols else np.zeros((T, 0), dtype=qpos.dtype)
  return qvel


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
  """Spherical linear interpolation between two quaternions (wxyz)."""
  dot = float(np.dot(q0, q1))
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  DOT_THRESHOLD = 0.9995
  if dot > DOT_THRESHOLD:
    # LERP
    res = q0 + t * (q1 - q0)
    res = res / np.linalg.norm(res)
    return res.astype(np.float32)
  theta_0 = np.arccos(dot)
  sin_theta_0 = np.sin(theta_0)
  theta = theta_0 * t
  sin_theta = np.sin(theta)
  s0 = np.sin(theta_0 - theta) / sin_theta_0
  s1 = sin_theta / sin_theta_0
  out = s0 * q0 + s1 * q1
  return out.astype(np.float32)


def _slerp_sequence(quats: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
  """Interpolate quaternion sequence at new time points."""
  T_old = quats.shape[0]
  out = np.zeros((t_new.shape[0], 4), dtype=np.float32)
  for i, tn in enumerate(t_new):
    if tn <= t_old[0]:
      out[i] = quats[0]
      continue
    if tn >= t_old[-1]:
      out[i] = quats[-1]
      continue
    j = np.searchsorted(t_old, tn) - 1
    j = np.clip(j, 0, T_old - 2)
    t0, t1 = t_old[j], t_old[j+1]
    alpha = float((tn - t0) / (t1 - t0)) if t1 > t0 else 0.0
    out[i] = _slerp(quats[j], quats[j+1], alpha)
  return out


def _with_default_qpos_global(mj_model: mujoco.MjModel) -> np.ndarray:
  """Returns a default qpos vector initialized from model keyframe 0 if present, else zeros."""
  qpos = np.zeros(mj_model.nq, dtype=np.float64)
  if mj_model.nkey > 0:
    kqpos = mj_model.key_qpos[0]
    if kqpos.shape[0] == mj_model.nq:
      qpos = kqpos.copy()
  return qpos


def _compute_global_from_local(
  mj_model: mujoco.MjModel,
  indexing: EntityIndexing,
  qpos_local: np.ndarray,
  qvel_local: np.ndarray,
  t: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Compose full-model qpos/qvel from entity-local arrays for frame t."""
  qpos_g = _with_default_qpos_global(mj_model)
  qvel_g = np.zeros(mj_model.nv, dtype=np.float64)
  if indexing.free_joint_q_adr.numel() > 0:
    qpos_g[indexing.free_joint_q_adr.cpu().numpy()] = qpos_local[t, :7]
    qvel_g[indexing.free_joint_v_adr.cpu().numpy()] = qvel_local[t, :6]
    qpos_local_joint = qpos_local[t, 7:]
    qvel_local_joint = qvel_local[t, 6:]
  else:
    qpos_local_joint = qpos_local[t]
    qvel_local_joint = qvel_local[t]
  if indexing.joint_q_adr.numel() > 0 and qpos_local_joint.size > 0:
    qpos_g[indexing.joint_q_adr.cpu().numpy()] = qpos_local_joint
  if indexing.joint_v_adr.numel() > 0 and qvel_local_joint.size > 0:
    qvel_g[indexing.joint_v_adr.cpu().numpy()] = qvel_local_joint
  return qpos_g, qvel_g


def _compute_rest_via_forward_per_frame(
  mj_model: mujoco.MjModel,
  indexing: EntityIndexing,
  qpos_local: np.ndarray,
  qvel_local: np.ndarray,
  t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Compute entity-local kinematics for frame t via mujoco forward."""
  data = mujoco.MjData(mj_model)
  qpos_g, qvel_g = _compute_global_from_local(mj_model, indexing, qpos_local, qvel_local, t)
  data.qpos[:] = qpos_g
  data.qvel[:] = qvel_g
  mujoco.mj_forward(mj_model, data)
  b_ids = indexing.body_ids.cpu().numpy()
  s_ids = indexing.site_ids.cpu().numpy()
  xpos = data.xpos[b_ids].astype(np.float32)
  xquat = data.xquat[b_ids].astype(np.float32)
  cvel = data.cvel[b_ids].astype(np.float32)
  subtree = data.subtree_com[b_ids].astype(np.float32)
  site_xpos = data.site_xpos[s_ids].astype(np.float32)
  site_xmat = data.site_xmat[s_ids].astype(np.float32)
  return xpos, xquat, cvel, subtree, site_xpos, site_xmat


