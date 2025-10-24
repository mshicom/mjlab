"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg, RecordCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.tasks.velocity.mdp import AMDDemoCfg

from mjlab.managers.manager_term_config import term, ObservationTermCfg as ObsTerm, EventTermCfg as EventTerm
from mjlab.rl.config import RslRlPpoAmpCfg
SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
  records=[],
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class CommandsCfg:
  twist: mdp.UniformVelocityCommandCfg = term(
    mdp.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=mdp.UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )

  
@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "twist"}
    )

    # joint state history (qvel, qpos_error)
    joint_state_history: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_state,
      hist_window_size=2,
      hist_func=mdp.aggregate_cat,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[
          r".*hip.*",
          r".*knee.*",
          r".*ankle.*",
        ]),   # can be overrided in robot cfg.
      }
    )
    
    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False
      

  @dataclass
  class AmpStateCfg(ObsGroup):
    joint_abs_hist: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_abs,
      hist_window_size=2,
      hist_func=mdp.aggregate_cat,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[
          r".*",
          # r".*hip.*",
          # r".*knee.*",
          # r".*ankle.*",
        ]),   # can be overrided in robot cfg.
      }
    )
    
    base_lin_vel_hist: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      hist_window_size=2,
      hist_func=mdp.aggregate_cat,
    )
    
    base_ang_vel_hist: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      hist_window_size=2,
      hist_func=mdp.aggregate_cat,
    )

    def __post_init__(self):
      self.enable_corruption = False
      self.concatenate_terms = True
      self.concatenate_dim = -1

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)
  amp_state: AmpStateCfg = field(default_factory=AmpStateCfg)


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
      "velocity_range": {},
    },
  )
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )
  push_robot: EventTerm | None = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
  )
  foot_friction: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )



@dataclass
class RewardCfg:
  # is_alive: RewardTerm = term(
  #   RewardTerm, 
  #   func=mdp.is_alive, 
  #   weight=1.0
  # )
  
  track_lin_vel_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_lin_vel_exp,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  track_ang_vel_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_ang_vel_exp,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  pose: RewardTerm = term(
    RewardTerm,
    func=mdp.posture,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "std": [],
    },
  )
  dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)
  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.1)

  air_time: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_air_time,
    weight=0.01,
    params={
      "asset_name": "robot",
      "threshold_min": 0.05,
      "threshold_max": 0.15,
      "command_name": "twist",
      "command_threshold": 0.05,
      "sensor_names": [],
      "reward_mode": "on_landing",
    },
  )
  
  feet_slide: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_slide,
    weight=-0.25,
    params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
        "sensor_names": [],  # Override in robot cfg.
    },
  )
  


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )
  root_height_below_minimum: DoneTerm = term(
    DoneTerm, func=mdp.root_height_below_minimum, params={"minimum_height": 0.1} # override in robot cfg.
  )


@dataclass
class CurriculumCfg:
  terrain_levels: CurrTerm | None = term(
    CurrTerm, func=mdp.terrain_levels_vel, params={"command_name": "twist"}
  )
  command_vel: CurrTerm | None = term(
    CurrTerm,
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 500 * 24, "range": (-3.0, 3.0)},
      ],
    },
  )


SIM_CFG = SimulationCfg(
  nconmax=140_000,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

AMP_CFG = AMDDemoCfg(
  enabled=False,
  obs_key="amp_state",
  logical_target_name="robot",
  exclude_terms=["actions", "command", "joint_state_history"], # "joint_vel", "joint_pos", "projected_gravity"],
  use_cache=True,
  cache_dir=None,
  cache_filename=None,
  prefer_cuda=True,
)

@dataclass
class LocomotionVelocityEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 4  # 50 Hz control frequency.
  episode_length_s: float = 20.0
  amp_demo: AMDDemoCfg = field(default_factory=lambda: AMP_CFG)

  def __post_init__(self):
    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True
    
    if len(SCENE_CFG.records) == 0:
      self.amp_demo.enabled = False
    
    if self.amp_demo.enabled:
      # check if obs_key in observations
      if not hasattr(self.observations, self.amp_demo.obs_key):
        raise ValueError(f"AMP obs_key '{self.amp_demo.obs_key}' not found in observations.")