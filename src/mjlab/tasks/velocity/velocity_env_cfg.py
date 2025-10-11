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
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.managers.manager_term_config import term, ObservationTermCfg as ObsTerm, EventTermCfg as EventTerm
from mjlab.amp.obs_terms import amp_features_reset
from mjlab.amp.config import AmpCfg, AmpFeatureSetCfg, FeatureTermCfg, JointSelectionCfg, AmpDatasetCfg, SymmetryAugmentCfg
from mjlab.amp.obs_terms import amp_features_obs

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
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

AMP_CFG = AmpCfg(
  reward_coef=0.5,
  discr_hidden_dims=(1024, 512, 256),
  task_reward_lerp=0.0,
  feature_set=AmpFeatureSetCfg(
    terms=[
      # RMS of joint speed/accel/jerk on informative joints
      FeatureTermCfg(
        name="joint_speed_rms",
        source="qvel",
        channels=["scalar"],  # interpret as scalar per selected DOF
        window_size=30,
        pre_diff="none",
        aggregators=["rms"],
        select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
      ),
      FeatureTermCfg(
        name="joint_accel_rms",
        source="qvel",
        channels=["scalar"],
        window_size=30,
        pre_diff="acceleration",
        aggregators=["rms"],
        select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
      ),
      FeatureTermCfg(
        name="joint_jerk_rms",
        source="qvel",
        channels=["scalar"],
        window_size=30,
        pre_diff="jerk",
        aggregators=["rms"],
        select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
      ),
      # Global energy proxy: mean speed^2 of selected bodies (or full robot)
      FeatureTermCfg(
        name="energy_proxy",
        source="cvel",  # body 6D velocities; use linear components
        channels=["lin.x", "lin.y", "lin.z"],  # assume selector will pick linear parts
        window_size=50,
        aggregators=["mean"],  # mean of squared speed can be implemented in term or via pre_diff mode
        select=JointSelectionCfg(bodies=["pelvis", "torso"]),
      ),
      # Foot-ground duty (L/R) and transition rate
      # FeatureTermCfg(
      #   name="duty_left",
      #   source="site_xpos",  # or derive from contacts; see preprocessing
      #   channels=["z<eps->contact"],  # custom channel routine in selector
      #   window_size=50,
      #   aggregators=["mean"],  # mean(contact) ~= duty factor
      #   select=JointSelectionCfg(sites=["left_foot"]),
      # ),
      # FeatureTermCfg(
      #   name="duty_right",
      #   source="site_xpos",
      #   channels=["z<eps->contact"],
      #   window_size=50,
      #   aggregators=["mean"],
      #   select=JointSelectionCfg(sites=["right_foot"]),
      # ),
      # FeatureTermCfg(
      #   name="contact_transition_rate",
      #   source="site_xpos",
      #   channels=["z<eps->contact"],
      #   window_size=50,
      #   aggregators=["rms"],  # rms of diff(contact) ~ transition rate
      #   pre_diff="velocity",
      #   select=JointSelectionCfg(sites=["left_foot", "right_foot"]),
      # ),
      # Left-right symmetry: ankle & wrist speed correlations
      FeatureTermCfg(
        name="lr_symmetry_ankle_wrist",
        source="cvel",
        channels=["lin.speed"],  # add "speed" support in selector
        window_size=50,
        aggregators=["mean"],  # implement correlation in term: replace "mean" with dedicated "corr" agg in your code
        select=JointSelectionCfg(bodies=["left_ankle", "right_ankle", "left_wrist", "right_wrist"]),
      ),
      # Upper-lower limb coordination
      FeatureTermCfg(
        name="upper_lower_coord",
        source="cvel",
        channels=["lin.speed"],
        window_size=50,
        aggregators=["mean"],  # implement cross-correlation or phase-lock value as a new agg if desired
        select=JointSelectionCfg(bodies=["left_wrist", "right_wrist", "left_ankle", "right_ankle"]),
      ),
      # Yaw-rate RMS (from shoulders/hips or pelvis angular z)
      FeatureTermCfg(
        name="yaw_rate_rms",
        source="cvel",
        channels=["ang.z"],
        window_size=30,
        aggregators=["rms"],
        select=JointSelectionCfg(bodies=["pelvis"]),
      ),
      # Dominant frequency + spectral entropy of COM vertical velocity
      FeatureTermCfg(
        name="com_vert_freq",
        source="subtree_com",
        channels=["lin.z"],  # z component of COM position
        window_size=64,
        pre_diff="velocity",  # differentiate to get velocity
        aggregators=["dominant_freq", "spectral_entropy"],
        select=JointSelectionCfg(bodies=["pelvis"]),  # or a virtual COM index
      ),
    ],
  ),
  dataset=AmpDatasetCfg(
    files=[], # Override in robot cfg.
    time_between_frames=0.05,
    preload_transitions=200_000,
    symmetry=SymmetryAugmentCfg(enabled=False),
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
    # Optional AMP features term to be defined in robot-specific cfg as a function:
    amp: ObsTerm | None = term(
      ObsTerm,
      func=amp_features_obs,
      params={
        "feature_set": AMP_CFG.feature_set,
        "sensor_names": [],  # Override in robot cfg.
      },
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)



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
  # Hook the AMP buffer reset into the reset event pipeline.
  reset_amp_obs: EventTerm | None = term(
    EventTerm,
    func=amp_features_reset, 
    mode="reset"
  )


@dataclass
class RewardCfg:
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
    weight=0.0,
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


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
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
  amp_cfg: AmpCfg = field(default_factory=lambda: AMP_CFG)

  def __post_init__(self):
    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True