from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg

# NEW: AMP features
from mjlab.managers.manager_term_config import term, ObservationTermCfg as ObsTerm
from mjlab.amp.obs_terms import AmpFeatureObs
from mjlab.amp.config import AmpFeatureSetCfg, FeatureTermCfg


@dataclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": g1_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names
    self.actions.joint_pos.scale = G1_ACTION_SCALE
    self.rewards.air_time.params["sensor_names"] = sensor_names

    self.rewards.pose.params["std"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.35,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.25,
      r".*wrist.*": 0.3,
    }

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 0.75

    self.curriculum.command_vel = None

    # NEW: enable AMP observation term with a minimal feature set
    self.observations.policy.amp = term(
      ObsTerm,
      func=AmpFeatureObs,
      params={
        "feature_set": AmpFeatureSetCfg(
          terms=[
            # Joint velocity RMS over a short window
            FeatureTermCfg(
              name="joint_speed_rms",
              source="qvel",
              channels=["scalar"],
              window_size=30,
              pre_diff="none",
              aggregators=["rms"],
            ),
            # Yaw-rate RMS from base angular velocity
            FeatureTermCfg(
              name="yaw_rate_rms",
              source="base_ang",
              channels=["ang.z"],
              window_size=30,
              aggregators=["rms"],
            ),
            # Foot-ground duty (L/R)
            FeatureTermCfg(
              name="duty_left_right",
              source="contacts",
              channels=["binary"],
              window_size=50,
              aggregators=["mean"],
            ),
          ]
        ),
        "sensor_names": sensor_names,
      },
    )


@dataclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0