from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.nubot_z1.z1_constants import (
  Z1_ACTION_SCALE,
  Z1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class NuBotZ1RoughEnvCfg(LocomotionVelocityEnvCfg):
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
    z1_cfg = replace(Z1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": z1_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    geom_names = []
    for i in range(1, 2):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 2):
      geom_names.append(f"right_foot{i}_collision")

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.actions.joint_pos.scale = Z1_ACTION_SCALE

    self.rewards.feet_slide.params["sensor_names"] = sensor_names
    self.rewards.feet_slide.params["asset_cfg"].geom_names = geom_names
    
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
      # r".*waist_yaw.*": 0.15,
      # r".*waist_roll.*": 0.08,
      # r".*waist_pitch.*": 0.1,
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



@dataclass
class NuBotZ1RoughEnvCfg_PLAY(NuBotZ1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 100.0
