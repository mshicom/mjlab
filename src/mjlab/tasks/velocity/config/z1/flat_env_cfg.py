from dataclasses import dataclass

from mjlab.tasks.velocity.config.z1.rough_env_cfg import (
  NuBotZ1RoughEnvCfg,
)


@dataclass
class NuBotZ1FlatEnvCfg(NuBotZ1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None

    self.curriculum.command_vel = None

    assert self.events.push_robot is not None
    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
    }


@dataclass
class NuBotZ1FlatEnvCfg_PLAY(NuBotZ1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
