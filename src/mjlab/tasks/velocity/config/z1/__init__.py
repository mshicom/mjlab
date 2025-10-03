import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-NuBot-Z1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:NuBotZ1RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:NuBotZ1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-NuBot-Z1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:NuBotZ1RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:NuBotZ1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-NuBot-Z1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:NuBotZ1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:NuBotZ1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-NuBot-Z1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:NuBotZ1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:NuBotZ1PPORunnerCfg",
  },
)
