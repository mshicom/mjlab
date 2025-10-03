import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-NuBot-Z1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Z1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:Z1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-NuBot-Z1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Z1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:Z1FlatPPORunnerCfg",
  },
)
