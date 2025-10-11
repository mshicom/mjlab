import os

import wandb
from rsl_rl.runners import OnPolicyRunner
from dataclasses import asdict
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper
  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    # copy amp_cfg from env.cfg to train_cfg to enable AMP training
    if env.cfg.amp_cfg is not None:
      train_cfg["algorithm"]["amp_cfg"] = env.cfg.amp_cfg

    super().__init__(env, train_cfg, log_dir, device)
    

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = policy_path.split("/")[-2] + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
