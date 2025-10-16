import os

import wandb
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)
from mjlab.utils.dataset.motion_dataset import prepare_amp_demo


class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper
  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    
    # TODO: make prepare_amp_demo and env.unwrapped.sample_amp_demos stuff in a env wrapper
    # Prepare AMP demo dataset (if configured) before constructing OnPolicyRunner.
    # This ensures env.sample_amp_demos exists when rsl_rl resolves AMP config.
    amp_ds_cfg = getattr(env.cfg, "amp_dataset", None)
    if amp_ds_cfg is not None and amp_ds_cfg.enabled:
      # build or reuse cached per-trajectory features and attach env.unwrapped.sample_amp_demos
      _ = prepare_amp_demo(
        env=env.unwrapped,  # use underlying mjlab env to access sim + observation_manager
        trajectories=amp_ds_cfg.trajectories,
        group_name=amp_ds_cfg.group_name,
        subsample_stride=amp_ds_cfg.subsample_stride,
        max_frames_per_traj=amp_ds_cfg.max_frames_per_traj,
        seed=amp_ds_cfg.seed
      )
      train_cfg["algorithm"]["amp_cfg"]['enabled'] = True

    # Proceed with standard rsl_rl on-policy runner initialization
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