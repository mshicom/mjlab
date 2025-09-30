"""Script to train RL agent with RSL-RL."""

import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import tyro

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.distributed import (
  DistributedContext,
  get_distributed_context,
  resolve_distributed_device,
  wait_for_path_update,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000


def _prepare_log_directory(
  log_root_path: Path,
  ctx: DistributedContext,
  start_time: float,
  run_name_suffix: str,
) -> Path:
  """Create or retrieve the shared logging directory for all ranks."""

  marker_path = log_root_path / ".latest_run"
  if ctx.is_main_process and marker_path.exists():
    marker_path.unlink()

  if ctx.is_main_process:
    log_dir_name = run_name_suffix
    log_dir = log_root_path / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(log_dir_name)
  else:
    wait_for_path_update(marker_path, start_time)
    log_dir_name = marker_path.read_text().strip()
    log_dir = log_root_path / log_dir_name
    wait_for_path_update(log_dir, start_time)

  return log_dir


def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  start_time = time.time()
  ctx = get_distributed_context()
  device = resolve_distributed_device(cfg.device, ctx)

  registry_name: str | None = None

  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.mkdir(parents=True, exist_ok=True)

  if isinstance(cfg.env, TrackingEnvCfg):
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    motion_marker = log_root_path / ".motion_file"
    if ctx.is_main_process and motion_marker.exists():
      motion_marker.unlink()

    if ctx.is_main_process:
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_path = Path(artifact.download()) / "motion.npz"
      cfg.env.commands.motion.motion_file = str(motion_path)
      motion_marker.write_text(motion_path.as_posix())
    else:
      wait_for_path_update(motion_marker, start_time)
      cfg.env.commands.motion.motion_file = motion_marker.read_text().strip()

  # Specify directory for logging experiments.
  run_name_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    run_name_suffix += f"_{cfg.agent.run_name}"
  log_dir = _prepare_log_directory(log_root_path, ctx, start_time, run_name_suffix)
  if ctx.is_main_process:
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

  env = gym.make(
    task, cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video and ctx.is_main_process else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  if cfg.video and ctx.is_main_process:
    video_kwargs = {
      "video_folder": os.path.join(log_dir, "videos", "train"),
      "step_trigger": lambda step: step % cfg.video_interval == 0,
      "video_length": cfg.video_length,
      "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if isinstance(cfg.env, TrackingEnvCfg):
    runner = MotionTrackingOnPolicyRunner(
      env, agent_cfg, str(log_dir), device, registry_name
    )
  else:
    runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if ctx.is_main_process:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  parsed_args = list(remaining_args)
  args = tyro.cli(
    TrainConfig,
    args=parsed_args,
    default=TrainConfig(env=env_cfg, agent=agent_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args, parsed_args

  run_train(chosen_task, args)


if __name__ == "__main__":
  main()
