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
  run_name_suffix: str,
  should_resume: bool,
  load_run_pattern: str,
  load_checkpoint_pattern: str,
) -> tuple[Path, Path | None]:
  """Create or retrieve the shared logging directory for all ranks.

  When resuming training, the resolved checkpoint path must remain identical
  across every process. The main rank resolves the checkpoint once and writes
  the selected run/checkpoint information to synchronisation markers that the
  remaining ranks wait on. This avoids races where a freshly created log
  directory would otherwise change the regex resolution order for
  ``get_checkpoint_path``.
  """

  marker_path = log_root_path / ".latest_run"
  resume_marker_path = log_root_path / ".latest_checkpoint"
  resume_path: Path | None = None

  if ctx.is_main_process:
    if marker_path.exists():
      marker_path.unlink()
    if resume_marker_path.exists():
      resume_marker_path.unlink()

    if should_resume:
      resume_path = get_checkpoint_path(
        log_root_path,
        load_run_pattern,
        load_checkpoint_pattern,
      )
      log_dir = resume_path.parent
      if not log_dir.exists():
        raise FileNotFoundError(
          f"Resolved resume directory '{log_dir}' does not exist for rank {ctx.global_rank}."
        )
      marker_path.write_text(log_dir.name)
      resume_marker_path.write_text(resume_path.as_posix())
    else:
      log_dir_name = run_name_suffix
      log_dir = log_root_path / log_dir_name
      log_dir.mkdir(parents=True, exist_ok=True)
      marker_path.write_text(log_dir_name)
    time.sleep(0.1)
  else:
    print(f"[INFO] Rank {ctx.global_rank} waiting for log directory...")
    wait_for_path_update(marker_path, timeout=120)
    log_dir_name = marker_path.read_text().strip()
    log_dir = log_root_path / log_dir_name
    if should_resume:
      wait_for_path_update(resume_marker_path, timeout=120)
      resume_path = Path(resume_marker_path.read_text().strip())
      if not log_dir.exists():
        raise FileNotFoundError(
          f"Resolved resume directory '{log_dir}' does not exist for rank {ctx.global_rank}."
        )
    else:
      wait_for_path_update(log_dir, timeout=120)

  if should_resume and resume_path is None:
    raise RuntimeError(
      "Failed to resolve a checkpoint path for resuming training; "
      f"rank {ctx.global_rank} did not receive synchronised metadata."
    )

  return log_dir, resume_path




def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

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
      wait_for_path_update(motion_marker, timeout=120)
      cfg.env.commands.motion.motion_file = motion_marker.read_text().strip()

  run_name_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    run_name_suffix += f"_{cfg.agent.run_name}"
  log_dir, resume_path = _prepare_log_directory(
    log_root_path,
    ctx,
    run_name_suffix,
    cfg.agent.resume,
    cfg.agent.load_run,
    cfg.agent.load_checkpoint,
  )
  if ctx.is_main_process:
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

  env = gym.make(
    task, cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video and ctx.is_main_process else None
  )

  if cfg.video and ctx.is_main_process:
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=os.path.join(log_dir, "videos", "train"),
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

    print("[INFO] Recording videos during training.")

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
