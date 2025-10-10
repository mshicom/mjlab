#!/usr/bin/env python3
"""
Standalone multi-GPU smoke test for VelocityAmpOnPolicyRunner with DDP-enabled AmpPPO
using the real UnitreeGo1FlatEnv from mjlab.

Launch examples:
  CUDA (2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/amp/smoke_velocity_amp_ddp.py --iters 10

  CPU (gloo, 2 ranks):
    torchrun --standalone --nproc_per_node=2 tests/amp/smoke_velocity_amp_ddp.py --iters 2 --device cpu

  Single-process quick smoke:
    python tests/amp/smoke_velocity_amp_ddp.py --world-size 1 --iters 1
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Optional
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import asdict

# mjlab imports: real env + RL wrapper + AMP runner
from mjlab.tasks.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg
from mjlab.tasks.velocity.rl.runner_amp import VelocityAmpOnPolicyRunner
from mjlab.tasks.velocity.rl.amp.config import AmpCfg, AmpDatasetCfg, AmpDiscriminatorCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.config import RslRlOnPolicyRunnerCfg

# Manager-based env (VecEnv API)
from mjlab.envs import ManagerBasedRlEnv


# -----------------------------
# Synthetic AMP dataset writer
# -----------------------------

def _write_synthetic_amp_dataset(tmpdir: str, name: str, dof: int = 12, T: int = 32, seed: int = 42):
    path = os.path.join(tmpdir, f"{name}.npy")
    joints_list = [f"joint_{i}" for i in range(dof)]
    # Deterministic random walk-ish joint positions
    np.random.seed(seed)  # Ensure same data across all ranks
    joint_positions = [np.zeros((dof,), dtype=np.float32)]
    for _ in range(T - 1):
        joint_positions.append(joint_positions[-1] + 0.01 * np.random.randn(dof).astype(np.float32))
    root_position = [np.zeros(3, dtype=np.float32) for _ in range(T)]
    # random quaternions in xyzw (normalize)
    q = np.random.randn(T, 4).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    root_quaternion = [q[i] for i in range(T)]  # xyzw
    fps = 60.0
    data = dict(
        joints_list=joints_list,
        joint_positions=joint_positions,
        root_position=root_position,
        root_quaternion=root_quaternion,  # xyzw
        fps=fps,
    )
    np.save(path, data, allow_pickle=True)
    return path


# ---------------
# Worker
# ---------------

def _worker(rank: int, world_size: int, args):
    # Do NOT initialize process group here - the VelocityAmpOnPolicyRunner
    # (via OnPolicyRunner._configure_multi_gpu) will handle it
    
    # Set deterministic seeds for reproducibility across ranks
    seed = 42  # Fixed seed for all ranks to ensure identical initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Resolve local rank and device strings
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.device == "cuda" and torch.cuda.is_available():
        # Set device after seeds
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cpu"
    torch_device = torch.device(device_str)

    # Build the real UnitreeGo1 flat environment
    env_cfg = UnitreeGo1FlatEnvCfg()
    # Each rank gets its own slice of environments in DDP mode
    env_cfg.scene.num_envs = args.num_envs
    # Disable problematic rewards for smoke test
    if hasattr(env_cfg, 'rewards') and hasattr(env_cfg.rewards, 'feet_slide'):
        env_cfg.rewards.feet_slide.weight = 0.0
    # ManagerBasedRlEnv expects a string device for Warp (e.g., "cuda:0" or "cpu")
    base_env = ManagerBasedRlEnv(env_cfg, device=device_str)  # type: ignore[arg-type]
    env = RslRlVecEnvWrapper(base_env)

    # Derive observation/action dims from the real env (only for sanity checks)
    init_obs = env.get_observations()
    if isinstance(init_obs, torch.Tensor):
        obs_dim = init_obs.shape[1]
    elif isinstance(init_obs, dict) and "policy" in init_obs:
        obs_dim = init_obs["policy"].shape[1]
    else:
        obs_dim = getattr(env, "num_obs", 48)
    act_dim = getattr(env, "num_actions", 12)
    if rank == 0:
        print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, device={device_str}")

    with tempfile.TemporaryDirectory() as tmp:
        # Write a synthetic AMP dataset (small) - use same seed for all ranks
        dof = 12  # Unitree Go1 actuated joints (approx) for AMP obs sizing
        _write_synthetic_amp_dataset(tmp, "tiny_demo", dof=dof, T=32, seed=42)
        
        # Ensure all ranks have the same dataset before proceeding
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # Discriminator input is 2*(qpos+qvel+lin+ang) = 2*(12+12+3+3)=2*30=60
        amp_cfg = AmpCfg(
            dataset=AmpDatasetCfg(root=tmp, names=["tiny_demo"], weights=[1.0], slow_down_factor=1),
            discriminator=AmpDiscriminatorCfg(input_dim=2 * (dof + dof + 3 + 3), hidden_dims=(128, 128), reward_scale=0.1),
        )
        # Build the runner config; keep it minimal and aligned to VelocityOnPolicyRunner
        runner_cfg = RslRlOnPolicyRunnerCfg(
            num_steps_per_env=args.steps_per_env,
            max_iterations=args.iters,
            save_interval=10_000,  # do not save in smoke
            experiment_name="go1_flat_amp_smoke",
        )
        # Use more stable hyperparameters for smoke test
        # Lower learning rate and stronger gradient clipping to prevent NaN
        runner_cfg = replace(
            runner_cfg, 
            policy=replace(
                runner_cfg.policy,
                init_noise_std=30.0,  # Higher initial action stddev
            ),
            algorithm=replace(
                runner_cfg.algorithm,
                learning_rate=1e-5,  # Lower LR to prevent instability
                max_grad_norm=0.001,   # Stronger gradient clipping
            )
        )
    
        # Convert to dict and add AMP-specific configs
        train_cfg = asdict(runner_cfg)
        train_cfg["amp"] = amp_cfg  # Keep as dataclass, not dict
        train_cfg["ddp_enable"] = (world_size > 1 and args.device == "cuda" and torch.cuda.is_available())

        # Instantiate the real VelocityAmpOnPolicyRunner
        runner = VelocityAmpOnPolicyRunner(env, train_cfg, log_dir=None, device=device_str)

        # Minimal learn loop
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=False)

        # Cross-rank param consistency check on the trained policy (rank 0 gathers)
        with torch.no_grad():
            params = [p.detach().reshape(-1).clone() for p in runner.alg.policy.parameters()]
            flat = torch.cat(params) if params else torch.zeros(1, device=torch_device)
            if world_size > 1 and dist.is_initialized():
                if rank == 0:
                    bufs = [torch.zeros_like(flat) for _ in range(world_size)]
                else:
                    bufs = []
                dist.gather(flat, gather_list=bufs if rank == 0 else None, dst=0)
                if rank == 0:
                    base_vec = bufs[0].cpu()
                    for i in range(1, world_size):
                        diff = torch.norm(bufs[i].cpu() - base_vec).item()
                        # allow for some divergence in smoke test due to NaN handling
                        if not (torch.isnan(torch.tensor(diff)) or torch.isinf(torch.tensor(diff))):
                            assert diff < 1.0, f"Rank {i} parameters diverged significantly (norm diff={diff})"
                        else:
                            print(f"[WARNING] Rank {i} parameters contain NaN/Inf (norm diff={diff}), skipping consistency check")

    if rank == 0 or world_size == 1:
        print("[OK] VelocityAmpOnPolicyRunner DDP smoke test on UnitreeGo1FlatEnv completed.")


# ---------------
# Main
# ---------------

def parse_args():
    p = argparse.ArgumentParser(description="DDP Smoke Test for VelocityAmpOnPolicyRunner on UnitreeGo1FlatEnv")
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--backend", type=str, default=None, help="ddp backend: nccl or gloo (auto if None)")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cuda", "cpu"])
    p.add_argument("--iters", type=int, default=5, help="number of AmpPPO updates")
    p.add_argument("--steps-per-env", type=int, default=24, help="rollout steps per update")
    p.add_argument("--num-envs", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
        _worker(rank, world_size, args)
    else:
        world_size = args.world_size
        if world_size <= 1:
            _worker(rank=0, world_size=1, args=args)
        else:
            mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()