import os
from pathlib import Path

import numpy as np
import pytest
import torch

scipy = pytest.importorskip("scipy")

from mjlab.tasks.velocity.rl.amp.motion_loader import MotionDataset


def _write_simple_npy_dataset(tmp_path: Path, name: str, T: int = 100):
    # Minimal dataset with 2 joints, linear pos, identity quats
    joints = ["j1", "j2"]
    joint_positions = [np.array([np.sin(0.1 * t), np.cos(0.1 * t)], dtype=np.float32) for t in range(T)]
    root_position = [np.array([0.01 * t, 0.0, 0.0], dtype=np.float32) for t in range(T)]
    root_quaternion = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) for _ in range(T)]  # xyzw (identity)
    fps = 50.0

    data = {
        "joints_list": joints,
        "joint_positions": joint_positions,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": fps,
    }
    out = tmp_path / f"{name}.npy"
    np.save(str(out), data, allow_pickle=True)
    return out


def test_motion_dataset_load_and_iter(tmp_path):
    name = "demo"
    _write_simple_npy_dataset(tmp_path, name, T=120)

    sim_dt = 1.0 / 50.0  # 50 Hz
    ds = MotionDataset(
        root=tmp_path,
        names=[name],
        weights=[1.0],
        simulation_dt=sim_dt,
        slow_down_factor=1,
        device="cpu",
        expected_joint_names=None,
    )

    # amp_dim = qpos(2) + qvel(2) + lin_vel_b(3) + ang_vel_b(3) = 10
    assert ds.amp_dim() == 10

    gen = ds.feed_forward_generator(num_mini_batch=3, mini_batch_size=8)
    batches = list(gen)
    assert len(batches) == 3
    for s, ns in batches:
        assert s.shape == (8, 10)
        assert ns.shape == (8, 10)
        assert torch.isfinite(s).all()
        assert torch.isfinite(ns).all()