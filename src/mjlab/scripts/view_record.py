#!/usr/bin/env python3
"""
Visualize an offline record (REC) attached to UnitreeG1FlatEnvCfg using mujoco.viewer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import tyro

from mjlab.scene.scene import RecordCfg
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.flat_env_cfg import UnitreeG1FlatEnvCfg


def main(
    npz: str = "/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/walk.npz",
    source_xml: Optional[Path] = "/workspaces/ws_rl/src/loco-mujoco/loco_mujoco/models/unitree_g1/g1_23dof.xml",
    speed: Optional[float] = 1.,
    device: str = "cpu"
):
   
    # Build env and load record using project env factory
    cfg = UnitreeG1FlatEnvCfg()
    cfg.scene.records=[
        RecordCfg(
            path=npz, 
            name="rec", 
            source_xml=source_xml
        )
    ]
    env = ManagerBasedRlEnv(cfg, device=device)
    rec:Entity = env.scene["rec"]
    freq = rec._rec_fps
    # Get model/data for rendering
    mj_model = env.scene.compile()
    mj_data = mujoco.MjData(mj_model)
    v = mujoco.viewer.launch_passive(mj_model, mj_data)

    # visualize_record_entity(rec, mj_model, mj_data)
    dt = 1.0 / freq
    for _ in rec.frames():
        mj_data.qpos[:] = rec.data.data.qpos
        mj_data.qvel[:] = rec.data.data.qvel
        mujoco.mj_forward(mj_model, mj_data)
        v.sync(state_only=True)
        time.sleep(dt)
    v.close()

if __name__ == "__main__":
    tyro.cli(main)