import os
import numpy as np
import mujoco
import pytest
import torch
from dataclasses import dataclass

from mjlab.utils.dataset.traj_class import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from mjlab.utils.dataset.motion_dataset import MotionDataset, TrajectorySpec
from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg


def simple_obs(env):
  # Expect joint_pos present (N, J)
  return env.scene["robot"].data.joint_pos

# Group config that contains terms (required by ObservationManager._prepare_terms).
@dataclass
class BaseGroupCfg(ObservationGroupCfg):
  base_pos: ObservationTermCfg | None = None

@dataclass
class ObsCfg:
  base: BaseGroupCfg


def _make_init_env_for_om(n_envs: int = 1, n_joints: int = 1, device: str = "cpu"):
  """Build a minimal env object so ObservationManager can infer shapes at init time."""
  class _Data:
    def __init__(self):
      self.device = device
      self.joint_pos = torch.zeros(n_envs, n_joints, device=device)
      self.default_joint_pos = torch.zeros(n_envs, n_joints, device=device)
      self.projected_gravity_b = torch.tensor([[0, 0, -1]] * n_envs, dtype=torch.float32, device=device)
  class _Entity:
    def __init__(self):
      self.data = _Data()
  class _Env:
    def __init__(self):
      self.device = device
      self.num_envs = n_envs
      self.scene = {"robot": _Entity()}
      self.action_manager = type("A", (), {"action": torch.zeros(n_envs, 1, device=device)})()
      self.command_manager = type("C", (), {"get_command": lambda self, name: torch.zeros(n_envs, 1, device=device)})()
  return _Env()


def test_motion_dataset_build_minimal():
  # Build a trivial mjModel with 1 hinge joint
  mjcf = """
  <mujoco>
    <worldbody>
      <body name="root">
        <joint name="hinge" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.001" density="1000"/>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(mjcf)

  # Minimal trajectory with 10 steps
  T = 10
  data = TrajectoryData(
    qpos=np.zeros((T, 1), dtype=np.float32),
    qvel=np.zeros((T, 1), dtype=np.float32),
    split_points=np.array([0, T], dtype=np.int32)
  )
  info = TrajectoryInfo(
    joint_names=["hinge"],
    model=TrajectoryModel(njnt=1, jnt_type=np.array([mujoco.mjtJoint.mjJNT_HINGE])),
    frequency=100.0
  )
  traj = Trajectory(info=info, data=data)

  # Save to temp npz
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "traj.npz")
    traj.save(path)

    # Obs manager configured to emit joint_pos concatenated; provide a minimal env for init
    cfg = ObsCfg(
      base=BaseGroupCfg(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=False,
        base_pos=ObservationTermCfg(func=simple_obs, params={})
      )
    )
    env_init = _make_init_env_for_om(n_envs=1, n_joints=1, device="cpu")
    om = ObservationManager(cfg, env_init)
    md = MotionDataset(mj_model=model, obs_manager=om, trajectories=[TrajectorySpec(path=path, weight=2.0)], device="cpu")
    md.build()
    obs, w = md.get_replay_buffer().as_tensors()
    assert obs.shape[0] == T
    assert obs.shape[1] == 1
    assert torch.allclose(w, torch.full_like(w, 2.0))


@pytest.mark.skipif(
  not os.path.exists("/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/stepinplace1.npz"),
  reason="Real dataset npz not found on this machine."
)
def test_motion_dataset_with_real_unitree_g1_dataset():
  """
  Load a real dataset npz and run MotionDataset to precompute observations.
  This test is skipped if the dataset path does not exist locally.
  """
  real_path = "/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/stepinplace1.npz"
  from mjlab.managers.manager_term_config import term, ObservationTermCfg as ObsTerm, EventTermCfg as EventTerm
  from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
  from mjlab.tasks.velocity import mdp
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.flat_env_cfg import (
    UnitreeG1FlatEnvCfg
  )
  from mjlab.tasks.velocity.velocity_env_cfg import ObservationCfg
  from dataclasses import replace
   
  # Load trajectory (numpy backend)
  traj = Trajectory.load(real_path, backend=np)
  assert isinstance(traj, Trajectory)
  assert traj.data.qpos.shape[0] > 0, "Empty trajectory data"

  
  cfg = UnitreeG1FlatEnvCfg()
  env = ManagerBasedRlEnv(cfg, device="cpu")

  # Build MotionDataset and precompute
  md = MotionDataset(
    mj_model=env.sim.mj_model,
    obs_manager=env.observation_manager,
    trajectories=[TrajectorySpec(path=real_path, weight=1.5)],
    device="cpu",
  )
  md.build()
  obs, w = md.get_replay_buffer().as_tensors()


  # Sanity checks
  assert obs.ndim == 2 and obs.shape[0] > 0, "No observations recorded"
  assert w.shape[0] == obs.shape[0], "Weights length mismatch"
  assert torch.allclose(w, torch.full_like(w, 1.5)), "Weights not applied correctly"