import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import pytest

from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.observation_manager import ObservationManager
from mjlab.utils.dataset.motion_dataset import (
    MotionDataset,
    TrajectorySpec,
    prepare_amp_demo,
)
from mjlab.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)


# -----------------------
# Helpers (shared across tests)
# -----------------------


def simple_obs(env, **kwargs):
    # Expect joint_pos present (N, J)
    return env.scene["robot"].data.joint_pos


# Group config that contains terms (required by ObservationManager._prepare_terms).
@dataclass
class BaseGroupCfg(ObservationGroupCfg):
    base_pos: ObservationTermCfg | None = None


@dataclass
class BaseObsCfg:
    base: BaseGroupCfg


@dataclass
class AmpGroupCfg(ObservationGroupCfg):
    amp_term: ObservationTermCfg | None = None


@dataclass
class AmpObsCfg:
    # expose an 'amp_state' group to match default AMP obs key
    amp_state: AmpGroupCfg


def _make_init_env_for_om(n_envs: int = 1, n_joints: int = 1, device: str = "cpu"):
    """Build a minimal env object so ObservationManager can infer shapes at init time."""

    class _Data:
        def __init__(self):
            self.device = device
            self.joint_pos = torch.zeros(n_envs, n_joints, device=device)
            self.default_joint_pos = torch.zeros(n_envs, n_joints, device=device)
            self.projected_gravity_b = torch.tensor(
                [[0, 0, -1]] * n_envs, dtype=torch.float32, device=device
            )

    class _Entity:
        def __init__(self):
            self.data = _Data()

    class _Env:
        def __init__(self):
            self.device = device
            self.num_envs = n_envs
            self.scene = {"robot": _Entity()}
            self.action_manager = type(
                "A", (), {"action": torch.zeros(n_envs, 1, device=device)}
            )()
            self.command_manager = type(
                "C", (), {"get_command": lambda self, name: torch.zeros(n_envs, 1, device=device)},
            )()

    return _Env()


def _make_mjmodel(n_joints: int = 1):
    assert n_joints >= 1, "n_joints must be >= 1"
    body_xml = []
    for i in range(n_joints):
        body_xml.append(
            f"""
          <body name="link_{i}">
            <joint name="hinge_{i}" type="hinge" axis="1 0 0"/>
            <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0 0.1 0"/>
          </body>"""
        )
    bodies_str = "\n".join(body_xml)
    mjcf = f"""
    <mujoco>
      <worldbody>
        <body name="pelvis">
          <freejoint name="root"/>
          <geom type="sphere" size="0.001" density="1000"/>
{bodies_str}
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(mjcf)


def _make_env_for_prepare(mj_model):
    """Minimal env exposing sim.mj_model, observation_manager and device (for prepare_amp_demo)."""

    class _Data:
        def __init__(self, device):
            self.device = device
            self.joint_pos = torch.zeros(1, 1, device=device)
            self.default_joint_pos = torch.zeros(1, 1, device=device)
            self.projected_gravity_b = torch.tensor(
                [[0, 0, -1]], dtype=torch.float32, device=device
            )

    class _Entity:
        def __init__(self, device):
            self.data = _Data(device)

    class _Sim:
        def __init__(self, mjm):
            self.mj_model = mjm

    class _Env:
        def __init__(self, mjm, device="cpu"):
            self.device = device
            self.sim = _Sim(mjm)
            self.scene = {"robot": _Entity(device)}
            # minimal managers
            self.action_manager = type(
                "A", (), {"action": torch.zeros(1, 1, device=device)}
            )()
            self.command_manager = type(
                "C", (), {"get_command": lambda self, name: torch.zeros(1, 1, device=device)},
            )()
            # set later
            self.observation_manager = None

    env = _Env(mj_model, device="cpu")
    # Build ObservationManager with group 'amp_state'
    cfg = AmpObsCfg(
        amp_state=AmpGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            amp_term=ObservationTermCfg(func=simple_obs, params={}),
        )
    )
    env.observation_manager = ObservationManager(cfg, env)
    return env


# -----------------------
# Tests
# -----------------------


def test_motion_dataset_build_and_sample_minimal(tmp_path):
    n_joints = 3
    model = _make_mjmodel(n_joints)

    # Minimal trajectory with 10 steps
    T = 10
    data = TrajectoryData(
        qpos=np.zeros((T, n_joints), dtype=np.float32),
        qvel=np.zeros((T, n_joints), dtype=np.float32),
        split_points=np.array([0, T], dtype=np.int32),
    )
    info = TrajectoryInfo(
        joint_names=[f"hinge_{i}" for i in range(n_joints)],
        model=TrajectoryModel(
            njnt=n_joints, jnt_type=np.array([3]*n_joints)
        ),
        frequency=100.0,
    )
    traj = Trajectory(info=info, data=data)

    # Save to temp npz
    path = os.path.join(tmp_path, "traj.npz")
    traj.save(path)

    # Obs manager configured to emit joint_pos concatenated; provide a minimal env for init
    cfg = BaseObsCfg(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(func=simple_obs, params={}),
        )
    )
    env_init = _make_init_env_for_om(n_envs=1, n_joints=n_joints, device="cpu")
    om = ObservationManager(cfg, env_init)

    # MotionDataset configured to use group "base"
    md = MotionDataset(
        mj_model=model,
        obs_manager=om,
        trajectories=[TrajectorySpec(path=path)],
        group_name="base",
        device="cpu",
    )
    # Use two-step API
    md.load()
    md.compute(force_recompute=True)

    # Basic introspection
    assert md.num_trajs() == 1
    assert md.num_samples() == T
    assert md.get_feature_dim() == n_joints

    # Sampling returns [N, F]
    samples = md.sample(5, device="cpu")
    assert samples.shape == (5, n_joints)
    # All zeros since trajectory qpos are zeros
    assert torch.allclose(samples, torch.zeros_like(samples))


def test_toggle_and_uniform_sampling(tmp_path):
    model = _make_mjmodel()

    # Create two trajectories with distinct constant qpos values: 0 and 1
    T = 20
    trajs = []
    for val, name in [(0.0, "zeros"), (1.0, "ones")]:
        data = TrajectoryData(
            qpos=np.full((T, 1), val, dtype=np.float32),
            qvel=np.zeros((T, 1), dtype=np.float32),
            split_points=np.array([0, T], dtype=np.int32),
        )
        info = TrajectoryInfo(
            joint_names=["hinge_0"],
            model=TrajectoryModel(
                njnt=1, jnt_type=np.array([3])
            ),
            frequency=100.0,
        )
        traj = Trajectory(info=info, data=data)
        p = os.path.join(tmp_path, f"{name}.npz")
        traj.save(p)
        trajs.append(TrajectorySpec(path=p, name=name))

    cfg = BaseObsCfg(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(func=simple_obs, params={}),
        )
    )
    om = ObservationManager(cfg, _make_init_env_for_om())

    md = MotionDataset(
        mj_model=model,
        obs_manager=om,
        trajectories=trajs,
        group_name="base",
        device="cpu",
    )
    md.build(force_recompute=True)

    # With equal frame counts, global uniform sampling across all frames -> ~50/50 zeros/ones
    N = 2000
    s = md.sample(N)
    mean_val = s.mean().item()
    assert 0.4 <= mean_val <= 0.6, f"Expected ~0.5 mix; got mean {mean_val:.3f}"

    # Disable 'ones' trajectory: all samples should be zeros
    md.enable("ones", on=False)
    s2 = md.sample(200)
    assert torch.allclose(
        s2, torch.zeros_like(s2)
    ), "Disabled trajectory still influencing samples."

    # Enable 'ones', disable 'zeros': all ones
    md.enable("ones", on=True)
    md.enable("zeros", on=False)
    s3 = md.sample(200)
    assert torch.allclose(
        s3, torch.ones_like(s3)
    ), "Disabled trajectory still influencing samples."


def test_cache_and_force_recompute(tmp_path):
    model = _make_mjmodel()
    T = 12
    data = TrajectoryData(
        qpos=np.arange(T, dtype=np.float32).reshape(T, 1),
        qvel=np.zeros((T, 1), dtype=np.float32),
        split_points=np.array([0, T], dtype=np.int32),
    )
    info = TrajectoryInfo(joint_names=["hinge_0"], model=TrajectoryModel(njnt=1,jnt_type=np.array([3])), frequency=100.0)
    traj = Trajectory(info=info, data=data)
    path = os.path.join(tmp_path, "lin.npz")
    traj.save(path)

    # Build ObservationManager
    cfg = BaseObsCfg(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(func=simple_obs, params={}),
        )
    )
    om = ObservationManager(cfg, _make_init_env_for_om())

    # First build: force recompute and create cache
    md = MotionDataset(
        mj_model=model,
        obs_manager=om,
        trajectories=[TrajectorySpec(path=path)],
        group_name="base",
        device="cpu",
    )
    md.build(force_recompute=True)
    # Verify cache exists
    cache_path = md._cache_path(path)  # type: ignore (internal method acceptable for test)
    assert cache_path.exists(), "Cache file not created."

    # Second build: ensure we can load from cache without calling ObservationManager.compute()
    class _BadOM:
        def compute(self):
            raise RuntimeError("Should not be called when loading from cache")

        def reset(self):
            pass

    md2 = MotionDataset(
        mj_model=model,
        obs_manager=_BadOM(),
        trajectories=[TrajectorySpec(path=path)],
        group_name="base",
        device="cpu",
    )
    md2.load()
    md2.compute(force_recompute=False)  # Should succeed via cache
    assert md2.num_samples() == T

    # Third build: force_recompute=True should recompute (and thus call compute); restore real OM
    md3 = MotionDataset(
        mj_model=model,
        obs_manager=om,
        trajectories=[TrajectorySpec(path=path)],
        group_name="base",
        device="cpu",
    )
    md3.build(force_recompute=True)
    assert md3.num_samples() == T


def test_prepare_amp_demo_attach_and_sample(tmp_path):
    mj_model = _make_mjmodel()
    env = _make_env_for_prepare(mj_model)

    # Create a tiny trajectory
    T = 6
    data = TrajectoryData(
        qpos=np.ones((T, 1), dtype=np.float32),  # ones to be identifiable
        qvel=np.zeros((T, 1), dtype=np.float32),
        split_points=np.array([0, T], dtype=np.int32),
    )
    info = TrajectoryInfo(joint_names=["hinge_0"], model=TrajectoryModel(njnt=1, jnt_type=np.array([3])), frequency=100.0)
    traj = Trajectory(info=info, data=data)
    path = os.path.join(tmp_path, "amp_demo.npz")
    traj.save(path)

    # prepare_amp_demo will build dataset, cache, and attach env.sample_amp_demos
    ds = prepare_amp_demo(
        env=env,
        trajectories=[TrajectorySpec(path=path, name="demo")],
        group_name="amp_state",
        force_recompute=True,
    )
    assert hasattr(
        env, "sample_amp_demos"
    ), "prepare_amp_demo did not attach sample_amp_demos."

    # sample and check shapes/values
    smp = env.sample_amp_demos(8, device="cpu")
    assert smp.shape == (8, 1)
    # since qpos were ones, all samples should be ones
    assert torch.allclose(smp, torch.ones_like(smp))
    

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
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.g1.flat_env_cfg import (
        UnitreeG1FlatEnvCfg
    )
    
    # Load trajectory (numpy backend)
    traj = Trajectory.load(real_path, backend=np)
    assert isinstance(traj, Trajectory)
    assert traj.data.qpos.shape[0] > 0, "Empty trajectory data"
    
    cfg = UnitreeG1FlatEnvCfg(episode_length_s=1e9)
    env = ManagerBasedRlEnv(cfg, device="cpu")

    # Build MotionDataset and precompute
    md = MotionDataset(
        env=env,
        trajectories=[TrajectorySpec(path=real_path)],
        device="cpu",
        auto_extend_incomplete=True,
    )
    md.build(force_recompute=True)
    
    # Basic introspection
    assert md.num_trajs() == 1
    assert md.num_samples() == traj.data.qpos.shape[0]
    assert md.get_feature_dim() > 0

    # Sampling returns [N, F]
    samples = md.sample(5, device="cpu")
    assert samples.shape == (5, md.get_feature_dim())