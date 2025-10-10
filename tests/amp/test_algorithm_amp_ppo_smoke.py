import math

import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")

from rsl_rl.modules import ActorCritic
from mjlab.tasks.velocity.rl.amp.algorithm import AmpPPO
from mjlab.tasks.velocity.rl.amp.config import AmpCfg, AmpDatasetCfg, AmpDiscriminatorCfg, AmpReplayBufferCfg, AmpLearningCfg
from mjlab.tasks.velocity.rl.amp.motion_loader import MotionDataset
from mjlab.tasks.velocity.rl.amp.normalizer import Normalizer


@pytest.fixture
def tiny_motion_dataset(tmp_path):
    # Build a trivial .npy dataset
    import numpy as np

    joints = ["j1", "j2"]
    T = 60
    joint_positions = [np.array([0.01 * t, -0.01 * t], dtype=np.float32) for t in range(T)]
    root_position = [np.array([0.0, 0.0, 0.0], dtype=np.float32) for _ in range(T)]
    root_quaternion = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) for _ in range(T)]
    fps = 60.0
    data = {
        "joints_list": joints,
        "joint_positions": joint_positions,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": fps,
    }
    out = tmp_path / "tiny.npy"
    np.save(str(out), data, allow_pickle=True)

    ds = MotionDataset(
        root=tmp_path,
        names=["tiny"],
        weights=[1.0],
        simulation_dt=1.0 / 60.0,
        slow_down_factor=1,
        device="cpu",
    )
    return ds


def test_amp_ppo_single_update_smoke(tiny_motion_dataset: MotionDataset):
    device = "cpu"
    amp_dim = tiny_motion_dataset.amp_dim()
    disc_input_dim = 2 * amp_dim

    # Minimal policy
    num_obs = 16
    num_critic_obs = 16
    num_actions = 4
    policy = ActorCritic(
        num_obs=num_obs,
        num_privileged_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    ).to(device)

    amp_cfg = AmpCfg(
        dataset=AmpDatasetCfg(root=tiny_motion_dataset.root, names=["tiny"], weights=[1.0], slow_down_factor=1),
        discriminator=AmpDiscriminatorCfg(
            input_dim=disc_input_dim,
            hidden_dims=[32, 16],
            reward_scale=0.1,
            loss_type="BCEWithLogits",
        ),
        replay=AmpReplayBufferCfg(size=512),
        learn=AmpLearningCfg(
            num_learning_epochs=2,
            num_mini_batches=2,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=5e-4,
            schedule="fixed",
        ),
    )
    amp = AmpPPO(
        actor_critic=policy,
        amp_cfg=amp_cfg,
        amp_dataset=tiny_motion_dataset,
        device=device,
        amp_normalizer=Normalizer(amp_dim, device=device),
    )

    num_envs = 2
    num_steps_per_env = 4
    amp.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=(num_obs,),
        critic_obs_shape=(num_critic_obs,),
        action_shape=(num_actions,),
    )

    # Fill storage by acting on random observations
    obs = torch.randn(num_envs, num_obs)
    critic_obs = torch.randn(num_envs, num_critic_obs)
    for _ in range(num_steps_per_env):
        with torch.no_grad():
            actions = amp.act(obs, critic_obs)
        rewards = torch.zeros(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        info = {}  # no bootstrapping on timeouts

        # AMP step: random AMP obs (same dim as dataset)
        amp_obs = torch.randn(num_envs, amp_dim)
        amp.act_amp(amp_obs)
        amp_next_obs = torch.randn(num_envs, amp_dim)
        amp.process_amp_step(amp_next_obs)

        amp.process_env_step(rewards, dones, info)

        # next obs/critic_obs
        obs = torch.randn_like(obs)
        critic_obs = torch.randn_like(critic_obs)

    with torch.no_grad():
        amp.compute_returns(critic_obs)

    stats = amp.update()
    # Assert stats are numeric
    for v in stats:
        assert isinstance(v, float) or (hasattr(v, "__float__"))
        assert math.isfinite(float(v))