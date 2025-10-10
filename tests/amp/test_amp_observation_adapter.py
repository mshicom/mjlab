import torch

from mjlab.tasks.velocity.rl.amp.observation import build_amp_obs_from_env


class _MockRobotData:
    def __init__(self, B=4, dof=6):
        self.joint_pos = torch.randn(B, dof)
        self.joint_vel = torch.randn(B, dof)
        self.root_link_lin_vel_b = torch.randn(B, 3)
        self.root_link_ang_vel_b = torch.randn(B, 3)


class _MockEntity:
    def __init__(self):
        self.data = _MockRobotData()


class _MockEnv:
    def __init__(self):
        self.scene = {"robot": _MockEntity()}


def test_amp_observation_adapter_shapes():
    env = _MockEnv()
    out = build_amp_obs_from_env(env)
    # dof=6 -> qpos 6 + qvel 6 + blv 3 + bav 3 = 18
    assert out.shape[1] == 18
    assert torch.isfinite(out).all()