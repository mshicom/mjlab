import math
from dataclasses import dataclass

import torch

from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg

# ----------------------
# Mocks and helpers
# ----------------------

class _MockEntityData:
    def __init__(self, device: str, num_envs: int, num_joints: int):
        self.device = device
        self.joint_pos = torch.zeros(num_envs, num_joints, device=device, dtype=torch.float32)
        self.default_joint_pos = torch.zeros(num_envs, num_joints, device=device, dtype=torch.float32)
        self.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32).repeat(num_envs, 1)

class _MockEnv:
    def __init__(self, device: str = "cpu", num_envs: int = 3, num_joints: int = 4):
        self.device = device
        self.num_envs = num_envs
        self.scene = {"robot": type("E", (), {"data": _MockEntityData(device, num_envs, num_joints)})()}
        self.action_manager = type("A", (), {"action": torch.zeros(num_envs, 1, device=device)})()
        self.command_manager = type("C", (), {"get_command": lambda self, name: torch.zeros(num_envs, 1, device=device)})()

def obs_joint_pos(env):
    # Returns (N, J) tensor
    return env.scene["robot"].data.joint_pos

# Hist func helpers (read params from term_cfg.params)
def hist_value(env, hist, window: int = 5, dt: float = 0.05, poly_degree: int = 2, **kwargs):
    # Endpoint SG smoothing (value at latest sample)
    return hist.get_hist_data_smooth_and_diffed(window_size=window, diff_order=0, diff_dt=dt, poly_degree=poly_degree)

def hist_vel(env, hist, window: int = 9, dt: float = 0.05, poly_degree: int = 2, **kwargs):
    # Endpoint SG 1st derivative
    return hist.get_hist_data_smooth_and_diffed(window_size=window, diff_order=1, diff_dt=dt, poly_degree=poly_degree)

def hist_acc(env, hist, window: int = 9, dt: float = 0.05, poly_degree: int = 2, **kwargs):
    # Endpoint SG 2nd derivative
    return hist.get_hist_data_smooth_and_diffed(window_size=window, diff_order=2, diff_dt=dt, poly_degree=poly_degree)

@dataclass
class BaseGroupCfg(ObservationGroupCfg):
    base_pos: ObservationTermCfg | None = None
    base_vel: ObservationTermCfg | None = None


@dataclass
class ObsCfgOne:
    base: BaseGroupCfg


# ----------------------
# Tests
# ----------------------


def test_partial_reset_replicate_padding():
  from mjlab.utils.sliding_window import SlidingWindow

  N, C, W = 4, 3, 5
  sw = SlidingWindow(num_envs=N, feature_shape=torch.Size([C]), max_window_size=W, device=torch.device("cpu"))

  # Push 3 steps with distinct values per env/channel
  for t in range(3):
    x = torch.full((N, C), float(t + 1))  # t=0 -> 1s, t=1 -> 2s, t=2 -> 3s
    sw.push(x)

  # Partially reset envs 1 and 3 (0-based)
  sw.reset(env_ids=torch.tensor([1, 3]))

  # Next push (first actual frame after reset)
  x4 = torch.full((N, C), 10.0)
  sw.push(x4)

  # Request T=4 history and check replicate padding for reset envs
  seq = sw.get_hist_data(4)  # (T=4, N, C)

  # For env 1 and 3: valid_len=1 -> pad_len = 4 - 1 = 3, first actual frame is value 10
  # So first 3 rows should be 10, last row is 10
  assert torch.allclose(seq[:3, 1, :], torch.full((3, C), 10.0))
  assert torch.allclose(seq[:3, 3, :], torch.full((3, C), 10.0))

  # For env 0 and 2: valid_len=4 -> pad_len=0, chronological values should reflect pushes [1,2,3,10]
  assert torch.allclose(seq[:, 0, 0], torch.tensor([1.0, 2.0, 3.0, 10.0]))
  assert torch.allclose(seq[:, 2, 0], torch.tensor([1.0, 2.0, 3.0, 10.0]))

  # SG deriv1 over T=4 should be finite and not crash
  v = sw.get_hist_data_smooth_and_diffed(window_size=4, diff_order=1, diff_dt=0.1, poly_degree=2)
  assert v.shape == (N, C)
  
  
def test_no_window_passthrough_and_get_term_hist_none():
    # hist_window_size = 0 -> no sliding window overhead, passthrough raw obs
    env = _MockEnv(num_envs=2, num_joints=3)
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={},  # no hist params
                hist_window_size=0,  # disabled
                hist_func=None,
            )
        )
    )
    om = ObservationManager(cfg, env)

    # Feed a sample
    env.scene["robot"].data.joint_pos[:] = torch.tensor([[1.0, 2.0, 3.0],
                                                         [4.0, 5.0, 6.0]], device=env.device)
    out = om.compute()
    assert "base" in out
    assert out["base"].shape == (2, 3)
    # Values equal raw
    assert torch.allclose(out["base"], env.scene["robot"].data.joint_pos)

    # get_term works
    term = om.get_term("base_pos", group="base")
    assert term.shape == (2, 3)
    assert torch.allclose(term, env.scene["robot"].data.joint_pos)

    # No sliding window instance
    hist = om.get_term_hist("base_pos", group="base")
    assert hist is None


def test_hist_value_smoothing_and_hist_shapes():
    # Sliding window enabled, hist_func returns SG-smoothed value
    N, J = 2, 4
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.05
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": 5, "dt": dt, "poly_degree": 2},
                hist_window_size=5,
                hist_func=hist_value,
            )
        )
    )
    om = ObservationManager(cfg, env)

    # Create a ramp in time for joint_pos: x(t) = b + s * t
    s = 0.3  # slope
    b = 1.0
    T = 7
    for t in range(T):
        env.scene["robot"].data.joint_pos[:] = b + s * t
        out = om.compute()

    # get_term returns latest smoothed, which should be close to latest value for a linear signal with deg>=1
    latest = b + s * (T - 1)
    term = om.get_term("base_pos", group="base")
    assert term.shape == (N, J)
    assert torch.allclose(term, torch.full((N, J), latest, device=env.device, dtype=torch.float32), atol=1e-4)

    # get_term_hist returns a SlidingWindow and its get_hist_data returns (T_hist, N, J)
    hist = om.get_term_hist("base_pos", group="base")
    assert hist is not None
    seq = hist.get_hist_data(window_size=5)
    assert seq.shape == (5, N, J)
    # Check chronological ordering matches the last 5 values of the ramp
    expected_tail = torch.stack([torch.full((N, J), b + s * t, device=env.device) for t in range(T - 5, T)], dim=0)
    assert torch.allclose(seq, expected_tail, atol=1e-6)


def test_hist_derivative_linear_exact():
    # For a linear signal and poly_degree>=1, endpoint SG derivative should equal the true slope
    N, J = 3, 4
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.02
    s = -0.7  # slope
    b = 2.0
    window = 9
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": window, "dt": dt, "poly_degree": 1},  # deg 1 sufficient for linear
                hist_window_size=window,
                hist_func=hist_vel,  # diff_order=1
            )
        )
    )
    om = ObservationManager(cfg, env)

    T = 12
    for t in range(T):
        env.scene["robot"].data.joint_pos[:] = b + s * t * dt  # continuous-time linear in t*dt
        out = om.compute()

    # Expected derivative: s (units per second)
    term = om.get_term("base_pos", group="base")
    assert term.shape == (N, J)
    assert torch.allclose(term, torch.full((N, J), s, device=env.device), atol=1e-4)


def test_hist_second_derivative_quadratic_exact():
    # For a quadratic signal and poly_degree>=2, endpoint SG 2nd derivative should equal true constant accel
    N, J = 2, 3
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.01
    a = 1.75  # constant accel
    v0 = -0.2
    x0 = 0.5
    window = 11
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": window, "dt": dt, "poly_degree": 2},
                hist_window_size=window,
                hist_func=hist_acc,  # diff_order=2
            )
        )
    )
    om = ObservationManager(cfg, env)

    T = 20
    for k in range(T):
        t = k * dt
        env.scene["robot"].data.joint_pos[:] = x0 + v0 * t + 0.5 * a * t * t
        out = om.compute()

    term = om.get_term("base_pos", group="base")
    assert term.shape == (N, J)
    assert torch.allclose(term, torch.full((N, J), a, device=env.device), atol=5e-3)


def test_partial_reset_replicate_padding_in_manager():
    # Verify partial reset in ObservationManager propagates to SlidingWindow
    N, J, W = 4, 3, 5
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.1
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": W, "dt": dt, "poly_degree": 2},
                hist_window_size=W,
                hist_func=hist_value,
            )
        )
    )
    om = ObservationManager(cfg, env)

    # Push 3 steps: 1, 2, 3
    for t in range(3):
        env.scene["robot"].data.joint_pos[:] = float(t + 1)
        om.compute()

    # Partial reset envs 1 and 3
    env_ids = torch.tensor([1, 3], dtype=torch.long)
    om.reset(env_ids=env_ids)

    # Next push: value 10
    env.scene["robot"].data.joint_pos[:] = 10.0
    om.compute()

    hist = om.get_term_hist("base_pos", group="base")
    assert hist is not None
    seq = hist.get_hist_data(window_size=4)  # (T=4, N, C)

    # For reset envs: valid_len=1 -> pad_len = 3, head should be replicated with 10.0
    assert torch.allclose(seq[:3, 1, :], torch.full((3, J), 10.0, device=env.device))
    assert torch.allclose(seq[:3, 3, :], torch.full((3, J), 10.0, device=env.device))
    # For non-reset envs (0,2): chronological tail should be [1,2,3,10]
    assert torch.allclose(seq[:, 0, :], torch.tensor([[1.0], [2.0], [3.0], [10.0]], device=env.device).repeat(1, J))
    assert torch.allclose(seq[:, 2, :], torch.tensor([[1.0], [2.0], [3.0], [10.0]], device=env.device).repeat(1, J))


def test_multiple_terms_concatenation_and_get_term_slices():
    # Two terms in the same group, concatenated along last dim
    N, J = 2, 2
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.05
    window = 7
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,  # last dim concat (feature dim)
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": window, "dt": dt, "poly_degree": 2},
                hist_window_size=window,
                hist_func=hist_value,
            ),
            base_vel=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": window, "dt": dt, "poly_degree": 2},
                hist_window_size=window,
                hist_func=hist_vel,
            ),
        )
    )
    om = ObservationManager(cfg, env)

    # Feed a few steps of a constant position to get zero velocity
    for _ in range(window):
        env.scene["robot"].data.joint_pos[:] = 3.14
        om.compute()

    buf = om.compute()
    assert isinstance(buf["base"], torch.Tensor)
    # Concatenated along last dim: shape (N, J + J)
    assert buf["base"].shape == (N, J * 2)

    # get_term should slice out each term range correctly
    t_pos = om.get_term("base_pos", group="base")
    t_vel = om.get_term("base_vel", group="base")
    assert t_pos.shape == (N, J)
    assert t_vel.shape == (N, J)

    # Numeric checks: pos ~ 3.14, vel ~ 0
    assert torch.allclose(t_pos, torch.full((N, J), 3.14, device=env.device), atol=1e-4)
    assert torch.allclose(t_vel, torch.zeros(N, J, device=env.device), atol=1e-4)


def test_clip_pipeline_applied_after_hist_func():
    # Verify clip is applied after hist aggregation
    N, J = 1, 3
    env = _MockEnv(num_envs=N, num_joints=J)
    dt = 0.05
    window = 5
    cfg = ObsCfgOne(
        base=BaseGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            base_pos=ObservationTermCfg(
                func=obs_joint_pos,
                params={"window": window, "dt": dt, "poly_degree": 2},
                hist_window_size=window,
                hist_func=hist_value,
                clip=(0.0, 1.0),
            )
        )
    )
    om = ObservationManager(cfg, env)

    # Feed values that exceed clip range
    for v in [2.5, 3.0, 4.0, -5.0, 0.8]:
        env.scene["robot"].data.joint_pos[:] = v
        om.compute()

    out = om.get_term("base_pos", group="base")
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= 0.0 - 1e-6)
    


# ----------------------
# Tests: cascaded get_term calls inside func and hist_func
# ----------------------

@dataclass
class CascGroupCfg(ObservationGroupCfg):
    # src term produces the raw signal; the other two terms will reference it via get_term
    src: ObservationTermCfg | None = None
    prev_src_func: ObservationTermCfg | None = None
    prev_src_hist: ObservationTermCfg | None = None

@dataclass
class CascCfg:
    base: CascGroupCfg


def _prev_src_in_func(env):
    # Return previous-step value of 'src' via ObservationManager.get_term (if available),
    # otherwise fall back to zeros on the very first compute call.
    om = getattr(env, "observation_manager", None)
    shape_like = env.scene["robot"].data.joint_pos
    if om is None or getattr(om, "_obs_buffer", None) is None:
        return torch.zeros_like(shape_like)
    return om.get_term("src", group="base")


def _prev_src_in_hist(env, hist, **kwargs):
    # Same idea as above, but invoked from a hist_func. We ignore hist content here on purpose
    # to ensure cascaded access works inside aggregators as well.
    om = getattr(env, "observation_manager", None)
    shape_like = env.scene["robot"].data.joint_pos
    if om is None or getattr(om, "_obs_buffer", None) is None:
        return torch.zeros_like(shape_like)
    return om.get_term("src", group="base")


def test_cascaded_get_term_in_func_and_histfunc_previous_step():
    N, J = 2, 3
    env = _MockEnv(num_envs=N, num_joints=J)

    cfg = CascCfg(
        base=CascGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            src=ObservationTermCfg(
                func=obs_joint_pos,
                params={},            # no hist params for source
                hist_window_size=0,   # raw passthrough
                hist_func=None,
            ),
            prev_src_func=ObservationTermCfg(
                func=_prev_src_in_func,  # reads src via om.get_term()
                params={},               # no extra params
                hist_window_size=0,
                hist_func=None,
            ),
            prev_src_hist=ObservationTermCfg(
                func=obs_joint_pos,      # push raw into window, but aggregator ignores hist and uses get_term
                params={},
                hist_window_size=3,
                hist_func=_prev_src_in_hist,
            ),
        )
    )
    om = ObservationManager(cfg, env)
    # Make manager accessible from env for cascaded calls
    env.observation_manager = om

    # Step 0: set v0, compute -> prev_* fall back to zeros
    v0 = 1.23
    env.scene["robot"].data.joint_pos[:] = v0
    om.compute()
    t_src = om.get_term("src", group="base")
    t_pf = om.get_term("prev_src_func", group="base")
    t_ph = om.get_term("prev_src_hist", group="base")
    assert torch.allclose(t_src, torch.full((N, J), v0, device=env.device))
    assert torch.allclose(t_pf, torch.zeros(N, J, device=env.device))
    assert torch.allclose(t_ph, torch.zeros(N, J, device=env.device))

    # Step 1: set v1, compute -> prev_* should equal v0
    v1 = -0.7
    env.scene["robot"].data.joint_pos[:] = v1
    om.compute()
    t_src = om.get_term("src", group="base")
    t_pf = om.get_term("prev_src_func", group="base")
    t_ph = om.get_term("prev_src_hist", group="base")
    assert torch.allclose(t_src, torch.full((N, J), v1, device=env.device))
    assert torch.allclose(t_pf, torch.full((N, J), v0, device=env.device))
    assert torch.allclose(t_ph, torch.full((N, J), v0, device=env.device))

    # Step 2: set v2, compute -> prev_* should equal v1
    v2 = 4.56
    env.scene["robot"].data.joint_pos[:] = v2
    om.compute()
    t_src = om.get_term("src", group="base")
    t_pf = om.get_term("prev_src_func", group="base")
    t_ph = om.get_term("prev_src_hist", group="base")
    assert torch.allclose(t_src, torch.full((N, J), v2, device=env.device))
    assert torch.allclose(t_pf, torch.full((N, J), v1, device=env.device))
    assert torch.allclose(t_ph, torch.full((N, J), v1, device=env.device))


def test_cascaded_chain_two_levels_hist_reads_func_reads_src():
    # Build a chain: src -> prev_src_func -> prev_prev_hist
    # prev_src_func (func) reads src(prev step), prev_prev_hist (hist_func) reads prev_src_func (prev step)
    N, J = 2, 2
    env = _MockEnv(num_envs=N, num_joints=J)

    def prev_src_in_func(env):
        om = getattr(env, "observation_manager", None)
        if om is None or getattr(om, "_obs_buffer", None) is None:
            return torch.zeros_like(env.scene["robot"].data.joint_pos)
        return om.get_term("src", group="base")

    def prev_prev_in_hist(env, hist, **kwargs):
        om = getattr(env, "observation_manager", None)
        if om is None or getattr(om, "_obs_buffer", None) is None:
            return torch.zeros_like(env.scene["robot"].data.joint_pos)
        return om.get_term("prev_src_func", group="base")

    @dataclass
    class ChainGroupCfg(ObservationGroupCfg):
        src: ObservationTermCfg | None = None
        prev_src_func: ObservationTermCfg | None = None
        prev_prev_hist: ObservationTermCfg | None = None

    @dataclass
    class ChainCfg:
        base: ChainGroupCfg

    cfg = ChainCfg(
        base=ChainGroupCfg(
            concatenate_terms=True,
            concatenate_dim=-1,
            enable_corruption=False,
            src=ObservationTermCfg(func=obs_joint_pos, params={}, hist_window_size=0, hist_func=None),
            prev_src_func=ObservationTermCfg(func=prev_src_in_func, params={}, hist_window_size=0, hist_func=None),
            prev_prev_hist=ObservationTermCfg(func=obs_joint_pos, params={}, hist_window_size=3, hist_func=prev_prev_in_hist),
        )
    )
    om = ObservationManager(cfg, env)
    env.observation_manager = om

    # Step 0
    env.scene["robot"].data.joint_pos[:] = 5.0
    om.compute()
    # Step 1
    env.scene["robot"].data.joint_pos[:] = 6.0
    om.compute()
    # Step 2
    env.scene["robot"].data.joint_pos[:] = 7.0
    om.compute()

    t_src = om.get_term("src", group="base")
    t_prev = om.get_term("prev_src_func", group="base")
    t_prev_prev = om.get_term("prev_prev_hist", group="base")

    # At step 2: prev_src_func == src at step 1, prev_prev_hist == prev_src_func at step 1 == src at step 0
    assert torch.allclose(t_src, torch.full((N, J), 7.0, device=env.device))
    assert torch.allclose(t_prev, torch.full((N, J), 6.0, device=env.device))
    assert torch.allclose(t_prev_prev, torch.full((N, J), 5.0, device=env.device))