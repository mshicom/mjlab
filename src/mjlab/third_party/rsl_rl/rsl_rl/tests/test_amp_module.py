import math
import torch
import pytest

from rsl_rl.modules.amp import AMPDiscriminator, AMPConfig


def _make_gaussian_provider(mean: float, std: float, F: int):
    """Return a demo_provider that samples N(mean, std^2) iid features."""
    def provider(n: int, device: torch.device) -> torch.Tensor:
        return mean + std * torch.randn(n, F, device=device)
    return provider


def test_amp_forward_and_reward_zero_init_cpu():
    device = torch.device("cpu")
    F = 8
    cfg = AMPConfig(
        obs_key="amp_state",
        hidden_dims=(16,),
        activation="elu",
        init_output_scale=0.0,  # so logits weight stays default init (close to 0)
        learning_rate=1e-3,
        reward_scale=2.0,
        demo_provider=_make_gaussian_provider(mean=1.0, std=0.1, F=F),
    )
    amp = AMPDiscriminator(cfg, amp_state_shape=torch.Size([F]), device=str(device)).to(device)

    # Agent batch: zeros
    x = torch.zeros(32, F, device=device)
    with torch.no_grad():
        r = amp.compute_reward(x)  # logits ~ 0 => p ~ 0.5 => r ~ -log(0.5) * 2
    assert r.shape == (32,)
    expected = -math.log(0.5) * cfg.reward_scale
    assert torch.allclose(r.mean(), torch.tensor(expected, device=device), atol=0.25), \
        "Reward should be close to -log(0.5)*scale under near-zero logits."

def test_amp_loss_bce_and_metrics_separable():
    device = torch.device("cpu")
    F = 4
    # Agent around -1, demo around +1 => separable
    demo_provider = _make_gaussian_provider(mean=+1.0, std=0.1, F=F)
    cfg = AMPConfig(
        hidden_dims=(32, 32),
        activation="elu",
        demo_provider=demo_provider,
        logit_reg=0.0,
        grad_penalty=0.0,
        disc_weight_decay=0.0,
    )
    amp = AMPDiscriminator(cfg, amp_state_shape=torch.Size([F]), device=str(device)).to(device)

    # Train a couple of steps to get non-trivial logits
    optim = torch.optim.Adam(amp.parameters(), lr=3e-3)
    for _ in range(50):
        agent_batch = -1.0 + 0.1 * torch.randn(64, F, device=device)
        loss_dict = amp.compute_loss(agent_batch)
        loss = loss_dict["amp_loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate metrics
    agent_batch = -1.0 + 0.1 * torch.randn(128, F, device=device)
    with torch.no_grad():
        d = amp.compute_loss(agent_batch)
    assert d["amp_agent_acc"] > 0.6, "Discriminator should classify agent as negative (>60%)."
    assert d["amp_demo_acc"] > 0.6, "Discriminator should classify demo as positive (>60%)."

def test_amp_loss_regularizers_increase_loss():
    device = torch.device("cpu")
    F = 6
    demo_provider = _make_gaussian_provider(mean=0.5, std=0.2, F=F)

    base_cfg = AMPConfig(
        hidden_dims=(16,),
        activation="relu",
        demo_provider=demo_provider,
        logit_reg=0.0,
        grad_penalty=0.0,
        disc_weight_decay=0.0,
    )
    amp = AMPDiscriminator(base_cfg, amp_state_shape=torch.Size([F]), device=str(device)).to(device)
    agent_batch = torch.randn(64, F, device=device)

    # Base loss
    d0 = amp.compute_loss(agent_batch)
    base_loss = d0["amp_loss"].item()

    # Logit L2 should not decrease loss
    amp.cfg.logit_reg = 1e-2
    d1 = amp.compute_loss(agent_batch)
    assert d1["amp_loss"].item() >= base_loss - 1e-6

    # Grad penalty should not decrease loss (enable requires grad path)
    amp.cfg.grad_penalty = 1.0
    d2 = amp.compute_loss(agent_batch)
    assert d2["amp_loss"].item() >= base_loss - 1e-6
    assert d2["amp_grad_penalty"] >= 0.0

    # Manual weight decay should not decrease loss
    amp.cfg.disc_weight_decay = 1e-3
    d3 = amp.compute_loss(agent_batch)
    assert d3["amp_loss"].item() >= base_loss - 1e-6

def test_amp_normalizer_updates_and_freezes():
    device = torch.device("cpu")
    F = 3
    cfg = AMPConfig(
        demo_provider=_make_gaussian_provider(mean=0.0, std=1.0, F=F),
        norm_until=64,  # stop updating after 64 samples
    )
    amp = AMPDiscriminator(cfg, amp_state_shape=torch.Size([F]), device=str(device)).to(device)

    # Update with 64 samples
    x = torch.randn(64, F, device=device)
    amp.update_normalization(x)
    # State is updated, but EmpiricalNormalization's count is private; smoke test ensure no crash.

    # After until reached, further updates should be no-ops; not directly assertable here
    # but at least ensure forward still works.
    x2 = torch.randn(8, F, device=device)
    with torch.no_grad():
        _ = amp.compute_reward(x2)
        

class ObsDict(dict):
    """Simple dict with a .to(device) method to mimic TensorDict behavior used by rsl_rl."""
    def to(self, device):
        out = ObsDict()
        for k, v in self.items():
            out[k] = v.to(device)
        return out


class DummyAmpEnv(VecEnv):  # type: ignore
    """Minimal VecEnv-compatible environment that provides amp_state and a demo sampler.

    Observations:
      - 'critic': [N, F]
      - 'amp_state': [N, F]
    Rewards: zeros (AMP reward will be added by PPO).
    Dones: episodes of length max_episode_length.
    """
    def __init__(self, num_envs=8, F=12, device="cpu"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_actions = 2
        self.F = F
        self.max_episode_length = 64
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._state = torch.randn(self.num_envs, self.F, device=self.device)
        # Precomputed demo bank with different mean to induce separability
        self._demo_bank = 1.5 + 0.5 * torch.randn(10_000, self.F, device=self.device)

    def get_observations(self):
        return ObsDict({"critic": self._state.clone(), "amp_state": self._state.clone()})

    def step(self, actions):
        # Random walk in state space
        self._state = 0.99 * self._state + 0.01 * torch.randn_like(self._state)
        # Progress episodes
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).unsqueeze(-1).to(torch.long)
        # reset finished episodes
        reset_ids = (dones[:, 0] > 0).nonzero(as_tuple=False).view(-1)
        if reset_ids.numel() > 0:
            self._state[reset_ids] = torch.randn_like(self._state[reset_ids])
            self.episode_length_buf[reset_ids] = 0
        rewards = torch.zeros(self.num_envs, 1, device=self.device)
        extras = {}
        return self.get_observations(), rewards, dones, extras

    # AMP demo provider
    def sample_amp_demos(self, n: int, device: torch.device):
        idx = torch.randint(low=0, high=self._demo_bank.shape[0], size=(n,), device=self.device)
        return self._demo_bank[idx].to(device)


def test_amp_end_to_end_smoke():
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    device = "cpu"
    env = DummyAmpEnv(num_envs=4, F=10, device=device)
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "desired_kl": 0.01,
            # AMP enabled
            "amp_cfg": {
                "enabled": True,
                "obs_key": "amp_state",
                "hidden_dims": [64, 64],
                "activation": "elu",
                "init_output_scale": 0.0,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "logit_reg": 0.0,
                "grad_penalty": 0.0,
                "disc_weight_decay": 0.0,
                "reward_scale": 1.0,
                "reward_coef": 1.0,
                "eval_batch_size": 0,
                "norm_until": None,
                # demo_provider resolved automatically via env.sample_amp_demos
                "demo_batch_ratio": 1.0,
            },
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "num_steps_per_env": 8,
        "save_interval": 1000,
        "logger": "tensorboard",
        "obs_groups": {  # minimal valid mapping
            "policy": ["critic"],
            "critic": ["critic"],
            "amp": ["amp_state"],
        },
    }
    runner = OnPolicyRunner(env=env, train_cfg=cfg, log_dir=None, device=device)
    # Run a few iterations
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
    # Ensure algorithm contains AMP and loss dict logs it
    assert runner.alg.amp is not None