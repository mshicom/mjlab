import math

import torch
import pytest

from mjlab.tasks.velocity.rl.amp.discriminator import Discriminator
from mjlab.tasks.velocity.rl.amp.normalizer import Normalizer


@pytest.mark.parametrize("loss_type", ["BCEWithLogits", "Wasserstein"])
def test_discriminator_forward_and_reward(loss_type: str):
    device = "cpu"
    input_dim = 16
    obs_dim = input_dim // 2
    disc = Discriminator(
        input_dim=input_dim,
        hidden_dims=(8, 8),
        reward_scale=0.1,
        reward_clamp_epsilon=1e-4,
        loss_type=loss_type,
        device=device,
    )
    x = torch.randn(32, input_dim)
    logits = disc(x)
    assert logits.shape == (32, 1)

    state = torch.randn(32, obs_dim)
    next_state = torch.randn(32, obs_dim)
    norm = Normalizer(obs_dim, device=device)

    with torch.no_grad():
        rew = disc.predict_reward(state, next_state, normalizer=norm)
        assert rew.shape == (32,)
        assert torch.isfinite(rew).all()

        # For BCE mode, reward is -log(1 - sigmoid(d)) scaled and should be non-negative.
        if loss_type == "BCEWithLogits":
            assert torch.all(rew >= 0.0)


@pytest.mark.parametrize("loss_type", ["BCEWithLogits", "Wasserstein"])
def test_discriminator_gradient_penalty(loss_type: str):
    device = "cpu"
    input_dim = 12
    obs_dim = input_dim // 2
    disc = Discriminator(
        input_dim=input_dim,
        hidden_dims=(16,),
        reward_scale=1.0,
        loss_type=loss_type,
        device=device,
    )
    expert_s = torch.randn(16, obs_dim)
    expert_ns = torch.randn(16, obs_dim)
    policy_s = torch.randn(16, obs_dim)
    policy_ns = torch.randn(16, obs_dim)

    gp = disc.gradient_penalty(
        expert_pair=(expert_s, expert_ns),
        policy_pair=(policy_s, policy_ns),
        lambda_gp=10.0,
    )
    assert isinstance(gp, torch.Tensor)
    assert gp.ndim == 0
    assert math.isfinite(gp.item())

    # Compute losses path
    with torch.no_grad():
        cat = torch.cat(
            (torch.cat([policy_s, policy_ns], dim=-1), torch.cat([expert_s, expert_ns], dim=-1)),
            dim=0,
        )
        logits = disc(cat)
        pol_logits, exp_logits = logits[: policy_s.size(0)], logits[policy_s.size(0) :]
        amp_loss, gp_loss = disc.compute_losses(pol_logits, exp_logits, gp)
        assert torch.isfinite(amp_loss).all()
        assert torch.isfinite(gp_loss).all()