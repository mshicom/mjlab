import torch

from mjlab.tasks.velocity.rl.rma import RmaCfg, RmaEncoder
from mjlab.tasks.velocity.rl.state_estimator import StateEstimatorCfg, ConcurrentStateEstimator


def test_rma_encoder_forward_shape():
    B, Din, Z = 8, 20, 12
    x = torch.randn(B, Din)
    enc = RmaEncoder(RmaCfg(input_dim=Din, hidden_dims=(32, 16), latent_dim=Z), device="cpu")
    z = enc(x)
    assert z.shape == (B, Z)
    assert torch.isfinite(z).all()


def test_concurrent_state_estimator_training_reduces_loss():
    B, Din, Dout = 64, 16, 3
    x = torch.randn(B, Din)
    # Ground truth: a linear map followed by clipping
    W = torch.randn(Din, Dout)
    y = (x @ W).clamp(-1.0, 1.0)

    cfg = StateEstimatorCfg(input_dim=Din, output_dim=Dout, hidden_dims=(64, 64), activation="elu", loss="l2", learning_rate=1e-2)
    se = ConcurrentStateEstimator(cfg, device="cpu")

    # Initial loss
    with torch.no_grad():
        pred0 = se(x)
        loss0 = torch.nn.functional.mse_loss(pred0, y).item()

    # Train for a few steps
    for _ in range(200):
        se.train_step(x, y)

    with torch.no_grad():
        pred1 = se(x)
        loss1 = torch.nn.functional.mse_loss(pred1, y).item()

    assert loss1 < loss0, f"Expected loss to decrease, but got {loss0} -> {loss1}"