import math
import numpy as np
import torch

from mjlab.amp.config import AmpFeatureSetCfg, FeatureTermCfg
from mjlab.amp.feature_manager import FeatureManager
from mjlab.amp.feature_terms import savgol_smooth, time_deriv, spectral_entropy
from mjlab.amp.loader import AmpMotionLoader


def test_savgol_and_time_deriv_shapes():
    B, T, C = 2, 10, 3
    x = torch.randn(B, T, C)
    xs = savgol_smooth(x)
    assert xs.shape == (B, T, C)
    dx = time_deriv(xs)
    assert dx.shape == (B, T, C)


def test_dominant_frequency_detection():
    # Build a pure sine at 5 Hz, sampled at 100 Hz (dt=0.01), T=100 frames -> 1 second.
    B, T, C = 4, 100, 1
    dt = 0.01
    t = torch.arange(T).float() * dt
    f = 5.0  # Hz
    sig = torch.sin(2 * math.pi * f * t).view(1, T, 1).repeat(B, 1, 1)
    # Feature: dominant frequency over time
    fm = FeatureManager(AmpFeatureSetCfg(terms=[
        FeatureTermCfg(
            name="domfreq",
            source="qvel",
            channels=["scalar"],
            window_size=T,
            pre_diff="none",
            aggregators=["dominant_freq"],
        )
    ]))
    # Map qvel:ALL to the single channel
    name_to_index = {"qvel:ALL": torch.tensor([0], dtype=torch.long)}
    fm.resolve(name_to_index, device=torch.device("cpu"))
    feats = fm.compute({"qvel": sig}, dt=dt)
    # Expect all rows ~ f within tolerance
    assert feats.shape == (B, 1)
    assert torch.allclose(feats.squeeze(), torch.full((B,), f), atol=0.25)


def test_spectral_entropy_bounds():
    # Random power spectra must yield entropy in [0, 1]
    B, Freq, C = 3, 16, 2
    power = torch.rand(B, Freq, C)
    ent = spectral_entropy(power, dim=1)
    assert ent.shape == (B, C)
    assert torch.all(ent >= 0.0) and torch.all(ent <= 1.0 + 1e-6)


def test_feature_manager_window_trimming():
    # FeatureTerm with window_size<Wmax: ensure manager uses last W frames per term
    B, Tq, Dq = 2, 8, 3
    dt = 0.02
    q = torch.arange(B * Tq * Dq, dtype=torch.float32).reshape(B, Tq, Dq)
    fm = FeatureManager(AmpFeatureSetCfg(terms=[
        FeatureTermCfg(
            name="mean_last4",
            source="qvel",
            channels=["scalar"],
            window_size=4,
            aggregators=["mean"],
        )
    ]))
    name_to_index = {"qvel:ALL": torch.arange(Dq, dtype=torch.long)}
    fm.resolve(name_to_index, device=torch.device("cpu"))
    feats = fm.compute({"qvel": q}, dt=dt)  # should average last 4 frames
    # Compute ground truth
    gt = q[:, -4:, :].mean(dim=1).reshape(B, -1)
    assert torch.allclose(feats, gt)


def test_contacts_from_site_z():
    # Simple square wave in z crossing threshold
    N, K = 10, 2
    z = np.ones((N, K), dtype=np.float32) * 0.03
    z[2:6, :] = 0.0  # below threshold for a span
    contacts = AmpMotionLoader.derive_contacts_from_site_z(z, z_threshold=0.02, hysteresis=0.005)
    # Expect zeros except 2..5 inclusive are ones
    assert contacts.shape == (N, K)
    assert np.all(contacts[:2] == 0.0)
    assert np.all(contacts[2:6] == 1.0)
    # Rising above threshold + hysteresis -> zero again
    assert np.all(contacts[6:] == 0.0)


def test_base_from_qvel_derivation():
    N = 5
    qv = np.zeros((N, 10), dtype=np.float32)
    qv[:, 0:3] = 1.0
    qv[:, 3:6] = 2.0
    bl, ba = AmpMotionLoader.derive_base_from_qvel(qv)
    assert bl.shape == (N, 3)
    assert ba.shape == (N, 3)
    assert np.allclose(bl, 1.0)
    assert np.allclose(ba, 2.0)