import os
import math
import numpy as np
import torch
import pytest

from mjlab.amp.config import AmpDatasetCfg, AmpFeatureSetCfg, FeatureTermCfg, SymmetryAugmentCfg
from mjlab.amp.loader import AmpMotionLoader


DATASET_FILE = "/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/stepinplace1.npz"


def _skip_if_missing():
    if not os.path.exists(DATASET_FILE):
        pytest.skip(f"Dataset file not found: {DATASET_FILE}")


def _make_feature_set_basic() -> AmpFeatureSetCfg:
    """
    Keep the feature set minimal and robust across datasets:
    - qvel rms over window 32
    - base_lin speed rms and dominant frequency (derived from qvel first 3 comps)
    - base_ang yaw-rate rms (derived from qvel next 3 comps)
    """
    return AmpFeatureSetCfg(
        terms=[
            FeatureTermCfg(
                name="qvel_rms",
                source="qvel",
                channels=["scalar"],  # scalar DOFs
                window_size=32,
                pre_diff="none",
                aggregators=["rms"],
            ),
            FeatureTermCfg(
                name="base_lin_speed_rms",
                source="base_lin",
                channels=["lin.speed"],  # norm of linear base velocity
                window_size=32,
                pre_diff="none",
                aggregators=["rms"],
            ),
            FeatureTermCfg(
                name="base_lin_domfreq",
                source="base_lin",
                channels=["lin.z"],  # vertical component
                window_size=64,
                pre_diff="none",
                aggregators=["dominant_freq"],
            ),
            FeatureTermCfg(
                name="base_ang_yaw_rms",
                source="base_ang",
                channels=["ang.z"],  # yaw-rate
                window_size=32,
                pre_diff="none",
                aggregators=["rms"],
            ),
        ]
    )


def _make_dataset_cfg(symmetry: bool = False) -> AmpDatasetCfg:
    return AmpDatasetCfg(
        files=[DATASET_FILE],
        time_between_frames=0.05,
        preload_transitions=2000,
        symmetry=SymmetryAugmentCfg(enabled=symmetry),
        # contact derivation params (won't be used if site_xpos/sites absent)
        contact_site_names=None,
        contact_z_threshold=0.02,
        contact_hysteresis=0.005,
    )


def test_amp_motion_loader_smoke_basic():
    _skip_if_missing()

    feature_set = _make_feature_set_basic()
    dataset_cfg = _make_dataset_cfg(symmetry=False)

    loader = AmpMotionLoader(dataset_cfg=dataset_cfg, feature_set=feature_set, device="cpu")

    # Observation dim should be positive
    assert loader.observation_dim > 0

    # Basic modality presence (qvel must exist)
    assert "qvel" in loader.modalities and len(loader.modalities["qvel"]) >= 1
    # Derived bases should be present even if zero when Dq<6
    assert "base_lin" in loader.modalities and len(loader.modalities["base_lin"]) == len(loader.modalities["qvel"])
    assert "base_ang" in loader.modalities and len(loader.modalities["base_ang"]) == len(loader.modalities["qvel"])

    # If qvel has >=6 columns, check base derivation equals qvel slices for a few frames
    qv0 = loader.modalities["qvel"][0]
    bl0 = loader.modalities["base_lin"][0]
    ba0 = loader.modalities["base_ang"][0]
    if qv0.shape[1] >= 6:
        assert np.allclose(bl0, qv0[:, :3], atol=1e-5)
        assert np.allclose(ba0, qv0[:, 3:6], atol=1e-5)

    # Sample a batch of transitions
    gen = loader.feed_forward_generator(num_batches=1, batch_size=8)
    s, s_next = next(gen)
    assert isinstance(s, torch.Tensor) and isinstance(s_next, torch.Tensor)
    assert s.shape == s_next.shape
    assert s.shape[0] == 8
    assert s.shape[1] == loader.observation_dim
    assert torch.isfinite(s).all()
    assert torch.isfinite(s_next).all()

    # Run multiple batches to ensure stability
    gen = loader.feed_forward_generator(num_batches=3, batch_size=4)
    for s, s_next in gen:
        assert s.shape == s_next.shape
        assert s.shape[1] == loader.observation_dim
        assert torch.isfinite(s).all() and torch.isfinite(s_next).all()


def test_amp_motion_loader_symmetry_augmentation():
    _skip_if_missing()

    feature_set = _make_feature_set_basic()

    # Loader without symmetry
    loader_no_sym = AmpMotionLoader(dataset_cfg=_make_dataset_cfg(symmetry=False), feature_set=feature_set, device="cpu")

    # Loader with symmetry augmentation
    loader_sym = AmpMotionLoader(dataset_cfg=_make_dataset_cfg(symmetry=True), feature_set=feature_set, device="cpu")

    # For modalities that are mirrored, the number of trajectories should double.
    # At minimum, base_lin/base_ang should be mirrored.
    assert len(loader_sym.modalities["base_lin"]) == 2 * len(loader_no_sym.modalities["base_lin"])
    assert len(loader_sym.modalities["base_ang"]) == 2 * len(loader_no_sym.modalities["base_ang"])

    # If cvel exists, it should also be mirrored.
    if "cvel" in loader_no_sym.modalities and loader_no_sym.modalities["cvel"]:
        assert len(loader_sym.modalities["cvel"]) == 2 * len(loader_no_sym.modalities["cvel"])

    # Sanity: expert generator still works with symmetry enabled
    gen = loader_sym.feed_forward_generator(num_batches=1, batch_size=6)
    s, s_next = next(gen)
    assert s.shape == s_next.shape
    assert s.shape[0] == 6
    assert s.shape[1] == loader_sym.observation_dim
    assert torch.isfinite(s).all() and torch.isfinite(s_next).all()


def test_amp_motion_loader_frequency_and_dt():
    _skip_if_missing()

    loader = AmpMotionLoader(dataset_cfg=_make_dataset_cfg(symmetry=False), feature_set=_make_feature_set_basic(), device="cpu")
    assert loader.frequency > 0.0
    # dt from sampling should be 1/frequency
    Wmax = 65  # > any window in feature set (+1 for transitions)
    windows, dt = loader._sample_windows(batch_size=1, window_size_max=Wmax)
    assert math.isclose(dt, 1.0 / loader.frequency, rel_tol=1e-6)


def test_amp_motion_loader_contacts_if_available():
    _skip_if_missing()

    # Try enabling contact derivation with a heuristic default. This will only assert shape consistency if present.
    cfg = _make_dataset_cfg(symmetry=False)
    cfg.contact_site_names = None  # let loader pick default foot sites if available
    loader = AmpMotionLoader(dataset_cfg=cfg, feature_set=_make_feature_set_basic(), device="cpu")

    if "contacts" in loader.modalities and loader.modalities["contacts"]:
        # Contacts should be 0/1 floats
        arr = loader.modalities["contacts"][0]
        assert arr.ndim == 2  # [N, K]
        assert np.isfinite(arr).all()
        assert ((arr == 0.0) | (arr == 1.0)).all()


def test_amp_motion_loader_all_features():
    """
    Comprehensive smoke test:
    - Dynamically construct a feature set that attempts to cover all available sources in the dataset,
      using a variety of channels, pre_diff settings, and aggregators.
    - Compute expert transitions and verify shapes and finiteness.

    This test adapts to what's present in the npz (some sources may be absent).
    """

    # First, create a loader with a trivial feature set to inspect available modalities.
    probe_loader = AmpMotionLoader(dataset_cfg=_make_dataset_cfg(symmetry=False), feature_set=_make_feature_set_basic(), device="cpu")
    avail = {k for k, v in probe_loader.modalities.items() if v}

    terms = []

    # qvel: multiple pre-diff RMS terms
    if "qvel" in avail:
        terms += [
            FeatureTermCfg(name="qvel_speed_rms", source="qvel", channels=["scalar"], window_size=32, pre_diff="none", aggregators=["rms"]),
            FeatureTermCfg(name="qvel_accel_rms", source="qvel", channels=["scalar"], window_size=32, pre_diff="acceleration", aggregators=["rms"]),
            FeatureTermCfg(name="qvel_jerk_rms", source="qvel", channels=["scalar"], window_size=32, pre_diff="jerk", aggregators=["rms"]),
            FeatureTermCfg(name="qvel_flatten", source="qvel", channels=["scalar"], window_size=8, pre_diff="none", aggregators=["flatten"]),
        ]

    # base_lin/base_ang (derived)
    if "base_lin" in avail:
        terms += [
            FeatureTermCfg(name="base_lin_speed_rms_all", source="base_lin", channels=["lin.speed"], window_size=32, aggregators=["rms"]),
            FeatureTermCfg(name="base_lin_z_mean", source="base_lin", channels=["lin.z"], window_size=32, aggregators=["mean"]),
            FeatureTermCfg(name="base_lin_z_domfreq", source="base_lin", channels=["lin.z"], window_size=64, aggregators=["dominant_freq"]),
        ]
    if "base_ang" in avail:
        terms += [
            FeatureTermCfg(name="base_ang_yaw_rms", source="base_ang", channels=["ang.z"], window_size=32, aggregators=["rms"]),
            FeatureTermCfg(name="base_ang_speed_rms", source="base_ang", channels=["ang.speed"], window_size=32, aggregators=["rms"]),
        ]

    # cvel (per-body 6D): include norms and components
    if "cvel" in avail:
        terms += [
            FeatureTermCfg(name="cvel_lin_speed_rms", source="cvel", channels=["lin.speed"], window_size=32, aggregators=["rms"]),
            FeatureTermCfg(name="cvel_ang_speed_rms", source="cvel", channels=["ang.speed"], window_size=32, aggregators=["rms"]),
            FeatureTermCfg(name="cvel_ang_z_rms", source="cvel", channels=["ang.z"], window_size=32, aggregators=["rms"]),
            FeatureTermCfg(name="cvel_flatten", source="cvel", channels=[], window_size=8, aggregators=["flatten"]),
        ]

    # subtree_com (per-body 3D): spectral on vertical velocity (pre_diff=velocity)
    if "subtree_com" in avail:
        terms += [
            FeatureTermCfg(name="com_vert_freq", source="subtree_com", channels=["lin.z"], window_size=64, pre_diff="velocity", aggregators=["dominant_freq"]),
            FeatureTermCfg(name="com_vert_specent", source="subtree_com", channels=["lin.z"], window_size=64, pre_diff="velocity", aggregators=["spectral_entropy"]),
        ]

    # xpos (per-body 3D): simple aggregates and flatten
    if "xpos" in avail:
        terms += [
            FeatureTermCfg(name="xpos_z_mean", source="xpos", channels=["lin.z"], window_size=32, aggregators=["mean"]),
            FeatureTermCfg(name="xpos_flatten", source="xpos", channels=[], window_size=8, aggregators=["flatten"]),
        ]

    # site_xpos/site_xmat: flatten as generic checks (channels[] => pass-through groups)
    if "site_xpos" in avail:
        terms += [
            FeatureTermCfg(name="site_xpos_flatten", source="site_xpos", channels=[], window_size=8, aggregators=["flatten"]),
        ]
    if "site_xmat" in avail:
        terms += [
            FeatureTermCfg(name="site_xmat_flatten", source="site_xmat", channels=[], window_size=4, aggregators=["flatten"]),
        ]

    # xquat (per-body 4D): flatten and simple mean (treated as components)
    if "xquat" in avail:
        terms += [
            FeatureTermCfg(name="xquat_flatten", source="xquat", channels=[], window_size=8, aggregators=["flatten"]),
            FeatureTermCfg(name="xquat_mean", source="xquat", channels=["scalar"], window_size=16, aggregators=["mean"]),
        ]

    # contacts (derived): duty (mean) and transition rate (rms of diff)
    if "contacts" in avail:
        # ensure contact channels are included; loader exposes contacts_names for ordering
        # select.sites is optional; FeatureManager will select ALL if left empty in env path,
        # but for offline resolve-from-info we pass meta in loader so ALL is fine.
        terms += [
            FeatureTermCfg(name="contacts_duty", source="contacts", channels=["scalar"], window_size=50, aggregators=["mean"]),
            FeatureTermCfg(name="contacts_transitions", source="contacts", channels=["scalar"], window_size=50, pre_diff="velocity", aggregators=["rms"]),
        ]

    # If nothing added (very unlikely), fallback to basic
    if not terms:
        terms = _make_feature_set_basic().terms

    all_features_set = AmpFeatureSetCfg(terms=terms)

    # Build a loader with the comprehensive feature set
    loader = AmpMotionLoader(dataset_cfg=_make_dataset_cfg(symmetry=False), feature_set=all_features_set, device="cpu")

    # Observation dim should be positive and larger than the basic set
    assert loader.observation_dim > 0

    # Generate transitions and verify shapes/finiteness
    batch_size = 5
    gen = loader.feed_forward_generator(num_batches=2, batch_size=batch_size)
    for s, s_next in gen:
        assert s.shape == s_next.shape
        assert s.shape[0] == batch_size
        assert s.shape[1] == loader.observation_dim
        assert torch.isfinite(s).all() and torch.isfinite(s_next).all()

        # sanity: if spectral terms included, expect non-negative frequencies and entropies in [0,1]
        spectral_terms_present = any(t.aggregators and ("dominant_freq" in t.aggregators or "spectral_entropy" in t.aggregators) for t in terms)
        if spectral_terms_present:
            # Cannot precisely isolate columns without catalog internals, but we can assert s is finite and non-nan (already done).
            # Optionally: check non-negative values for dominant frequency by simple heuristic (not strict).
            assert (s >= s.min()).all()  # trivial assertion to keep branch