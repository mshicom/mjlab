from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# What is a SignalSource?
# - Signal sources denote the time-series modalities that the FeatureManager can consume from either:
#   (a) TrajectoryData loaded from npz (offline expert dataset) or
#   (b) live environment state (online policy rollout).
# - The "generic" sources map to TrajectoryData fields (qpos, qvel, xpos, xquat, cvel, subtree_com, site_xpos, site_xmat).
# - The "convenience" sources (base_lin, base_ang, contacts) can be derived in AmpMotionLoader from generic sources
#   or produced directly by an env observation term (AmpFeatureObs).
SignalSource = Literal[
    # generic npz/trajectory sources
    "qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat",
    # convenience derived sources (offline loader and/or env observation term can provide)
    "base_lin", "base_ang", "contacts",
]


# What is an Aggregator?
# - A post-temporal aggregation operator that converts a [B, T, C] slice into [B, F] statistics.
# - "flatten" keeps the time dimension by concatenation; others collapse time.
Aggregator = Literal[
    "mean", "rms", "std", "max", "min",
    "dominant_freq", "spectral_entropy",
    "flatten",
]


# Pre-differentiation setting:
# - Differences computed over time AFTER optional Savitzky–Golay smoothing and BEFORE aggregation.
# - velocity: 1st difference; acceleration: 2nd difference; jerk: 3rd difference.
PreDiff = Literal["none", "velocity", "acceleration", "jerk"]

# Savitzky–Golay smoothing preset:
# - "poly2_w5": order-2 polynomial over a 5-tap window (classic coefficients).
# - Applied per channel across the time axis before differences.
SavgolMode = Literal["none", "poly2_w5"]


@dataclass
class JointSelectionCfg:
    """
    Selects subsets (by name) for joints, bodies, or sites.
    Used by: FeatureTermCfg.select to scope which indices the term extracts from the source modality.
    Resolved by: the mapping passed to FeatureManager.resolve (name->index).
    """
    joints: list[str] = field(default_factory=list)
    bodies: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    presets: list[str] = field(default_factory=list)  # e.g. ["ankles", "wrists", "knees"]


@dataclass
class FeatureTermCfg:
    """
    Declarative configuration for one feature term in the FeatureManager.

    Fields:
    - name: a unique label for this term (for debugging/logging).
    - source: which SignalSource to draw the time-series from.
    - channels: semantics for channel selection within a modality. Interpreted by the selector in FeatureManager.
                Examples: ["scalar"], ["ang.z"], ["lin.speed"]. The skeleton gathers indices directly.
    - window_size: temporal window length (number of frames) for this term.
    - savgol: whether to apply Savitzky–Golay smoothing before differencing.
    - pre_diff: order of time difference (velocity/acceleration/jerk) applied after smoothing.
    - aggregators: list of aggregation ops over time ("rms", "mean", etc.). "flatten" retains time.
    - select: subset of named elements to include (resolved to indices via FeatureManager.resolve).
    """
    name: str
    source: SignalSource
    channels: list[str] = field(default_factory=list)
    window_size: int = 20
    savgol: SavgolMode = "poly2_w5"
    pre_diff: PreDiff = "none"
    aggregators: list[Aggregator] = field(default_factory=lambda: ["rms"])
    select: JointSelectionCfg = field(default_factory=JointSelectionCfg)


@dataclass
class SymmetryAugmentCfg:
    """
    Dataset-side symmetry augmentation policy.

    Fields:
    - enabled: turn augmentation on/off.
    - left_right_prefixes: list of (left_prefix, right_prefix) pairs used to swap names.
    - lateral_axes: indices of axes to flip sign for linear quantities (e.g., y for lateral).
    - explicit_maps: optional explicit renaming dicts per modality if prefix swapping is insufficient.
    Used by: AmpMotionLoader when duplicating trajectories with mirrored data.
    """
    enabled: bool = False
    left_right_prefixes: list[tuple[str, str]] = field(default_factory=lambda: [("left_", "right_"), ("l_", "r_")])
    lateral_axes: list[int] = field(default_factory=lambda: [1])
    explicit_maps: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class AmpFeatureSetCfg:
    """
    Collection of feature terms to be concatenated by FeatureManager.

    Fields:
    - terms: list of FeatureTermCfg entries composing the final feature vector.
    - default_select: default selection applied to all terms unless overridden.
    - default_savgol: default smoothing option applied to all terms unless overridden.
    - normalize: optional toggle to apply an external normalizer at the consumer (e.g., AMP discriminator).
    Used by: FeatureManager (both env and offline).
    """
    terms: list[FeatureTermCfg]
    default_select: JointSelectionCfg = field(default_factory=JointSelectionCfg)
    default_savgol: SavgolMode = "poly2_w5"
    normalize: bool = False


@dataclass
class AmpDatasetCfg:
    """
    Dataset configuration for the AmpMotionLoader (offline expert).

    Fields:
    - files: paths to Trajectory.save() npz files (see utils/dataset/traj_class.py).
    - mujoco_xml: optional path if you want to filter/extend/reorder via TrajectoryHandler (not implemented here).
    - time_between_frames: dt for transition sampling (used for s, s_next pairing in window sampling).
    - preload_transitions: number of expert transitions to preload if building caches (not implemented here).
    - symmetry: symmetry augmentation config (duplication with left/right swapping).
    - precache: enable building cached numpy arrays per modality for fast sampling.
    - contact_site_names: site names used to derive contact binary signals from site_z (if available).
    - contact_z_threshold: threshold on site_z below which contact is 1.0, else 0.0.
    - contact_hysteresis: small band to reduce flickering around threshold (simple Schmitt trigger).
    Used by: AmpMotionLoader.
    """
    files: list[str] = field(default_factory=list)
    mujoco_xml: str | None = None
    time_between_frames: float = 0.05
    preload_transitions: int = 200_000
    symmetry: SymmetryAugmentCfg = field(default_factory=SymmetryAugmentCfg)
    precache: bool = True
    contact_site_names: list[str] | None = None
    contact_z_threshold: float = 0.02
    contact_hysteresis: float = 0.005


@dataclass
class AmpCfg:
    """
    Top-level AMP configuration for constructing the AdversarialMotionPrior.

    Fields:
    - reward_coef: scale for AMP reward.
    - discr_hidden_dims: MLP hidden sizes for the discriminator.
    - task_reward_lerp: optional blend between AMP and task reward inside discriminator.
    - feature_set: the feature terms used both online (env) and offline (expert). If not provided for expert-side,
                   AmpMotionLoader can default to consuming 'ALL' channels or a separate feature set can be passed.
    - dataset: AmpDatasetCfg for expert data access (npz trajectories).
    - replay_buffer_size: policy replay capacity for AMP transitions.
    - state_normalization: whether to use EmpiricalNormalization for AMP inputs.
    - grad_penalty_lambda: gradient penalty coefficient for discriminator regularizer.
    Used by: rsl_rl.modules.amp.AdversarialMotionPrior
    """
    reward_coef: float = 0.5
    discr_hidden_dims: tuple[int, ...] = (1024, 512, 256)
    task_reward_lerp: float = 0.0
    feature_set: AmpFeatureSetCfg = field(default_factory=AmpFeatureSetCfg)
    dataset: AmpDatasetCfg = field(default_factory=AmpDatasetCfg)
    replay_buffer_size: int = 10000
    state_normalization: bool = True
    grad_penalty_lambda: float = 10.0
    learning_rate: float = 1e-3  # learning rate for the discriminator optimizer
