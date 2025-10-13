from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal




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
    discr_hidden_dims: tuple[int, ...] = (512, 256)
    task_reward_lerp: float = 0.0
    observation_group: str | None = None  # If None, use all observations.
    dataset: AmpDatasetCfg = field(default_factory=AmpDatasetCfg)
    replay_buffer_size: int = 2000
    state_normalization: bool = False
    grad_penalty_lambda: float = 10.0
    learning_rate: float = 1e-3  # learning rate for the discriminator optimizer
    amp_grad_pen_interval: int = 5  # interval (in policy updates) to update the AMP discriminator
    amp_num_mini_batches: int = 4  # number of mini-batches to split the AMP batch into for discriminator updates
    amp_batch_size: int = 256  # batch size of AMP transitions to use per discriminator update
    