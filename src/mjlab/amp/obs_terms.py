from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from mjlab.amp.config import AmpFeatureSetCfg
from mjlab.amp.feature_manager import FeatureManager
from mjlab.managers.manager_term_config import ObservationTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity import mdp


@dataclass
class EnvInfo:
    """
    Lightweight adapter that mimics the subset of TrajectoryInfo used by FeatureManager.

    This lets FeatureManager.resolve() build per-term index mappings from a live env,
    without needing a full TrajectoryInfo from a dataset.

    Fields expected by FeatureManager._resolve_indices_from_info:
    - joint_names: ordered names of policy DOFs.
    - joint_name2ind_qvel: dict[name] -> np.ndarray of indices into qvel vector.
      For 1-DoF joints (hinge/slide), map to a single index; we assume env qvel_rel excludes base 6 DOFs.
    - body_names, body_name2ind: optional; empty list/dict if not needed by current feature set.
    - site_names, site_name2ind: optional; empty list/dict if not needed by current feature set.

    Notes:
    - If your AMP feature terms select body/site subsets (xpos/xquat/cvel/site_xpos/site_xmat),
      consider populating body/site names from your scene Entity (if available) to enable name-based selection.
    """

    joint_names: List[str]
    joint_name2ind_qvel: dict[str, np.ndarray]
    body_names: Optional[List[str]] = None
    body_name2ind: Optional[dict[str, np.ndarray]] = None
    site_names: Optional[List[str]] = None
    site_name2ind: Optional[dict[str, np.ndarray]] = None

    # Keep an attribute shape similar to traj_info.model if needed by other code paths
    class _ModelShim:
        def __init__(self, njnt: int):
            self.njnt = njnt

    @property
    def model(self):
        return EnvInfo._ModelShim(njnt=len(self.joint_names))


@dataclass
class AmpFeatureObs:
    """
    Observation term that computes AMP feature vector per step and writes to env.extras["amp_observations"].

    What this term does:
    - Maintains per-env rolling windows for selected modalities:
        qvel (relative joint velocities), base_lin (root linear velocity),
        base_ang (root angular velocity), contacts (binary, from sensor hits).
    - Uses FeatureManager to compute per-term features with per-term window sizes, smoothing, and aggregations.
    - Publishes a per-env feature vector each step, accessible to AMP via extras["amp_observations"].

    How indices are resolved:
    - Builds an EnvInfo adapter that mimics TrajectoryInfo fields required by FeatureManager.
    - Calls FeatureManager.resolve(env_info, device, meta={"contacts_names": sensor_names}) so that
      name-based selections in FeatureTermCfg.select (e.g., select.sites for contacts) work.

    Limitations and assumptions:
    - For qvel mapping: we assume mdp.joint_vel_rel(env) returns policy DOFs (excluding floating base 6 DOFs).
      If your feature set targets specific joint names, ensure EnvInfo.joint_names is ordered accordingly.
    - For body/site selection: this adapter leaves body/site names empty by default. If your feature set uses
      body- or site-based sources, extend _build_env_info() to query names from your Entity (e.g., asset.body_names).
    - The feature_set used for env should only include sources provided by this term (qvel, base_lin, base_ang, contacts),
      unless you extend this term to buffer additional modalities.

    Params expected in ObservationTermCfg.params:
      - feature_set: AmpFeatureSetCfg
      - sensor_names: list[str] (optional) used for contacts channel ordering
    """

    cfg: ObservationTermCfg
    env: ManagerBasedRlEnv

    def __post_init__(self):
        params = self.cfg.params
        self.feature_set: AmpFeatureSetCfg = params["feature_set"]
        self.sensor_names: List[str] = params.get("sensor_names", [])
        self.window_size_max = max(t.window_size for t in self.feature_set.terms)

        self.manager = FeatureManager(self.feature_set)
        self.dt = self.env.step_dt

        # Build EnvInfo adapter to leverage FeatureManager's resolve-from-info path.
        self.env_info = self._build_env_info()

        # Resolve indices from EnvInfo; pass contacts meta for site-name-based selection in 'contacts' source.
        meta = {"contacts_names": self.sensor_names} if self.sensor_names else None
        self.manager.resolve(self.env_info, torch.device(self.env.device), meta=meta)

        N = self.env.num_envs
        self.buffers = {
            "qvel": torch.zeros(N, self.window_size_max, self._qvel_dim(), device=self.env.device),
            "base_lin": torch.zeros(N, self.window_size_max, 3, device=self.env.device),
            "base_ang": torch.zeros(N, self.window_size_max, 3, device=self.env.device),
            "contacts": torch.zeros(N, self.window_size_max, max(1, len(self.sensor_names)), device=self.env.device),
        }
        self.filled = torch.zeros(N, dtype=torch.long, device=self.env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        for k in self.buffers:
            self.buffers[k][env_ids].zero_()
        if isinstance(env_ids, slice):
            self.filled.zero_()
        else:
            self.filled[env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        # Gather current-frame modalities.
        qvel = mdp.joint_vel_rel(env)  # [N,Dq]
        base_lin = mdp.base_lin_vel(env)  # [N,3]
        base_ang = mdp.base_ang_vel(env)  # [N,3]
        contacts = self._compute_contacts(env)  # [N,K]

        # Roll and insert into windows.
        self._append_frame("qvel", qvel)
        self._append_frame("base_lin", base_lin)
        self._append_frame("base_ang", base_ang)
        self._append_frame("contacts", contacts)

        self.filled = torch.clamp(self.filled + 1, max=self.window_size_max)

        # Compute features with full buffers (FeatureManager trims per-term with its own window sizes).
        windows = {k: v for k, v in self.buffers.items()}
        feats = self.manager.compute(windows, self.dt)

        # Expose to AMP via extras.
        env.extras["amp_observations"] = feats
        return feats

    # ----- helpers -----

    def _append_frame(self, key: str, val: torch.Tensor):
        """
        Append current frame to the per-env rolling buffer for the given modality.
        Pads or truncates channels to match buffer's last dimension.
        """
        B = self.buffers[key]
        B[:] = torch.roll(B, shifts=-1, dims=1)
        if val.ndim == 1:
            val = val.unsqueeze(0)
        if val.ndim == 2:
            val = val.unsqueeze(1)  # [N,1,D]
        if val.shape[-1] != B.shape[-1]:
            D = B.shape[-1]
            v = torch.zeros(val.shape[0], 1, D, device=B.device, dtype=B.dtype)
            v[..., : min(D, val.shape[-1])] = val[..., : min(D, val.shape[-1])]
            val = v
        B[:, -1, :] = val[:, 0, :]

    def _compute_contacts(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """
        Build a binary contact vector per env.

        If sensor_names were provided, this reads each named sensor's 'found' flag (bool->float).
        Otherwise, returns a single zero channel as a placeholder.
        """
        if not self.sensor_names:
            return torch.zeros(env.num_envs, 1, device=env.device)
        asset = env.scene["robot"]
        vals = []
        for s in self.sensor_names:
            sensor = asset.data.sensor_data[s]  # [N,1] "found"
            vals.append((sensor[:, 0] > 0).float())
        return torch.stack(vals, dim=1)

    def _qvel_dim(self) -> int:
        # derive dimension from entity articulation
        asset = self.env.scene["robot"]
        return asset.num_joints

    def _build_env_info(self) -> EnvInfo:
        """
        Construct an EnvInfo object to allow FeatureManager to perform name-based selections.

        Attempts to query DOF names from the robot entity; if not available, generates generic names.
        Body/site names are left empty by default; extend here if your feature set uses body/site-based sources.
        """
        asset = self.env.scene["robot"]

        # Try to fetch dof/joint names from the entity or its data.
        names: Optional[List[str]] = None
        for attr in ("dof_names", "joint_names"):
            n = getattr(asset, attr, None)
            if n is not None:
                names = list(n)
                break
        if names is None:
            n = getattr(asset.data, "dof_names", None)
            if n is not None:
                names = list(n)

        D = self._qvel_dim()
        if names is None or len(names) != D:
            # Fallback to generic names if unavailable or mismatched
            names = [f"dof_{i}" for i in range(D)]

        # Build a trivial mapping name -> single qvel index (hinge/slide).
        joint_name2ind_qvel: dict[str, np.ndarray] = {}
        for i, nm in enumerate(names):
            joint_name2ind_qvel[nm] = np.array([i], dtype=np.int64)

        # Body/site names (optional): leave empty by default.
        body_names: Optional[List[str]] = getattr(asset, "body_names", None)
        site_names: Optional[List[str]] = getattr(asset, "site_names", None)

        body_name2ind = {bn: np.array([i], dtype=np.int64) for i, bn in enumerate(body_names or [])}
        site_name2ind = {sn: np.array([i], dtype=np.int64) for i, sn in enumerate(site_names or [])}

        return EnvInfo(
            joint_names=names,
            joint_name2ind_qvel=joint_name2ind_qvel,
            body_names=body_names or [],
            body_name2ind=body_name2ind,
            site_names=site_names or [],
            site_name2ind=site_name2ind,
        )