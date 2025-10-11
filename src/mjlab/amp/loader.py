from __future__ import annotations

import glob
from typing import Dict, List, Tuple

import numpy as np
import torch

from mjlab.utils.dataset.traj_class import Trajectory
from mjlab.amp.config import AmpDatasetCfg, AmpFeatureSetCfg, SymmetryAugmentCfg
from mjlab.amp.feature_manager import FeatureManager

class AmpMotionLoader:
    """
    Trajectory npz-based expert feature loader using FeatureManager.

    Responsibilities:
    - Load Trajectory.save() npz files (TrajectoryData + TrajectoryInfo).
    - Build per-modality numpy arrays for fast sampling (precache).
    - Compute derived convenience sources:
        base_lin: from free joint qvel (first 3 components)
        base_ang: from free joint qvel (next 3 components)
        contacts: proxy from site_xpos z-height threshold for selected sites
    - Optionally apply symmetry augmentation (left/right swap + lateral sign flips).
    - Sample aligned multi-modality sliding windows and compute feature pairs (s, s_next).

    Used by: rsl_rl.modules.amp.AdversarialMotionPrior.expert_generator().
    """

    def __init__(self, dataset_cfg: AmpDatasetCfg, feature_set: AmpFeatureSetCfg, device: str):
        self.dataset_cfg = dataset_cfg
        self.feature_set = feature_set 
        self.device = torch.device(device)

        self.trajs: List[Trajectory] = []
        self.frequency: float = 0.0
        self.modalities: Dict[str, List[np.ndarray]] = {}
        self.contacts_names: List[str] | None = None
        self.manager = FeatureManager(feature_set)

        self._load_all()
        self._precompute()
        self._derive_convenience_modalities()

        # Symmetry augmentation (duplicates modalities in-place)
        if self.dataset_cfg.symmetry.enabled:
            self._apply_symmetry_augmentation(self.dataset_cfg.symmetry)

        self._resolve_manager()

    def _load_all(self):
        files = self.dataset_cfg.files or glob.glob("datasets/motion_traj/*.npz")
        for f in files:
            self.trajs.append(Trajectory.load(f))
        if not self.trajs:
            raise FileNotFoundError("No trajectory npz files found for AMP dataset.")
        self.frequency = float(self.trajs[0].info.frequency)

    def _precompute(self):
        keys = ["qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat"]
        for key in keys:
            self.modalities[key] = [np.array(getattr(t.data, key)) for t in self.trajs if hasattr(t.data, key)]

    def _derive_convenience_modalities(self):
        # base_lin/base_ang from qvel
        if "qvel" in self.modalities and self.modalities["qvel"]:
            base_lin_list: List[np.ndarray] = []
            base_ang_list: List[np.ndarray] = []
            for qv in self.modalities["qvel"]:
                bl, ba = self.derive_base_from_qvel(qv)
                base_lin_list.append(bl)
                base_ang_list.append(ba)
            self.modalities["base_lin"] = base_lin_list
            self.modalities["base_ang"] = base_ang_list

        # contacts from site_xpos if we have site names
        site_names = getattr(self.trajs[0].info, "site_names", None)
        if "site_xpos" in self.modalities and self.modalities["site_xpos"] and site_names:
            contact_sites = self.dataset_cfg.contact_site_names or self._default_contact_sites(site_names)
            self.contacts_names = contact_sites
            indices = [site_names.index(n) for n in contact_sites if n in site_names]
            if indices:
                contacts_list: List[np.ndarray] = []
                for sx in self.modalities["site_xpos"]:
                    # sx: [N, nsite, 3]
                    z = sx[:, indices, 2]  # [N, K]
                    contacts = self.derive_contacts_from_site_z(
                        z,
                        z_threshold=self.dataset_cfg.contact_z_threshold,
                        hysteresis=self.dataset_cfg.contact_hysteresis,
                    )  # float32 0/1
                    contacts_list.append(contacts.astype(np.float32))
                self.modalities["contacts"] = contacts_list

    def _apply_symmetry_augmentation(self, sym: SymmetryAugmentCfg):
        """
        Duplicate all modalities by mirroring:
          - swap left/right body and site indices using prefixes (if names exist)
          - flip lateral axis sign for linear components (e.g., y)
          - flip yaw-rate sign for angular z components in cvel/base_ang
        Notes:
          - This function augments body/site/base modalities. Joint-level (qpos/qvel) remapping is robot-specific for
            sign conventions and is not applied here. If you rely on joint features, consider adding a joint remapper.
        """
        info = self.trajs[0].info
        body_names = info.body_names or []
        site_names = info.site_names or []

        body_perm = self._build_swap_perm(body_names, sym.left_right_prefixes)
        site_perm = self._build_swap_perm(site_names, sym.left_right_prefixes)

        # Helper to mirror arrays with shape [N, n, C] by permuting n and flipping specified axes
        def mirror_vec(arrs: List[np.ndarray], perm: List[int], flip_axes: List[int], C: int, flip_ang_z: bool = False, is_cvel: bool = False):
            out = []
            for A in arrs:
                # A: [N, n, C]
                B = A[:, perm, :].copy()
                for ax in flip_axes:
                    if ax < C:
                        B[..., ax] *= -1.0
                if is_cvel:
                    # cvel: [lin(0..2), ang(3..5)]; flip yaw(z) sign
                    B[..., 5] *= -1.0  # ang.z
                out.append(B)
            return out

        # Bodies
        if "xpos" in self.modalities and body_perm:
            # flip lateral axis (e.g., y=1)
            self.modalities["xpos"].extend(mirror_vec(self.modalities["xpos"], body_perm, sym.lateral_axes, 3))
        if "subtree_com" in self.modalities and body_perm:
            self.modalities["subtree_com"].extend(mirror_vec(self.modalities["subtree_com"], body_perm, sym.lateral_axes, 3))
        if "cvel" in self.modalities and body_perm:
            # flip lin lateral axes and ang.z
            out = []
            for A in self.modalities["cvel"]:
                B = A[:, body_perm, :].copy()
                for ax in sym.lateral_axes:
                    if ax < 3:
                        B[..., ax] *= -1.0  # linear part
                B[..., 5] *= -1.0  # ang.z
                out.append(B)
            self.modalities["cvel"].extend(out)

        # Sites
        if "site_xpos" in self.modalities and site_perm:
            self.modalities["site_xpos"].extend(mirror_vec(self.modalities["site_xpos"], site_perm, sym.lateral_axes, 3))

        # Derived base_* and contacts (no permutation dimension)
        if "base_lin" in self.modalities:
            out = []
            for A in self.modalities["base_lin"]:
                B = A.copy()
                for ax in sym.lateral_axes:
                    if ax < 3:
                        B[..., ax] *= -1.0
                out.append(B)
            self.modalities["base_lin"].extend(out)
        if "base_ang" in self.modalities:
            out = []
            for A in self.modalities["base_ang"]:
                B = A.copy()
                # flip yaw-rate sign
                B[..., 2] *= -1.0
                out.append(B)
            self.modalities["base_ang"].extend(out)
        if "contacts" in self.modalities and self.contacts_names:
            # Swap left/right order by rebuilding using perm from site names
            site_idx = [site_names.index(n) for n in self.contacts_names if n in site_names]
            contact_perm = [site_idx.index(p) for p in [site_perm[i] for i in site_idx]]
            out = []
            for A in self.modalities["contacts"]:
                B = A[:, contact_perm].copy()
                out.append(B)
            self.modalities["contacts"].extend(out)

    @staticmethod
    def _build_swap_perm(names: List[str], lr_prefixes: List[tuple[str, str]]) -> List[int]:
        """
        Build a permutation that swaps left/right items for a name list.
        If no partner is found, keep index unchanged.
        """
        n = len(names)
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        perm = list(range(n))
        for i, nm in enumerate(names):
            partner = None
            for lpre, rpre in lr_prefixes:
                if nm.startswith(lpre):
                    cand = nm.replace(lpre, rpre, 1)
                    if cand in name_to_idx:
                        partner = name_to_idx[cand]
                        break
                if nm.startswith(rpre):
                    cand = nm.replace(rpre, lpre, 1)
                    if cand in name_to_idx:
                        partner = name_to_idx[cand]
                        break
            if partner is not None:
                perm[i] = partner
        return perm

    def _resolve_manager(self):
        # Resolve using TrajectoryInfo; pass contacts_names for site-based contacts mapping if needed
        meta = {"contacts_names": self.contacts_names} if self.contacts_names else None
        self.manager.resolve(self.trajs[0].info, self.device, meta=meta)

    @property
    def observation_dim(self) -> int:
        assert self.manager.catalog is not None
        return self.manager.catalog.out_dim

    def _sample_windows(self, batch_size: int, window_size_max: int) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Sample aligned windows of length T=window_size_max for every available modality.
        Each batch row can come from a different trajectory and time index; time alignment across modalities
        is done by taking the same [start:start+T] slice.

        Returns:
          windows: dict of source -> [B, T, D]
          dt: seconds per frame (1/frequency)
        """
        B, T = batch_size, window_size_max
        windows: Dict[str, torch.Tensor] = {}
        dt = 1.0 / self.frequency
        for src, per_traj in self.modalities.items():
            if not per_traj:
                continue
            xs = []
            for _ in range(B):
                tid = np.random.randint(len(per_traj))
                X = per_traj[tid]  # [N,D] or [N,n,C]; we flatten second dim if needed
                if X.ndim == 3:
                    N, n, C = X.shape
                    Xf = X.reshape(N, n * C)
                else:
                    Xf = X
                    N = Xf.shape[0]
                if N < T:
                    pad = np.repeat(Xf[:1], T - N, axis=0)
                    seg = np.concatenate([pad, Xf], axis=0)
                else:
                    start = np.random.randint(0, N - T + 1)
                    seg = Xf[start:start + T]
                xs.append(torch.from_numpy(seg).to(self.device, dtype=torch.float32))
            windows[src] = torch.stack(xs, dim=0)
        return windows, dt

    def feed_forward_generator(self, num_batches: int, batch_size: int):
        """
        Generator yielding expert transition pairs (s, s_next) using sliding windows.
        Windows are sampled with size Wmax+1 to produce consecutive pairs by shifting by 1.

        Yields:
          s: [B, F], s_next:[B, F]
        """
        Wmax = max(t.cfg.window_size for t in self.manager.catalog.terms)
        for _ in range(num_batches):
            windows, dt = self._sample_windows(batch_size, Wmax + 1)
            s = self.manager.compute({k: v[:, :-1, :] for k, v in windows.items()}, dt)
            s_next = self.manager.compute({k: v[:, 1:, :] for k, v in windows.items()}, dt)
            yield s, s_next

    # ---- utilities for derived signals ----

    @staticmethod
    def derive_base_from_qvel(qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility to derive base linear and angular velocities from qvel.

        Args:
          qvel: [N, Dq] with a free joint at head (at least 6-dim).

        Returns:
          (base_lin, base_ang): each [N, 3].

        Used by: _derive_convenience_modalities() and can be unit-tested independently.
        """
        if qvel.shape[1] < 6:
            N = qvel.shape[0]
            return np.zeros((N, 3), dtype=qvel.dtype), np.zeros((N, 3), dtype=qvel.dtype)
        return qvel[:, 0:3], qvel[:, 3:6]

    @staticmethod
    def derive_contacts_from_site_z(z: np.ndarray, z_threshold: float, hysteresis: float) -> np.ndarray:
        """
        Utility to derive binary contact signals from site_z(t) height.

        Args:
          z: [N, K] site vertical positions.
          z_threshold: threshold below which a site is considered in contact.
          hysteresis: Schmitt-trigger band to reduce flicker; we use a running state:
              - if prev == 1, new contact persists until z > z_threshold + hysteresis
              - if prev == 0, new contact starts when z <= z_threshold

        Returns:
          contacts: float32 array [N, K] with 0.0 or 1.0.

        Used by: _derive_convenience_modalities() and unit tests.
        """
        N, K = z.shape
        out = np.zeros((N, K), dtype=np.float32)
        if N == 0:
            return out
        out[0] = (z[0] <= z_threshold).astype(np.float32)
        for t in range(1, N):
            prev = out[t - 1]
            curr = prev.copy()
            curr[z[t] <= z_threshold] = 1.0
            curr[z[t] > (z_threshold + hysteresis)] = 0.0
            out[t] = curr
        return out

    @staticmethod
    def _default_contact_sites(site_names: List[str]) -> List[str]:
        candidates = []
        for s in site_names:
            s_lower = s.lower()
            if ("foot" in s_lower) or ("toe" in s_lower) or ("heel" in s_lower):
                candidates.append(s)
        return candidates or [site_names[0]]