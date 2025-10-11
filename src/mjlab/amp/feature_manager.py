from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from mjlab.amp.config import AmpFeatureSetCfg, FeatureTermCfg, JointSelectionCfg
from mjlab.amp.feature_terms import FeatureTerm, build_selector

try:
    # TrajectoryInfo type from OpenTrack-style dataset (optional import)
    from src.utils.dataset.traj_class import TrajectoryInfo
except Exception:
    TrajectoryInfo = object  # type: ignore


@dataclass
class FeatureCatalog:
    terms: List[FeatureTerm]
    out_dim: int


_GROUP_SIZE = {
    "qpos": 1,
    "qvel": 1,
    "xpos": 3,
    "xquat": 4,
    "cvel": 6,
    "subtree_com": 3,
    "site_xpos": 3,
    "site_xmat": 9,
    "base_lin": 3,
    "base_ang": 3,
    "contacts": 1,
}


class FeatureManager:
    """
    FeatureManager composes feature terms into a single vector.
    It can resolve index mappings either from:
      - TrajectoryInfo (preferred): builds per-joint/body/site indices for each modality; or
      - an explicit name_to_index dict (backward compatible): maps "source:ALL" -> indices, etc.

    Typical usage:
      1) manager = FeatureManager(feature_set)
      2) manager.resolve(traj_info, device)  # or explicit dict
      3) feats = manager.compute(windows_dict, dt)
    """
    def __init__(self, cfg: AmpFeatureSetCfg):
        self.cfg = cfg
        self.catalog: FeatureCatalog | None = None
        self._info: Optional[TrajectoryInfo] = None
        self._meta: Dict[str, object] = {}

    def resolve(
        self,
        info_or_map: Union[TrajectoryInfo, Dict[str, torch.Tensor]],
        device: torch.device,
        meta: Optional[Dict[str, object]] = None,
    ) -> FeatureCatalog:
        """
        Resolve feature terms into a FeatureCatalog.

        Args:
          info_or_map: either a TrajectoryInfo instance to auto-build indices, or a dict mapping "source:key" to indices.
          device: torch device to place index tensors.
          meta: optional extra metadata; currently supports {"contacts_names": List[str]} for contacts channel ordering.

        Returns:
          FeatureCatalog with assembled terms and total output dimension.
        """
        self._meta = meta or {}
        if isinstance(info_or_map, dict):
            # legacy path: use the provided indices directly
            name_to_index = {k: v.to(device) for k, v in info_or_map.items()}
            terms, out_dim = self._build_terms_from_map(name_to_index, device)
        else:
            self._info = info_or_map
            terms, out_dim = self._build_terms_from_info(info_or_map, device)
        self.catalog = FeatureCatalog(terms=terms, out_dim=out_dim)
        return self.catalog

    def compute(self, windows: Dict[str, torch.Tensor], dt: float) -> torch.Tensor:
        assert self.catalog is not None
        parts = []
        for term in self.catalog.terms:
            x = windows[term.cfg.source]  # [B,T,D]
            T = x.shape[1]
            W = term.cfg.window_size
            xw = x[:, T - W:T, :]
            parts.append(term.compute(xw, dt))
        return torch.cat(parts, dim=-1)

    # ---- internal: resolve from explicit map ----

    def _build_terms_from_map(self, name_to_index: Dict[str, torch.Tensor], device: torch.device):
        terms: List[FeatureTerm] = []
        out_dim = 0
        for tcfg in self.cfg.terms:
            key = f"{tcfg.source}:ALL"
            if key not in name_to_index:
                raise ValueError(f"Missing index mapping for {key}")
            idx = name_to_index[key]
            selector = build_selector(tcfg.source, idx, tcfg.channels, _GROUP_SIZE.get(tcfg.source, 1))
            term = FeatureTerm(cfg=tcfg, indices=idx, channel_selector=selector)
            terms.append(term)
            out_dim += self._estimate_term_out_dim(tcfg, idx.numel())
        return terms, out_dim

    # ---- internal: resolve using TrajectoryInfo ----

    def _build_terms_from_info(self, info: TrajectoryInfo, device: torch.device):
        terms: List[FeatureTerm] = []
        out_dim = 0
        for tcfg in self.cfg.terms:
            idx = self._resolve_indices_from_info(tcfg, info, device)
            selector = build_selector(tcfg.source, idx, tcfg.channels, _GROUP_SIZE.get(tcfg.source, 1))
            term = FeatureTerm(cfg=tcfg, indices=idx, channel_selector=selector)
            terms.append(term)
            out_dim += self._estimate_term_out_dim(tcfg, idx.numel(), group_size=_GROUP_SIZE.get(tcfg.source, 1))
        return terms, out_dim

    def _estimate_term_out_dim(self, tcfg: FeatureTermCfg, n_indices: int, group_size: int = 1) -> int:
        # Rough estimator (channels can change the per-group output; we conservatively keep n_indices for scalar,
        # and for vector cases multiply group count by selected components count).
        if "flatten" in tcfg.aggregators:
            return n_indices * tcfg.window_size
        # non-flatten aggregations collapse time:
        if group_size <= 1:
            comps_per_group = 1
        else:
            # channels may change components per group; approximate:
            if not tcfg.channels:
                comps_per_group = group_size
            else:
                # count scalar outputs contributed by channels
                comps_per_group = 0
                for ch in tcfg.channels:
                    if ch in ("lin", "ang"):
                        comps_per_group += 3
                    elif ch in ("lin.x", "lin.y", "lin.z", "ang.x", "ang.y", "ang.z", "lin.speed", "ang.speed", "speed", "norm"):
                        comps_per_group += 1
                    elif ch in ("scalar",):
                        comps_per_group += group_size
                    else:
                        comps_per_group += 0
            n_groups = max(1, n_indices // group_size)
            return n_groups * comps_per_group
        return n_indices

    def _resolve_indices_from_info(self, tcfg: FeatureTermCfg, info: TrajectoryInfo, device: torch.device) -> torch.Tensor:
        src = tcfg.source
        sel: JointSelectionCfg = tcfg.select

        def concat(arrs: List[torch.Tensor]) -> torch.Tensor:
            if not arrs:
                return torch.zeros(0, dtype=torch.long, device=device)
            return torch.cat(arrs).to(device)

        # Joints: qpos/qvel -> direct indices from info maps
        if src == "qpos":
            if sel.joints:
                idxs = [torch.as_tensor(info.joint_name2ind_qpos[j], dtype=torch.long) for j in sel.joints if j in info.joint_name2ind_qpos]
            else:
                # ALL qpos in order
                idxs = [torch.arange(info.model.njnt * 0) ]  # dummy, replaced below
                # flatten in order of joint_names
                idxs = [torch.as_tensor(info.joint_name2ind_qpos[j], dtype=torch.long) for j in info.joint_names]
            return concat(idxs)

        if src == "qvel":
            if sel.joints:
                idxs = [torch.as_tensor(info.joint_name2ind_qvel[j], dtype=torch.long) for j in sel.joints if j in info.joint_name2ind_qvel]
            else:
                idxs = [torch.as_tensor(info.joint_name2ind_qvel[j], dtype=torch.long) for j in info.joint_names]
            return concat(idxs)

        # Bodies: per-body vector blocks (xpos 3, xquat 4, cvel 6, subtree_com 3)
        if src in ("xpos", "xquat", "cvel", "subtree_com"):
            g = _GROUP_SIZE[src]
            body_names = info.body_names or []
            if sel.bodies:
                names = [b for b in sel.bodies if b in body_names]
            else:
                names = body_names
            idxs: List[torch.Tensor] = []
            for b in names:
                bi = int(info.body_name2ind[b][0])
                start = bi * g
                idxs.append(torch.arange(start, start + g, dtype=torch.long))
            return concat(idxs)

        # Sites: per-site vector blocks (site_xpos 3, site_xmat 9)
        if src in ("site_xpos", "site_xmat"):
            g = _GROUP_SIZE[src]
            site_names = info.site_names or []
            if sel.sites:
                names = [s for s in sel.sites if s in site_names]
            else:
                names = site_names
            idxs: List[torch.Tensor] = []
            for s in names:
                si = int(info.site_name2ind[s][0])
                start = si * g
                idxs.append(torch.arange(start, start + g, dtype=torch.long))
            return concat(idxs)

        # Derived sources (base_* and contacts) are already shaped as D=3 or D=K; select ALL or specific sites for contacts
        if src in ("base_lin", "base_ang"):
            return torch.arange(_GROUP_SIZE[src], dtype=torch.long, device=device)

        if src == "contacts":
            names = self._meta.get("contacts_names", None)
            if isinstance(names, list) and sel.sites:
                idxs = [torch.tensor([names.index(s)], dtype=torch.long) for s in sel.sites if s in names]
                return concat(idxs)
            # Fallback: ALL
            # contacts D is provided by windows dict, select all channels
            # We can only determine D at runtime; select a placeholder range and clip in selector
            # Instead, return empty here; FeatureManager.compute uses direct x[..., :] if idx empty -> fix by selecting 0..D-1 later.
            # To avoid ambiguity, return a single dummy index; the selector must handle this by ignoring indices when group_size=1 and idx empty.
            return torch.tensor([], dtype=torch.long, device=device)

        # Default: ALL for unknown source
        return torch.tensor([], dtype=torch.long, device=device)