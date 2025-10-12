from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable

import torch
import numpy as np

from mjlab.amp.config import AmpFeatureSetCfg, FeatureTermCfg, JointSelectionCfg, Aggregator, PreDiff
from mjlab.utils.dataset.traj_class import TrajectoryInfo


@dataclass
class FeatureCatalog:
    terms: dict[str, FeatureTerm]
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


# Classic Savitzky–Golay coefficients for polynomial order 2 and window size 5.
# This is a "tiny" smoother that preserves local quadratic trends while denoising.
_SAVGOL_W5_POLY2 = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0


def savgol_smooth(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Savitzky–Golay smoothing across time to each channel independently.

    Args:
      x: [B, T, C] time-series batch.

    Returns:
      Smoothed tensor with same shape. If T < 5, returns input unchanged.

    Used by: FeatureTerm.compute() before applying time differences.
    """
    if x.shape[1] < 5 or x.shape[2] == 0:
        return x
    kernel = _SAVGOL_W5_POLY2.to(x.device, x.dtype).view(1, 1, -1)
    x_perm = x.permute(0, 2, 1)  # [B, C, T]
    sm = torch.nn.functional.conv1d(x_perm, kernel.expand(x_perm.shape[1], 1, -1), padding=2, groups=x_perm.shape[1])
    return sm.permute(0, 2, 1)


def time_deriv(x: torch.Tensor) -> torch.Tensor:
    """
    First-order temporal difference along T with same-length output via front padding.

    Args:
      x: [B, T, C].

    Returns:
      dx: [B, T, C] with dx[:, t, :] = x[:, t, :] - x[:, t-1, :] and dx[:, 0, :] == dx[:, 1, :].

    Used by: pre_diff_apply().
    """
    if x.shape[2] == 0:
        return x
    dx = x[:, 1:, :] - x[:, :-1, :]
    return torch.nn.functional.pad(dx, (0, 0, 1, 0), mode="replicate")


def pre_diff_apply(x: torch.Tensor, mode: PreDiff) -> torch.Tensor:
    """
    Apply pre-differentiation (velocity/acceleration/jerk) after smoothing.

    Args:
      x: [B, T, C].
      mode: "none" | "velocity" | "acceleration" | "jerk".

    Returns:
      [B, T, C] with appropriate order of differences applied.

    Used by: FeatureTerm.compute().
    """
    if mode == "none" or x.shape[2] == 0:
        return x
    x1 = time_deriv(x)
    if mode == "velocity":
        return x1
    x2 = time_deriv(x1)
    if mode == "acceleration":
        return x2
    x3 = time_deriv(x2)
    return x3  # jerk


def rms(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Root-mean-square along dim.

    Args:
      x: input tensor.
      dim: dimension to reduce.

    Returns:
      sqrt(mean(x^2)) with epsilon for stability.

    Used by: FeatureTerm.compute() when "rms" aggregator is set.
    """
    return torch.sqrt(torch.mean(x * x, dim=dim) + 1e-12)


def spectral_entropy(power: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Shannon spectral entropy of a power spectrum.

    Args:
      power: non-negative spectrum (e.g., |rFFT|^2) [B, Freq, C].
      dim: frequency dimension (typically 1).

    Returns:
      Entropy normalized to [0, 1] by dividing with log(N).

    Used by: FeatureTerm.compute() for "spectral_entropy".
    """
    p = power / (power.sum(dim=dim, keepdim=True) + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum(dim=dim)
    n = power.shape[dim]
    return ent / np.log(n + 1e-12)


@dataclass
class FeatureTerm:
    """
    Runtime object for a configured feature term. It binds:
      - cfg: FeatureTermCfg
      - indices: LongTensor of channel indices into the source modality
      - channel_selector: callable mapping a [B, T, D] modality tensor to [B, T, C_selected]

    Used by: FeatureManager.compute() to produce descriptor blocks per term.
    """
    cfg: FeatureTermCfg
    indices: torch.Tensor
    channel_selector: Callable[[torch.Tensor], torch.Tensor]

    def compute(self, series: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute this term on the provided modality window.

        Args:
          series: [B, T, D] modality window (FeatureManager already trimmed T to cfg.window_size).
          dt: seconds per frame for frequency computations.

        Returns:
          [B, F_term] descriptor chunk.
        """
        x = self.channel_selector(series)
        if self.cfg.savgol != "none":
            x = savgol_smooth(x)
        if self.cfg.pre_diff != "none":
            x = pre_diff_apply(x, self.cfg.pre_diff)

        feats = []
        for agg in self.cfg.aggregators:
            if agg == "mean":
                feats.append(x.mean(dim=1))
            elif agg == "rms":
                feats.append(rms(x, dim=1))
            elif agg == "std":
                feats.append(x.std(dim=1))
            elif agg == "max":
                feats.append(x.max(dim=1).values)
            elif agg == "min":
                feats.append(x.min(dim=1).values)
            elif agg == "flatten":
                feats.append(x.reshape(x.shape[0], -1))
            elif agg in ("dominant_freq", "spectral_entropy"):
                fft_vals = torch.fft.rfft(x, dim=1)
                power = (fft_vals.real**2 + fft_vals.imag**2)
                if agg == "dominant_freq":
                    idx = power.argmax(dim=1)  # [B,C]
                    T = x.shape[1]
                    hz = idx * (1.0 / (T * dt))
                    feats.append(hz)
                else:
                    se = spectral_entropy(power, dim=1)
                    feats.append(se)
            else:
                raise ValueError(f"Unknown aggregator: {agg}")
        return torch.cat(feats, dim=-1)


def build_selector(
    source: str,
    indices: torch.Tensor,
    channels: List[str],
    group_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a selector that supports semantic channel tags and vector norms.

    Args:
      source: SignalSource name (informational).
      indices: channel indices within the flattened modality vector (size D).
               For vector modalities (e.g., cvel), 'indices' is concatenated group blocks.
      channels: semantic hints for selection:
                - "lin", "lin.x", "lin.y", "lin.z", "lin.speed"
                - "ang", "ang.x", "ang.y", "ang.z", "ang.speed"
                - "speed" (norm of entire group), "scalar" (pass-through)
      group_size: number of scalars per group (e.g., 6 for cvel, 3 for xpos, 1 for qvel).

    Returns:
      Callable mapping [B, T, D] -> [B, T, C_out].

    Notes:
      - For scalar modalities (group_size=1), this function simply gathers 'indices'.
      - For vector modalities, this function reshapes gathered channels to [B, T, G, group_size] and
        implements selections/norms per group, finally flattening G and component axes.
    """
    indices = indices.view(-1)
    if group_size <= 1:
        def _sel_scalar(x: torch.Tensor) -> torch.Tensor:
            return x[..., indices]
        return _sel_scalar

    def _sel_vec(x: torch.Tensor) -> torch.Tensor:
        xg = x[..., indices]  # [B,T,G*group_size]
        G = xg.shape[-1] // group_size
        if G * group_size != xg.shape[-1]:
            # fallback: not an integer number of groups; return gathered scalars
            return xg
        xg = xg.view(xg.shape[0], xg.shape[1], G, group_size)  # [B,T,G,S]
        outs = []
        # Helper: select components
        def get_comp(tag: str) -> torch.Tensor:
            # Supports lin/ang split for S==6; otherwise use first 3 for "lin", remainder for "ang"
            if tag == "lin":
                return xg[..., : min(3, group_size)]
            if tag == "ang":
                if group_size >= 6:
                    return xg[..., 3:6]
                # if we don't have explicit ang part, fallback to last 3 if present
                return xg[..., max(0, group_size - 3):group_size]
            # Handle .speed cases before splitting on axis components
            if tag in ("lin.speed", "ang.speed"):
                arr = get_comp(tag.split(".")[0])
                return torch.linalg.norm(arr, dim=-1, keepdim=True)
            if tag.startswith("lin.") or tag.startswith("ang."):
                base, axis = tag.split(".")
                arr = get_comp(base)
                comp = {"x": 0, "y": 1, "z": 2}[axis]
                if arr.shape[-1] <= comp:
                    # if missing, fallback to zeros
                    return torch.zeros(arr.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
                return arr[..., comp:comp + 1]
            if tag in ("speed", "norm"):
                return torch.linalg.norm(xg, dim=-1, keepdim=True)
            if tag in ("scalar",):
                # pass-through (flatten group)
                return xg
            # unknown tag: ignore, return empty
            return torch.zeros(xg.shape[:-1] + (0,), device=x.device, dtype=x.dtype)

        if not channels:
            outs.append(xg)  # default: pass-through groups
        else:
            for ch in channels:
                outs.append(get_comp(ch))

        y = torch.cat(outs, dim=-1)  # [B,T,G,Csel]
        return y.reshape(y.shape[0], y.shape[1], -1)  # flatten groups

    return _sel_vec


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
        
        self._info = info_or_map
        terms = self._build_terms_from_info(info_or_map, device)
        
        # Compute actual dimension by doing a dummy forward pass
        out_dim = self._compute_actual_dim(terms, device)
        
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

    def _compute_actual_dim(self, terms: List[FeatureTerm], device: torch.device) -> int:
        """
        Compute actual output dimension by running a dummy forward pass.
        
        Args:
          terms: list of FeatureTerm objects
          device: torch device
          
        Returns:
          actual total output dimension
        """
        total_dim = 0
        for term in terms:
            # Create dummy input with shape [1, window_size, D_source] 
            # where D_source is inferred from max index + 1 (or 1 if no indices)
            if term.indices.numel() > 0:
                D_source = int(term.indices.max().item()) + 1
            else:
                D_source = 1
            dummy_input = torch.zeros(1, term.cfg.window_size, D_source, device=device, dtype=torch.float32)
            # Call selector to get the actual selected dimensions
            dummy_selected = term.channel_selector(dummy_input)
            # Now compute features on selected data
            dummy_output = self._compute_term_features(dummy_selected, term.cfg, dt=0.02)
            total_dim += dummy_output.shape[-1]
        return total_dim
    
    def _compute_term_features(self, x: torch.Tensor, cfg: FeatureTermCfg, dt: float) -> torch.Tensor:
        """Helper to compute features without calling the full FeatureTerm.compute (to avoid circular dependency)."""
        if cfg.savgol != "none":
            x = savgol_smooth(x)
        if cfg.pre_diff != "none":
            x = pre_diff_apply(x, cfg.pre_diff)

        feats = []
        for agg in cfg.aggregators:
            if agg == "mean":
                feats.append(x.mean(dim=1))
            elif agg == "rms":
                feats.append(rms(x, dim=1))
            elif agg == "std":
                feats.append(x.std(dim=1))
            elif agg == "max":
                feats.append(x.max(dim=1).values)
            elif agg == "min":
                feats.append(x.min(dim=1).values)
            elif agg == "flatten":
                feats.append(x.reshape(x.shape[0], -1))
            elif agg in ("dominant_freq", "spectral_entropy"):
                fft_vals = torch.fft.rfft(x, dim=1)
                power = (fft_vals.real**2 + fft_vals.imag**2)
                if agg == "dominant_freq":
                    idx = power.argmax(dim=1)
                    T = x.shape[1]
                    hz = idx * (1.0 / (T * dt))
                    feats.append(hz)
                else:
                    se = spectral_entropy(power, dim=1)
                    feats.append(se)
            else:
                raise ValueError(f"Unknown aggregator: {agg}")
        return torch.cat(feats, dim=-1)


    # ---- internal: resolve using TrajectoryInfo ----

    def _build_terms_from_info(self, info: TrajectoryInfo, device: torch.device):
        terms: list[FeatureTerm] = []
        for k, tcfg in self.cfg.terms.items():
            idx = self._resolve_indices_from_info(tcfg, info, device)
            selector = build_selector(tcfg.source, idx, tcfg.channels, _GROUP_SIZE.get(tcfg.source, 1))
            term = FeatureTerm(cfg=tcfg, indices=idx, channel_selector=selector)
            terms.append(term)
        return terms

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
                assert idxs, f"No valid joints found in selection {sel.joints}"
            else:
                # ALL qpos
                idxs = [torch.as_tensor(info.joint_name2ind_qpos[j], dtype=torch.long) for j in info.joint_names]
            return concat(idxs)

        if src == "qvel":
            if sel.joints:
                idxs = [torch.as_tensor(info.joint_name2ind_qvel[j], dtype=torch.long) for j in sel.joints if j in info.joint_name2ind_qvel]
                assert idxs, f"No valid joints found in selection {sel.joints}"
            else: # all joints if not specified
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