"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Dict, Any

import hashlib
import importlib
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class AMDDemoCfg:
  """Configuration for AMP demo provider."""
  enabled: bool = False
  obs_key: str = "amp_state"
  logical_target_name: str = "robot"
  exclude_terms: List[str] = field(default_factory=lambda: ["actions", "command"])
  use_cache: bool = True
  cache_dir: Optional[Path] = None
  cache_filename: Optional[str] = None
  prefer_cuda: bool = True

class AMPDemoProvider:
  """Precomputes AMP demo states from Scene records and serves random samples.

  Workflow:
    - Iterate over all record entities registered in env.scene (e.g., via SceneCfg.records).
    - For each record, stream frames through Entity.frames() and compute the observation dict
      using env.observation_manager.compute_on_record(record_entity_name=..., logical_target_name=...).
    - Extract amp_state tensor by key (obs_key) and collect rows (shape [1, F]) across frames and records.
    - Exposes __call__(n, device) -> Tensor [n, F] to uniformly sample stored demos during AMP training.

  Notes:
    - This uses the ObservationManager's scene aliasing path to keep observation functions unchanged.
    - Sliding windows are reset at the start of every record to avoid cross-clip leakage.
    - Noise is disabled during offline feature computation for determinism.
    - Terms like actions/commands can be excluded (zero-filled) via exclude_terms to ensure parity.

  Caching:
    - Supports disk caching to a .pt file with a fingerprint that depends on:
        * record file paths + (size, mtime) metadata,
        * observation settings (obs_key, logical_target_name, exclude_terms),
        * observation group schema (term names and output dims).
    - Default cache location is the directory containing the environment cfg class module.
    - On load: if cache exists and fingerprint matches, dataset is loaded directly.
    - Dataset is pre-transferred to CUDA if available; otherwise pinned on CPU for faster H2D copies.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    obs_key: str = "amp_state",
    logical_target_name: str = "robot",
    exclude_terms: Optional[Iterable[str]] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    cache_filename: Optional[str] = None,
    prefer_cuda: bool = True,
  ) -> None:
    self.env = env
    self.obs_key = obs_key
    self.logical_target_name = logical_target_name
    self.exclude_terms = list(exclude_terms) if exclude_terms is not None else []
    self.use_cache = use_cache
    self._dataset: Optional[torch.Tensor] = None  # [N, F]

    # Device preference
    self._prefer_cuda = prefer_cuda and torch.cuda.is_available()
    # Resolve default cache directory to where the env cfg class is defined
    self._cache_dir = cache_dir or self._default_cache_dir_for_env_cfg()
    self._cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache filename
    env_cfg_name = self.env.cfg.__class__.__name__
    self._cache_file = self._cache_dir / (cache_filename or f"amp_demos_{env_cfg_name}_{self.obs_key}.pt")

  # -------------------------------
  # Public API
  # -------------------------------
  def build(self) -> None:
    """Precompute and cache demo AMP states from all record entities in the scene."""
    # Attempt cache load
    if self.use_cache and self._try_load_cache():
      return

    # Otherwise build from scratch
    record_names: List[str] = self._collect_record_names()
    assert len(record_names) > 0, "No record entities found in scene to build AMP demos."

    amp_rows: List[torch.Tensor] = []
    device = torch.device(self.env.device)

    # Ensure observation manager exclusion knobs set for offline computation
    if len(self.exclude_terms) > 0:
      self.env.observation_manager.set_offline_excluded_terms(self.exclude_terms)

    for rec_name in record_names:
      # Try to resolve the record; skip if missing
      try:
        rec_ent: Entity = self.env.scene[rec_name]
      except Exception:
        continue
      print(f"AMPDemoProvider: processing record '{rec_name}' for AMP demos...")
      n_frames = getattr(rec_ent, "_rec_len", None) or getattr(rec_ent, "_traj_len", None)
      for _ in rec_ent.frames(0, n_frames):
        obs = self.env.observation_manager.compute_on_record(
          record_entity_name=rec_name,
          logical_target_name=self.logical_target_name,
          disable_noise=True,
          exclude_terms=self.exclude_terms,
        )
        # Extract amp_state group
        assert self.obs_key in obs, (
          f"AMP obs_key '{self.obs_key}' missing from observation dict; "
          f"ensure your ObservationCfg defines this group and concatenate_terms as required."
        )
        amp_group = obs[self.obs_key]
        # Expected shape [1, F] for offline nworld=1
        if isinstance(amp_group, torch.Tensor):
          row = amp_group.reshape(amp_group.shape[0], -1)[0]  # [F]
        else:
          parts = []
          for name in self.env.observation_manager._group_obs_term_names[self.obs_key]:
            parts.append(amp_group[name].reshape(1, -1))
          row = torch.cat(parts, dim=1)[0]
        amp_rows.append(row.detach().to(device=device, dtype=torch.float32))

    assert len(amp_rows) > 0, "No AMP demo frames were collected from records."
    data = torch.stack(amp_rows, dim=0)  # [N, F]

    # Move to preferred memory/device
    data = self._prepare_storage(data)

    # Assign
    self._dataset = data

    # Save cache
    if self.use_cache:
      self._save_cache()

  def __len__(self) -> int:
    return 0 if self._dataset is None else int(self._dataset.shape[0])

  def __call__(self, n: int, device: torch.device) -> torch.Tensor:
    """Uniformly sample n AMP states from the cached dataset.

    Args:
      n: Number of samples to draw.
      device: Target device for returned tensor.

    Returns:
      Tensor of shape [n, F].
    """
    assert self._dataset is not None and self._dataset.numel() > 0, (
      "AMPDemoProvider not built or empty. Call .build() after the env is initialized."
    )
    N = self._dataset.shape[0]
    n = max(1, int(n))
    # Index on the dataset's device to keep it fast, then .to(device) if needed.
    idx = torch.randint(low=0, high=N, size=(n,), device=self._dataset.device)
    out = self._dataset[idx]
    if out.device != device:
      out = out.to(device, non_blocking=True)
    return out

  # -------------------------------
  # Cache support
  # -------------------------------
  def _default_cache_dir_for_env_cfg(self) -> Path:
    """Default cache directory: where the environment cfg class module is defined."""
    # env.cfg.__class__.__module__ -> module with __file__
    mod_name = self.env.cfg.__class__.__module__
    try:
      mod = importlib.import_module(mod_name)
      mod_file = Path(getattr(mod, "__file__", ""))
      if mod_file.exists():
        return mod_file.parent
    except Exception:
      pass
    # Fallback to current working dir if module path not resolvable
    return Path.cwd()

  def _prepare_storage(self, data: torch.Tensor) -> torch.Tensor:
    """Place dataset on CUDA if available (by default), else pin on CPU."""
    if self._prefer_cuda:
      try:
        return data.to("cuda", non_blocking=True)
      except Exception:
        # If CUDA move fails, fall back to pinned CPU
        pass
    # Pin memory if possible on CPU
    try:
      return data.pin_memory()
    except Exception:
      return data.contiguous()

  def _collect_record_names(self) -> List[str]:
    names: List[str] = []
    if hasattr(self.env.scene, "_records"):
      names.extend(list(getattr(self.env.scene, "_records").keys()))
    # Also include names from cfg.scene.records if available
    try:
      cfg_recs = getattr(self.env.cfg.scene, "records", [])
      for rc in cfg_recs:
        if rc.name is not None:
          names.append(rc.name)
    except Exception:
      pass
    # Dedup preserving order
    seen = set()
    return [r for r in names if not (r in seen or seen.add(r))]

  def _get_record_file_meta(self) -> List[Dict[str, Any]]:
    """Collect per-record file metadata to include in fingerprint."""
    metas: List[Dict[str, Any]] = []
    # Prefer runtime-loaded record entities (paths reflect actual npz used)
    if hasattr(self.env.scene, "_records"):
      for name, ent in getattr(self.env.scene, "_records").items():
        p = None
        try:
          npz_info = getattr(ent, "_rec_npz", None)
          if npz_info and isinstance(npz_info, dict):
            p = npz_info.get("path", None)
        except Exception:
          p = None
        metas.append(self._file_meta(name, p))
    # Also include configured records
    try:
      for rc in getattr(self.env.cfg.scene, "records", []):
        metas.append(self._file_meta(getattr(rc, "name", None), getattr(rc, "path", None)))
    except Exception:
      pass
    # Deduplicate by (name, path)
    dedup = {}
    for m in metas:
      key = (m.get("name", ""), m.get("path", ""))
      if key not in dedup:
        dedup[key] = m
    return list(dedup.values())

  def _file_meta(self, name: Optional[str], path_like: Optional[os.PathLike | str]) -> Dict[str, Any]:
    p_str = str(path_like) if path_like is not None else ""
    meta: Dict[str, Any] = {"name": name or "", "path": p_str, "size": 0, "mtime": 0.0}
    try:
      if p_str:
        st = os.stat(p_str)
        meta["size"] = int(st.st_size)
        meta["mtime"] = float(st.st_mtime)
    except Exception:
      pass
    return meta

  def _obs_schema_meta(self) -> Dict[str, Any]:
    """Observation schema (term names and dims) for the AMP group."""
    om = self.env.observation_manager
    names = om._group_obs_term_names.get(self.obs_key, [])
    dims = om._group_obs_term_dim.get(self.obs_key, [])
    return {"group": self.obs_key, "names": list(names), "dims": [tuple(d) for d in dims]}

  def _fingerprint(self) -> str:
    """Compute a stable fingerprint string for cache validation."""
    payload = {
      "env_cfg_module": self.env.cfg.__class__.__module__,
      "env_cfg_name": self.env.cfg.__class__.__name__,
      "obs_key": self.obs_key,
      "logical_target_name": self.logical_target_name,
      "exclude_terms": sorted(self.exclude_terms),
      "records": self._get_record_file_meta(),
      "schema": self._obs_schema_meta(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

  def _try_load_cache(self) -> bool:
    """Try to load dataset from cache; returns True if loaded and valid."""
    try:
      if not self._cache_file.exists():
        return False
      obj = torch.load(self._cache_file, map_location="cpu")
      if not isinstance(obj, dict):
        return False
      cached_fp = obj.get("fingerprint", None)
      data = obj.get("dataset", None)
      if cached_fp is None or data is None or not isinstance(data, torch.Tensor):
        return False
      cur_fp = self._fingerprint()
      if str(cached_fp) != str(cur_fp):
        return False
      # Prepare storage (move to cuda or pin)
      data = self._prepare_storage(data)
      self._dataset = data
      return True
    except Exception:
      return False

  def _save_cache(self) -> None:
    """Save dataset and fingerprint to cache .pt file."""
    if self._dataset is None:
      return
    try:
      payload = {"fingerprint": self._fingerprint(), "dataset": self._dataset.detach().to("cpu")}
      # Save CPU copy to keep cache portable; device placement is handled on load
      torch.save(payload, self._cache_file)
    except Exception:
      # Ignore cache save failures
      pass


def wire_amp_demo_provider(
  env: "ManagerBasedRlEnv",
  *,
  obs_key: str = "amp_state",
  logical_target_name: str = "robot",
  exclude_terms: Optional[Iterable[str]] = ("actions", "command"),
  use_cache: bool = True,
  cache_dir: Optional[Path] = None,
  cache_filename: Optional[str] = None,
  prefer_cuda: bool = True,
) -> AMPDemoProvider:
  """Construct, build, and attach an AMPDemoProvider to the env.

  - Computes offline AMP observations (obs_key='amp_state') from Scene records.
  - Excludes action- and command-dependent terms from offline computation.
  - Exposes env.sample_amp_demos(n, device) for rsl-rl AMP integration.
  - Supports disk caching with fingerprinting; pre-places dataset on CUDA if available, else pinned CPU.

  Returns:
    The constructed and built AMPDemoProvider instance.
  """
  provider = AMPDemoProvider(
    env=env,
    obs_key=obs_key,
    logical_target_name=logical_target_name,
    exclude_terms=list(exclude_terms) if exclude_terms is not None else [],
    use_cache=use_cache,
    cache_dir=cache_dir,
    cache_filename=cache_filename,
    prefer_cuda=prefer_cuda,
  )
  provider.build()
  # Attach to env and expose callable for AMP resolve
  setattr(env, "_amp_demo_provider", provider)
  setattr(env, "sample_amp_demos", lambda n, device=None: provider(n, device=torch.device(env.device) if device is None else device))
  return provider