"""Observation manager for computing observations."""

from typing import Sequence

import numpy as np
import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.dataclasses import get_terms
from mjlab.utils.noise import noise_cfg, noise_model
from mjlab.utils.sliding_window import SlidingWindow


class ObservationManager(ManagerBase):
  def __init__(self, cfg: object, env):
    self.cfg = cfg

    self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
    # Sliding window instances (per group -> per term)
    self._sliding_windows: dict[str, dict[str, SlidingWindow]] = {}
    # Map of per-term output dims (post-aggregation) and raw dims (pre-aggregation)
    self._group_obs_term_raw_dim: dict[str, list[tuple[int, ...]]] = {}

    super().__init__(env=env)
    
    # Compute final per-group dims for presentation/concatenation
    for group_name, group_term_dims in self._group_obs_term_dim.items():
      if self._group_obs_concatenate[group_name]:
        try:
          term_dims = torch.stack(
            [torch.tensor(dims, device="cpu") for dims in group_term_dims], dim=0
          )
          if len(term_dims.shape) > 1:
            if self._group_obs_concatenate_dim[group_name] >= 0:
              dim = self._group_obs_concatenate_dim[group_name] - 1
            else:
              dim = self._group_obs_concatenate_dim[group_name]
            dim_sum = torch.sum(term_dims[:, dim], dim=0)
            term_dims[0, dim] = dim_sum
            term_dims = term_dims[0]
          else:
            term_dims = torch.sum(term_dims, dim=0)
          self._group_obs_dim[group_name] = tuple(term_dims.tolist())
        except RuntimeError:
          raise RuntimeError(
            f"Unable to concatenate observation terms in group {group_name}."
          ) from None
      else:
        self._group_obs_dim[group_name] = group_term_dims

    self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

    # Initialize sliding windows (only for enabled terms), using RAW dims (pre-aggregation)
    self._sliding_windows = {g: {} for g in self._group_obs_term_names.keys()}
    for group_name in self._group_obs_term_names.keys():
      for idx, (term_name, term_cfg) in enumerate(
        zip(
          self._group_obs_term_names[group_name],
          self._group_obs_term_cfgs[group_name],
          strict=False,
        )
      ):
        if term_cfg.hist_window_size > 0:
          raw_feat_shape = torch.Size(self._group_obs_term_raw_dim[group_name][idx])
          self._sliding_windows[group_name][term_name] = SlidingWindow(
            num_envs=self._env.num_envs,
            feature_shape=raw_feat_shape,
            max_window_size=term_cfg.hist_window_size,
            device=torch.device(self._env.device),
          )

  def __str__(self) -> str:
    msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
    for group_name, group_dim in self._group_obs_dim.items():
      table = PrettyTable()
      table.title = f"Active Observation Terms in Group: '{group_name}'"
      if self._group_obs_concatenate[group_name]:
        table.title += f" (shape: {group_dim})"  # type: ignore
      table.field_names = ["Index", "Name", "Shape"]
      table.align["Name"] = "l"
      obs_terms = zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
        strict=False,
      )
      for index, (name, dims) in enumerate(obs_terms):
        tab_dims = tuple(dims)
        table.add_row([index, name, tab_dims])
      msg += table.get_string()
      msg += "\n"
    return msg

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []

    if self._obs_buffer is None:
      self.compute()
    assert self._obs_buffer is not None
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

    for group_name, _ in self.group_obs_dim.items():
      if not self.group_obs_concatenate[group_name]:
        buffers = obs_buffer[group_name]
        assert isinstance(buffers, dict)
        for name, term in buffers.items():
          terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
        continue

      idx = 0
      data = obs_buffer[group_name]
      assert isinstance(data, torch.Tensor)
      for name, shape in zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
        strict=False,
      ):
        data_length = np.prod(shape)
        term = data[env_idx, idx : idx + data_length]
        terms.append((group_name + "-" + name, term.cpu().tolist()))
        idx += data_length

    return terms

  @property
  def active_terms(self) -> dict[str, list[str]]:
    return self._group_obs_term_names

  @property
  def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
    return self._group_obs_dim

  @property
  def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
    return self._group_obs_term_dim

  @property
  def group_obs_concatenate(self) -> dict[str, bool]:
    return self._group_obs_concatenate

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    for _group_name, group_cfg in self._group_obs_class_term_cfgs.items():
      for term_cfg in group_cfg:
        if hasattr(term_cfg.func, "reset"):
          term_cfg.func.reset(env_ids=env_ids)
    for mod in self._group_obs_class_instances.values():
      mod.reset(env_ids=env_ids)

    for group_windows in self._sliding_windows.values():
      for win in group_windows.values():
        win.reset(env_ids=env_ids)
    return {}

  def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
    for group_name in self._group_obs_term_names:
      obs_buffer[group_name] = self.compute_group(group_name)
    self._obs_buffer = obs_buffer
    return obs_buffer

  def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
    group_term_names = self._group_obs_term_names[group_name]
    group_obs: dict[str, torch.Tensor] = {}
    obs_terms = zip(
      group_term_names, self._group_obs_term_cfgs[group_name], strict=False
    )
    for term_name, term_cfg in obs_terms:
      # 1) Compute raw per-step observation from source function.
      # Always pass params at runtime (per request).
      obs_raw: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()

      # 2) If sliding window enabled, push raw sample and optionally aggregate via hist_func.
      sw = self._sliding_windows[group_name].get(term_name, None)
      if sw is not None:
        sw.push(obs_raw)
        if term_cfg.hist_func is not None:
          obs = term_cfg.hist_func(self._env, sw, **term_cfg.params)
        else:
          obs = obs_raw
      else:
        obs = obs_raw

      # 3) Noise / corruption.
      if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
        obs = term_cfg.noise.apply(obs)
      elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
        obs = self._group_obs_class_instances[term_name](obs)

      # 4) Optional clip.
      if term_cfg.clip is not None:
        obs = torch.clamp(obs, term_cfg.clip[0], term_cfg.clip[1])

      group_obs[term_name] = obs

    if self._group_obs_concatenate[group_name]:
      return torch.cat(
        list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
      )
    return group_obs

  def get_term(self, term_name: str, group: str | None = None) -> torch.Tensor:
    if self._obs_buffer is None:
      self.compute()
    assert self._obs_buffer is not None

    if group is None:
      groups_with_term = [g for g, names in self._group_obs_term_names.items() if term_name in names]
      assert len(groups_with_term) == 1, "term_name must be unique or specify group"
      group = groups_with_term[0]
    assert group in self._group_obs_term_names, f"Unknown group '{group}'"
    assert term_name in self._group_obs_term_names[group], f"Unknown term '{term_name}' in group '{group}'"

    buf = self._obs_buffer[group]
    if not self._group_obs_concatenate[group]:
      assert isinstance(buf, dict)
      return buf[term_name]

    assert isinstance(buf, torch.Tensor)
    slc = self._group_term_slices[group][term_name]
    return buf[:, slc]

  def get_term_hist(self, term_name: str, group: str | None = None) -> SlidingWindow | None:
    if group is None:
      groups_with_term = [g for g, names in self._group_obs_term_names.items() if term_name in names]
      assert len(groups_with_term) == 1, "term_name must be unique or specify group"
      group = groups_with_term[0]
    assert group in self._group_obs_term_names, f"Unknown group '{group}'"
    assert term_name in self._group_obs_term_names[group], f"Unknown term '{term_name}' in group '{group}'"
    return self._sliding_windows.get(group, {}).get(term_name, None)

  def _prepare_terms(self) -> None:
    self._group_obs_term_names: dict[str, list[str]] = dict()
    self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
    self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_concatenate: dict[str, bool] = dict()
    self._group_obs_concatenate_dim: dict[str, int] = dict()
    self._group_obs_class_instances: dict[str, noise_model.NoiseModel] = {}
    self._group_term_slices: dict[str, dict[str, slice]] = {}

    group_cfg_items = get_terms(self.cfg, ObservationGroupCfg).items()
    for group_name, group_cfg in group_cfg_items:
      if group_cfg is None:
        print(f"group: {group_name} set to None, skipping...")
        continue
      group_cfg: ObservationGroupCfg

      self._group_obs_term_names[group_name] = []
      self._group_obs_term_dim[group_name] = []
      self._group_obs_term_raw_dim[group_name] = []
      self._group_obs_term_cfgs[group_name] = []
      self._group_obs_class_term_cfgs[group_name] = []

      self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
      self._group_obs_concatenate_dim[group_name] = (
        group_cfg.concatenate_dim + 1
        if group_cfg.concatenate_dim >= 0
        else group_cfg.concatenate_dim
      )

      group_cfg_items = get_terms(group_cfg, ObservationTermCfg).items()
      for term_name, term_cfg in group_cfg_items:
        if term_cfg is None:
          print(f"term: {term_name} set to None, skipping...")
          continue

        self._resolve_common_term_cfg(term_name, term_cfg)

        if not group_cfg.enable_corruption:
          term_cfg.noise = None

        # Register term
        self._group_obs_term_names[group_name].append(term_name)
        self._group_obs_term_cfgs[group_name].append(term_cfg)

        # 1) Probe RAW feature dims via func, passing params
        raw_sample = term_cfg.func(self._env, **term_cfg.params)
        assert isinstance(raw_sample, torch.Tensor), f"Observation func '{term_name}' must return a Tensor"
        raw_dims = tuple(raw_sample.shape[1:])
        self._group_obs_term_raw_dim[group_name].append(raw_dims)

        # 2) Infer OUTPUT dims:
        #    If hist_func present and hist_window_size>0, pre-fill a temp SlidingWindow with raw samples
        #    (using same params) then call hist_func(env, temp_sw, **params) to get output shape.
        if term_cfg.hist_func is not None and term_cfg.hist_window_size > 0:
          temp_sw = SlidingWindow(
            num_envs=self._env.num_envs,
            feature_shape=torch.Size(raw_dims),
            max_window_size=term_cfg.hist_window_size,
            device=torch.device(self._env.device),
          )
          for _ in range(term_cfg.hist_window_size):
            temp_sw.push(raw_sample)
          agg_sample = term_cfg.hist_func(self._env, temp_sw, **term_cfg.params)
          assert isinstance(agg_sample, torch.Tensor), f"hist_func for '{term_name}' must return a Tensor"
          out_dims = tuple(agg_sample.shape[1:])
        else:
          out_dims = raw_dims

        self._group_obs_term_dim[group_name].append(out_dims)

      # Precompute slices for concatenated groups
      if self._group_obs_concatenate[group_name]:
        offsets: dict[str, slice] = {}
        start = 0
        for name, shp in zip(
          self._group_obs_term_names[group_name],
          self._group_obs_term_dim[group_name],
          strict=False,
        ):
          length = int(np.prod(shp))
          offsets[name] = slice(start, start + length)
          start += length
        self._group_term_slices[group_name] = offsets

    # Prepare noise model classes last
    for group_name in self._group_obs_term_names.keys():
      for term_name, term_cfg in zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_cfgs[group_name],
        strict=False,
      ):
        if term_cfg.noise is not None and isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
          noise_model_cls = term_cfg.noise.class_type
          assert issubclass(noise_model_cls, noise_model.NoiseModel), (
            f"Class type for observation term '{term_name}' NoiseModelCfg"
            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
          )
          self._group_obs_class_instances[term_name] = noise_model_cls(
            term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
          )