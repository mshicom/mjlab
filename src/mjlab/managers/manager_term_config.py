from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ParamSpec, TypeVar

from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg

P = ParamSpec("P")
T = TypeVar("T")


def term(term_cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return field(default_factory=lambda: term_cls(*args, **kwargs))


@dataclass
class ManagerTermBaseCfg:
  func: Any
  params: dict[str, Any] = field(default_factory=lambda: {})


##
# Action manager.
##


@dataclass(kw_only=True)
class ActionTermCfg:
  """Configuration for an action term."""

  class_type: type[ActionTerm]
  asset_name: str
  clip: dict[str, tuple] | None = None


##
# Command manager.
##


@dataclass(kw_only=True)
class CommandTermCfg:
  """Configuration for a command generator term."""

  class_type: type[CommandTerm]
  resampling_time_range: tuple[float, float]
  debug_vis: bool = False


##
# Curriculum manager.
##


@dataclass(kw_only=True)
class CurriculumTermCfg(ManagerTermBaseCfg):
  pass


##
# Event manager.
##


EventMode = Literal["startup", "reset", "interval"]


@dataclass(kw_only=True)
class EventTermCfg(ManagerTermBaseCfg):
  """Configuration for an event term."""

  mode: EventMode
  interval_range_s: tuple[float, float] | None = None
  is_global_time: bool = False
  min_step_count_between_reset: int = 0


##
# Observation manager.
##


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
  """Configuration for an observation term.

  Fields:
    func: Callable that produces the raw per-step observation tensor (N, ...).
    params: Keyword arguments passed to `func` (and to `hist_func` if provided).
    noise: Optional corruption/noise to apply after term computation.
    clip: Optional min/max clipping applied after noise.

    hist_window_size: Length of the per-term SlidingWindow. 0 disables SlidingWindow and
      incurs no overhead; the term behaves like before (stateless).
    hist_func: Optional aggregator callable. If provided, `func` becomes the input source
      that is pushed into the per-term SlidingWindow and `hist_func` is used to generate
      the final term output for this step by consuming the SlidingWindow contents.
      Signature is expected as:
        hist_func(env, hist_window: SlidingWindow, **params) -> torch.Tensor
  """

  noise: NoiseCfg | NoiseModelCfg | None = None
  clip: tuple[float, float] | None = None

  # SlidingWindow controls
  hist_window_size: int = 0
  hist_func: Any | None = None


@dataclass
class ObservationGroupCfg:
  """Configuration for an observation group."""

  concatenate_terms: bool = True
  concatenate_dim: int = -1
  enable_corruption: bool = False


##
# Reward manager.
##


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Any
  weight: float


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whethher the term contributes towards episodic timeouts. Defaults to False."""
