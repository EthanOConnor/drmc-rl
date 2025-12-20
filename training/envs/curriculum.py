from __future__ import annotations

"""Scripted curriculum support for Dr. Mario training.

The curriculum requested for the placement macro environment uses "synthetic"
negative levels as a convenience encoding:

  - level  0: 4 viruses (vanilla)
  - level -1: 3 viruses
  - level -2: 2 viruses
  - level -3: 1 virus
  - level -15..-4: 0 viruses, goal = clear N matches (4-match events)
      - level -15: 1 match ("any match")
      - level -14: 2 matches
      ...
      - level  -4: 12 matches

The environment implements the negative-level semantics by patching the bottle
RAM at reset time (see `envs/retro/drmario_env.py`).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def wilson_lower_bound(*, successes: int, n: int, sigmas: float) -> float:
    """One-sided Wilson score lower bound for Bernoulli p.

    Args:
        successes: Number of successes (k).
        n: Number of trials.
        sigmas: Z-score (e.g., 2.0 for "2-sigma").
    """

    n_i = int(n)
    if n_i <= 0:
        return 0.0
    k_i = int(max(0, min(int(successes), n_i)))
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return float(k_i) / float(n_i)

    p_hat = float(k_i) / float(n_i)
    z2 = z * z
    denom = 1.0 + z2 / float(n_i)
    center = p_hat + z2 / (2.0 * float(n_i))
    adj = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n_i))) / float(n_i))
    return _clamp01((center - adj) / denom)


def confidence_window_size(*, target: float, sigmas: float) -> int:
    """Return a principled rolling window size for proving p > target.

    The curriculum uses a rolling window of Bernoulli outcomes (success/fail)
    and advances when a `sigmas`-Wilson lower bound for `p` within that window
    exceeds `target`.

    We pick a window size that is reasonable *when skill is near the target*.
    Concretely, assume a "near-target" success probability:

        p_assumed = (1 + target) / 2

    i.e. halfway between the threshold and perfect play. Under a normal
    approximation, the sample size required to separate `p_assumed` from
    `target` at `sigmas` is:

        n ~= z^2 * p(1-p) / (p - target)^2

    With p=(1+target)/2 this simplifies to:

        n ~= z^2 * (1 + target) / (1 - target)
    """

    t = float(target)
    if t <= 0.0:
        return 1
    if t >= 1.0:
        # Impossible to certify p>1; callers should clamp targets.
        return 1_000_000_000
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return 1

    p = 0.5 * (1.0 + t)
    denom = max(1e-12, 1.0 - t)
    # z^2 * (1+t)/(1-t) with a tiny epsilon to avoid ceil() floating artifacts.
    n_est = (z * z) * (1.0 + t) / denom
    return int(max(1, int(math.ceil(n_est - 1e-12))))


def perfect_streak_window_size(*, target: float, sigmas: float) -> int:
    """Window size for certifying p > target given an all-success streak.

    This is used for "mastery" checks that explicitly require 100% success
    within the window (k=n). For k=n, the Wilson lower bound simplifies to:

        LB = n / (n + z^2)

    Solve LB > target -> n > z^2 * target / (1 - target).
    """

    t = float(target)
    if t <= 0.0:
        return 1
    if t >= 1.0:
        return 1_000_000_000
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return 1
    z2 = z * z
    n = int(math.floor((z2 * t) / max(1e-12, (1.0 - t)))) + 1
    return int(max(1, n))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.median(arr))


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation (MAD), unscaled."""

    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = False
    mode: str = "linear"  # linear|ln_hop_back
    start_level: int = -10
    max_level: int = 0
    success_threshold: float = 0.9
    probe_threshold: float = 0.01  # used by ln_hop_back
    # Confidence-based curriculum evaluation:
    # If `confidence_sigmas > 0`, window sizes are computed from
    # `confidence_window_size(target, confidence_sigmas)` and advancement
    # requires `wilson_lower_bound(...) > target` within that rolling window.
    confidence_sigmas: float = 2.0
    # Fallback / compatibility knobs (used only when confidence_sigmas <= 0).
    window_episodes: int = 100
    min_episodes: int = 50
    rehearsal_prob: float = 0.1
    seed: Optional[int] = None
    ln_hop_back_max_k: int = 5  # caps 1-exp(-k) thresholds for feasibility
    # Optional time-based constraints applied by the curriculum wrapper.
    # These are forwarded to env wrapper attributes (e.g. DrMarioPlacementEnv).
    task_max_frames: Optional[int] = None
    task_max_spawns: Optional[int] = None
    task_max_frames_by_level: Dict[int, int] = field(default_factory=dict)
    task_max_spawns_by_level: Dict[int, int] = field(default_factory=dict)
    # Time-budget curricula (activated after mastery).
    time_budget_enabled: bool = True
    time_budget_mastery_sigmas: float = 3.0
    time_budget_mastery_target: float = 0.99
    time_budget_time_window: int = 128
    time_budget_min_frames: int = 1
    time_budget_max_drop_fraction_of_mad: float = 0.1
    time_budget_min_drop_frames: int = 1

    @classmethod
    def from_cfg(cls, cfg: Any) -> "CurriculumConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, dict):
            data = cfg
        elif hasattr(cfg, "to_dict"):
            data = cfg.to_dict()
        else:
            data = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

        def _get(name: str, default: Any) -> Any:
            val = data.get(name, default)
            return default if val is None else val

        def _parse_budget(value: Any) -> tuple[Optional[int], Dict[int, int]]:
            if value is None:
                return None, {}
            if isinstance(value, (int, float)):
                return int(value), {}
            if isinstance(value, dict):
                out: Dict[int, int] = {}
                for k, v in value.items():
                    try:
                        out[int(k)] = int(v)
                    except Exception:
                        continue
                return None, out
            return None, {}

        max_frames_raw = data.get("task_max_frames", data.get("max_frames"))
        max_spawns_raw = data.get("task_max_spawns", data.get("max_spawns"))
        max_frames, max_frames_by_level = _parse_budget(max_frames_raw)
        max_spawns, max_spawns_by_level = _parse_budget(max_spawns_raw)

        return cls(
            enabled=bool(_get("enabled", False)),
            mode=str(_get("mode", "linear")),
            start_level=int(_get("start_level", -10)),
            max_level=int(_get("max_level", 0)),
            success_threshold=float(_get("success_threshold", 0.9)),
            probe_threshold=float(_get("probe_threshold", 0.01)),
            confidence_sigmas=float(_get("confidence_sigmas", 2.0)),
            window_episodes=int(_get("window_episodes", 100)),
            min_episodes=int(_get("min_episodes", 50)),
            rehearsal_prob=float(_get("rehearsal_prob", 0.1)),
            seed=None if data.get("seed") is None else int(data["seed"]),
            ln_hop_back_max_k=int(_get("ln_hop_back_max_k", 5)),
            task_max_frames=max_frames,
            task_max_spawns=max_spawns,
            task_max_frames_by_level=max_frames_by_level,
            task_max_spawns_by_level=max_spawns_by_level,
            time_budget_enabled=bool(_get("time_budget_enabled", True)),
            time_budget_mastery_sigmas=float(_get("time_budget_mastery_sigmas", 3.0)),
            time_budget_mastery_target=float(_get("time_budget_mastery_target", 0.99)),
            time_budget_time_window=int(_get("time_budget_time_window", 128)),
            time_budget_min_frames=int(_get("time_budget_min_frames", 1)),
            time_budget_max_drop_fraction_of_mad=float(_get("time_budget_max_drop_fraction_of_mad", 0.1)),
            time_budget_min_drop_frames=int(_get("time_budget_min_drop_frames", 1)),
        )


class ScriptedCurriculum:
    """Tracks success rates and advances the active curriculum level."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self.current_level = int(cfg.start_level)
        self._histories: Dict[int, Deque[bool]] = {}
        self._episodes_total: Dict[int, int] = {}
        self._time_samples: Dict[int, Deque[int]] = {}
        self._time_budget_frames: Dict[int, int] = {}

        self._sigmas = float(max(0.0, float(getattr(cfg, "confidence_sigmas", 0.0) or 0.0)))
        self._stage_window = (
            confidence_window_size(target=float(cfg.success_threshold), sigmas=self._sigmas)
            if self._sigmas > 0.0
            else int(max(1, int(cfg.window_episodes)))
        )
        mastery_sigmas = float(
            max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0))
        )
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        self._mastery_window = (
            perfect_streak_window_size(target=mastery_target, sigmas=mastery_sigmas)
            if mastery_sigmas > 0.0
            else 0
        )
        self._history_maxlen = int(max(1, self._stage_window, self._mastery_window))
        self._time_window = int(max(1, int(getattr(cfg, "time_budget_time_window", 128) or 128)))

    def _history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._histories:
            self._histories[lvl] = deque(maxlen=int(self._history_maxlen))
            self._episodes_total[lvl] = 0
        return self._histories[lvl]

    def _times(self, level: int) -> Deque[int]:
        lvl = int(level)
        if lvl not in self._time_samples:
            self._time_samples[lvl] = deque(maxlen=int(self._time_window))
        return self._time_samples[lvl]

    def task_max_frames_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_frames.get(int(level))

    def note_episode(
        self,
        *,
        level: int,
        success: bool,
        episode_frames: Optional[int] = None,
        objective_met: Optional[bool] = None,
    ) -> bool:
        """Record an episode outcome; returns True if this advanced the curriculum."""

        lvl = int(level)
        hist = self._history(lvl)
        hist.append(bool(success))
        self._episodes_total[lvl] = int(self._episodes_total.get(lvl, 0)) + 1

        if objective_met is None:
            objective_met = bool(success)
        if objective_met and episode_frames is not None:
            try:
                frames_i = int(episode_frames)
            except Exception:
                frames_i = 0
            if frames_i > 0:
                self._times(lvl).append(frames_i)
                self._maybe_update_time_budget(lvl)

        if lvl != int(self.current_level):
            return False
        if int(self.current_level) >= int(self.cfg.max_level):
            return False

        if self._meets_confidence_target(lvl, float(self.cfg.success_threshold)):
            self.current_level = min(int(self.cfg.max_level), int(self.current_level) + 1)
            return True
        return False

    def _meets_confidence_target(self, level: int, target: float) -> bool:
        t = float(target)
        if self._sigmas <= 0.0:
            if int(self._episodes_total.get(int(level), 0)) < int(self.cfg.min_episodes):
                return False
            return self.success_rate(int(level)) >= t

        window_size = int(max(1, int(self._stage_window)))
        hist = self._histories.get(int(level))
        if not hist:
            return False
        tail = list(hist)[-window_size:]
        if len(tail) < window_size:
            return False
        successes = int(sum(1 for x in tail if x))
        lb = wilson_lower_bound(successes=successes, n=int(window_size), sigmas=float(self._sigmas))
        return bool(lb > t)

    def _maybe_update_time_budget(self, level: int) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "time_budget_enabled", True)):
            return
        mastery_sigmas = float(max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0)))
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        if mastery_sigmas <= 0.0 or mastery_target <= 0.0:
            return
        window = int(max(1, int(self._mastery_window)))
        hist = self._histories.get(int(level))
        if not hist:
            return
        tail = list(hist)[-window:]
        if len(tail) < window:
            return
        if int(sum(1 for x in tail if x)) != window:
            return
        times = list(self._time_samples.get(int(level), deque()))
        if not times:
            return
        mean_frames = float(np.mean(np.asarray(times, dtype=np.float64)))
        mad_frames = _mad([float(t) for t in times])
        budget_now = self._time_budget_frames.get(int(level))

        budget_candidate = int(max(1, int(round(mean_frames))))
        min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
        budget_candidate = int(max(min_frames, budget_candidate))

        if budget_now is None:
            self._time_budget_frames[int(level)] = int(budget_candidate)
            return

        if budget_candidate >= int(budget_now):
            return
        drop_frac = float(getattr(cfg, "time_budget_max_drop_fraction_of_mad", 0.1) or 0.1)
        min_drop = int(max(1, int(getattr(cfg, "time_budget_min_drop_frames", 1) or 1)))
        max_drop = int(max(min_drop, int(round(float(mad_frames) * float(drop_frac)))))
        new_budget = int(max(budget_candidate, int(budget_now) - max_drop))
        self._time_budget_frames[int(level)] = int(max(min_frames, new_budget))

    def success_rate(self, level: int) -> float:
        hist = self._histories.get(int(level))
        if not hist:
            return 0.0
        return float(sum(1 for x in hist if x)) / float(len(hist))

    def episodes_seen(self, level: int) -> int:
        return int(self._episodes_total.get(int(level), 0))

    def sample_level(self, rng: np.random.Generator) -> int:
        cur = int(self.current_level)
        start = int(self.cfg.start_level)
        if cur <= start:
            return cur
        p = float(self.cfg.rehearsal_prob)
        if p <= 0.0:
            return cur
        if rng.random() >= p:
            return cur
        lower = list(range(start, cur))
        if not lower:
            return cur
        return int(rng.choice(lower))

    def snapshot(self) -> Dict[str, Any]:
        cur = int(self.current_level)
        hist = self._histories.get(cur)
        window_size = int(max(1, int(self._stage_window)))
        tail = list(hist)[-window_size:] if hist is not None else []
        window_n = int(len(tail))
        successes = int(sum(1 for x in tail if x))
        rate = float(successes) / float(window_n) if window_n > 0 else 0.0
        lb = wilson_lower_bound(successes=successes, n=window_n, sigmas=float(self._sigmas))
        out: Dict[str, Any] = {
            "current_level": cur,
            "rate_current": float(rate),
            "window_n": window_n,
            "window_size": window_size,
            "window_successes": int(successes),
            "confidence_sigmas": float(self._sigmas),
            "confidence_lower_bound": float(lb),
            "episodes_current_total": int(self.episodes_seen(cur)),
            "start_level": int(self.cfg.start_level),
            "max_level": int(self.cfg.max_level),
            "success_threshold": float(self.cfg.success_threshold),
            "mode": "linear",
        }
        budget = self._time_budget_frames.get(cur)
        if budget is not None:
            out["time_budget_frames"] = int(budget)
        times = self._time_samples.get(cur)
        if times:
            times_list = [float(t) for t in list(times)]
            out["time_mean_frames"] = float(np.mean(np.asarray(times_list, dtype=np.float64)))
            out["time_mad_frames"] = float(_mad(times_list))
        return out


@dataclass(slots=True, frozen=True)
class _LnStage:
    level: int
    threshold: float


class LnHopBackCurriculum:
    """Curriculum that alternates between new levels and ln-tightened hop-backs.

    Pattern (matching the request):
      - When introducing a new frontier level L: require a small probe success rate
        on L (e.g. 1%).
      - Then "hop back" through all previously introduced levels (start..L) with
        thresholds: 1 - exp(-k), where k increases the further back you hop.

    This yields a schedule like:
      -15@0.01, -15@0.632
      -14@0.01, -15@0.865, -14@0.632
      -13@0.01, -15@0.950, -14@0.865, -13@0.632
      ...
    """

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self._histories: Dict[int, Deque[bool]] = {}
        self._episodes_total: Dict[int, int] = {}
        self._stages: List[_LnStage] = self._build_stages()
        self._stage_idx = 0
        self._time_samples: Dict[int, Deque[int]] = {}
        self._time_budget_frames: Dict[int, int] = {}

        self._sigmas = float(max(0.0, float(getattr(cfg, "confidence_sigmas", 0.0) or 0.0)))
        if self._sigmas > 0.0:
            self._max_stage_window = int(
                max(confidence_window_size(target=float(s.threshold), sigmas=self._sigmas) for s in self._stages)
            )
        else:
            self._max_stage_window = int(max(1, int(cfg.window_episodes)))

        mastery_sigmas = float(
            max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0))
        )
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        self._mastery_window = (
            perfect_streak_window_size(target=mastery_target, sigmas=mastery_sigmas)
            if mastery_sigmas > 0.0
            else 0
        )
        self._history_maxlen = int(max(1, self._max_stage_window, self._mastery_window))
        self._time_window = int(max(1, int(getattr(cfg, "time_budget_time_window", 128) or 128)))

    def _build_stages(self) -> List[_LnStage]:
        start = int(self.cfg.start_level)
        end = int(self.cfg.max_level)
        if end < start:
            raise ValueError("curriculum.max_level must be >= curriculum.start_level")
        probe = float(self.cfg.probe_threshold)
        probe = float(min(1.0, max(0.0, probe)))
        max_k = int(max(1, int(getattr(self.cfg, "ln_hop_back_max_k", 5) or 5)))

        stages: List[_LnStage] = []
        for frontier in range(start, end + 1):
            stages.append(_LnStage(level=int(frontier), threshold=probe))
            for lvl in range(start, frontier + 1):
                k = int(min(max_k, int(frontier - lvl + 1)))
                thr = float(1.0 - math.exp(-float(k)))
                stages.append(_LnStage(level=int(lvl), threshold=float(min(1.0, max(0.0, thr)))))
        return stages

    @property
    def current_level(self) -> int:
        return int(self._stages[self._stage_idx].level)

    @property
    def stage_index(self) -> int:
        return int(self._stage_idx)

    @property
    def stage_count(self) -> int:
        return int(len(self._stages))

    def _history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._histories:
            self._histories[lvl] = deque(maxlen=int(self._history_maxlen))
            self._episodes_total[lvl] = 0
        return self._histories[lvl]

    def _times(self, level: int) -> Deque[int]:
        lvl = int(level)
        if lvl not in self._time_samples:
            self._time_samples[lvl] = deque(maxlen=int(self._time_window))
        return self._time_samples[lvl]

    def task_max_frames_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_frames.get(int(level))

    def note_episode(
        self,
        *,
        level: int,
        success: bool,
        episode_frames: Optional[int] = None,
        objective_met: Optional[bool] = None,
    ) -> bool:
        lvl = int(level)
        hist = self._history(lvl)
        hist.append(bool(success))
        self._episodes_total[lvl] = int(self._episodes_total.get(lvl, 0)) + 1

        if objective_met is None:
            objective_met = bool(success)
        if objective_met and episode_frames is not None:
            try:
                frames_i = int(episode_frames)
            except Exception:
                frames_i = 0
            if frames_i > 0:
                self._times(lvl).append(frames_i)
                self._maybe_update_time_budget(lvl)

        # Only advance when the active stage level completes.
        if lvl != self.current_level:
            return False
        if self._stage_idx >= len(self._stages) - 1:
            return False

        stage = self._stages[self._stage_idx]
        if self._meets_confidence_target(lvl, float(stage.threshold)):
            self._stage_idx += 1
            return True
        return False

    def _meets_confidence_target(self, level: int, target: float) -> bool:
        t = float(target)
        if self._sigmas <= 0.0:
            if int(self._episodes_total.get(int(level), 0)) < int(self.cfg.min_episodes):
                return False
            return self.success_rate(int(level)) >= t

        window_size = confidence_window_size(target=t, sigmas=float(self._sigmas))
        hist = self._histories.get(int(level))
        if not hist:
            return False
        tail = list(hist)[-int(window_size):]
        if len(tail) < int(window_size):
            return False
        successes = int(sum(1 for x in tail if x))
        lb = wilson_lower_bound(successes=successes, n=int(window_size), sigmas=float(self._sigmas))
        return bool(lb > t)

    def _maybe_update_time_budget(self, level: int) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "time_budget_enabled", True)):
            return
        mastery_sigmas = float(max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0)))
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        if mastery_sigmas <= 0.0 or mastery_target <= 0.0:
            return
        window = int(max(1, int(self._mastery_window)))
        hist = self._histories.get(int(level))
        if not hist:
            return
        tail = list(hist)[-window:]
        if len(tail) < window:
            return
        if int(sum(1 for x in tail if x)) != window:
            return
        times = list(self._time_samples.get(int(level), deque()))
        if not times:
            return
        mean_frames = float(np.mean(np.asarray(times, dtype=np.float64)))
        mad_frames = _mad([float(t) for t in times])
        budget_now = self._time_budget_frames.get(int(level))

        budget_candidate = int(max(1, int(round(mean_frames))))
        min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
        budget_candidate = int(max(min_frames, budget_candidate))

        if budget_now is None:
            self._time_budget_frames[int(level)] = int(budget_candidate)
            return

        if budget_candidate >= int(budget_now):
            return
        drop_frac = float(getattr(cfg, "time_budget_max_drop_fraction_of_mad", 0.1) or 0.1)
        min_drop = int(max(1, int(getattr(cfg, "time_budget_min_drop_frames", 1) or 1)))
        max_drop = int(max(min_drop, int(round(float(mad_frames) * float(drop_frac)))))
        new_budget = int(max(budget_candidate, int(budget_now) - max_drop))
        self._time_budget_frames[int(level)] = int(max(min_frames, new_budget))

    def success_rate(self, level: int) -> float:
        hist = self._histories.get(int(level))
        if not hist:
            return 0.0
        return float(sum(1 for x in hist if x)) / float(len(hist))

    def episodes_seen(self, level: int) -> int:
        return int(self._episodes_total.get(int(level), 0))

    def sample_level(self, rng: np.random.Generator) -> int:
        del rng
        return self.current_level

    def snapshot(self) -> Dict[str, Any]:
        stage = self._stages[self._stage_idx]
        cur = int(stage.level)
        hist = self._histories.get(cur)
        window_size = (
            confidence_window_size(target=float(stage.threshold), sigmas=float(self._sigmas))
            if self._sigmas > 0.0
            else int(max(1, int(self.cfg.window_episodes)))
        )
        tail = list(hist)[-int(window_size):] if hist is not None else []
        window_n = int(len(tail))
        successes = int(sum(1 for x in tail if x))
        rate = float(successes) / float(window_n) if window_n > 0 else 0.0
        lb = wilson_lower_bound(successes=successes, n=window_n, sigmas=float(self._sigmas))
        out: Dict[str, Any] = {
            "current_level": cur,
            "rate_current": float(rate),
            "success_threshold": float(stage.threshold),
            "window_n": window_n,
            "window_size": window_size,
            "window_successes": int(successes),
            "confidence_sigmas": float(self._sigmas),
            "confidence_lower_bound": float(lb),
            "episodes_current_total": int(self.episodes_seen(cur)),
            "start_level": int(self.cfg.start_level),
            "max_level": int(self.cfg.max_level),
            "stage_index": int(self._stage_idx),
            "stage_count": int(len(self._stages)),
            "mode": "ln_hop_back",
            "probe_threshold": float(self.cfg.probe_threshold),
        }
        budget = self._time_budget_frames.get(cur)
        if budget is not None:
            out["time_budget_frames"] = int(budget)
        times = self._time_samples.get(cur)
        if times:
            times_list = [float(t) for t in list(times)]
            out["time_mean_frames"] = float(np.mean(np.asarray(times_list, dtype=np.float64)))
            out["time_mad_frames"] = float(_mad(times_list))
        return out


class CurriculumVecEnv:
    """Vector-env wrapper that assigns per-episode curriculum levels."""

    def __init__(self, env: Any, cfg: CurriculumConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.num_envs = int(getattr(env, "num_envs", 1))
        self._rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))
        mode = str(getattr(cfg, "mode", "linear") or "linear").strip().lower()
        self._mode = mode
        if mode == "ln_hop_back":
            self._curriculum = LnHopBackCurriculum(cfg)
        else:
            self._curriculum = ScriptedCurriculum(cfg)
        self._env_levels: List[int] = [int(cfg.start_level) for _ in range(self.num_envs)]

        if not hasattr(env, "set_attr"):
            raise TypeError("CurriculumVecEnv requires an env that supports `set_attr`.")

    def reset(self, *args: Any, **kwargs: Any):
        self._env_levels = [self._curriculum.sample_level(self._rng) for _ in range(self.num_envs)]
        self.env.set_attr("level", list(self._env_levels))
        self._set_task_budgets(self._env_levels)
        obs, infos = self.env.reset(*args, **kwargs)
        infos_list = self._ensure_info_list(infos)
        self._inject_curriculum_info(infos_list)
        return obs, infos_list

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        infos_list = self._ensure_info_list(infos)

        term = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
        trunc = np.asarray(truncated, dtype=bool).reshape(self.num_envs)

        levels_this_step = list(self._env_levels)
        next_levels = list(self._env_levels)
        updated = False
        advanced_from: Optional[int] = None
        advanced_to: Optional[int] = None
        for i in range(self.num_envs):
            if not bool(term[i] or trunc[i]):
                continue
            info = infos_list[i] if i < len(infos_list) else {}
            success = bool(
                info.get("goal_achieved")
                if "goal_achieved" in info
                else info.get("cleared", info.get("drm", {}).get("cleared", False))
            )
            objective_met = bool(info.get("task/objective_met", success))
            episode_frames: Optional[int] = None
            episode = info.get("episode")
            if isinstance(episode, dict):
                try:
                    episode_frames = int(episode.get("l")) if episode.get("l") is not None else None
                except Exception:
                    episode_frames = None
            prev_level = int(getattr(self._curriculum, "current_level", self.cfg.start_level))
            advanced = self._curriculum.note_episode(
                level=int(levels_this_step[i]),
                success=success,
                episode_frames=episode_frames,
                objective_met=objective_met,
            )
            if advanced and advanced_to is None:
                advanced_from = prev_level
                advanced_to = int(getattr(self._curriculum, "current_level", prev_level))
            next_levels[i] = int(self._curriculum.sample_level(self._rng))
            updated = True

        if updated:
            self._env_levels = list(next_levels)
            self.env.set_attr("level", list(self._env_levels))
            self._set_task_budgets(self._env_levels)

        self._inject_curriculum_info(
            infos_list,
            levels=levels_this_step,
            next_levels=next_levels,
            advanced_from=advanced_from,
            advanced_to=advanced_to,
        )
        return obs, rewards, terminated, truncated, infos_list

    def _set_task_budgets(self, levels: List[int]) -> None:
        cfg = self.cfg
        frames_default = getattr(cfg, "task_max_frames", None)
        spawns_default = getattr(cfg, "task_max_spawns", None)
        frames_by_level = getattr(cfg, "task_max_frames_by_level", {}) or {}
        spawns_by_level = getattr(cfg, "task_max_spawns_by_level", {}) or {}

        dyn_frames_fn = getattr(self._curriculum, "task_max_frames_for_level", None)

        if (
            frames_default is None
            and not frames_by_level
            and spawns_default is None
            and not spawns_by_level
            and not callable(dyn_frames_fn)
        ):
            return

        def _resolve(default: Optional[int], mapping: Dict[int, int], level: int) -> Optional[int]:
            if mapping:
                if level in mapping:
                    return int(mapping[level])
                # Fall back to scalar default if provided.
            return None if default is None else int(default)

        try:
            frames_values = []
            for lvl in list(levels):
                dyn_val: Optional[int] = None
                if callable(dyn_frames_fn):
                    try:
                        dyn_val = dyn_frames_fn(int(lvl))
                    except Exception:
                        dyn_val = None
                if dyn_val is not None:
                    frames_values.append(int(dyn_val))
                else:
                    frames_values.append(_resolve(frames_default, frames_by_level, int(lvl)))
            self.env.set_attr("task_max_frames", frames_values)
        except Exception:
            pass
        try:
            spawns_values = [
                _resolve(spawns_default, spawns_by_level, int(lvl)) for lvl in list(levels)
            ]
            self.env.set_attr("task_max_spawns", spawns_values)
        except Exception:
            pass

    def _inject_curriculum_info(
        self,
        infos: List[Dict[str, Any]],
        *,
        levels: Optional[List[int]] = None,
        next_levels: Optional[List[int]] = None,
        advanced_from: Optional[int] = None,
        advanced_to: Optional[int] = None,
    ) -> None:
        snap = self._curriculum.snapshot()
        window_n = int(snap.get("window_n", 0) or 0)
        window_size = int(snap.get("window_size", 1) or 1)
        episodes_total = int(snap.get("episodes_current_total", 0) or 0)
        start_level = int(snap.get("start_level", self.cfg.start_level) or self.cfg.start_level)
        max_level = int(snap.get("max_level", self.cfg.max_level) or self.cfg.max_level)
        success_threshold = float(snap.get("success_threshold", self.cfg.success_threshold))
        mode = str(snap.get("mode", getattr(self.cfg, "mode", "linear")) or "linear")
        confidence_sigmas = snap.get("confidence_sigmas")
        confidence_lower_bound = snap.get("confidence_lower_bound")
        window_successes = snap.get("window_successes")
        time_budget_frames = snap.get("time_budget_frames")
        time_mean_frames = snap.get("time_mean_frames")
        time_mad_frames = snap.get("time_mad_frames")
        stage_idx = snap.get("stage_index")
        stage_count = snap.get("stage_count")
        probe_threshold = snap.get("probe_threshold")
        for i in range(min(len(infos), self.num_envs)):
            info = infos[i]
            if not isinstance(info, dict):
                continue
            level_i = (
                int(levels[i])
                if levels is not None and i < len(levels)
                else int(self._env_levels[i])
            )
            info["curriculum/env_level"] = int(level_i)
            if next_levels is not None and i < len(next_levels):
                info["curriculum/next_env_level"] = int(next_levels[i])
            info["curriculum/current_level"] = int(snap["current_level"])
            info["curriculum/rate_current"] = float(snap["rate_current"])
            info["curriculum/window_n"] = int(window_n)
            info["curriculum/window_size"] = int(window_size)
            info["curriculum/episodes_current_total"] = int(episodes_total)
            info["curriculum/start_level"] = int(start_level)
            info["curriculum/max_level"] = int(max_level)
            info["curriculum/success_threshold"] = float(success_threshold)
            info["curriculum/min_episodes"] = int(self.cfg.min_episodes)
            info["curriculum/rehearsal_prob"] = float(self.cfg.rehearsal_prob)
            info["curriculum/mode"] = str(mode)
            if confidence_sigmas is not None:
                try:
                    info["curriculum/confidence_sigmas"] = float(confidence_sigmas)
                except Exception:
                    pass
            if confidence_lower_bound is not None:
                try:
                    info["curriculum/confidence_lower_bound"] = float(confidence_lower_bound)
                except Exception:
                    pass
            if window_successes is not None:
                try:
                    info["curriculum/window_successes"] = int(window_successes)
                except Exception:
                    pass
            if time_budget_frames is not None:
                try:
                    info["curriculum/time_budget_frames"] = int(time_budget_frames)
                except Exception:
                    pass
            if time_mean_frames is not None:
                try:
                    info["curriculum/time_mean_frames"] = float(time_mean_frames)
                except Exception:
                    pass
            if time_mad_frames is not None:
                try:
                    info["curriculum/time_mad_frames"] = float(time_mad_frames)
                except Exception:
                    pass
            if stage_idx is not None:
                try:
                    info["curriculum/stage_index"] = int(stage_idx)
                except Exception:
                    pass
            if stage_count is not None:
                try:
                    info["curriculum/stage_count"] = int(stage_count)
                except Exception:
                    pass
            if probe_threshold is not None:
                try:
                    info["curriculum/probe_threshold"] = float(probe_threshold)
                except Exception:
                    pass
            if advanced_to is not None:
                info["curriculum/advanced"] = True
                if advanced_from is not None:
                    info["curriculum/advanced_from"] = int(advanced_from)
                info["curriculum/advanced_to"] = int(advanced_to)

    def _ensure_info_list(self, infos: Any) -> List[Dict[str, Any]]:
        if infos is None:
            return [{} for _ in range(self.num_envs)]
        if isinstance(infos, list):
            return [dict(i) if isinstance(i, dict) else {} for i in infos]
        if isinstance(infos, tuple):
            return [dict(i) if isinstance(i, dict) else {} for i in list(infos)]
        if isinstance(infos, dict):
            # Best-effort: broadcast dict to every env (caller is likely already wrapped).
            return [dict(infos) for _ in range(self.num_envs)]
        return [{} for _ in range(self.num_envs)]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.env, name)
