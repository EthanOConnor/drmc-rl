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


def wilson_lower_bound_fractional(*, successes: float, n: float, sigmas: float) -> float:
    """One-sided Wilson score lower bound for pseudo-counts.

    This is the same closed form as :func:`wilson_lower_bound` but accepts
    non-integer (successes, n). We use this for exponentially-weighted moving
    averages, where the effective sample size is fractional.
    """

    n_f = float(n)
    if not math.isfinite(n_f) or n_f <= 0.0:
        return 0.0
    k_f = float(successes)
    if not math.isfinite(k_f):
        return 0.0
    if k_f < 0.0:
        k_f = 0.0
    if k_f > n_f:
        k_f = n_f
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return _clamp01(k_f / n_f)

    p_hat = float(k_f) / float(n_f)
    z2 = z * z
    denom = 1.0 + z2 / float(n_f)
    center = p_hat + z2 / (2.0 * float(n_f))
    adj = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n_f))) / float(n_f))
    return _clamp01((center - adj) / denom)


def _ema_decay_from_half_life(half_life_episodes: float) -> float:
    """Return EMA decay factor per episode for a given half-life in episodes."""

    hl = float(half_life_episodes)
    if not math.isfinite(hl) or hl <= 0.0:
        return 0.0
    # After `hl` updates, weight should halve: decay**hl = 0.5.
    return float(0.5 ** (1.0 / hl))


def _ema_n_limit(decay: float) -> float:
    d = float(decay)
    if not math.isfinite(d) or d <= 0.0:
        return 1.0
    if d >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - d))


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
    probe_threshold: float = 0.16  # used by ln_hop_back
    # Confidence-based curriculum evaluation:
    # If `confidence_sigmas > 0`, window sizes are computed from
    # `confidence_window_size(target, confidence_sigmas)` and advancement
    # requires `wilson_lower_bound(...) > target` within that rolling window.
    confidence_sigmas: float = 1.0
    # Confidence gate uses an exponentially weighted moving success rate
    # (non-stationary friendly) with a Wilson-style lower bound computed from
    # pseudo-counts.
    confidence_ema_half_life_episodes: float = 256.0
    confidence_min_effective_episodes: float = 128.0
    # For ln_hop_back, also require a minimum number of macro-decisions to be
    # executed in a stage before the gate can advance. This helps avoid
    # "micro-stages" that complete within a single PPO rollout batch.
    min_stage_decisions: int = 512
    # Fallback / compatibility knobs (used only when confidence_sigmas <= 0).
    window_episodes: int = 100
    min_episodes: int = 50
    rehearsal_prob: float = 0.1
    seed: Optional[int] = None
    # Shared exponent multiplier applied to schedules of the form 1 - exp(-k).
    # Smaller values slow the ramp toward 1.0.
    pass_ramp_exponent_multiplier: float = 1.0 / 3.0
    ln_hop_back_max_k: int = 5  # caps 1-exp(-m*k) thresholds for feasibility
    # ln_hop_back hop-back behavior:
    # - Hop back to the Nth-highest mastered level (default: 3rd-highest).
    ln_hop_back_hop_back_mastered_rank: int = 3
    # - Bail out of a stuck probe stage after this fraction of the run's total
    #   steps/frames budget (0 disables). Requires `run_total_steps`.
    ln_hop_back_bailout_fraction: float = 0.01
    # Populated by the training runner/env factory (used for bailout thresholds).
    run_total_steps: Optional[int] = None
    # Optional time-based constraints applied by the curriculum wrapper.
    # These are forwarded to env wrapper attributes (e.g. DrMarioPlacementEnv).
    task_max_frames: Optional[int] = None
    task_max_spawns: Optional[int] = None
    task_max_frames_by_level: Dict[int, int] = field(default_factory=dict)
    task_max_spawns_by_level: Dict[int, int] = field(default_factory=dict)
    # Time-budget curricula (activated after mastery).
    time_budget_enabled: bool = True
    time_budget_mastery_sigmas: float = 2.0
    time_budget_mastery_target: float = 0.99
    time_budget_time_window: int = 128
    time_budget_min_frames: int = 1
    time_budget_max_drop_fraction_of_mad: float = 0.1
    time_budget_min_drop_frames: int = 1
    time_budget_max_k: int = 3
    # Persistent best-times DB (stored under data/, git-ignored).
    best_times_db_path: Optional[str] = None
    best_times_topk: int = 50

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
            probe_threshold=float(_get("probe_threshold", 0.16)),
            confidence_sigmas=float(_get("confidence_sigmas", 1.0)),
            confidence_ema_half_life_episodes=float(_get("confidence_ema_half_life_episodes", 256.0)),
            confidence_min_effective_episodes=float(_get("confidence_min_effective_episodes", 128.0)),
            min_stage_decisions=int(_get("min_stage_decisions", 512)),
            window_episodes=int(_get("window_episodes", 100)),
            min_episodes=int(_get("min_episodes", 50)),
            rehearsal_prob=float(_get("rehearsal_prob", 0.1)),
            seed=None if data.get("seed") is None else int(data["seed"]),
            pass_ramp_exponent_multiplier=float(_get("pass_ramp_exponent_multiplier", 1.0 / 3.0)),
            ln_hop_back_max_k=int(_get("ln_hop_back_max_k", 5)),
            ln_hop_back_hop_back_mastered_rank=int(_get("ln_hop_back_hop_back_mastered_rank", 3)),
            ln_hop_back_bailout_fraction=float(_get("ln_hop_back_bailout_fraction", 0.01)),
            run_total_steps=None if data.get("run_total_steps") is None else int(data["run_total_steps"]),
            task_max_frames=max_frames,
            task_max_spawns=max_spawns,
            task_max_frames_by_level=max_frames_by_level,
            task_max_spawns_by_level=max_spawns_by_level,
            time_budget_enabled=bool(_get("time_budget_enabled", True)),
            time_budget_mastery_sigmas=float(_get("time_budget_mastery_sigmas", 2.0)),
            time_budget_mastery_target=float(_get("time_budget_mastery_target", 0.99)),
            time_budget_time_window=int(_get("time_budget_time_window", 128)),
            time_budget_min_frames=int(_get("time_budget_min_frames", 1)),
            time_budget_max_drop_fraction_of_mad=float(_get("time_budget_max_drop_fraction_of_mad", 0.1)),
            time_budget_min_drop_frames=int(_get("time_budget_min_drop_frames", 1)),
            time_budget_max_k=int(_get("time_budget_max_k", 3)),
            best_times_db_path=data.get("best_times_db_path"),
            best_times_topk=int(_get("best_times_topk", 50)),
        )


class ScriptedCurriculum:
    """Tracks success rates and advances the active curriculum level."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self.current_level = int(cfg.start_level)
        # Base objective success (ignores time budgets): used for curriculum
        # advancement and mastery detection.
        self._histories: Dict[int, Deque[bool]] = {}
        self._episodes_total: Dict[int, int] = {}
        self._time_samples: Dict[int, Deque[int]] = {}
        self._time_samples_spawns: Dict[int, Deque[int]] = {}
        self._time_budget_frames: Dict[int, int] = {}
        self._time_budget_spawns: Dict[int, int] = {}
        # Time-goal success (under the currently active budgets) is tracked in a
        # separate rolling window that is reset whenever we tighten goals.
        self._time_success_histories: Dict[int, Deque[bool]] = {}
        self._time_k: Dict[int, int] = {}

        self._sigmas = float(max(0.0, float(getattr(cfg, "confidence_sigmas", 0.0) or 0.0)))
        hl_raw = getattr(cfg, "confidence_ema_half_life_episodes", 256.0)
        hl = float(256.0) if hl_raw is None else float(hl_raw)
        self._ema_decay = _ema_decay_from_half_life(hl)
        self._ema_success: Dict[int, float] = {}
        self._ema_weight: Dict[int, float] = {}
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
        self._time_max_k = int(max(1, int(getattr(cfg, "time_budget_max_k", 3) or 3)))
        # Max history needed to evaluate time success targets.
        exp_mult_raw = getattr(cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
        exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
        exp_mult = float(max(0.0, exp_mult))
        time_targets = [
            float(1.0 - math.exp(-float(k) * float(exp_mult))) for k in range(1, self._time_max_k + 1)
        ]
        if self._sigmas > 0.0:
            self._time_success_maxlen = int(
                max(confidence_window_size(target=t, sigmas=self._sigmas) for t in time_targets)
            )
        else:
            self._time_success_maxlen = int(max(1, int(cfg.window_episodes)))

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

    def _times_spawns(self, level: int) -> Deque[int]:
        lvl = int(level)
        if lvl not in self._time_samples_spawns:
            self._time_samples_spawns[lvl] = deque(maxlen=int(self._time_window))
        return self._time_samples_spawns[lvl]

    def _time_success_history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._time_success_histories:
            self._time_success_histories[lvl] = deque(maxlen=int(self._time_success_maxlen))
        return self._time_success_histories[lvl]

    def task_max_frames_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_frames.get(int(level))

    def task_max_spawns_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_spawns.get(int(level))

    def note_episode(
        self,
        *,
        level: int,
        success: bool,
        episode_frames: Optional[int] = None,
        episode_spawns: Optional[int] = None,
        objective_met: Optional[bool] = None,
        stage_token: Optional[object] = None,
    ) -> bool:
        """Record an episode outcome; returns True if this advanced the curriculum."""

        del stage_token
        lvl = int(level)
        base_success = bool(success if objective_met is None else objective_met)
        hist = self._history(lvl)
        hist.append(bool(base_success))
        self._episodes_total[lvl] = int(self._episodes_total.get(lvl, 0)) + 1
        if self._sigmas > 0.0 and float(self._ema_decay) > 0.0:
            prev_s = float(self._ema_success.get(lvl, 0.0) or 0.0)
            prev_w = float(self._ema_weight.get(lvl, 0.0) or 0.0)
            d = float(self._ema_decay)
            prev_s = d * prev_s + (1.0 if bool(base_success) else 0.0)
            prev_w = d * prev_w + 1.0
            self._ema_success[lvl] = float(prev_s)
            self._ema_weight[lvl] = float(prev_w)

        timed_success = bool(success)
        if base_success and episode_frames is not None:
            try:
                frames_i = int(episode_frames)
            except Exception:
                frames_i = 0
            if frames_i > 0:
                self._times(lvl).append(frames_i)
        if base_success and episode_spawns is not None:
            try:
                spawns_i = int(episode_spawns)
            except Exception:
                spawns_i = 0
            if spawns_i >= 0:
                self._times_spawns(lvl).append(max(0, spawns_i))

        # Track time-goal success only while budgets are active for this level.
        if int(lvl) in self._time_budget_frames or int(lvl) in self._time_budget_spawns:
            self._time_success_history(lvl).append(bool(timed_success))

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

        if int(self._episodes_total.get(int(level), 0)) < int(self.cfg.min_episodes):
            return False

        if float(self._ema_decay) <= 0.0:
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

        eff_n = float(self._ema_weight.get(int(level), 0.0) or 0.0)
        eff_s = float(self._ema_success.get(int(level), 0.0) or 0.0)
        min_eff_raw = getattr(self.cfg, "confidence_min_effective_episodes", 128.0)
        min_eff = float(128.0) if min_eff_raw is None else float(min_eff_raw)
        if eff_n < float(min_eff):
            return False
        lb = wilson_lower_bound_fractional(successes=eff_s, n=eff_n, sigmas=float(self._sigmas))
        return bool(lb > t)

    def _maybe_update_time_budget(self, level: int) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "time_budget_enabled", True)):
            return
        mastery_sigmas = float(max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0)))
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        if mastery_sigmas <= 0.0 or mastery_target <= 0.0:
            return
        lvl = int(level)
        window = int(max(1, int(self._mastery_window)))
        hist = self._histories.get(lvl)
        if not hist or len(hist) < window:
            return
        tail = list(hist)[-window:]
        base_mastered = int(sum(1 for x in tail if x)) == window

        if not base_mastered:
            # Falling below base mastery: disable time budgets and require
            # re-mastering before we re-enable time goals.
            self._time_budget_frames.pop(lvl, None)
            self._time_budget_spawns.pop(lvl, None)
            self._time_k.pop(lvl, None)
            if lvl in self._time_success_histories:
                self._time_success_histories[lvl].clear()
            return

        frames_samples = list(self._time_samples.get(lvl, deque()))
        spawns_samples = list(self._time_samples_spawns.get(lvl, deque()))
        if not frames_samples and not spawns_samples:
            return

        # Initialize budgets on first mastery.
        if lvl not in self._time_budget_frames and frames_samples:
            mean_frames = float(np.mean(np.asarray(frames_samples, dtype=np.float64)))
            budget_init = int(max(1, int(round(mean_frames))))
            min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
            self._time_budget_frames[lvl] = int(max(min_frames, budget_init))
        if lvl not in self._time_budget_spawns and spawns_samples:
            mean_spawns = float(np.mean(np.asarray(spawns_samples, dtype=np.float64)))
            budget_init = int(max(1, int(round(mean_spawns))))
            self._time_budget_spawns[lvl] = int(max(1, budget_init))

        if lvl not in self._time_k:
            self._time_k[lvl] = 1
            if lvl in self._time_success_histories:
                self._time_success_histories[lvl].clear()
            return

        # Tighten budgets when time-goal success exceeds the current time target.
        k = int(max(1, min(self._time_max_k, int(self._time_k.get(lvl, 1)))))
        exp_mult_raw = getattr(cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
        exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
        exp_mult = float(max(0.0, exp_mult))
        target = float(1.0 - math.exp(-float(k) * float(exp_mult)))
        if self._sigmas <= 0.0:
            return
        n = int(confidence_window_size(target=target, sigmas=float(self._sigmas)))
        hist_time = self._time_success_histories.get(lvl)
        if not hist_time or len(hist_time) < n:
            return
        tail_time = list(hist_time)[-n:]
        successes = int(sum(1 for x in tail_time if x))
        lb = wilson_lower_bound(successes=successes, n=int(n), sigmas=float(self._sigmas))
        if not bool(lb > target):
            return

        # Compute a capped drop based on MAD of observed successful clear times.
        drop_frac = float(getattr(cfg, "time_budget_max_drop_fraction_of_mad", 0.1) or 0.1)
        min_drop_frames = int(max(1, int(getattr(cfg, "time_budget_min_drop_frames", 1) or 1)))
        mad_frames = _mad([float(t) for t in frames_samples]) if frames_samples else 0.0
        max_drop_frames = int(max(min_drop_frames, int(round(float(mad_frames) * float(drop_frac)))))

        if lvl in self._time_budget_frames and frames_samples:
            cur_budget = int(self._time_budget_frames[lvl])
            new_budget = int(max(1, cur_budget - max_drop_frames))
            min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
            self._time_budget_frames[lvl] = int(max(min_frames, new_budget))

        if lvl in self._time_budget_spawns and spawns_samples:
            cur_budget = int(self._time_budget_spawns[lvl])
            self._time_budget_spawns[lvl] = int(max(1, cur_budget - 1))

        if int(self._time_k.get(lvl, 1)) < int(self._time_max_k):
            self._time_k[lvl] = int(k) + 1
        hist_time.clear()

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

    def stage_token_for_level(self, level: int) -> int:
        # The linear curriculum has one stage per level.
        return int(level)

    def snapshot(self) -> Dict[str, Any]:
        cur = int(self.current_level)
        episodes_total = int(self.episodes_seen(cur))
        if float(self._ema_decay) <= 0.0 or self._sigmas <= 0.0:
            hist = self._histories.get(cur)
            window_size = int(max(1, int(self._stage_window)))
            tail = list(hist)[-window_size:] if hist is not None else []
            window_n = int(len(tail))
            successes = int(sum(1 for x in tail if x))
            rate = float(successes) / float(window_n) if window_n > 0 else 0.0
            lb = wilson_lower_bound(successes=successes, n=window_n, sigmas=float(self._sigmas))
        else:
            eff_n = float(self._ema_weight.get(cur, 0.0) or 0.0)
            eff_s = float(self._ema_success.get(cur, 0.0) or 0.0)
            n_limit = float(_ema_n_limit(float(self._ema_decay)))
            window_size = (
                int(max(1, int(round(n_limit))))
                if math.isfinite(n_limit)
                else int(max(1, int(round(eff_n))))
            )
            window_n = (
                int(max(0, int(round(min(float(window_size), float(eff_n)))))) if eff_n > 0.0 else 0
            )
            successes = int(max(0, int(round(eff_s))))
            rate = float(eff_s) / float(eff_n) if eff_n > 0.0 else 0.0
            lb = (
                wilson_lower_bound_fractional(successes=eff_s, n=eff_n, sigmas=float(self._sigmas))
                if eff_n > 0.0
                else 0.0
            )
        out: Dict[str, Any] = {
            "current_level": cur,
            "rate_current": float(rate),
            "window_n": window_n,
            "window_size": window_size,
            "window_successes": int(successes),
            "confidence_sigmas": float(self._sigmas),
            "confidence_lower_bound": float(lb),
            "episodes_current_total": int(episodes_total),
            "start_level": int(self.cfg.start_level),
            "max_level": int(self.cfg.max_level),
            "success_threshold": float(self.cfg.success_threshold),
            "mode": "linear",
        }
        budget = self._time_budget_frames.get(cur)
        if budget is not None:
            out["time_budget_frames"] = int(budget)
        budget_spawns = self._time_budget_spawns.get(cur)
        if budget_spawns is not None:
            out["time_budget_spawns"] = int(budget_spawns)
        k = self._time_k.get(cur)
        if k is not None:
            out["time_k"] = int(k)
            exp_mult_raw = getattr(self.cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
            exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
            exp_mult = float(max(0.0, exp_mult))
            out["time_target"] = float(1.0 - math.exp(-float(int(k)) * float(exp_mult)))
        times = self._time_samples.get(cur)
        if times:
            times_list = [float(t) for t in list(times)]
            out["time_mean_frames"] = float(np.mean(np.asarray(times_list, dtype=np.float64)))
            out["time_mad_frames"] = float(_mad(times_list))
        spawns = self._time_samples_spawns.get(cur)
        if spawns:
            spawns_list = [float(t) for t in list(spawns)]
            out["time_mean_spawns"] = float(np.mean(np.asarray(spawns_list, dtype=np.float64)))
            out["time_mad_spawns"] = float(_mad(spawns_list))
        return out


@dataclass(slots=True, frozen=True)
class _LnStage:
    level: int
    threshold: float
    frontier: int
    is_probe: bool


class LnHopBackCurriculum:
    """Curriculum that alternates between new levels and ln-tightened hop-backs.

    Pattern (matching the request):
      - When introducing a new frontier level L: require a small probe success rate
        on L (default: 16%).
      - Then "hop back" through all previously introduced levels (start..L) with
        thresholds: 1 - exp(-m·k), where k increases the further back you hop and
        m = pass_ramp_exponent_multiplier.

    This yields a schedule like:
      -15@0.16, -15@0.283
      -14@0.16, -15@0.487, -14@0.283
      -13@0.16, -15@0.632, -14@0.487, -13@0.283
      ...
    """

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        # Base objective success (ignores time budgets).
        self._histories: Dict[int, Deque[bool]] = {}
        self._episodes_total: Dict[int, int] = {}
        # Stage-local histories (fresh stats each time we revisit a level with a
        # different threshold).
        self._stage_histories: Dict[int, Deque[bool]] = {}
        self._stage_episodes_total: Dict[int, int] = {}
        self._stage_frames_total: Dict[int, int] = {}
        self._probe_idx_by_frontier: Dict[int, int] = {}
        self._hop_idx_by_frontier_level: Dict[tuple[int, int], int] = {}
        self._stages: List[_LnStage] = self._build_stages()
        self._stage_idx = 0
        # For ln_hop_back, "mastery" is defined as passing a hop-back stage
        # (not just probing a new frontier). This list is used to choose a
        # shallower hop-back start point.
        self._mastered_levels: set[int] = set()
        self._time_samples: Dict[int, Deque[int]] = {}
        self._time_samples_spawns: Dict[int, Deque[int]] = {}
        self._time_budget_frames: Dict[int, int] = {}
        self._time_budget_spawns: Dict[int, int] = {}
        self._time_success_histories: Dict[int, Deque[bool]] = {}
        self._time_k: Dict[int, int] = {}
        self._stage_decisions_total: Dict[int, int] = {}

        # Bailout: if we can't pass a probe stage within a fraction of the total
        # run budget, hop back to a mastered level.
        bailout_frac_raw = getattr(cfg, "ln_hop_back_bailout_fraction", 0.01)
        bailout_frac = float(0.01) if bailout_frac_raw is None else float(bailout_frac_raw)
        bailout_frac = float(max(0.0, bailout_frac))
        run_total_raw = getattr(cfg, "run_total_steps", None)
        self._bailout_frames: Optional[int]
        if run_total_raw is None:
            self._bailout_frames = None
        else:
            try:
                run_total = int(run_total_raw)
            except Exception:
                run_total = 0
            self._bailout_frames = (
                int(max(1, int(round(float(run_total) * float(bailout_frac)))))
                if run_total > 0 and bailout_frac > 0.0
                else None
            )

        hop_rank_raw = getattr(cfg, "ln_hop_back_hop_back_mastered_rank", 3)
        try:
            hop_rank = int(hop_rank_raw) if hop_rank_raw is not None else 3
        except Exception:
            hop_rank = 3
        self._hop_back_rank = int(max(1, hop_rank))

        self._sigmas = float(max(0.0, float(getattr(cfg, "confidence_sigmas", 0.0) or 0.0)))
        hl_raw = getattr(cfg, "confidence_ema_half_life_episodes", 256.0)
        hl = float(256.0) if hl_raw is None else float(hl_raw)
        self._ema_decay = _ema_decay_from_half_life(hl)
        self._stage_ema_success: Dict[int, float] = {}
        self._stage_ema_weight: Dict[int, float] = {}
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
        self._time_max_k = int(max(1, int(getattr(cfg, "time_budget_max_k", 3) or 3)))
        exp_mult_raw = getattr(cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
        exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
        exp_mult = float(max(0.0, exp_mult))
        # "Pass target" used to decide whether to skip the hop-back chain after
        # probing a new frontier.
        self._pass_target_k1 = float(1.0 - math.exp(-1.0 * float(exp_mult)))
        time_targets = [
            float(1.0 - math.exp(-float(k) * float(exp_mult))) for k in range(1, self._time_max_k + 1)
        ]
        if self._sigmas > 0.0:
            self._time_success_maxlen = int(
                max(confidence_window_size(target=t, sigmas=self._sigmas) for t in time_targets)
            )
        else:
            self._time_success_maxlen = int(max(1, int(cfg.window_episodes)))

    def _build_stages(self) -> List[_LnStage]:
        start = int(self.cfg.start_level)
        end = int(self.cfg.max_level)
        if end < start:
            raise ValueError("curriculum.max_level must be >= curriculum.start_level")
        probe = float(self.cfg.probe_threshold)
        probe = float(min(1.0, max(0.0, probe)))
        max_k = int(max(1, int(getattr(self.cfg, "ln_hop_back_max_k", 5) or 5)))
        exp_mult_raw = getattr(self.cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
        exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
        exp_mult = float(max(0.0, exp_mult))

        stages: List[_LnStage] = []
        for frontier in range(start, end + 1):
            probe_idx = int(len(stages))
            self._probe_idx_by_frontier[int(frontier)] = int(probe_idx)
            stages.append(
                _LnStage(
                    level=int(frontier),
                    threshold=float(probe),
                    frontier=int(frontier),
                    is_probe=True,
                )
            )
            for lvl in range(start, frontier + 1):
                k = int(min(max_k, int(frontier - lvl + 1)))
                thr = float(1.0 - math.exp(-float(k) * float(exp_mult)))
                hop_idx = int(len(stages))
                self._hop_idx_by_frontier_level[(int(frontier), int(lvl))] = int(hop_idx)
                stages.append(
                    _LnStage(
                        level=int(lvl),
                        threshold=float(min(1.0, max(0.0, thr))),
                        frontier=int(frontier),
                        is_probe=False,
                    )
                )
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

    def stage_token_for_level(self, level: int) -> int:
        # The ln-hop-back curriculum's "stage" is the current stage index.
        del level
        return int(self._stage_idx)

    def note_decisions(self, decisions: int) -> None:
        """Record executed macro-decisions for the currently active stage."""

        try:
            d = int(decisions)
        except Exception:
            return
        if d <= 0:
            return
        sidx = int(self._stage_idx)
        self._stage_decisions_total[sidx] = int(self._stage_decisions_total.get(sidx, 0) or 0) + int(d)

    def _hop_back_start_level(self, *, frontier: int) -> int:
        """Return the level to start a hop-back from (Nth-highest mastered)."""

        f = int(frontier)
        mastered = sorted(int(lvl) for lvl in self._mastered_levels if int(lvl) < f)
        if len(mastered) >= int(self._hop_back_rank):
            return int(mastered[-int(self._hop_back_rank)])
        return int(self.cfg.start_level)

    def _hop_back_start_stage_idx(self, *, frontier: int) -> Optional[int]:
        """Return the stage index to jump to when hopping back within a frontier block."""

        f = int(frontier)
        start = int(self.cfg.start_level)
        target_level = int(self._hop_back_start_level(frontier=f))
        target_level = int(max(start, min(target_level, f)))
        idx = self._hop_idx_by_frontier_level.get((f, target_level))
        if idx is None:
            idx = self._hop_idx_by_frontier_level.get((f, start))
        return None if idx is None else int(idx)

    def _history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._histories:
            self._histories[lvl] = deque(maxlen=int(self._history_maxlen))
            self._episodes_total[lvl] = 0
        return self._histories[lvl]

    def _stage_history(self, stage_idx: int) -> Deque[bool]:
        sidx = int(stage_idx)
        if sidx not in self._stage_histories:
            self._stage_histories[sidx] = deque(maxlen=int(self._max_stage_window))
            self._stage_episodes_total[sidx] = 0
        return self._stage_histories[sidx]

    def _times(self, level: int) -> Deque[int]:
        lvl = int(level)
        if lvl not in self._time_samples:
            self._time_samples[lvl] = deque(maxlen=int(self._time_window))
        return self._time_samples[lvl]

    def _times_spawns(self, level: int) -> Deque[int]:
        lvl = int(level)
        if lvl not in self._time_samples_spawns:
            self._time_samples_spawns[lvl] = deque(maxlen=int(self._time_window))
        return self._time_samples_spawns[lvl]

    def _time_success_history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._time_success_histories:
            self._time_success_histories[lvl] = deque(maxlen=int(self._time_success_maxlen))
        return self._time_success_histories[lvl]

    def task_max_frames_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_frames.get(int(level))

    def task_max_spawns_for_level(self, level: int) -> Optional[int]:
        return self._time_budget_spawns.get(int(level))

    def note_episode(
        self,
        *,
        level: int,
        success: bool,
        episode_frames: Optional[int] = None,
        episode_spawns: Optional[int] = None,
        objective_met: Optional[bool] = None,
        stage_token: Optional[object] = None,
    ) -> bool:
        lvl = int(level)
        base_success = bool(success if objective_met is None else objective_met)
        hist = self._history(lvl)
        hist.append(bool(base_success))
        self._episodes_total[lvl] = int(self._episodes_total.get(lvl, 0)) + 1

        # Record stage-local stats (use objective_met if provided, otherwise fall
        # back to success).
        if stage_token is None:
            stage_idx = int(self._stage_idx)
        else:
            try:
                stage_idx = int(stage_token)  # type: ignore[arg-type]
            except Exception:
                stage_idx = int(self._stage_idx)
        if episode_frames is not None:
            try:
                frames_i = int(episode_frames)
            except Exception:
                frames_i = 0
            if frames_i > 0:
                self._stage_frames_total[stage_idx] = int(
                    self._stage_frames_total.get(stage_idx, 0) or 0
                ) + int(frames_i)
        stage_hist = self._stage_history(stage_idx)
        stage_hist.append(bool(base_success))
        self._stage_episodes_total[stage_idx] = int(self._stage_episodes_total.get(stage_idx, 0)) + 1
        if self._sigmas > 0.0 and float(self._ema_decay) > 0.0:
            prev_s = float(self._stage_ema_success.get(stage_idx, 0.0) or 0.0)
            prev_w = float(self._stage_ema_weight.get(stage_idx, 0.0) or 0.0)
            d = float(self._ema_decay)
            prev_s = d * prev_s + (1.0 if bool(base_success) else 0.0)
            prev_w = d * prev_w + 1.0
            self._stage_ema_success[stage_idx] = float(prev_s)
            self._stage_ema_weight[stage_idx] = float(prev_w)

        timed_success = bool(success)
        if base_success and episode_frames is not None:
            try:
                frames_i = int(episode_frames)
            except Exception:
                frames_i = 0
            if frames_i > 0:
                self._times(lvl).append(frames_i)
        if base_success and episode_spawns is not None:
            try:
                spawns_i = int(episode_spawns)
            except Exception:
                spawns_i = 0
            if spawns_i >= 0:
                self._times_spawns(lvl).append(max(0, spawns_i))

        if int(lvl) in self._time_budget_frames or int(lvl) in self._time_budget_spawns:
            self._time_success_history(lvl).append(bool(timed_success))

        self._maybe_update_time_budget(lvl)

        # Only advance when the active stage level completes.
        if lvl != self.current_level:
            return False
        if self._stage_idx >= len(self._stages) - 1:
            return False
        if int(stage_idx) != int(self._stage_idx):
            # Episode belonged to a previous stage; it should not influence
            # advancement of the current stage.
            return False

        stage = self._stages[self._stage_idx]
        passed = bool(self._meets_confidence_target(int(stage_idx), float(stage.threshold)))
        if not passed:
            # Bail out of a stuck probe stage by jumping into the hop-back chain.
            if bool(stage.is_probe) and self._bailout_frames is not None:
                frames_total = int(self._stage_frames_total.get(int(stage_idx), 0) or 0)
                if frames_total >= int(self._bailout_frames):
                    hop_idx = self._hop_back_start_stage_idx(frontier=int(stage.frontier))
                    if hop_idx is not None and int(hop_idx) != int(self._stage_idx):
                        self._stage_idx = int(hop_idx)
                        return True
            return False

        # Stage passed: advance or jump depending on whether this is a probe stage.
        if not bool(stage.is_probe):
            self._mastered_levels.add(int(stage.level))
            self._stage_idx += 1
            return True

        # Probe stage passed. If we're already confidently above the pass target
        # for this frontier, skip the hop-back chain and move to the next frontier.
        next_frontier = int(stage.frontier) + 1
        next_probe_idx = self._probe_idx_by_frontier.get(int(next_frontier))
        if next_probe_idx is not None and self._meets_confidence_target(int(stage_idx), float(self._pass_target_k1)):
            self._stage_idx = int(next_probe_idx)
            return True

        # Otherwise, start hop-back at the Nth-highest mastered level instead of
        # always returning to the curriculum start.
        hop_idx = self._hop_back_start_stage_idx(frontier=int(stage.frontier))
        if hop_idx is not None:
            self._stage_idx = int(hop_idx)
        else:
            self._stage_idx += 1
        return True

    def _meets_confidence_target(self, stage_idx: int, target: float) -> bool:
        t = float(target)
        min_decisions = int(max(0, int(getattr(self.cfg, "min_stage_decisions", 0) or 0)))
        if min_decisions > 0:
            decisions_total = int(self._stage_decisions_total.get(int(stage_idx), 0) or 0)
            if decisions_total < int(min_decisions):
                return False
        if self._sigmas <= 0.0:
            if int(self._stage_episodes_total.get(int(stage_idx), 0)) < int(self.cfg.min_episodes):
                return False
            hist = self._stage_histories.get(int(stage_idx))
            if not hist:
                return False
            return (float(sum(1 for x in hist if x)) / float(len(hist))) >= t

        if int(self._stage_episodes_total.get(int(stage_idx), 0)) < int(self.cfg.min_episodes):
            return False

        if float(self._ema_decay) <= 0.0:
            window_size = confidence_window_size(target=t, sigmas=float(self._sigmas))
            hist = self._stage_histories.get(int(stage_idx))
            if not hist:
                return False
            tail = list(hist)[-int(window_size):]
            if len(tail) < int(window_size):
                return False
            successes = int(sum(1 for x in tail if x))
            lb = wilson_lower_bound(successes=successes, n=int(window_size), sigmas=float(self._sigmas))
            return bool(lb > t)

        eff_n = float(self._stage_ema_weight.get(int(stage_idx), 0.0) or 0.0)
        eff_s = float(self._stage_ema_success.get(int(stage_idx), 0.0) or 0.0)
        min_eff_raw = getattr(self.cfg, "confidence_min_effective_episodes", 128.0)
        min_eff = float(128.0) if min_eff_raw is None else float(min_eff_raw)
        if eff_n < float(min_eff):
            return False
        lb = wilson_lower_bound_fractional(successes=eff_s, n=eff_n, sigmas=float(self._sigmas))
        return bool(lb > t)

    def _maybe_update_time_budget(self, level: int) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "time_budget_enabled", True)):
            return
        mastery_sigmas = float(max(0.0, float(getattr(cfg, "time_budget_mastery_sigmas", 0.0) or 0.0)))
        mastery_target = float(getattr(cfg, "time_budget_mastery_target", 0.99) or 0.99)
        if mastery_sigmas <= 0.0 or mastery_target <= 0.0:
            return
        lvl = int(level)
        window = int(max(1, int(self._mastery_window)))
        hist = self._histories.get(lvl)
        if not hist or len(hist) < window:
            return
        tail = list(hist)[-window:]
        base_mastered = int(sum(1 for x in tail if x)) == window

        if not base_mastered:
            self._time_budget_frames.pop(lvl, None)
            self._time_budget_spawns.pop(lvl, None)
            self._time_k.pop(lvl, None)
            if lvl in self._time_success_histories:
                self._time_success_histories[lvl].clear()
            return

        frames_samples = list(self._time_samples.get(lvl, deque()))
        spawns_samples = list(self._time_samples_spawns.get(lvl, deque()))
        if not frames_samples and not spawns_samples:
            return

        if lvl not in self._time_budget_frames and frames_samples:
            mean_frames = float(np.mean(np.asarray(frames_samples, dtype=np.float64)))
            budget_init = int(max(1, int(round(mean_frames))))
            min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
            self._time_budget_frames[lvl] = int(max(min_frames, budget_init))
        if lvl not in self._time_budget_spawns and spawns_samples:
            mean_spawns = float(np.mean(np.asarray(spawns_samples, dtype=np.float64)))
            budget_init = int(max(1, int(round(mean_spawns))))
            self._time_budget_spawns[lvl] = int(max(1, budget_init))

        if lvl not in self._time_k:
            self._time_k[lvl] = 1
            if lvl in self._time_success_histories:
                self._time_success_histories[lvl].clear()
            return

        k = int(max(1, min(self._time_max_k, int(self._time_k.get(lvl, 1)))))
        exp_mult_raw = getattr(cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
        exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
        exp_mult = float(max(0.0, exp_mult))
        target = float(1.0 - math.exp(-float(k) * float(exp_mult)))
        if self._sigmas <= 0.0:
            return
        n = int(confidence_window_size(target=target, sigmas=float(self._sigmas)))
        hist_time = self._time_success_histories.get(lvl)
        if not hist_time or len(hist_time) < n:
            return
        tail_time = list(hist_time)[-n:]
        successes = int(sum(1 for x in tail_time if x))
        lb = wilson_lower_bound(successes=successes, n=int(n), sigmas=float(self._sigmas))
        if not bool(lb > target):
            return

        drop_frac = float(getattr(cfg, "time_budget_max_drop_fraction_of_mad", 0.1) or 0.1)
        min_drop_frames = int(max(1, int(getattr(cfg, "time_budget_min_drop_frames", 1) or 1)))
        mad_frames = _mad([float(t) for t in frames_samples]) if frames_samples else 0.0
        max_drop_frames = int(max(min_drop_frames, int(round(float(mad_frames) * float(drop_frac)))))

        if lvl in self._time_budget_frames and frames_samples:
            cur_budget = int(self._time_budget_frames[lvl])
            new_budget = int(max(1, cur_budget - max_drop_frames))
            min_frames = int(max(1, int(getattr(cfg, "time_budget_min_frames", 1) or 1)))
            self._time_budget_frames[lvl] = int(max(min_frames, new_budget))

        if lvl in self._time_budget_spawns and spawns_samples:
            cur_budget = int(self._time_budget_spawns[lvl])
            self._time_budget_spawns[lvl] = int(max(1, cur_budget - 1))

        if int(self._time_k.get(lvl, 1)) < int(self._time_max_k):
            self._time_k[lvl] = int(k) + 1
        hist_time.clear()

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
        if float(self._ema_decay) <= 0.0 or self._sigmas <= 0.0:
            hist = self._stage_histories.get(int(self._stage_idx))
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
        else:
            sidx = int(self._stage_idx)
            eff_n = float(self._stage_ema_weight.get(sidx, 0.0) or 0.0)
            eff_s = float(self._stage_ema_success.get(sidx, 0.0) or 0.0)
            n_limit = float(_ema_n_limit(float(self._ema_decay)))
            window_size = (
                int(max(1, int(round(n_limit))))
                if math.isfinite(n_limit)
                else int(max(1, int(round(eff_n))))
            )
            window_n = (
                int(max(0, int(round(min(float(window_size), float(eff_n)))))) if eff_n > 0.0 else 0
            )
            successes = int(max(0, int(round(eff_s))))
            rate = float(eff_s) / float(eff_n) if eff_n > 0.0 else 0.0
            lb = (
                wilson_lower_bound_fractional(successes=eff_s, n=eff_n, sigmas=float(self._sigmas))
                if eff_n > 0.0
                else 0.0
            )
        out: Dict[str, Any] = {
            "current_level": cur,
            "rate_current": float(rate),
            "success_threshold": float(stage.threshold),
            "window_n": window_n,
            "window_size": window_size,
            "window_successes": int(successes),
            "confidence_sigmas": float(self._sigmas),
            "confidence_lower_bound": float(lb),
            "episodes_current_total": int(self._stage_episodes_total.get(int(self._stage_idx), 0)),
            "decisions_current_total": int(self._stage_decisions_total.get(int(self._stage_idx), 0) or 0),
            "min_stage_decisions": int(max(0, int(getattr(self.cfg, "min_stage_decisions", 0) or 0))),
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
        budget_spawns = self._time_budget_spawns.get(cur)
        if budget_spawns is not None:
            out["time_budget_spawns"] = int(budget_spawns)
        k = self._time_k.get(cur)
        if k is not None:
            out["time_k"] = int(k)
            exp_mult_raw = getattr(self.cfg, "pass_ramp_exponent_multiplier", 1.0 / 3.0)
            exp_mult = float(1.0 / 3.0) if exp_mult_raw is None else float(exp_mult_raw)
            exp_mult = float(max(0.0, exp_mult))
            out["time_target"] = float(1.0 - math.exp(-float(int(k)) * float(exp_mult)))
        times = self._time_samples.get(cur)
        if times:
            times_list = [float(t) for t in list(times)]
            out["time_mean_frames"] = float(np.mean(np.asarray(times_list, dtype=np.float64)))
            out["time_mad_frames"] = float(_mad(times_list))
        spawns = self._time_samples_spawns.get(cur)
        if spawns:
            spawns_list = [float(t) for t in list(spawns)]
            out["time_mean_spawns"] = float(np.mean(np.asarray(spawns_list, dtype=np.float64)))
            out["time_mad_spawns"] = float(_mad(spawns_list))
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
        self._env_stage_tokens: List[object] = [0 for _ in range(self.num_envs)]
        self._best_times = None
        self._best_times_topk = int(max(1, int(getattr(cfg, "best_times_topk", 50) or 50)))
        try:
            from pathlib import Path

            from training.diagnostics.best_times import BestTimesDB

            db_path_raw = getattr(cfg, "best_times_db_path", None)
            db_path = (
                Path(str(db_path_raw)).expanduser()
                if db_path_raw
                else BestTimesDB.default_path()
            )
            self._best_times = BestTimesDB(db_path)
        except Exception:
            self._best_times = None

        if not hasattr(env, "set_attr"):
            raise TypeError("CurriculumVecEnv requires an env that supports `set_attr`.")

    def reset(self, *args: Any, **kwargs: Any):
        self._env_levels = [self._curriculum.sample_level(self._rng) for _ in range(self.num_envs)]
        try:
            self._env_stage_tokens = [
                getattr(self._curriculum, "stage_token_for_level")(int(lvl)) for lvl in list(self._env_levels)
            ]
        except Exception:
            self._env_stage_tokens = [0 for _ in range(self.num_envs)]
        self.env.set_attr("level", list(self._env_levels))
        self._set_task_budgets(self._env_levels)
        obs, infos = self.env.reset(*args, **kwargs)
        infos_list = self._ensure_info_list(infos)
        self._inject_curriculum_info(infos_list)
        return obs, infos_list

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        infos_list = self._ensure_info_list(infos)

        if self._mode == "ln_hop_back":
            note_decisions = getattr(self._curriculum, "note_decisions", None)
            if callable(note_decisions):
                try:
                    note_decisions(int(self.num_envs))
                except Exception:
                    pass

        term = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
        trunc = np.asarray(truncated, dtype=bool).reshape(self.num_envs)

        levels_this_step = list(self._env_levels)
        stage_tokens_this_step = list(self._env_stage_tokens)
        next_levels = list(self._env_levels)
        next_stage_tokens = list(self._env_stage_tokens)
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
            episode_spawns: Optional[int] = None
            if "task/spawns_used" in info:
                try:
                    episode_spawns = int(info.get("task/spawns_used")) if info.get("task/spawns_used") is not None else None
                except Exception:
                    episode_spawns = None
            if self._best_times is not None and objective_met:
                rng_seed = info.get("rng_seed")
                frames_used = info.get("task/frames_used", episode_frames)
                spawns_used = info.get("task/spawns_used", 0)
                try:
                    self._best_times.record_clear(
                        level=int(levels_this_step[i]),
                        rng_seed=rng_seed,
                        frames=int(frames_used or 0),
                        spawns=int(spawns_used or 0),
                        run_id=str(getattr(self.cfg, "run_id", "") or ""),
                    )
                except Exception:
                    pass
            prev_level = int(getattr(self._curriculum, "current_level", self.cfg.start_level))
            advanced = self._curriculum.note_episode(
                level=int(levels_this_step[i]),
                success=success,
                episode_frames=episode_frames,
                episode_spawns=episode_spawns,
                objective_met=objective_met,
                stage_token=stage_tokens_this_step[i] if i < len(stage_tokens_this_step) else None,
            )
            if advanced and advanced_to is None:
                advanced_from = prev_level
                advanced_to = int(getattr(self._curriculum, "current_level", prev_level))
            next_levels[i] = int(self._curriculum.sample_level(self._rng))
            try:
                next_stage_tokens[i] = getattr(self._curriculum, "stage_token_for_level")(int(next_levels[i]))
            except Exception:
                next_stage_tokens[i] = stage_tokens_this_step[i] if i < len(stage_tokens_this_step) else 0
            updated = True

        if updated:
            self._env_levels = list(next_levels)
            self._env_stage_tokens = list(next_stage_tokens)
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

    def close(self) -> None:
        if self._best_times is not None:
            try:
                self._best_times.close()
            except Exception:
                pass
        if hasattr(self.env, "close"):
            self.env.close()

    def _set_task_budgets(self, levels: List[int]) -> None:
        cfg = self.cfg
        frames_default = getattr(cfg, "task_max_frames", None)
        spawns_default = getattr(cfg, "task_max_spawns", None)
        frames_by_level = getattr(cfg, "task_max_frames_by_level", {}) or {}
        spawns_by_level = getattr(cfg, "task_max_spawns_by_level", {}) or {}

        dyn_frames_fn = getattr(self._curriculum, "task_max_frames_for_level", None)
        dyn_spawns_fn = getattr(self._curriculum, "task_max_spawns_for_level", None)

        if (
            frames_default is None
            and not frames_by_level
            and spawns_default is None
            and not spawns_by_level
            and not callable(dyn_frames_fn)
            and not callable(dyn_spawns_fn)
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
                    v = int(dyn_val)
                    if self._best_times is not None:
                        try:
                            floor = self._best_times.best_frames_floor(level=int(lvl))
                        except Exception:
                            floor = None
                        if floor is not None:
                            v = max(int(v), int(floor))
                    frames_values.append(int(v))
                else:
                    frames_values.append(_resolve(frames_default, frames_by_level, int(lvl)))
            self.env.set_attr("task_max_frames", frames_values)
        except Exception:
            pass
        try:
            spawns_values = []
            for lvl in list(levels):
                dyn_val: Optional[int] = None
                if callable(dyn_spawns_fn):
                    try:
                        dyn_val = dyn_spawns_fn(int(lvl))
                    except Exception:
                        dyn_val = None
                if dyn_val is not None:
                    v = int(dyn_val)
                    if self._best_times is not None:
                        try:
                            floor = self._best_times.best_spawns_floor(level=int(lvl))
                        except Exception:
                            floor = None
                        if floor is not None:
                            v = max(int(v), int(floor))
                    spawns_values.append(int(v))
                else:
                    spawns_values.append(_resolve(spawns_default, spawns_by_level, int(lvl)))
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
        time_budget_spawns = snap.get("time_budget_spawns")
        time_mean_frames = snap.get("time_mean_frames")
        time_mad_frames = snap.get("time_mad_frames")
        time_mean_spawns = snap.get("time_mean_spawns")
        time_mad_spawns = snap.get("time_mad_spawns")
        time_k = snap.get("time_k")
        time_target = snap.get("time_target")
        stage_idx = snap.get("stage_index")
        stage_count = snap.get("stage_count")
        probe_threshold = snap.get("probe_threshold")
        decisions_current_total = snap.get("decisions_current_total")
        min_stage_decisions = snap.get("min_stage_decisions")
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
            if time_budget_spawns is not None:
                try:
                    info["curriculum/time_budget_spawns"] = int(time_budget_spawns)
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
            if time_mean_spawns is not None:
                try:
                    info["curriculum/time_mean_spawns"] = float(time_mean_spawns)
                except Exception:
                    pass
            if time_mad_spawns is not None:
                try:
                    info["curriculum/time_mad_spawns"] = float(time_mad_spawns)
                except Exception:
                    pass
            if time_k is not None:
                try:
                    info["curriculum/time_k"] = int(time_k)
                except Exception:
                    pass
            if time_target is not None:
                try:
                    info["curriculum/time_target"] = float(time_target)
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
            if decisions_current_total is not None:
                try:
                    info["curriculum/decisions_current_total"] = int(decisions_current_total)
                except Exception:
                    pass
            if min_stage_decisions is not None:
                try:
                    info["curriculum/min_stage_decisions"] = int(min_stage_decisions)
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
