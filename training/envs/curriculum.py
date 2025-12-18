from __future__ import annotations

"""Scripted curriculum support for Dr. Mario training.

The curriculum requested for the placement macro environment uses "synthetic"
negative levels as a convenience encoding:

  - level  0: 4 viruses (vanilla)
  - level -1: 3 viruses
  - level -2: 2 viruses
  - level -3: 1 virus
  - level -10..-4: 0 viruses, goal = clear N matches (4-match events)
      - level -10: 1 match ("any match")
      - level  -9: 2 matches
      ...
      - level  -4: 7 matches

The environment implements the negative-level semantics by patching the bottle
RAM at reset time (see `envs/retro/drmario_env.py`).
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = False
    start_level: int = -10
    max_level: int = 0
    success_threshold: float = 0.9
    window_episodes: int = 100
    min_episodes: int = 50
    rehearsal_prob: float = 0.1
    seed: Optional[int] = None

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

        return cls(
            enabled=bool(_get("enabled", False)),
            start_level=int(_get("start_level", -10)),
            max_level=int(_get("max_level", 0)),
            success_threshold=float(_get("success_threshold", 0.9)),
            window_episodes=int(_get("window_episodes", 100)),
            min_episodes=int(_get("min_episodes", 50)),
            rehearsal_prob=float(_get("rehearsal_prob", 0.1)),
            seed=None if data.get("seed") is None else int(data["seed"]),
        )


class ScriptedCurriculum:
    """Tracks success rates and advances the active curriculum level."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self.current_level = int(cfg.start_level)
        self._histories: Dict[int, Deque[bool]] = {}
        self._episodes_total: Dict[int, int] = {}

    def _history(self, level: int) -> Deque[bool]:
        lvl = int(level)
        if lvl not in self._histories:
            self._histories[lvl] = deque(maxlen=max(1, int(self.cfg.window_episodes)))
            self._episodes_total[lvl] = 0
        return self._histories[lvl]

    def note_episode(self, *, level: int, success: bool) -> bool:
        """Record an episode outcome; returns True if this advanced the curriculum."""

        lvl = int(level)
        hist = self._history(lvl)
        hist.append(bool(success))
        self._episodes_total[lvl] = int(self._episodes_total.get(lvl, 0)) + 1

        if lvl != int(self.current_level):
            return False
        if int(self.current_level) >= int(self.cfg.max_level):
            return False
        if int(self._episodes_total.get(lvl, 0)) < int(self.cfg.min_episodes):
            return False

        rate = self.success_rate(lvl)
        if rate >= float(self.cfg.success_threshold):
            self.current_level = min(int(self.cfg.max_level), int(self.current_level) + 1)
            return True
        return False

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
        window_n = int(len(hist)) if hist is not None else 0
        window_size = int(max(1, int(self.cfg.window_episodes)))
        return {
            "current_level": cur,
            "rate_current": float(self.success_rate(cur)),
            "window_n": window_n,
            "window_size": window_size,
            "episodes_current_total": int(self.episodes_seen(cur)),
            "start_level": int(self.cfg.start_level),
            "max_level": int(self.cfg.max_level),
        }


class CurriculumVecEnv:
    """Vector-env wrapper that assigns per-episode curriculum levels."""

    def __init__(self, env: Any, cfg: CurriculumConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.num_envs = int(getattr(env, "num_envs", 1))
        self._rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))
        self._curriculum = ScriptedCurriculum(cfg)
        self._env_levels: List[int] = [int(cfg.start_level) for _ in range(self.num_envs)]

        if not hasattr(env, "set_attr"):
            raise TypeError("CurriculumVecEnv requires an env that supports `set_attr`.")

    def reset(self, *args: Any, **kwargs: Any):
        self._env_levels = [self._curriculum.sample_level(self._rng) for _ in range(self.num_envs)]
        self.env.set_attr("level", list(self._env_levels))
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
        for i in range(self.num_envs):
            if not bool(term[i] or trunc[i]):
                continue
            info = infos_list[i] if i < len(infos_list) else {}
            success = bool(
                info.get("goal_achieved")
                if "goal_achieved" in info
                else info.get("cleared", info.get("drm", {}).get("cleared", False))
            )
            self._curriculum.note_episode(level=int(levels_this_step[i]), success=success)
            next_levels[i] = int(self._curriculum.sample_level(self._rng))
            updated = True

        if updated:
            self._env_levels = list(next_levels)
            self.env.set_attr("level", list(self._env_levels))

        self._inject_curriculum_info(infos_list, levels=levels_this_step, next_levels=next_levels)
        return obs, rewards, terminated, truncated, infos_list

    def _inject_curriculum_info(
        self,
        infos: List[Dict[str, Any]],
        *,
        levels: Optional[List[int]] = None,
        next_levels: Optional[List[int]] = None,
    ) -> None:
        snap = self._curriculum.snapshot()
        window_n = int(snap.get("window_n", 0) or 0)
        window_size = int(snap.get("window_size", 1) or 1)
        episodes_total = int(snap.get("episodes_current_total", 0) or 0)
        for i in range(min(len(infos), self.num_envs)):
            info = infos[i]
            if not isinstance(info, dict):
                continue
            level_i = int(levels[i]) if levels is not None and i < len(levels) else int(self._env_levels[i])
            info["curriculum/env_level"] = int(level_i)
            if next_levels is not None and i < len(next_levels):
                info["curriculum/next_env_level"] = int(next_levels[i])
            info["curriculum/current_level"] = int(snap["current_level"])
            info["curriculum/rate_current"] = float(snap["rate_current"])
            info["curriculum/window_n"] = int(window_n)
            info["curriculum/window_size"] = int(window_size)
            info["curriculum/episodes_current_total"] = int(episodes_total)

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
