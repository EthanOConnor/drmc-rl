from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from training.envs.curriculum import CurriculumConfig, CurriculumVecEnv


@dataclass
class _MiniEnv:
    level: int = 0


class _MiniVecEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = int(num_envs)
        self.envs = [_MiniEnv() for _ in range(self.num_envs)]
        self._steps = np.zeros(self.num_envs, dtype=np.int64)

    def set_attr(self, name: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        assert len(values) == self.num_envs
        for env, value in zip(self.envs, values):
            setattr(env, name, int(value))

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self._steps.fill(0)
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        infos = [{"level": env.level} for env in self.envs]
        return obs, infos

    def step(self, actions: Any):
        self._steps += 1
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = self._steps >= 1  # terminate every step for simplicity
        truncated = np.zeros_like(terminated, dtype=bool)
        infos = []
        for i, env in enumerate(self.envs):
            infos.append({"cleared": True, "level": env.level})
        return obs, rewards, terminated, truncated, infos


def test_curriculum_vec_env_sets_levels_and_injects_info() -> None:
    base = _MiniVecEnv(num_envs=2)
    cfg = CurriculumConfig(
        enabled=True,
        start_level=-4,
        max_level=-3,
        success_threshold=1.0,
        window_episodes=2,
        min_episodes=2,
        rehearsal_prob=0.0,
        seed=0,
    )
    env = CurriculumVecEnv(base, cfg)

    _, infos = env.reset()
    assert len(infos) == 2
    assert infos[0]["curriculum/env_level"] == -4
    assert infos[0]["curriculum/current_level"] == -4

    # Two episodes where env0 is always successful should advance from -4 -> -3.
    for _ in range(2):
        _, _, terminated, _, infos = env.step([0, 0])
        assert bool(np.asarray(terminated).all())
        assert infos[0]["curriculum/env_level"] in (-4, -3)

    # After advancement, the current_level snapshot reflects -3.
    _, infos = env.reset()
    assert infos[0]["curriculum/current_level"] == -3
