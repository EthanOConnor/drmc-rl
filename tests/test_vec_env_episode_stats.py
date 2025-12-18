from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from training.envs.dr_mario_vec import _InfoListWrapper


class _FakeVectorEnv:
    """Tiny vector-env stub returning dict-of-arrays infos (Gymnasium style)."""

    def __init__(self) -> None:
        self.num_envs = 2
        self._step = 0

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step = 0
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        infos = {"viruses_remaining": np.array([10, 10], dtype=np.int32)}
        return obs, infos

    def step(self, actions: Any):  # noqa: ANN401
        self._step += 1
        obs = np.full((self.num_envs, 1), float(self._step), dtype=np.float32)

        if self._step == 1:
            rewards = np.array([1.0, 2.0], dtype=np.float32)
            terminated = np.array([False, False])
            truncated = np.array([False, False])
            infos = {
                "placements/tau": np.array([3, 4], dtype=np.int32),
                "viruses_remaining": np.array([10, 9], dtype=np.int32),
                "topout": np.array([False, False]),
                "cleared": np.array([False, False]),
                "level": np.array([0, 0], dtype=np.int32),
            }
        elif self._step == 2:
            rewards = np.array([3.0, 4.0], dtype=np.float32)
            terminated = np.array([True, False])
            truncated = np.array([False, True])
            infos = {
                "placements/tau": np.array([2, 1], dtype=np.int32),
                "viruses_remaining": np.array([7, 0], dtype=np.int32),
                "topout": np.array([True, False]),
                "cleared": np.array([False, True]),
                "level": np.array([0, 0], dtype=np.int32),
            }
        else:
            rewards = np.array([5.0, 6.0], dtype=np.float32)
            terminated = np.array([True, True])
            truncated = np.array([False, False])
            infos = {
                "placements/tau": np.array([1, 1], dtype=np.int32),
                "viruses_remaining": np.array([7, 0], dtype=np.int32),
                "topout": np.array([False, False]),
                "cleared": np.array([False, True]),
                "level": np.array([0, 0], dtype=np.int32),
            }

        return obs, rewards, terminated, truncated, infos


def test_info_list_wrapper_attaches_episode_stats_using_tau() -> None:
    env = _InfoListWrapper(_FakeVectorEnv())
    _obs, infos = env.reset()
    assert isinstance(infos, list)
    assert len(infos) == 2

    _obs, _rewards, _term, _trunc, infos = env.step(np.zeros((2,), dtype=np.int64))
    assert "episode" not in infos[0]
    assert "episode" not in infos[1]

    _obs, _rewards, term, trunc, infos = env.step(np.zeros((2,), dtype=np.int64))
    assert bool(term[0]) is True
    assert bool(trunc[1]) is True

    ep0 = infos[0]["episode"]
    ep1 = infos[1]["episode"]
    assert ep0["r"] == 4.0
    assert ep1["r"] == 6.0
    assert ep0["l"] == 5
    assert ep1["l"] == 5
    assert ep0["decisions"] == 2
    assert ep1["decisions"] == 2

    drm0 = infos[0]["drm"]
    drm1 = infos[1]["drm"]
    assert drm0["viruses_cleared"] == 3
    assert drm1["viruses_cleared"] == 10
    assert drm0["top_out"] is True
    assert drm1["top_out"] is False
    assert drm0["cleared"] is False
    assert drm1["cleared"] is True

    # Counters reset after episode end.
    _obs, _rewards, _term, _trunc, infos = env.step(np.zeros((2,), dtype=np.int64))
    assert infos[0]["episode"]["r"] == 5.0
    assert infos[1]["episode"]["r"] == 6.0
    assert infos[0]["episode"]["l"] == 1
    assert infos[1]["episode"]["l"] == 1
