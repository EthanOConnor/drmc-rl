from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from training.envs.interactive import PlaybackControl, RateLimitedVecEnv


class _FakeVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        return obs, [{}]

    def step(
        self, actions: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        infos = [
            {
                "placements/tau": 5,
                "perf/planner_build_sec": 0.002,
                "perf/planner_plan_sec": 0.001,
            }
        ]
        return obs, rewards, terminated, truncated, infos


def test_rate_limited_vec_env_accumulates_perf_counters():
    env = RateLimitedVecEnv(_FakeVecEnv(), PlaybackControl())
    env.reset()

    env.record_inference(0.010)
    env.record_update(0.050, frames=50)
    env.step(np.zeros((1,), dtype=np.int64))

    perf = env.perf_snapshot()
    assert int(perf.get("inference_calls", 0)) == 1
    assert float(perf.get("last_inference_ms", 0.0)) > 0.0

    assert int(perf.get("planner_build_calls", 0)) == 1
    assert int(perf.get("planner_plan_calls", 0)) == 1
    assert int(perf.get("planner_calls", 0)) == 2
    assert float(perf.get("planner_ms_per_frame", 0.0)) > 0.0

    assert int(perf.get("update_calls", 0)) == 1
    assert int(perf.get("update_frames_last", 0)) == 50
    assert float(perf.get("update_ms_per_frame", 0.0)) > 0.0
