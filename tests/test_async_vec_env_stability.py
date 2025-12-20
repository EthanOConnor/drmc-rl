from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from training.envs.dr_mario_vec import VecEnvConfig, make_vec_env


ENGINE_PATH = Path("game_engine/drmario_engine")


def _sample_action(info: dict, rng: np.random.Generator) -> int:
    mask = info.get("placements/feasible_mask")
    if mask is not None:
        try:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            idxs = np.flatnonzero(m)
            if idxs.size > 0:
                return int(rng.choice(idxs))
        except Exception:
            pass
    return int(rng.integers(0, 1))


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_async_vec_env_autoreset_does_not_lose_state_cache() -> None:
    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=8,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-engine",
        state_repr="bitplane_reduced_mask",
        vectorization="async",
        emit_raw_ram=False,
    )
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=0)
        assert obs is not None
        assert isinstance(infos, (list, tuple))
        infos_list = list(infos)
        rng = np.random.default_rng(0)

        # Run a short loop to exercise autoresets and backend fast-forwarding.
        for _ in range(64):
            actions = np.array(
                [_sample_action(infos_list[i] if i < len(infos_list) else {}, rng) for i in range(cfg.num_envs)],
                dtype=np.int64,
            )
            obs, rewards, terminated, truncated, infos = env.step(actions)
            assert obs is not None
            assert rewards is not None
            assert terminated is not None
            assert truncated is not None
            assert isinstance(infos, (list, tuple))
            infos_list = list(infos)
    finally:
        env.close()

