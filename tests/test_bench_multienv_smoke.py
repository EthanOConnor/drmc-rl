from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.bench_multienv import _run_bench
from training.envs.dr_mario_vec import VecEnvConfig


ENGINE_PATH = Path("game_engine/drmario_engine")


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_bench_multienv_smoke() -> None:
    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=1,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-engine",
        state_repr="bitplane_bottle",
        vectorization="sync",
        emit_raw_ram=False,
    )
    metrics = _run_bench(cfg, duration_sec=0.2, warmup_steps=1, seed=0)
    assert float(metrics.get("fps_total", 0.0)) > 0.0
    assert float(metrics.get("fps_per_env", 0.0)) > 0.0
    assert float(metrics.get("dps_total", 0.0)) > 0.0
