from __future__ import annotations

import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present
from tools.bench_multienv import _run_bench
from drmc_rl.training.envs.dr_mario_vec import VecEnvConfig


def test_bench_multienv_dummy_reports_component_metrics() -> None:
    cfg = VecEnvConfig(
        id="Dummy-v0",
        obs_mode="state",
        num_envs=2,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="mock",
        state_repr="bitplane_bottle",
        vectorization="sync",
        emit_raw_ram=False,
    )
    metrics = _run_bench(
        cfg,
        duration_sec=0.01,
        warmup_steps=1,
        seed=0,
        action_mode="first",
        max_batches=2,
    )
    assert float(metrics["fps_total"]) > 0.0
    assert float(metrics["dps_total"]) > 0.0
    assert int(metrics["batches_total"]) == 2
    assert float(metrics["env_step_ms_mean"]) >= 0.0
    assert float(metrics["batch_wall_ms_p95"]) >= 0.0
    assert 0.0 <= float(metrics["harness_overhead_frac"]) <= 1.0


@pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
def test_bench_multienv_cpp_pool_smoke() -> None:
    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=1,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-pool",
        state_repr="bitplane_bottle_conn_mask",
        vectorization="sync",
        emit_raw_ram=False,
    )
    metrics = _run_bench(
        cfg,
        duration_sec=0.05,
        warmup_steps=1,
        seed=0,
        action_mode="first",
        max_batches=2,
    )
    assert float(metrics.get("fps_total", 0.0)) > 0.0
    assert float(metrics.get("fps_per_env", 0.0)) > 0.0
    assert float(metrics.get("dps_total", 0.0)) > 0.0
