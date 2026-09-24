"""CUDA twin of drm_reach_bfs_full: byte parity and arena routing."""
from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.execution.pace import BY_ID
from tools.trainer_event_rollout import planner_backend, unconstrained


def test_only_unconstrained_profiles_route_to_cuda():
    assert unconstrained(BY_ID["frame_perfect"].planner_args(0))
    assert unconstrained(BY_ID["frame_perfect"].planner_args(8))
    assert unconstrained({})
    for pace in BY_ID.values():
        if pace.id != "frame_perfect":
            for delay in (0, 4, 60):
                assert not unconstrained(pace.planner_args(delay)), pace.id


def test_planner_backend_knob():
    config = {"variants": {"a": {}, "b": {"planner_backend": "cuda"}}}
    assert planner_backend(config) == "cpu"
    assert planner_backend(config, "a") == "cpu"
    assert planner_backend(config, "b") == "cuda"
    config["planner_backend"] = "cuda"
    assert planner_backend(config, "a") == "cuda"
    with pytest.raises(ValueError):
        planner_backend({"variants": {}, "planner_backend": "gpu"})


def _cuda():
    try:
        from drmc_rl.planning.cuda.full import CudaReachFull
        from drmc_rl.planning.native_reach import is_library_present
        if not is_library_present():
            pytest.skip("native reach library not built")
        return CudaReachFull(max_batch=256)
    except Exception as error:  # no driver, no GPU, no cuda-bindings
        pytest.skip(f"CUDA unavailable: {error}")


def test_cuda_full_matches_cpu_bytes():
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from tools.test_reach_full_cuda_parity import random_board, random_spawn

    solver = _cuda()
    try:
        rng = np.random.default_rng(7)
        boards, spawns, thresholds = [], [], []
        for i in range(300):
            threshold = int(rng.integers(0, 40))
            boards.append(random_board(rng, i % 4))
            spawns.append(random_spawn(rng, threshold))
            thresholds.append(threshold)
        batch = solver.solve(solver.pack(np.stack(boards), spawns, thresholds))
    finally:
        solver.close()
    cpu = NativeReachabilityRunner()
    for i, (board, spawn, threshold) in enumerate(zip(boards, spawns, thresholds)):
        assert batch.status[i] == 0
        expected = cpu.bfs_full(board, spawn, speed_threshold=threshold)
        actual = batch.reach(i)
        for field in ("costs_u16", "offsets_u16", "lengths_u16", "script_buf"):
            assert getattr(expected, field).tobytes() == getattr(actual, field).tobytes(), (i, field)
