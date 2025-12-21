from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from envs.retro.drmario_env import DrMarioRetroEnv
from envs.retro.placement_env import DrMarioPlacementEnv


ENGINE_PATH = Path("game_engine/drmario_engine")


def _sample_feasible_action(info: dict) -> int:
    mask = info.get("placements/feasible_mask")
    assert mask is not None
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idxs = np.flatnonzero(m)
    assert idxs.size > 0
    return int(idxs[0])


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_cpp_engine_timeout_in_run_frames_mask_truncates_instead_of_raising(monkeypatch) -> None:
    base = DrMarioRetroEnv(
        obs_mode="state",
        backend="cpp-engine",
        backend_kwargs={"engine_path": str(ENGINE_PATH), "build_if_missing": False},
        state_repr="bitplane_bottle_mask",
        auto_start=False,
        emit_raw_ram=False,
    )
    env = DrMarioPlacementEnv(base)
    try:
        _obs, info = env.reset(seed=0)
        assert bool(info.get("placements/needs_action", False))
        action = _sample_feasible_action(info)

        backend = base._backend
        assert backend is not None

        calls = {"n": 0}

        def boom(*, buttons_mask: int, frames: int) -> dict[str, int]:
            calls["n"] += 1
            raise TimeoutError("synthetic timeout")

        monkeypatch.setattr(backend, "run_frames_mask", boom)

        _obs2, _r, _terminated, truncated, info2 = env.step(action)
        assert bool(truncated) is True
        assert int(calls["n"]) > 0
        assert "placements/backend_error" in info2
        assert info2.get("placements/backend_error_phase") == "run_frames_mask"
    finally:
        env.close()


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_cpp_engine_timeout_in_run_until_decision_truncates_instead_of_raising(monkeypatch) -> None:
    base = DrMarioRetroEnv(
        obs_mode="state",
        backend="cpp-engine",
        backend_kwargs={"engine_path": str(ENGINE_PATH), "build_if_missing": False},
        state_repr="bitplane_bottle_mask",
        auto_start=False,
        emit_raw_ram=False,
    )
    env = DrMarioPlacementEnv(base)
    try:
        _obs, info = env.reset(seed=0)
        assert bool(info.get("placements/needs_action", False))
        action = _sample_feasible_action(info)

        backend = base._backend
        assert backend is not None

        calls = {"n": 0}
        original = backend.run_until_decision

        def boom(*, last_spawn_id: int | None, max_frames: int) -> dict[str, int]:
            calls["n"] += 1
            raise TimeoutError("synthetic timeout")

        monkeypatch.setattr(backend, "run_until_decision", boom)

        _obs2, _r, _terminated, truncated, info2 = env.step(action)
        assert bool(truncated) is True
        assert int(calls["n"]) > 0
        assert "placements/backend_error" in info2
        assert info2.get("placements/backend_error_phase") == "run_until_decision"
        # Restore for clean shutdown paths that may reuse the backend.
        monkeypatch.setattr(backend, "run_until_decision", original)
    finally:
        env.close()

