"""Gate-best snapshot protection: collapsed-looking policies must not enter
the opponent pool; healthy ones must."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from drmc_rl.envs.backends.drmario_pool import is_library_present

pytestmark = pytest.mark.skipif(
    not is_library_present(), reason="native pool library missing"
)

SEEDS = [Path("runs/bc_opponents/bc_lt1600.pt.gz")]


def _env(tmp_path, gate):
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    if not SEEDS[0].is_file():
        pytest.skip("seed checkpoint not staged")
    return DrMarioVsPoolVecEnv(
        num_pairs=2,
        level=14,
        opponent_pool_cfg={
            "enabled": True,
            "max_pool": 6,
            "dir": str(tmp_path / "pool"),
            "seed_paths": [str(SEEDS[0])],
            "snapshot_every_matches": 1,
            "snapshot_gate": gate,
            "device": "cpu",
        },
    )


def _fake_sd():
    return {"w": torch.zeros(1)}


def test_gate_blocks_collapsed_matches(tmp_path):
    env = _env(tmp_path, {"enabled": True, "min_match_len_sec": 60.0})
    env.reset(seed=0)
    env._matches_since_snapshot = 10
    env._match_len_sec.extend([12.0] * 30)  # collapsed: 12s median
    n0 = len(env._opp_pool.entries)
    assert env.maybe_snapshot(_fake_sd) is False
    assert len(env._opp_pool.entries) == n0
    assert env.get_vs_metrics()["vs/snapshots_blocked"] == 1.0

    # Recovery: healthy match lengths -> snapshot admitted on retry.
    env._match_len_sec.clear()
    env._match_len_sec.extend([200.0] * 30)
    assert env.maybe_snapshot(_fake_sd) is True
    assert len(env._opp_pool.entries) == n0 + 1
    env.close()


def test_gate_disabled_preserves_behavior(tmp_path):
    env = _env(tmp_path, {"enabled": False})
    env.reset(seed=0)
    env._matches_since_snapshot = 10
    env._match_len_sec.extend([12.0] * 30)
    n0 = len(env._opp_pool.entries)
    assert env.maybe_snapshot(_fake_sd) is True  # old behavior: no gate
    assert len(env._opp_pool.entries) == n0 + 1
    env.close()
