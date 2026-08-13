from __future__ import annotations

"""League roles for the VS opponent pool (docs/LEAGUE.md).

Unit tests: config parsing, exploiter/mixed sampling, PFSP weighting over
per-target win rates, manifest round-trip. Integration smoke: exploiter mode
against a real champion checkpoint on the real vspool.
"""

from pathlib import Path

import numpy as np
import pytest

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.training.envs.vs_opponents import (
    LeagueConfig,
    OpponentPool,
    parse_league_config,
)

CHAMPION_TARGET = Path("runs/best_agents/smdp_ppo_step535164979.pt.gz")


def _has_vspool_symbols() -> bool:
    if not is_library_present():
        return False
    try:
        import ctypes

        from drmc_rl.envs.backends.drmario_pool import resolve_library_path

        lib = ctypes.CDLL(str(resolve_library_path()))
        return hasattr(lib, "drm_vspool_create")
    except Exception:
        return False


# ------------------------------------------------------------------ config parsing
def test_parse_league_config_defaults_to_pfsp() -> None:
    for cfg in (None, {}):
        league = parse_league_config(cfg)
        assert league.mode == "pfsp"
        assert league.main_agents == []


def test_parse_league_config_full() -> None:
    league = parse_league_config(
        {"mode": "mixed", "main_agents": ["a.pt.gz", "b.pt.gz"], "exploiter_fraction": 0.25}
    )
    assert league.mode == "mixed"
    assert league.main_agents == [Path("a.pt.gz"), Path("b.pt.gz")]
    assert league.exploiter_fraction == pytest.approx(0.25)


def test_parse_league_config_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        parse_league_config({"mode": "bogus"})
    with pytest.raises(ValueError):
        parse_league_config("exploiter")
    with pytest.raises(ValueError):
        parse_league_config({"mode": "exploiter"})  # no main_agents
    with pytest.raises(ValueError):
        parse_league_config({"mode": "mixed"})  # no main_agents
    with pytest.raises(ValueError):
        parse_league_config({"mode": "mixed", "main_agents": ["a"], "exploiter_fraction": 1.5})


# ------------------------------------------------------------------ sampling
def _league_pool(tmp_path, mode: str, fraction: float = 0.3) -> OpponentPool:
    pool = OpponentPool(
        tmp_path / "pool",
        league=LeagueConfig(mode=mode, main_agents=[Path("x")], exploiter_fraction=fraction),
    )
    pool.add_loaded("target_a", object(), league_target=True)
    pool.add_loaded("target_b", object(), league_target=True)
    pool.add_loaded("snap_1", object())
    pool.add_loaded("snap_2", object())
    return pool


def test_exploiter_never_samples_self_snapshots(tmp_path) -> None:
    pool = _league_pool(tmp_path, "exploiter")
    rng = np.random.default_rng(0)
    picks = {pool.sample(rng).id for _ in range(300)}
    assert picks == {"target_a", "target_b"}


def test_mixed_fraction_respected(tmp_path) -> None:
    frac = 0.3
    pool = _league_pool(tmp_path, "mixed", fraction=frac)
    rng = np.random.default_rng(7)
    n = 3000
    picks = [pool.sample(rng).id for _ in range(n)]
    target_share = sum(p.startswith("target") for p in picks) / n
    assert target_share == pytest.approx(frac, abs=0.04)
    # Both subsets actually get play.
    assert any(p.startswith("snap") for p in picks)


def test_mixed_falls_back_to_targets_when_no_history(tmp_path) -> None:
    pool = OpponentPool(
        tmp_path / "pool",
        league=LeagueConfig(mode="mixed", main_agents=[Path("x")], exploiter_fraction=0.0),
    )
    pool.add_loaded("target_a", object(), league_target=True)
    rng = np.random.default_rng(0)
    assert pool.sample(rng).id == "target_a"


def test_exploiter_pfsp_weighting_over_targets(tmp_path) -> None:
    pool = OpponentPool(
        tmp_path / "pool",
        league=LeagueConfig(mode="exploiter", main_agents=[Path("x")]),
    )
    close = pool.add_loaded("close", object(), league_target=True)
    close.wins, close.games = 45.0, 100  # ~45%: closest to cracking
    crushed_by = pool.add_loaded("crushed_by", object(), league_target=True)
    crushed_by.wins, crushed_by.games = 2.0, 100  # learner rarely wins

    w = pool.pfsp_weights(pool.league_targets())
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]
    # Same (p(1-p))^2 + floor math as the history pool.
    p = close.smoothed_winrate()
    assert close.pfsp_weight() == pytest.approx((p * (1 - p)) ** 2 + 0.05)

    rng = np.random.default_rng(3)
    picks = [pool.sample(rng).id for _ in range(400)]
    assert picks.count("close") > picks.count("crushed_by")
    assert picks.count("crushed_by") > 0  # floor: never starved


# ------------------------------------------------------------------ manifest
def test_manifest_roundtrip_with_league_state(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    target_ckpt = tmp_path / "champ_step530.pt.gz"
    save_checkpoint({"state_dict": {"w": torch.zeros(1)}, "cfg": {}, "step": 530}, target_ckpt)

    league = LeagueConfig(mode="exploiter", main_agents=[target_ckpt])
    pool_dir = tmp_path / "pool"
    pool = OpponentPool(pool_dir, league=league)
    pool.seed_league_targets(league.main_agents)
    assert len(pool.entries) == 1
    entry = pool.entries[0]
    assert entry.league_target and entry.protected

    pool.record(entry.id, True)
    pool.record(entry.id, False)
    pool.record(entry.id, 0.5)

    reloaded = OpponentPool(pool_dir, league=league)
    re_entry = reloaded.entries[0]
    assert re_entry.id == entry.id
    assert re_entry.league_target and re_entry.protected
    assert re_entry.wins == pytest.approx(1.5)
    assert re_entry.games == 3

    # Re-seeding on restart is idempotent: no duplicate entry, counts kept.
    reloaded.seed_league_targets(league.main_agents)
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].games == 3


# ------------------------------------------------------------------ env integration
@pytest.mark.skipif(
    not _has_vspool_symbols(),
    reason="cpp-vs-pool library missing (build with: make -C vendor/drmario_native libdrmario_pool)",
)
@pytest.mark.skipif(
    not CHAMPION_TARGET.is_file(),
    reason=f"champion checkpoint missing: {CHAMPION_TARGET}",
)
def test_exploiter_env_smoke_vs_real_champion(tmp_path) -> None:
    pytest.importorskip("torch")
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    num_pairs = 2
    env = DrMarioVsPoolVecEnv(
        num_pairs=num_pairs,
        state_repr="bitplane_bottle_conn_mask",
        level=0,
        speed_setting=2,
        randomize_rng=True,
        opponent_pool_cfg={
            "enabled": True,
            "dir": str(tmp_path / "pool"),
            "snapshot_every_matches": 1,
            "league": {"mode": "exploiter", "main_agents": [str(CHAMPION_TARGET)]},
        },
    )
    rng = np.random.default_rng(0)
    try:
        pool = env._opp_pool
        assert pool.league.mode == "exploiter"
        assert [e.league_target for e in pool.entries] == [True]
        target_id = pool.entries[0].id

        obs, infos = env.reset(seed=0)
        assert obs.shape == (num_pairs, 12, 16, 8)
        # 12-channel arch path: the loaded champion reads board_channels<=12.
        assert all(e is not None and e.id == target_id for e in env._pair_opponents)

        def learner_actions(infos_):
            actions = []
            for info in infos_:
                mask = np.asarray(info["placements/feasible_mask"], dtype=np.uint8)
                idxs = np.flatnonzero(mask.reshape(-1))
                if bool(info.get("placements/needs_action")) and idxs.size > 0:
                    actions.append(int(rng.choice(idxs)))
                else:
                    actions.append(-1)
            return np.asarray(actions, dtype=np.int32)

        matches_target = 2
        for _ in range(5000):
            obs, rewards, terminated, truncated, infos = env.step(learner_actions(infos))
            if env.get_vs_metrics().get("vs/matches_total", 0.0) >= matches_target:
                break

        metrics = env.get_vs_metrics()
        assert metrics["vs/matches_total"] >= matches_target
        assert metrics["vs/league_targets"] == 1.0
        assert 0.0 <= metrics["vs/league_win_rate"] <= 1.0
        assert metrics["vs/league_win_rate_min"] <= metrics["vs/league_win_rate_max"]
        assert f"vs/league_wr_{target_id}" in metrics
        # Per-target results recorded (and persisted in the manifest).
        assert pool.entries[0].games >= matches_target
        reloaded = OpponentPool(tmp_path / "pool")
        assert reloaded.entries[0].games == pool.entries[0].games

        # Exploiter mode never snapshots the learner into the pool.
        assert not env.maybe_snapshot(lambda: {}, cfg={}, step=1)
        assert len(pool.entries) == 1
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)
