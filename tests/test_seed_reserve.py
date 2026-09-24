"""The permanent evaluation-seed reserve and its enforcement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drmc_rl.program import seed_reserve as sr
from drmc_rl.seedlab import rng as seedlab

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "runs/review-20260909/afterstate-core-v1/ppo-ppo-v1.json"


def seedlab_seed(arena: int) -> int:
    r0, r1 = sr.rng_state(arena)
    return seedlab.bytes_to_seed(r0, r1)


def test_orbit_matches_the_seedlab_catalog_universe():
    assert len(sr.orbit_seeds()) == 32767
    assert {seedlab_seed(s) for s in sr.orbit_seeds()} == set(seedlab.orbit())


def test_twin_seeds_play_identical_games_and_form_the_only_classes():
    members = sr.class_members()
    assert len(members) == 32768
    assert members[0] == (0x100,)  # steps into the 0x0000 lockup
    pairs = [m for m in members.values() if len(m) == 2]
    assert len(pairs) == 32767
    assert all(b == sr.twin(a) and (a in sr.orbit_seeds()) != (b in sr.orbit_seeds()) for a, b in pairs)
    for a, b in pairs[::2048]:
        for level in (14, 20):
            ga, gb = (seedlab.generate_game(level, seedlab_seed(s)) for s in (a, b))
            assert (ga.board, ga.pills) == (gb.board, gb.pills)


def test_committed_reserve_is_consistent_and_disjoint_from_registered_seeds():
    reserve = sr.load_reserve()
    assert len(reserve.seeds) == 4096 and len(reserve.blocked) == 8192
    assert reserve.blocked == {m for s in reserve.seeds for m in (s, sr.twin(s))}
    assert set(reserve.seeds) <= sr.orbit_seeds()
    registered = set(json.loads(LEGACY_CONFIG.read_text())["holdout_seeds"])
    for name in ("seeds.json", "confirmation-seeds.json"):
        groups = json.loads((LEGACY_CONFIG.parent / name).read_text())
        registered |= {s for g in groups.values() for v in g.values() for s in v}
    assert not reserve.blocked & (registered | {sr.twin(s) for s in registered})
    assert sr.is_legacy(json.loads(LEGACY_CONFIG.read_text()))


def test_training_pool_excludes_reserve_and_twins_but_legacy_runs_keep_their_draws():
    reserve = sr.load_reserve()
    holdout = [5, 700]
    pool = sr.training_seed_pool(holdout, config={"output": "/x/new-run"})
    assert not set(pool.tolist()) & (reserve.blocked | {5, 700, sr.twin(5), sr.twin(700)})
    assert len(pool) == 65535 - len(reserve.blocked | {5, 700, sr.twin(5), sr.twin(700)})
    legacy = json.loads(LEGACY_CONFIG.read_text())
    np.testing.assert_array_equal(
        sr.training_seed_pool(legacy["holdout_seeds"], config=legacy),
        np.setdiff1d(np.arange(1, 65536), legacy["holdout_seeds"]))
    assert sr.is_legacy({"seed_reserve": "legacy"})


def test_explicit_training_seeds_and_random_states_avoid_the_reserve():
    reserve = sr.load_reserve()
    seed = reserve.seeds[0]
    with pytest.raises(ValueError, match="evaluation-reserve"):
        sr.require_training_seeds([1, sr.twin(seed)])
    sr.require_training_seeds([sr.twin(seed)], config={"seed_reserve": "legacy"})
    rng = np.random.default_rng(3)
    states = [sr.draw_training_state(rng) for _ in range(4000)]
    assert not any(sr.arena_seed(*s) in reserve.blocked for s in states)
    assert states[:50] == [sr.draw_training_state(r) for r in [np.random.default_rng(3)] for _ in range(50)]


def test_allocations_are_disjoint_recorded_and_never_reissued(tmp_path):
    reserve, path = sr.load_reserve(), tmp_path / "allocations.json"
    first = sr.allocate("study-a", 256, "panel", reserve=reserve, path=path)
    second = sr.allocate("study-b", 64, "quick read", reserve=reserve, path=path)
    assert first == list(reserve.seeds[:256]) and second == list(reserve.seeds[256:320])
    assert sr.allocated_seeds("study-b", reserve=reserve, path=path) == second
    with pytest.raises(ValueError, match="already holds"):
        sr.allocate("study-a", 8, "again", reserve=reserve, path=path)
    with pytest.raises(ValueError, match="remain"):
        sr.allocate("study-c", 4096, "too many", reserve=reserve, path=path)
    data = json.loads(path.read_text())
    data["allocations"][1]["start"] = 100
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contiguous"):
        sr.allocated_seeds(reserve=reserve, path=path)


def test_check_flags_unallocated_reserve_seeds(tmp_path):
    reserve, path = sr.load_reserve(), tmp_path / "allocations.json"
    mine = sr.allocate("study-a", 4, "panel", reserve=reserve, path=path)
    other = sr.allocate("study-b", 4, "panel", reserve=reserve, path=path)
    free = reserve.seeds[8]
    kwargs = dict(reserve=reserve, allocations_path=path)
    config = {"schedule": [{"id": "m", "seeds": mine}], "holdout_seeds": mine + other}
    assert sr.check_config(config, study="study-a", **kwargs) == []
    assert sr.check_config({"schedule": [{"seeds": other}]}, study="study-a", **kwargs)
    assert sr.check_config({"holdout_seeds": [free]}, **kwargs)
    assert sr.check_config({"schedule": [{"seeds": [sr.twin(free)]}]}, **kwargs)
    assert sr.check_config({"excluded_reset_seeds": [list(sr.rng_state(free))]}, **kwargs)
    assert sr.check_config({"quick": {"normal": [free]}}, seed_file=True, **kwargs)
    assert sr.check_config({"board": [free]}, **kwargs) == []  # not a seed list


def test_committed_configs_and_seed_lists_use_no_unallocated_reserve_seeds():
    assert sr.check_paths([ROOT / "runs"]) == []


def test_arena_random_schedules_avoid_the_reserve_and_explicit_banks_need_an_allocation():
    from tools.trainer_planning_arena import paired_jobs

    reserve = sr.load_reserve()
    jobs = paired_jobs({}, {"games": 2000, "seed": 7})
    assert not {seed for seed, _, _ in jobs} & reserve.blocked
    with pytest.raises(ValueError, match="unallocated"):
        paired_jobs({}, {"games": 2, "seeds": [reserve.seeds[-1]]})
