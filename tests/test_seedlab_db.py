from __future__ import annotations

from datetime import datetime, timedelta, timezone

from drmc_rl.seedlab.db import (
    RESERVOIR_CAP,
    Attempt,
    CatalogDB,
    pack_actions,
    unpack_actions,
    unpack_reservoir,
)


def _mk(tmp_path):
    return CatalogDB(tmp_path / "catalog.sqlite3")


def test_record_attempts_aggregates_and_best(tmp_path) -> None:
    db = _mk(tmp_path)
    a = lambda frames, cleared=True, actions=None: Attempt(
        level=5, speed=2, seed=0x8988, cleared=cleared, frames=frames,
        spawns=frames // 60, solver="test", actions=actions,
    )
    new = db.record_attempts([a(3000, actions=pack_actions([1, 2, 3])), a(2500), a(0, cleared=False)])
    assert new == 1

    row = db._conn.execute(
        "SELECT n_attempts, n_clears, min_frames, max_frames, best_frames, best_spawns FROM seed_stats;"
    ).fetchone()
    assert row == (3, 2, 2500, 3000, 2500, 2500 // 60)

    # Improvement with a trace replaces the stored solution.
    new = db.record_attempts([a(2000, actions=pack_actions([7, 8]))])
    assert new == 1
    sol = db.solution(level=5, speed=2, seed=0x8988)
    assert sol is not None
    frames, spawns, actions, solver, _at, verified = sol
    assert frames == 2000 and unpack_actions(actions) == [7, 8] and verified == 0

    # Worse attempt: aggregates move, best/solution don't.
    new = db.record_attempts([a(4000)])
    assert new == 0
    sol2 = db.solution(level=5, speed=2, seed=0x8988)
    assert sol2[0] == 2000
    attempted, cleared = db.coverage(level=5, speed=2)
    assert (attempted, cleared) == (1, 1)
    db.close()


def test_reservoir_caps(tmp_path) -> None:
    db = _mk(tmp_path)
    attempts = [
        Attempt(level=0, speed=2, seed=7, cleared=True, frames=1000 + i, spawns=10, solver="t")
        for i in range(RESERVOIR_CAP * 3)
    ]
    db.record_attempts(attempts)
    blob = db._conn.execute("SELECT reservoir FROM seed_stats;").fetchone()[0]
    res = unpack_reservoir(blob)
    assert len(res) == RESERVOIR_CAP
    assert all(1000 <= v < 1000 + RESERVOIR_CAP * 3 for v in res)
    db.close()


def test_work_queue_lease_cycle(tmp_path) -> None:
    db = _mk(tmp_path)
    n = db.enqueue_units(level=10, speed=2, pass_idx=0, total_seeds=100, chunk=32)
    assert n == 4
    # Idempotent re-enqueue.
    db.enqueue_units(level=10, speed=2, pass_idx=0, total_seeds=100, chunk=32)
    assert db.unit_counts() == {"todo": 4}

    u1 = db.claim_unit(worker_id="w1")
    u2 = db.claim_unit(worker_id="w2")
    assert u1 is not None and u2 is not None and u1.id != u2.id
    assert (u1.seed_lo, u1.seed_hi) == (0, 32)
    assert db.unit_counts() == {"todo": 2, "leased": 2}

    db.complete_unit(u1.id)
    db.release_unit(u2.id)
    assert db.unit_counts() == {"todo": 3, "done": 1}

    # Stale lease reclaim.
    u3 = db.claim_unit(worker_id="w3")
    assert u3 is not None
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds")
    assert db.reclaim_stale_leases(older_than_iso=future) == 1
    assert db.unit_counts() == {"todo": 3, "done": 1}

    # Drain remaining units.
    seen = set()
    while (u := db.claim_unit(worker_id="w")) is not None:
        seen.add((u.seed_lo, u.seed_hi))
        db.complete_unit(u.id)
    assert db.unit_counts() == {"done": 4}
    db.close()


def test_queries(tmp_path) -> None:
    db = _mk(tmp_path)
    db.upsert_games([(3, 0x8988, b"\x01" * 8, 16, 0), (3, 0x4C4C, b"\x02" * 8, 16, 1)])
    assert db.census_count(level=3) == 2

    db.record_attempts(
        [
            Attempt(level=3, speed=2, seed=0x8988, cleared=True, frames=1500, spawns=12, solver="t"),
            Attempt(level=3, speed=2, seed=0x4C4C, cleared=True, frames=1200, spawns=11, solver="t"),
            Attempt(level=3, speed=2, seed=0x4C4C, cleared=True, frames=1800, spawns=14, solver="t"),
        ]
    )
    assert db.levels_present(speed=2) == [3]
    assert sorted(db.best_frames_array(level=3, speed=2)) == [1200, 1500]
    assert db.fastest_seeds(level=3, speed=2, k=1) == [(0x4C4C, 1200)]
    assert sorted(db.pooled_reservoir(level=3, speed=2)) == [1200, 1500, 1800]
    assert db.pooled_reservoir(level=3, speed=2, seed=0x4C4C) == [1200, 1800]
    recs = db.recent_records(limit=5)
    assert len(recs) == 2 and all(r[1] == 3 for r in recs)

    db.mark_solution_verified(level=3, speed=2, seed=0x4C4C, ok=True)
    db.close()
