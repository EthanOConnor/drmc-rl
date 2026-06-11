from __future__ import annotations

import glob

import pytest

from seedlab.db import CatalogDB


def _is_pool_present() -> bool:
    try:
        from envs.backends.drmario_pool import is_library_present

        return bool(is_library_present())
    except Exception:
        return False


def _latest_checkpoint() -> str | None:
    paths = sorted(glob.glob("runs/best_agents/*.pt.gz"))
    return paths[-1] if paths else None


@pytest.mark.skipif(
    not _is_pool_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
def test_worker_machinery_end_to_end(tmp_path) -> None:
    """Queue lifecycle + aggregates with a weak baseline (clears not required)."""

    from seedlab.worker import CatalogWorker

    db = CatalogDB(tmp_path / "catalog.sqlite3")
    n_seeds = 24
    db.enqueue_units(level=0, speed=2, pass_idx=0, total_seeds=n_seeds, chunk=16)

    worker = CatalogWorker(
        db=db,
        worker_id="test",
        policy="greedy-cost",
        attempts_per_seed=2,
        num_envs=8,
        max_decisions=300,
        seed=7,
    )
    worker.run()

    assert db.unit_counts() == {"done": 2}
    attempted, _cleared = db.coverage(level=0, speed=2)
    assert attempted == n_seeds
    assert worker.total_attempts == n_seeds * 2

    row = db._conn.execute(
        "SELECT SUM(n_attempts) FROM seed_stats WHERE level=0 AND speed=2;"
    ).fetchone()
    assert int(row[0]) == n_seeds * 2
    db.close()


@pytest.mark.skipif(
    not _is_pool_present() or _latest_checkpoint() is None,
    reason="needs cpp-pool library and a runs/best_agents checkpoint",
)
def test_worker_checkpoint_clears_and_verify(tmp_path) -> None:
    """Trained policy clears level 0 seeds; stored solutions replay exactly."""

    from seedlab.verify import verify_and_mark
    from seedlab.worker import CatalogWorker

    db = CatalogDB(tmp_path / "catalog.sqlite3")
    n_seeds = 8
    db.enqueue_units(level=0, speed=2, pass_idx=0, total_seeds=n_seeds, chunk=8)

    worker = CatalogWorker(
        db=db,
        worker_id="test-ckpt",
        policy="checkpoint",
        checkpoint=_latest_checkpoint(),
        device="cpu",
        attempts_per_seed=1,
        num_envs=8,
        max_decisions=300,
        seed=11,
    )
    worker.run()

    assert db.unit_counts() == {"done": 1}
    attempted, cleared = db.coverage(level=0, speed=2)
    assert attempted == n_seeds
    assert cleared >= n_seeds // 2, "trained checkpoint should clear most level-0 seeds"

    fastest = db.fastest_seeds(level=0, speed=2, k=1)
    assert fastest
    seed, frames = fastest[0]
    sol = db.solution(level=0, speed=2, seed=seed)
    assert sol is not None and int(sol[0]) == frames

    ok, msg = verify_and_mark(db, level=0, speed=2, seed=seed)
    assert ok, msg
    assert int(db.solution(level=0, speed=2, seed=seed)[5]) == 1
    db.close()
