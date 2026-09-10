from contextlib import closing
import json
import sqlite3

import pytest

from drmc_rl.arena.store import ArenaStore, SCHEMA
from tools.trainer_arena_sync import completed_journal_matches, refresh_results, sync
from tools.trainer_planning_arena import publish


def register(store):
    for name in ("a", "b"):
        store.register(agent_id=name, name=name, family="test", generation=0,
                       checkpoint=f"{name}.pt", status="active")


def test_match_condition_migration_preserves_rows_and_scoped_idempotency(tmp_path):
    path = tmp_path / "legacy.sqlite"
    legacy = SCHEMA.replace("  condition_key TEXT NOT NULL DEFAULT '',\n", "").replace(
        "UNIQUE(agent_a, agent_b, seed, side_assignment, condition_key)",
        "UNIQUE(agent_a, agent_b, seed, side_assignment)",
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(legacy)
        for name in ("a", "b"):
            connection.execute("INSERT INTO agents(id,name,family,generation,checkpoint,created) "
                               "VALUES(?,?,?,0,?,?)", (name, name, "test", f"{name}.pt", "now"))
        connection.execute("INSERT INTO matches(id,agent_a,agent_b,seed,side_assignment,winner,created,match_key,provenance) "
                           "VALUES(41,'a','b',7,0,'a','now','old','{\"kept\":true}')")
        connection.execute("CREATE INDEX retained_index ON matches(seed)")
        connection.commit()
    store = ArenaStore(path)
    try:
        row = store.conn.execute("SELECT id,condition_key,provenance FROM matches").fetchone()
        assert tuple(row) == (41, "", '{"kept":true}')
        assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='retained_index'").fetchone()
        kwargs = dict(seed=7, side=0, winner="a", match_len_sec=1, decisions=1)
        assert not store.record("a", "b", **kwargs, match_key="another-legacy-key")
        assert store.record("a", "b", **kwargs, condition_key="fast", match_key="fast-0")
        assert not store.record("a", "b", **kwargs, condition_key="fast", match_key="fast-0")
        assert not store.record("a", "b", **kwargs, condition_key="fast", match_key="fast-duplicate")
    finally:
        store.close()
    reopened = ArenaStore(path)
    try:
        assert reopened.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
        assert reopened.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()


def journal():
    matches, games = [], []
    for pace in ("relaxed", "normal", "fast"):
        matches.append(dict(id=pace, a="a", b="b", level=14, pace=pace, target=2, games=2))
        for side in (0, 1):
            games.append(dict(comparison=pace, seed=7, side=side, index=side, level=14,
                              pace=pace, score=1.0, winner="a", reason="clear", frames=60,
                              a_stats=dict(decisions=1), b_stats=dict(decisions=1)))
    return dict(updated_at="2026-09-09T00:00:00Z", tournaments=matches), games


@pytest.mark.parametrize("defer", [False, True])
def test_sync_recovers_games_across_paces_and_is_idempotent(tmp_path, defer):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    report, games = journal()
    original = ArenaStore(source / "arena.sqlite")
    register(original)
    for row in games:
        # Reproduce the old writer: only the first pace survives its unscoped key.
        original.record("a", "b", seed=7, side=row["side"], winner="a", match_len_sec=1,
                        decisions=2, match_key=f"{row['comparison']}-{row['index']}")
    assert original.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
    original.close()
    (source / "results.json").write_text(json.dumps(report))
    (source / "games.jsonl").write_text("\n".join(map(json.dumps, games)) + "\n")
    assert sync(source, target, refresh=not defer) == 6
    if defer:
        assert not (target / "results.json").exists()
        refresh_results(target)
    before = (target / "results.json").read_text()
    assert sum(row["played"] for row in json.loads(before)["tournaments"]) == 6
    assert sync(source, target) == 0
    assert (target / "results.json").read_text() == before
    with closing(sqlite3.connect(target / "arena.sqlite")) as connection:
        assert connection.execute("SELECT condition_key,COUNT(*) FROM matches GROUP BY condition_key ORDER BY condition_key").fetchall() == [
            ("fast", 2), ("normal", 2), ("relaxed", 2)
        ]
        assert connection.execute("SELECT COUNT(*) FROM matches WHERE provenance LIKE '%recovered_from_game_journal%'").fetchone()[0] == 4


@pytest.mark.parametrize("censored", [False, True])
def test_journal_recovery_waits_for_complete_uncensored_pairs(censored):
    report, games = journal()
    if censored:
        games[-1].update(reason="timeout", score=None, winner=None)
    else:
        games.pop()
    rows = completed_journal_matches(report, games)
    assert len(rows) == 4
    assert not any(key.startswith("fast-") for key in rows)


def test_published_snapshot_releases_its_connection(tmp_path, monkeypatch):
    store = ArenaStore(tmp_path / "working.sqlite")
    captured = []
    connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        captured.append(connection)
        return connection

    monkeypatch.setattr("tools.trainer_planning_arena.sqlite3.connect", tracking_connect)
    config = dict(schedule=[], variants={}, reactive_compute_frames=4,
                  preparation_compute_frames=6)
    publish(config, {}, tmp_path, store)
    store.close()
    assert len(captured) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured[0].execute("SELECT 1")
    with closing(connect(tmp_path / "arena.sqlite")) as snapshot:
        assert snapshot.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
