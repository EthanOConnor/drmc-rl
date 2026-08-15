from pathlib import Path
import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from drmc_rl.arena.ratings import RatingConfig
from drmc_rl.arena.remote import ArenaRemoteClient, PROTOCOL_VERSION
from drmc_rl.arena.store import ArenaStore
from tools.arena import (
    DashboardHandler,
    discover_once,
    matchup_schedule,
    maybe_promote,
    pair_priority,
    parse_telemetry,
    paired_specs,
    scheduler_snapshot,
    eligible_agents,
)


def add(store: ArenaStore, agent_id: str, status: str, generation: int = 0) -> None:
    store.register(agent_id=agent_id, name=agent_id.title(), family="central",
                   generation=generation, checkpoint=f"/{agent_id}.pt.gz", status=status)


FAST_RATING = RatingConfig(
    chains=2, warmup=80, samples=120, require_convergence=False, seed=7
)


def test_arena_match_horizon_defaults_to_one_thousand_per_side() -> None:
    import inspect

    from tools.tournament import ARENA_MAX_DECISIONS_PER_SIDE, VsMatchRunner

    assert ARENA_MAX_DECISIONS_PER_SIDE == 1000
    assert (
        inspect.signature(VsMatchRunner).parameters["max_decisions_per_side"].default
        == ARENA_MAX_DECISIONS_PER_SIDE
    )


def test_snapshot_rates_and_keeps_lineage(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "old", "lineage")
    add(store, "champ", "champion", 1)
    add(store, "new", "candidate", 2)
    for seed in range(12):
        store.record("new", "champ", seed=seed, side=seed % 2, winner="a",
                     match_len_sec=60, decisions=20)
    store.refit_ratings(FAST_RATING)
    snap = store.snapshot()
    assert snap["games"] == 12
    assert snap["agents"][0]["id"] == "new"
    assert {a["status"] for a in snap["agents"]} == {"lineage", "champion", "candidate"}


def test_replays_are_content_addressed_outside_sqlite(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    frames = [{"frame": 1, "a": [[1]], "b": [[2]]}]
    assert store.record(
        "alpha", "beta", seed=1, side=0, winner="a", match_len_sec=1,
        decisions=1, replay=frames, match_key="match-1",
    )
    row = store.conn.execute(
        "SELECT replay,replay_ref,match_key FROM matches"
    ).fetchone()
    assert row["replay"] is None
    assert row["match_key"] == "match-1"
    assert (store.replay_dir / row["replay_ref"]).is_file()
    assert store.replay(1)["replay"] == frames
    assert not store.record(
        "alpha", "beta", seed=1, side=0, winner="a", match_len_sec=1,
        decisions=1, replay=frames, match_key="match-1",
    )


def test_expired_lease_is_reclaimed_with_a_new_claim(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    payload = {"protocol_version": PROTOCOL_VERSION, "specs": [{"match_id": "m1"}]}
    first = store.create_lease(
        lease_id="lease", worker_id="one", agent_a="alpha", agent_b="beta",
        payload=payload, now=10, ttl_seconds=5,
    )
    reclaimed, _token = store.claim_expired_lease(
        worker_id="two", now=16, ttl_seconds=10
    )
    assert reclaimed["lease_id"] == "lease"
    assert reclaimed["claim_token"] != first["claim_token"]
    row = store.conn.execute("SELECT worker_id,attempts FROM leases").fetchone()
    assert (row["worker_id"], row["attempts"]) == ("two", 2)


def test_legacy_inline_replays_can_be_externalized(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite", replay_dir=tmp_path / "blobs")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    frames = [{"frame": 7}]
    store.conn.execute(
        "INSERT INTO matches(agent_a,agent_b,seed,side_assignment,winner,match_len_sec,"
        "decisions,terminal_reason,replay,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("alpha", "beta", 1, 0, "a", 1.0, 2, "clear", json.dumps(frames), "now"),
    )
    store.conn.commit()
    assert store.externalize_replays(limit=10) == 1
    row = store.conn.execute("SELECT replay,replay_ref FROM matches").fetchone()
    assert row["replay"] is None and row["replay_ref"]
    assert store.replay(1)["replay"] == frames


def test_remote_coordinator_lease_checkpoint_and_idempotent_submit(tmp_path: Path) -> None:
    db = tmp_path / "arena.sqlite"
    checkpoint_a = tmp_path / "alpha.pt.gz"
    checkpoint_b = tmp_path / "beta.pt.gz"
    checkpoint_a.write_bytes(b"alpha-checkpoint")
    checkpoint_b.write_bytes(b"beta-checkpoint")
    store = ArenaStore(db)
    store.register(
        agent_id="alpha", name="Alpha", family="test", generation=0,
        checkpoint=str(checkpoint_a), status="lineage",
    )
    store.register(
        agent_id="beta", name="Beta", family="test", generation=1,
        checkpoint=str(checkpoint_b), status="candidate",
    )
    store.close()

    DashboardHandler.db = db
    DashboardHandler.replay_dir = tmp_path / "replays"
    DashboardHandler.worker_token = "test-token"
    DashboardHandler.lease_ttl = 60
    DashboardHandler.arena_config = {
        "level": 14, "speed_setting": 2,
        "state_repr": "bitplane_bottle_conn_mask_vs",
        "max_decisions_per_side": 1000,
        "policy_run_seed": 27182,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ArenaRemoteClient(
            f"http://127.0.0.1:{server.server_port}", "test-token",
            checkpoint_cache=tmp_path / "cache",
        )
        assert client.capabilities()["protocol_version"] == PROTOCOL_VERSION
        lease = client.lease({
            "protocol_version": PROTOCOL_VERSION, "worker_id": "worker-1",
            "device": "cpu", "threads": 1, "batch_size": 2,
            "arena_config": DashboardHandler.arena_config,
        })
        assert lease is not None and len(lease["specs"]) == 2
        renewed = client.renew(lease["lease_id"], lease["claim_token"])
        assert renewed["expires"] > lease["expires"]
        assert client.materialize_checkpoint(lease["agent_a"]).is_file()
        results = [{
            "match_id": spec["match_id"], "winner": "a",
            "match_len_sec": 1.5, "decisions": 3,
            "terminal_reason": "clear", "frames": 90,
            "replay": [{"frame": 1}],
        } for spec in lease["specs"]]
        submission = {
            "protocol_version": PROTOCOL_VERSION,
            "claim_token": lease["claim_token"],
            "results": results,
            "worker_sample": {
                "worker_id": "worker-1", "device": "cpu", "threads": 1,
                "batch_size": 2, "agent_a": lease["agent_a"]["id"],
                "agent_b": lease["agent_b"]["id"], "games": 2,
                "simulated_frames": 180, "decisions": 6, "wall_seconds": 2.0,
            },
        }
        assert client.submit(lease["lease_id"], submission)["accepted"] is True
        assert client.submit(lease["lease_id"], submission)["accepted"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    verify = ArenaStore(db, replay_dir=tmp_path / "replays")
    assert verify.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
    assert verify.conn.execute("SELECT COUNT(*) FROM worker_samples").fetchone()[0] == 1
    assert verify.conn.execute("SELECT status FROM leases").fetchone()[0] == "complete"


def test_paired_specs_share_every_reset_field_and_swap_sides() -> None:
    specs = paired_specs(
        schedule_seed=7, agent_a="alpha", agent_b="beta", start=40, count=4,
        level=14, speed_setting=2, state_repr="bitplane_bottle_conn_mask_vs",
        max_decisions_per_side=1000,
        policy_run_seed=27182,
    )
    assert [spec["a_side"] for spec in specs] == [0, 1, 0, 1]
    reset_fields = {
        "seed", "frame_counter_base", "level", "speed_setting", "state_repr",
        "max_decisions_per_side",
    }
    assert {key: specs[0][key] for key in reset_fields} == {
        key: specs[1][key] for key in reset_fields
    }
    assert {key: specs[2][key] for key in reset_fields} == {
        key: specs[3][key] for key in reset_fields
    }
    assert specs == paired_specs(
        schedule_seed=7, agent_a="alpha", agent_b="beta", start=40, count=4,
        level=14, speed_setting=2, state_repr="bitplane_bottle_conn_mask_vs",
        max_decisions_per_side=1000,
        policy_run_seed=27182,
    )


def test_match_provenance_round_trips_with_replay(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    assert store.record(
        "alpha", "beta", seed=3, side=0, winner="a", match_len_sec=1,
        decisions=2, replay=[{"frame": 1}], match_key="complete",
        game_index=9, frame_counter_base=1234, level=14, speed_setting=2,
        state_repr="bitplane_bottle_conn_mask_vs", max_decisions_per_side=1000,
        provenance={"protocol_version": PROTOCOL_VERSION, "worker_id": "green"},
    )
    replay = store.replay(1)
    assert replay["frame_counter_base"] == 1234
    assert replay["provenance"]["worker_id"] == "green"


def test_incomplete_lease_submission_is_rejected_without_telemetry(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    lease = store.create_lease(
        lease_id="legacy", worker_id="old-worker", agent_a="alpha", agent_b="beta",
        payload={
            "protocol_version": 1,
            "specs": [{"match_id": "old", "game_idx": 0, "seed": 1, "a_side": 0}],
        },
        now=1, ttl_seconds=100,
    )
    with pytest.raises(ValueError, match="incomplete reset spec"):
        store.submit_lease(
            lease_id="legacy", claim_token=lease["claim_token"], submission_sha256="x",
            results=[{
                "match_id": "old", "winner": "a", "match_len_sec": 1,
                "decisions": 1,
            }],
            worker_sample={
                "worker_id": "old-worker", "device": "cpu", "threads": 1,
                "batch_size": 1, "agent_a": "alpha", "agent_b": "beta",
                "games": 1, "simulated_frames": 60, "decisions": 1,
                "wall_seconds": 1,
            },
            now=2,
        )
    assert store.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM worker_samples").fetchone()[0] == 0
    assert store.conn.execute("SELECT status FROM leases").fetchone()[0] == "leased"


def test_concurrent_store_open_serializes_additive_migration(tmp_path: Path) -> None:
    db = tmp_path / "arena.sqlite"
    store = ArenaStore(db)
    for column in (
        "game_index", "frame_counter_base", "level", "speed_setting", "state_repr",
        "max_decisions_per_side", "policy_run_seed", "provenance",
    ):
        store.conn.execute(f"ALTER TABLE matches DROP COLUMN {column}")
    store.conn.commit()
    store.close()

    barrier = threading.Barrier(8)

    def open_store(_index: int) -> set[str]:
        barrier.wait()
        concurrent_store = ArenaStore(db)
        try:
            return {
                str(row[1])
                for row in concurrent_store.conn.execute("PRAGMA table_info(matches)")
            }
        finally:
            concurrent_store.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        schemas = list(executor.map(open_store, range(8)))
    required = {
        "game_index", "frame_counter_base", "level", "speed_setting", "state_repr",
        "max_decisions_per_side", "policy_run_seed", "provenance",
    }
    assert all(required <= schema for schema in schemas)


def test_scheduler_focus_is_reversible_and_preserves_agents(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "old", "lineage")
    add(store, "middle", "candidate", 1)
    add(store, "new", "candidate", 2)
    store.set_scheduler_focus(("middle", "new"))
    assert {agent.id for agent in eligible_agents(store)} == {"middle", "new"}
    assert len(store.agents()) == 3
    store.set_scheduler_focus(())
    assert {agent.id for agent in eligible_agents(store)} == {"old", "middle", "new"}


def test_snapshot_initializes_unplayed_child_at_parent_rating(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "anchor", "lineage")
    add(store, "parent", "champion", 1)
    store.register(
        agent_id="child", name="Child", family="central", generation=2,
        parent_id="parent", checkpoint="/child.pt.gz", status="candidate",
    )
    for seed in range(20):
        store.record(
            "parent", "anchor", seed=seed, side=seed % 2,
            winner="a" if seed < 14 else "b", match_len_sec=60, decisions=20,
        )
    store.refit_ratings(FAST_RATING)
    by_id = {agent["id"]: agent for agent in store.snapshot()["agents"]}
    assert by_id["child"]["games"] == 0
    assert abs(by_id["child"]["rating"] - by_id["parent"]["rating"]) < 35
    assert by_id["child"]["rating95"] > by_id["parent"]["rating95"]
    assert by_id["child"]["rating_low"] < by_id["child"]["rating_high"]


def test_snapshot_uses_cached_posterior_and_reports_staleness(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    for seed in range(12):
        store.record(
            "alpha", "beta", seed=seed, side=seed % 2,
            winner="a" if seed < 8 else "b", match_len_sec=60, decisions=20,
        )
    fit = store.refit_ratings(FAST_RATING)
    snapshot = store.snapshot()
    assert snapshot["ratings"]["model"] == "hierarchical-davidson-hmc-v1"
    assert snapshot["ratings"]["fit_id"] == fit["id"]
    assert snapshot["ratings"]["status"] == "current"
    store.record("alpha", "beta", seed=99, side=1, winner="a",
                 match_len_sec=60, decisions=20)
    stale = store.snapshot()
    assert stale["ratings"]["status"] == "updating"
    assert stale["ratings"]["pending_games"] == 1
    update = store.update_ratings(
        FAST_RATING, min_new_matches=1, full_refresh_matches=100,
        min_importance_ess_fraction=0.05,
    )
    assert update is not None
    assert update["method"] == "sequential"
    current = store.snapshot()
    assert current["ratings"]["status"] == "current"
    assert current["ratings"]["method"] == "sequential"


def test_snapshot_exposes_record_and_terminal_causes(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    store.record("alpha", "beta", seed=1, side=0, winner="a",
                 match_len_sec=60, decisions=20, terminal_reason="clear")
    store.record("alpha", "beta", seed=2, side=1, winner="b",
                 match_len_sec=60, decisions=20, terminal_reason="topout")
    store.record("alpha", "beta", seed=3, side=0, winner="draw",
                 match_len_sec=60, decisions=20, terminal_reason="horizon")

    by_id = {agent["id"]: agent for agent in store.snapshot()["agents"]}
    assert (by_id["alpha"]["wins"], by_id["alpha"]["losses"],
            by_id["alpha"]["draws"], by_id["alpha"]["clears"],
            by_id["alpha"]["topouts"]) == (1, 1, 1, 1, 1)
    assert (by_id["beta"]["wins"], by_id["beta"]["losses"],
            by_id["beta"]["draws"], by_id["beta"]["clears"],
            by_id["beta"]["topouts"]) == (1, 1, 1, 0, 0)


def test_promotion_demotes_champion_to_active_lineage(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "old", "champion")
    add(store, "new", "candidate", 1)
    for seed in range(400):
        store.record("new", "old", seed=seed, side=seed % 2, winner="a",
                     match_len_sec=60, decisions=20)
    verdict = maybe_promote(store, store.agent("new"), store.agent("old"), elo0=0,
                            elo1=10, alpha=.05, beta=.05, max_games=400)
    assert verdict == "promoted"
    assert store.agent("new").status == "champion"
    assert store.agent("old").status == "lineage"
    assert {a.id for a in store.agents(("champion", "lineage"))} == {"new", "old"}


def test_scheduler_eventually_prefers_underserved_historical_pair(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "old", "lineage")
    add(store, "champ", "champion", 1)
    add(store, "new", "candidate", 2)
    for seed in range(100):
        store.record("new", "champ", seed=seed, side=seed % 2, winner="a",
                     match_len_sec=60, decisions=20)
    a, b = pair_priority(store, store.agents())
    assert "old" in {a.id, b.id}


def test_scheduler_weights_posterior_information_gain(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    for agent_id in ("alpha", "beta", "gamma"):
        add(store, agent_id, "lineage")
    store.matchup_information = lambda: {  # type: ignore[method-assign]
        ("alpha", "beta"): 0.20,
        ("alpha", "gamma"): 0.01,
        ("beta", "gamma"): 0.01,
    }
    store.matchup_counts = lambda: {}  # type: ignore[method-assign]
    import random
    random.seed(5)
    selections = [
        frozenset(agent.id for agent in pair_priority(store, store.agents()))
        for _ in range(300)
    ]
    assert selections.count(frozenset(("alpha", "beta"))) > 225


def test_scheduler_snapshot_exposes_exact_worker_distribution(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "candidate")
    add(store, "gamma", "lineage")
    store.matchup_information = lambda: {  # type: ignore[method-assign]
        ("alpha", "beta"): 0.20,
        ("alpha", "gamma"): 0.01,
        ("beta", "gamma"): 0.03,
    }
    store.matchup_counts = lambda: {("alpha", "beta"): 17}  # type: ignore[method-assign]
    schedule = matchup_schedule(store, store.agents())
    snapshot = scheduler_snapshot(store)
    assert snapshot["mode"] == "bayesian_information"
    assert snapshot["eligible_pairs"] == 3
    assert snapshot["matchups"][0]["a"] == schedule[0]["a"].id
    assert snapshot["matchups"][0]["b"] == schedule[0]["b"].id
    assert snapshot["matchups"][0]["games"] == 17
    assert abs(sum(item["selection_probability"] for item in schedule) - 1.0) < 1e-12
    assert {factor["label"] for factor in snapshot["matchups"][0]["factors"]} == {
        "new entrant", "temperature", "coverage floor"
    }


def test_scheduler_drops_new_entry_boost_at_cap_without_status_boost(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    store.register(
        agent_id="beta", name="Beta", family="central", generation=1,
        checkpoint="/beta.pt.gz", status="candidate",
        metadata={"scheduler_boost": {"multiplier": 2.0, "max_games": 10,
                                      "los_target": 0.95}},
    )
    add(store, "gamma", "candidate")
    store.matchup_information = lambda: {  # type: ignore[method-assign]
        ("alpha", "beta"): 0.20,
        ("alpha", "gamma"): 0.20,
        ("beta", "gamma"): 0.20,
    }
    store.matchup_counts = lambda: {("alpha", "beta"): 10}  # type: ignore[method-assign]
    schedule = matchup_schedule(store, store.agents())
    beta_gamma = next(
        item for item in schedule
        if {item["a"].id, item["b"].id} == {"beta", "gamma"}
    )
    labels = {factor["label"] for factor in beta_gamma["factors"]}
    assert labels == {"temperature", "new entrant", "coverage floor"}
    assert beta_gamma["weight"] > 0


def test_matchup_counts_canonicalizes_both_agent_orders(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "alpha", "lineage")
    add(store, "beta", "lineage")
    store.record("alpha", "beta", seed=1, side=0, winner="a",
                 match_len_sec=10, decisions=4)
    store.record("beta", "alpha", seed=2, side=1, winner="b",
                 match_len_sec=10, decisions=4)
    assert store.matchup_counts() == {("alpha", "beta"): 2}


def test_discovery_names_and_links_candidate(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    checkpoint = tmp_path / "run" / "checkpoints" / "smdp_ppo_step25000000.pt.gz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    add(store, "champ", "champion", 2)
    config = tmp_path / "campaigns.json"
    config.write_text(
        '{"campaigns":[{"id":"az","family":"central",'
        '"name":"Capsule Prime G{generation} · {step}",'
        f'"root":"{tmp_path}","glob":"run/checkpoints/*.pt.gz",'
        '"settle_seconds":0}]}'
    )
    assert discover_once(store, config) == 1
    candidate = store.agents(("candidate",))[0]
    assert candidate.name == "Capsule Prime G3 · 25000000"
    assert candidate.parent_id == "champ"
    assert discover_once(store, config) == 0


def test_replay_and_training_metrics_feed_dashboard(tmp_path: Path) -> None:
    db = tmp_path / "runs" / "arena" / "arena.sqlite"
    store = ArenaStore(db)
    add(store, "old", "lineage")
    add(store, "new", "champion", 1)
    frames = [{"boards": [[[0] * 128], [[0] * 128]], "decision": 0}]
    store.record("new", "old", seed=7, side=0, winner="a", match_len_sec=20,
                 decisions=4, replay=frames)
    run = tmp_path / "runs" / "campaign" / "run-1"
    run.mkdir(parents=True)
    with gzip.open(run / "metrics.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"step": 10, "type": "scalar", "name": "perf/sps",
                                 "value": 123456}) + "\n")
        handle.write(json.dumps({"step": 10, "type": "scalar", "name": "perf/dps",
                                 "value": 3456}) + "\n")
    snap = store.snapshot()
    assert snap["recent"][0]["has_replay"] == 1
    assert store.replay(snap["recent"][0]["id"])["replay"] == frames
    assert snap["training"]["latest"]["perf/sps"] == 123456


def test_worker_throughput_uses_simulated_work_not_game_count(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.record_worker_sample(
        worker_id="cpu-123", device="cpu", threads=6, batch_size=4,
        agent_a="alpha", agent_b="beta", games=4,
        simulated_frames=120_000, decisions=800, wall_seconds=20.0,
    )
    worker = store.snapshot()["workers"][0]
    assert worker["games_per_min"] == 12.0
    assert worker["frames_per_sec"] == 6_000.0
    assert worker["frames_per_min"] == 360_000.0
    assert worker["decisions_per_sec"] == 40.0
    assert worker["frames_per_game"] == 30_000.0


def test_telemetry_parses_live_afterstate_training_and_validation() -> None:
    text = """@@AFTERSTATE runs/human_policy/human_afterstate_v3_train.log
schema=drmc-human-afterstate-v3 parameters=10,897,863 shards=48 bf16=True
epoch=1 step=100 decisions/s=2,586 loss=3.3554 style=2.6901 outcome=0.5450
epoch=1 step=200 decisions/s=2,666 loss=3.0865 style=2.4608 outcome=0.5871
{
  "epoch": 1,
    "metrics": {
      "validation_objective": 2.2,
      "validation_top1": 0.42,
      "validation_quality_top1": 0.37,
    "validation_outcome_brier": 0.19,
    "validation_mean_regret": 0.31
  }
}
"""
    tasks = parse_telemetry(text)
    assert len(tasks) == 1
    latest = tasks[0]["latest"]
    assert latest["train/epoch"] == 1
    assert latest["train/step"] == 200
    assert latest["perf/dps"] == 2666
    assert latest["train/style"] == 2.4608
    assert latest["validation/top1"] == 0.42
    assert latest["validation/quality_top1"] == 0.37
    assert tasks[0]["history"]["perf/dps"] == [[100, 2586.0], [200, 2666.0]]


def test_telemetry_parses_full_corpus_extraction() -> None:
    tasks = parse_telemetry(
        """@@CORPUS /store/human-v5/full-corpus/extract-v2.log
scanned=12,288 sampled=12,288 kept=12,223 rate=3,814/s
scanned=20,422 sampled=20,422 kept=20,337 rate=3,880/s
"""
    )
    assert len(tasks) == 1
    latest = tasks[0]["latest"]
    assert latest["global_step"] == 20_422
    assert latest["corpus/kept"] == 20_337
    assert latest["corpus/total"] == 54_873_706
    assert latest["perf/dps"] == 3_880


def test_telemetry_preserves_latest_scalar_training_step() -> None:
    tasks = parse_telemetry(
        """@@JSON runs/campaign/run/metrics.jsonl.gz
{"step":1900000000,"type":"scalar","name":"perf/sps","value":120000}
{"step":1900012800,"type":"scalar","name":"loss/total","value":0.125}
"""
    )
    assert tasks[0]["latest"]["global_step"] == 1_900_012_800
