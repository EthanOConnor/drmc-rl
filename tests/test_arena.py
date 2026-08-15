from pathlib import Path
import gzip
import json

from drmc_rl.arena.ratings import RatingConfig
from drmc_rl.arena.store import ArenaStore
from tools.arena import discover_once, maybe_promote, pair_priority, parse_telemetry


def add(store: ArenaStore, agent_id: str, status: str, generation: int = 0) -> None:
    store.register(agent_id=agent_id, name=agent_id.title(), family="central",
                   generation=generation, checkpoint=f"/{agent_id}.pt.gz", status=status)


FAST_RATING = RatingConfig(
    chains=2, warmup=80, samples=120, require_convergence=False, seed=7
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
    assert selections.count(frozenset(("alpha", "beta"))) > 240


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
