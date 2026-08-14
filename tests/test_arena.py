from pathlib import Path
import gzip
import json

from drmc_rl.arena.store import ArenaStore
from tools.arena import discover_once, maybe_promote, pair_priority


def add(store: ArenaStore, agent_id: str, status: str, generation: int = 0) -> None:
    store.register(agent_id=agent_id, name=agent_id.title(), family="central",
                   generation=generation, checkpoint=f"/{agent_id}.pt.gz", status=status)


def test_snapshot_rates_and_keeps_lineage(tmp_path: Path) -> None:
    store = ArenaStore(tmp_path / "arena.sqlite")
    add(store, "old", "lineage")
    add(store, "champ", "champion", 1)
    add(store, "new", "candidate", 2)
    for seed in range(12):
        store.record("new", "champ", seed=seed, side=seed % 2, winner="a",
                     match_len_sec=60, decisions=20)
    snap = store.snapshot()
    assert snap["games"] == 12
    assert snap["agents"][0]["id"] == "new"
    assert {a["status"] for a in snap["agents"]} == {"lineage", "champion", "candidate"}


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
