"""Distributed trainer studies journal exactly what the single-host arena does."""
import json
import random
import sys
import threading

import pytest

import tools.trainer_planning_arena as arena
from tools import trainer_arena_distributed as dist


def fake_game(seed, side, index):
    rng = random.Random(seed * 7 + side)
    # Side A wins most seeds so the sequential comparison stops early.
    score = 1.0 if rng.random() < 0.9 else 0.0
    row = dict(seed=seed, side=side, index=index, score=score, winner="a" if score else "b",
               reason="topout", frames=1000 + seed % 97,
               a_stats=dict(decisions=10 + side), b_stats=dict(decisions=9))
    moves = [dict(frame=f, side=f % 2, placement=dict(action=(seed + f) % 512), board=[seed % 7] * 4)
             for f in range(3 + seed % 5)]
    return row, moves, []


class FakeRuntime:
    played = []

    def __init__(self, config):
        self.config = config

    def play(self, match, jobs):
        FakeRuntime.played.append((match["id"], len(jobs)))
        return [fake_game(*job) for job in jobs], 0.01

    def close(self):
        pass


def study(tmp_path, name):
    checkpoint = tmp_path / "model.pt"
    other = tmp_path / "other.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"model")
        other.write_bytes(b"other")
    output = tmp_path / name
    return dict(
        checkpoint=str(checkpoint), device="cpu", native_library=str(checkpoint), pairs=8,
        output=str(output), working_db=str(output / "working" / "arena.sqlite"),
        reactive_compute_frames=4, preparation_compute_frames=6, replay_games=0, strict_fp32=True,
        rollout_backend="frames", skip_identical=True, identity_probe_games=4,
        variants=dict(
            base=dict(name="base", checkpoint=str(checkpoint), delay=4),
            twin=dict(name="twin", checkpoint=str(checkpoint), delay=3),
            other=dict(name="other", checkpoint=str(other), delay=6),
        ),
        schedule=[
            dict(id="seq", a="other", b="base", games=96, seed=5, level=14, pace="fast",
                 sequential=dict(question="threshold", threshold=0.45), look_games=8),
            dict(id="twin", a="twin", b="base", games=16, seed=6, level=14, pace="normal"),
            dict(id="plain", a="base", b="other", games=24, seed=7, level=14, pace="frame_perfect"),
        ],
    )


def single_host(tmp_path, monkeypatch):
    config = study(tmp_path, "single")
    path = tmp_path / "single.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(arena, "ArenaRuntime", FakeRuntime)
    monkeypatch.setattr(sys, "argv", ["arena", "--config", str(path)])
    arena.main()
    return tmp_path / "single"


def test_out_of_order_completion_journals_the_single_host_study(tmp_path, monkeypatch):
    reference = single_host(tmp_path, monkeypatch)
    coordinator = dist.StudyCoordinator(study(tmp_path, "dist"), log=lambda *_: None)
    worker = dict(protocol=dist.PROTOCOL, worker_id="w", host="h", numerics="cpu/test", device="cpu", threads=1,
                  source=coordinator.source)
    rng = random.Random(1)
    outstanding = []
    while True:
        while len(outstanding) < 5:
            lease = coordinator.lease(dict(worker, worker_id=f"w{len(outstanding)}"))
            if lease["status"] != "lease":
                break
            outstanding.append(lease)
        if not outstanding:
            break
        lease = outstanding.pop(rng.randrange(len(outstanding)))
        played = [fake_game(*job) for job in lease["jobs"]]
        payload = dict(claim_token=lease["claim_token"], batch=lease["batch"], elapsed=0.01, worker=worker,
                       rows=[p[0] for p in played], moves=[p[1] for p in played], replays=[[] for _ in played])
        coordinator.submit(lease["lease_id"], json.loads(json.dumps(payload)))
    assert coordinator.complete()
    verdicts = coordinator.stopping.verdicts
    assert verdicts["seq"]["decision"] == "pass" and verdicts["seq"]["early"]
    assert "twin" not in verdicts or verdicts["twin"]["decision"] != "identical"
    assert any(b.status == "discarded" for b in coordinator.plan["seq"])
    coordinator.close()
    report = dist.compare_outputs(reference, tmp_path / "dist")
    assert report["identical"], report
    assert report["games"] == sum(1 for _ in (reference / "games.jsonl").open())


def test_http_workers_reproduce_the_single_host_study(tmp_path, monkeypatch):
    reference = single_host(tmp_path, monkeypatch)
    config_path = tmp_path / "dist.json"
    config_path.write_text(json.dumps(study(tmp_path, "http")))
    args = type("Args", (), dict(config=config_path, lease_ttl=60.0, replicate_every=2, calibration_games=0,
                                 max_ahead=0, allow_source_mismatch=False, trust=[], fidelity="strict",
                                 min_agreement=0.99, host="127.0.0.1",
                                 port=0, token="secret", exit_when_done=True, linger=0.5))()
    ready = threading.Event()
    port = {}
    server = threading.Thread(target=dist.serve, args=(args,),
                              kwargs=dict(on_ready=lambda p: (port.setdefault("p", p), ready.set())))
    server.start()
    assert ready.wait(30)
    workers = []
    for i in range(2):
        worker_args = type("Args", (), dict(coordinator=f"http://127.0.0.1:{port['p']}", token="secret",
                                            token_file=None, worker_id=f"w{i}", device=None, threads=None,
                                            planner_workers=None, native_library=None, reach_library=None,
                                            cache=str(tmp_path / f"cache{i}"), poll=0.05, max_batches=0,
                                            allow_source_mismatch=False, shared_artifacts=i == 1))()
        workers.append(threading.Thread(target=dist.run_worker, args=(worker_args,)))
        workers[-1].start()
    for thread in workers:
        thread.join(60)
    server.join(60)
    assert not server.is_alive()
    report = dist.compare_outputs(reference, tmp_path / "http")
    assert report["identical"], report
    audits = [json.loads(line) for line in (tmp_path / "http" / "distributed" / "audit.jsonl").open()]
    assert audits and all(a["equal"] for a in audits)


def test_leases_expire_and_mismatched_replicas_are_audited(tmp_path):
    config = study(tmp_path, "expire")
    config["schedule"] = config["schedule"][2:]
    coordinator = dist.StudyCoordinator(config, lease_ttl=0.0, replicate_every=1, log=lambda *_: None)
    worker = dict(protocol=dist.PROTOCOL, worker_id="a", host="h", numerics="cpu/x", device="cpu", threads=1,
                  source=coordinator.source)
    first = coordinator.lease(worker)
    second = coordinator.lease(dict(worker, worker_id="b"))
    assert first["batch"] == second["batch"]  # the first lease expired immediately
    played = [fake_game(*job) for job in second["jobs"]]
    payload = dict(claim_token=second["claim_token"], batch=second["batch"], elapsed=0.5, worker=worker,
                   rows=[p[0] for p in played], moves=[p[1] for p in played], replays=[[] for _ in played])
    assert coordinator.submit(second["lease_id"], json.loads(json.dumps(payload)))["accepted"]
    # A late identical upload from the expired lease is an idempotent duplicate.
    late = dict(payload, claim_token=first["claim_token"])
    assert coordinator.submit(first["lease_id"], json.loads(json.dumps(late)))["duplicate"]
    replica = coordinator.lease(dict(worker, worker_id="c", numerics="cuda/y"))
    assert replica["purpose"] == "replicate" and replica["batch"] == second["batch"]
    changed = json.loads(json.dumps(payload))
    changed["moves"][0][0]["placement"]["action"] = 511
    changed["claim_token"] = replica["claim_token"]
    assert not coordinator.submit(replica["lease_id"], changed)["accepted"]
    audit = json.loads((tmp_path / "expire" / "distributed" / "audit.jsonl").read_text().splitlines()[0])
    assert not audit["equal"] and audit["differing"][0]["first_move"] == 0
    with pytest.raises(ValueError):
        coordinator.submit("unknown", dict(payload, batch=second["batch"], rows=payload["rows"][:1]))
    coordinator.close()


def test_seed_pairs_share_a_batch_and_tolerant_fidelity_scores_decisions(tmp_path):
    coordinator = dist.StudyCoordinator(study(tmp_path, "pairs"), log=lambda *_: None)
    for batch in coordinator.batches.values():
        sides = {}
        for seed, side, _ in batch.jobs:
            sides.setdefault(seed, set()).add(side)
        assert all(v == {0, 1} for v in sides.values())  # both side-swapped games on one worker
    games = [fake_game(*job) for job in coordinator.plan["plain"][0].jobs]
    rows, moves = [g[0] for g in games], [g[1] for g in games]
    other = json.loads(json.dumps(moves))
    other[0][1]["placement"]["action"] = 511  # diverge at the second decision of one game
    comparison = coordinator._compare(rows, moves, rows, other)
    total = sum(len(m) for m in moves)
    assert comparison["divergent_games"] == 1
    assert comparison["compared_decisions"] == total - len(moves[0]) + 2
    assert comparison["agreed_decisions"] == total - len(moves[0]) + 1
    assert coordinator._acceptable(comparison) == (comparison["agreement"] >= 0.99)
    coordinator.fidelity = "strict"
    assert not coordinator._acceptable(comparison)
    coordinator.close()
