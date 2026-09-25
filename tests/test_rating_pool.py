"""Rating pool: rating math, conditions, journal replay, scheduler priorities, intentions, end to end."""
from __future__ import annotations

import json
import math
import random
import threading
import time
from types import SimpleNamespace

import pytest

from drmc_rl.pool import intentions as intent
from drmc_rl.pool.conditions import condition_key, make_condition, normalize_decision, study_condition
from drmc_rl.pool.coordinator import PROTOCOL, PoolCoordinator
from drmc_rl.pool.ratings import ELO_SCALE, PairStats, fit, pooled
from drmc_rl.pool.report import build_report, stop_rule, summary_text
from drmc_rl.pool.store import PoolState, game_id

SHA = {n: (f"{i:x}" * 64)[:64] for i, n in enumerate(["anchor", "a", "b", "c", "d", "e", "f", "old", "run-f1",
                                                     "run-f2", "run-f3", "run-f4"], start=1)}
BANK = list(range(1000, 1400))
UNIFORM, MIXTURE = list(range(2000, 2400)), list(range(3000, 3400))
SEED_SETS = dict(uniform=dict(seed_mix=0.0, seeds=UNIFORM), mixture=dict(seed_mix=0.5, seeds=MIXTURE))


def condition(pace="normal", **kw):
    return make_condition(backend=kw.pop("backend", "events"), engine="19f292c", level=kw.pop("level", 14),
                          pace=pace, decision=kw.pop("decision", dict(delay=4)), **kw)


def simulate(strength, pairs, rng):
    """Complete side-swapped seed pairs with a Bradley-Terry truth."""
    out = {}
    for (i, j), count in pairs.items():
        p = 1 / (1 + math.exp(-(strength[i] - strength[j])))
        s = PairStats()
        for _ in range(count):
            s.add(float(rng.random() < p) + float(rng.random() < p))
        out[(i, j)] = s
    return out


# ------------------------------------------------------------------------------ ratings


def test_fit_recovers_bradley_terry_strengths_and_fixes_the_anchor():
    truth = dict(anchor=0.0, a=1.0, b=-0.5, c=0.3)
    pairs = simulate(truth, {("a", "anchor"): 400, ("anchor", "b"): 400, ("anchor", "c"): 400, ("a", "c"): 200},
                     random.Random(1))
    result = fit(pairs, "anchor", anchor_rating=1500)
    assert result.ratings["anchor"].rating == 1500 and result.ratings["anchor"].se == 0
    for e, t in truth.items():
        r = result.ratings[e]
        assert abs((r.rating - 1500) / ELO_SCALE - t) < 3 * max(r.se / ELO_SCALE, 1e-9) + 1e-9
    assert result.ratings["a"].rating > result.ratings["c"].rating > result.ratings["b"].rating
    # Deterministic: a refit of the same data is bit-identical.
    again = fit(pairs, "anchor", anchor_rating=1500)
    assert all(again.ratings[e].rating == result.ratings[e].rating for e in truth)
    assert 0.5 < result.expected("a", "anchor") < 1
    assert result.difference_se("a", "c") > 0


def test_unconnected_entrants_are_unanchored_not_guessed():
    pairs = {("a", "anchor"): PairStats(pairs=10, score=10, square=10), ("b", "c"): PairStats(pairs=5, score=5, square=5)}
    result = fit(pairs, "anchor")
    assert set(result.ratings) == {"a", "anchor"} and set(result.unanchored) == {"b", "c"}


def test_sandwich_widens_uncertainty_for_correlated_seed_pairs():
    # Same totals; in one record every pair is split 1-1, in the other pairs are 2-0 or 0-2.
    split = {("a", "anchor"): PairStats(pairs=100, score=100.0, square=100 * 1.0)}
    clustered = {("a", "anchor"): PairStats(pairs=100, score=100.0, square=50 * 4.0)}
    assert fit(clustered, "anchor").ratings["a"].se > fit(split, "anchor").ratings["a"].se
    assert fit(split, "anchor").ratings["a"].rating == pytest.approx(1500, abs=1e-6)


def test_perfect_records_stay_finite_and_pooled_needs_every_condition():
    sweep = {("a", "anchor"): PairStats(pairs=32, score=64.0, square=32 * 4.0)}
    r = fit(sweep, "anchor").ratings["a"]
    assert math.isfinite(r.rating) and r.rating > 1800
    f1 = fit({("a", "anchor"): PairStats(pairs=10, score=12, square=16)}, "anchor")
    f2 = fit({("anchor", "b"): PairStats(pairs=10, score=12, square=16),
              ("a", "anchor"): PairStats(pairs=10, score=8, square=8)}, "anchor")
    view = pooled({"x": f1, "y": f2}, ["x", "y"])
    assert set(view) == {"a", "anchor"}                    # b is not rated under x
    assert view["a"]["rating"] == pytest.approx((f1.ratings["a"].rating + f2.ratings["a"].rating) / 2, abs=0.1)
    assert pooled({"x": f1}, ["x", "missing"]) == {}


# ------------------------------------------------------------------------------ conditions


def test_condition_key_normalizes_inert_settings_and_separates_contracts():
    spawn = condition(decision=dict(delay=4, decision_point="spawn", early_preview="repeat"))
    assert spawn["decision"] == dict(delay=4) and condition_key(spawn) == condition_key(condition())
    lock = condition(backend="frames", decision=dict(delay=4, decision_point="lock_safe", early_preview="repeat"))
    frames_spawn = condition(backend="frames")
    assert len({condition_key(spawn), condition_key(lock), condition_key(frames_spawn),
                condition_key(condition(level=20)), condition_key(condition("fast"))}) == 5
    with pytest.raises(ValueError):
        condition(decision=dict(delay=4, decision_point="lock_safe"))   # events backend is spawn-only
    with pytest.raises(ValueError):
        normalize_decision(dict(delay=4, anticipation=True))
    with pytest.raises(ValueError):
        condition(speed=1)


def test_study_rows_with_asymmetric_contracts_are_not_imported():
    config = dict(rollout_backend="frames", native_commit="19f292c")
    match = dict(level=14, pace="normal")
    spec, why = study_condition(config, match, dict(delay=4), dict(delay=4))
    assert spec is not None and not why
    spec, why = study_condition(config, match, dict(delay=4, decision_point="lock_safe"), dict(delay=4))
    assert spec is None and "different decision contracts" in why
    spec, why = study_condition(config, match, dict(delay=4, anticipation=True), dict(delay=4))
    assert spec is None
    spec, why = study_condition(dict(rollout_backend="events"), match, dict(delay=4), dict(delay=4))
    assert spec is None and "native engine" in why


# ------------------------------------------------------------------------------ journal and state


def entrant(eid, era="test", status="active", **kw):
    return dict(id=eid, name=eid, loader="plain", checkpoint=dict(sha256=SHA.get(eid, SHA["a"]), paths=[]), era=era,
                status=status, **kw)


def setup_state(tmp_path, entrants=("anchor", "a", "b"), paces=("normal",)):
    state = PoolState(tmp_path, settings=dict(seed_bank=BANK, default_anchor="anchor"))
    keys = []
    for pace in paces:
        spec = condition(pace)
        state.record("condition", dict(spec=spec, name=f"ev-{pace}"))
        keys.append(condition_key(spec))
    for e in entrants:
        state.record("entrant", entrant(e))
    state.record("condition_set", dict(name="main", conditions=keys, anchor="anchor", primary=True))
    return state, keys


def rows_for(key, a, b, seeds, score_a=1.0, numerics="mps/test", source="pool"):
    out = []
    for seed in seeds:
        for side in (0, 1):
            x, y = sorted((a, b))
            s = score_a if x == a else 1 - score_a
            out.append(dict(id=game_id(key, x, y, seed, side), condition=key, a=x, b=y, seed=seed, side=side, score=s,
                            winner="a" if s == 1 else "b" if s == 0 else "draw", reason="topout", frames=100,
                            source=source, numerics=numerics))
    return out


def test_journal_repairs_a_torn_tail_and_replay_is_idempotent(tmp_path):
    state, (key,) = setup_state(tmp_path)
    assert len(state.add_games(rows_for(key, "a", "anchor", BANK[:4]))) == 8
    assert state.add_games(rows_for(key, "a", "anchor", BANK[:4])) == []          # idempotent by id
    with open(tmp_path / "games.jsonl", "a") as stream:
        stream.write('{"id": "torn')                                            # crash mid-write
    replay = PoolState(tmp_path)
    assert len(replay.games) == 8 and replay.pair_stats(key)[("a", "anchor")].pairs == 4
    assert (tmp_path / "games.jsonl").read_text().endswith("\n")
    assert replay.entrants.keys() == state.entrants.keys() and replay.condition_sets == state.condition_sets


def test_only_complete_uncensored_admitted_seed_pairs_are_rated(tmp_path):
    state, (key,) = setup_state(tmp_path)
    rows = rows_for(key, "a", "anchor", BANK[:3])
    rows[5]["score"] = None                          # third pair censored
    state.add_games(rows + rows_for(key, "a", "anchor", [BANK[3]])[:1])         # a half pair
    assert state.pair_stats(key)[("a", "anchor")].pairs == 2
    state.add_games(rows_for(key, "b", "anchor", BANK[:2], numerics="cuda/bad"))
    assert ("anchor", "b") in state.pair_stats(key)
    state.record("admission", dict(numerics="cuda/bad", verdict="replica failed"))
    assert ("anchor", "b") not in state.pair_stats(key)                          # rejected class leaves ratings


def test_registry_rejects_invalid_records(tmp_path):
    state, (key,) = setup_state(tmp_path)
    with pytest.raises(ValueError):
        state.record("entrant", dict(entrant("x"), settings=dict(delay=6)))   # a condition key is not a setting
    with pytest.raises(ValueError):
        state.record("job", dict(id="j", status="active", games=3, conditions=[key], entrants=["a"]))
    with pytest.raises(ValueError):
        state.add_games([dict(rows_for(key, "a", "anchor", [BANK[0]])[0], a="zzz")])
    other = condition("fast")
    state.record("condition", dict(spec=other, name="ev-fast"))
    state.record("entrant", entrant("c"))
    with pytest.raises(ValueError):             # one condition, one anchor
        state.record("condition_set", dict(name="other", conditions=[key], anchor="c"))


# ------------------------------------------------------------------------------ scheduler


def coordinator(tmp_path, **kw):
    return PoolCoordinator(tmp_path, source="test", capabilities=None, log=lambda *_: None,
                           settings={**dict(seed_bank=BANK, seed_sets=SEED_SETS, default_anchor="anchor",
                                            calibration_games=0, replicate_every=0), **kw.pop("settings", {})}, **kw)


def worker(wid="w0", numerics="mps/test"):
    from drmc_rl.pool.conditions import runtime_capabilities
    from drmc_rl.pool.coordinator import REPO
    return dict(protocol=PROTOCOL, worker_id=wid, host="h", numerics=numerics, device="mps", threads=1,
                source="test", capabilities=sorted(runtime_capabilities(REPO) | {"engine:19f292c"}))


def available_everything(c):
    c.available = lambda e: True
    c.scheduler.available = c.available


def submit(c, lease, strengths):
    spec = lease["batch"]
    a, b = spec["a"], spec["b"]
    p = 1 / (1 + math.exp(-(strengths[a] - strengths[b])))
    rows, moves = [], []
    for seed, side, index in spec["jobs"]:
        u = random.Random(f"{spec['condition']}{a}{b}{seed}{side}").random()
        s = 1.0 if u < p else 0.0
        rows.append(dict(seed=seed, side=side, index=index, score=s, winner="a" if s else "b", reason="topout",
                         frames=10, a_stats={}, b_stats={}))
        moves.append([dict(frame=1, placement=dict(action=seed % 7))])
    return c.submit(lease["lease_id"], dict(claim_token=lease["claim_token"], elapsed=1.0, worker=worker(
        lease.get("_wid", "w0")), rows=rows, moves=moves))


def test_new_entrant_plays_the_anchor_first_in_whole_side_swapped_pairs(tmp_path):
    state, (key,) = setup_state(tmp_path)
    state.add_games(rows_for(key, "a", "anchor", BANK[:100], score_a=1.0)[:0])
    for i in range(100):
        state.add_games(rows_for(key, "a", "anchor", [BANK[i]], score_a=float(i % 2)))
    c = coordinator(tmp_path)
    available_everything(c)
    lease = c.lease(worker())
    spec = lease["batch"]
    assert lease["status"] == "lease" and {spec["a"], spec["b"]} == {"anchor", "b"} and "new entrant" in spec["why"]
    seeds = [j[0] for j in spec["jobs"]]
    assert all(seeds.count(s) == 2 for s in seeds) and sorted({j[1] for j in spec["jobs"]}) == [0, 1]
    assert seeds[0] == MIXTURE[0]                # a seed set in order: every pairing plays the same games
    second = c.lease(worker("w1"))["batch"]      # in-flight seeds are never leased twice
    if {second["a"], second["b"]} == {"anchor", "b"}:
        assert not set(second["seeds"]) & set(spec["seeds"])


def test_focused_jobs_precede_background_and_background_keeps_a_share(tmp_path):
    state, (key,) = setup_state(tmp_path, entrants=("anchor", "a", "b", "c"))
    state.record("job", dict(id="focus", status="active", games=512, conditions=["set:main"], entrants=["c"],
                             opponents=["a"], priority=80))
    c = coordinator(tmp_path, settings=dict(background_min_share=0.25, max_inflight_per_pairing=100))
    available_everything(c)
    kinds = [c.lease(worker(f"w{i}"))["batch"]["job"] for i in range(8)]
    assert kinds.count("focus") == 6 and kinds.count(None) == 2
    state.record("job", dict(id="later", status="active", games=64, conditions=["set:main"], entrants=["b"],
                             opponents=["a"], priority=5))           # below background priority


def test_job_completes_and_explicit_seeds_are_honored(tmp_path):
    state, (key,) = setup_state(tmp_path)
    c = coordinator(tmp_path, settings=dict(background_min_share=0))
    available_everything(c)
    c.register(dict(type="job", id="confirm", games=8, conditions=["ev-normal"], entrants=["a"], opponents=["b"],
                    priority=90, seeds=dict(explicit={"ev-normal": [7, 8, 9, 10]})))
    strengths = dict(anchor=0, a=0.5, b=0)
    lease = c.lease(worker())
    assert lease["batch"]["job"] == "confirm" and sorted(lease["batch"]["seeds"]) == [7, 8, 9, 10]
    assert submit(c, lease, strengths)["new_games"] == 8
    c.tick()
    assert c.state.jobs["confirm"]["status"] == "done"


def test_failures_block_an_incompatible_entrant(tmp_path):
    state, (key,) = setup_state(tmp_path)
    c = coordinator(tmp_path)
    available_everything(c)
    for i in range(2):
        lease = c.lease(worker(f"w{i}"))
        c.fail(lease["lease_id"], dict(claim_token=lease["claim_token"], kind="incompatible",
                                       error="ValueError: historical public opponents must ..."))
    assert c.state.blocks
    blocked = next(iter(c.state.blocks.values()))
    assert blocked["a"] in ("a", "b") and "historical" in blocked["reason"]


# ------------------------------------------------------------------------------ intentions


def test_intention_lifecycle_submits_its_job_when_entrants_appear(tmp_path):
    state, (key,) = setup_state(tmp_path)
    c = coordinator(tmp_path)
    available_everything(c)
    c.register(dict(type="intention", id="arm-c", title="Full-size afterstate arm", entrants=["armc-*"],
                    conditions=["set:main"], depends=["capability:backend:events", "external:real-player data"],
                    due="2000-01-01", job=dict(games=16, conditions=["set:main"], entrants=["armc-*"], priority=70)))
    road = {r["id"]: r for r in intent.roadmap(c.state, c.capabilities)}
    assert road["arm-c"]["view"] == "blocked" and road["arm-c"]["overdue"]
    assert set(road["arm-c"]["waiting_on"]) == {"external:real-player data", "entrant:armc-*"}
    c.register(dict(type="intention", id="arm-c", resolved=["external:real-player data"]))
    c.register(dict(type="entrant", **entrant("armc-f1")))
    assert c.state.intentions["arm-c"]["status"] == "running" and c.state.jobs["arm-c"]["intention"] == "arm-c"
    assert c.lease(worker())["batch"]["job"] == "arm-c"
    c.register(dict(type="intention", id="arm-c", status="done", notes="decided"))
    assert not intent.roadmap(c.state, c.capabilities)[0]["overdue"]


# ------------------------------------------------------------------------------ fidelity and stop rule


def test_calibration_admits_a_matching_class_and_replicas_reject_a_bad_one(tmp_path):
    state, (key,) = setup_state(tmp_path)
    c = coordinator(tmp_path, settings=dict(calibration_games=4, replicate_every=1, trace_every=1))
    available_everything(c)
    strengths = dict(anchor=0, a=0.3, b=-0.3)
    lease = c.lease(worker("mac"))
    assert lease["purpose"] == "play"            # mps/ is trusted
    submit(c, lease, strengths)
    lease = c.lease(dict(worker("green", "cuda/x"), worker_id="green"))
    assert lease["purpose"] == "calibrate"
    lease["_wid"] = "green"
    spec = lease["batch"]
    rows, moves = [], []
    for seed, side, index in spec["jobs"]:
        trace = json.loads(__import__("gzip").decompress((tmp_path / c.state.games[game_id(
            spec["condition"], spec["a"], spec["b"], seed, side)]["trace"]).read_bytes()))
        rows.append(trace["game"])
        moves.append(trace["moves"])
    reply = c.submit(lease["lease_id"], dict(claim_token=lease["claim_token"], elapsed=1.0,
                                             worker=worker("green", "cuda/x"), rows=rows, moves=moves))
    assert reply["accepted"] and c.state.admitted["cuda/x"] is True
    # The trusted Mac batch was sampled for audit; green replays it with different decisions.
    lease = c.lease(dict(worker("green", "cuda/x"), worker_id="green"))
    assert lease["purpose"] == "replicate"
    spec = lease["batch"]
    bad = [dict(seed=s, side=d, index=i, score=0.0, winner="b", reason="topout", frames=10)
           for s, d, i in spec["jobs"]]
    c.submit(lease["lease_id"], dict(claim_token=lease["claim_token"], elapsed=1.0, worker=worker("green", "cuda/x"),
                                     rows=bad, moves=[[dict(frame=1, placement=dict(action=99))]] * len(bad)))
    assert c.state.admitted["cuda/x"] != True        # noqa: E712
    assert c.lease(dict(worker("green", "cuda/x"), worker_id="green"))["status"] == "rejected"
    assert len(list(c.state.audit_journal.read())) == 2


def test_stop_rule_fires_after_two_non_improving_snapshots(tmp_path):
    state, (key,) = setup_state(tmp_path, entrants=("anchor",))
    for i, s in enumerate([0.5, 0.9, 0.6, 0.7], start=1):
        state.record("entrant", entrant(f"run-f{i}", lineage=dict(run="run", step=i, parent="anchor")))
    rng = random.Random(3)
    truth = {"run-f1": 0.2, "run-f2": 0.8, "run-f3": 0.1, "run-f4": 0.3}
    for e, t in truth.items():
        p = 1 / (1 + math.exp(-t))
        for seed in BANK[:200]:
            s = [float(rng.random() < p) for _ in (0, 1)]
            rows = rows_for(key, e, "anchor", [seed])
            for row, value in zip(rows, s):
                row["score"] = value if row["a"] == e else 1 - value
            state.add_games(rows)
    c = coordinator(tmp_path)
    result = stop_rule(c, run="run", min_games=128)
    statuses = [s["status"] for s in result["snapshots"]]
    assert statuses == ["improved", "improved", "not improved", "not improved"]
    assert result["fired"] and result["selected"]["entrant"] == "run-f2"
    report = build_report(c)
    run_row = [r for r in report["condition_sets"][0]["pooled"] if r.get("lineage") == "run"]
    assert len(run_row) == 1 and run_row[0]["entrant"] == "run-f4"     # active run: its newest snapshot
    assert len(run_row[0]["trajectory"]) == 4 and "run · " in summary_text(report)


# ------------------------------------------------------------------------------ end to end over HTTP


def test_end_to_end_http_workers_restart_and_recompute(tmp_path, monkeypatch):
    from tools import rating_pool
    from drmc_rl.pool import worker as pool_worker
    from drmc_rl.pool.client import PoolClient
    state, keys = setup_state(tmp_path, entrants=("anchor", "a", "b", "c"), paces=("normal", "fast"))
    for e in ("anchor", "a", "b", "c"):
        (tmp_path / "artifacts" / SHA[e]).write_bytes(b"x")          # servable artifacts
        state.entrants[e]["checkpoint"]["sha256"] = SHA[e]
    del state
    monkeypatch.setattr("tools.trainer_arena_distributed.source_revision", lambda root=None: "test")
    token = "t0ken"
    strengths = dict(anchor=0.0, a=1.2, b=-0.8, c=0.4)
    stopped, ready = threading.Event(), {}
    args = SimpleNamespace(data=tmp_path, settings=json.dumps(dict(seed_bank=BANK, seed_sets=SEED_SETS,
                                                                   default_anchor="anchor",
                                                                   calibration_games=0, replicate_every=0,
                                                                   batch_games=dict(events=8))),
                           allow_source_mismatch=False, report_port=0, report_host=None, host="127.0.0.1", port=0,
                           token_file="unused")
    server = threading.Thread(target=rating_pool.serve, args=(args,), kwargs=dict(
        token=token, stopped=stopped, on_ready=lambda p: ready.setdefault("port", p)), daemon=True)
    server.start()
    for _ in range(200):
        if "port" in ready:
            break
        time.sleep(0.05)
    url = f"http://127.0.0.1:{ready['port']}"
    client = PoolClient(url, token=token)
    wargs = SimpleNamespace(worker_id=None, device="cpu", slot=0, reach_library=None, fake=json.dumps(strengths),
                            native_library=None, numerics="mps/fake", engine="19f292c", threads=1,
                            planner_workers=None, allow_source_mismatch=False, cache=str(tmp_path / "wcache"),
                            artifact_dir=[], cache_gb=1.0, max_loaded=2, poll=0.1, host_budget=0, min_free_gb=0,
                            max_batches=40, max_failures=3)
    threads = []
    for i in range(2):
        w = SimpleNamespace(**{**vars(wargs), "slot": i, "worker_id": f"fake{i}"})
        t = threading.Thread(target=pool_worker.run_worker, args=(w, client), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(120)
    report = client.get("/api/v1/pool/report")
    assert report["totals"]["games"] == 2 * 40 * 8
    order = [r["entrant"] for r in report["condition_sets"][0]["pooled"]]
    assert order[0] == "a" and order[-1] == "b"        # strengths 1.2 > 0.4 ~ 0 > -0.8
    stopped.set()
    server.join(30)
    # Restart-resume and offline recomputation from the journal alone give identical ratings.
    again = PoolCoordinator(tmp_path, source="test", log=lambda *_: None,
                            settings=dict(seed_bank=BANK, seed_sets=SEED_SETS, default_anchor="anchor"))
    assert len(again.state.games) == report["totals"]["games"]
    view = again.ratings_for("main")
    for row in report["condition_sets"][0]["pooled"]:
        assert round(view[row["entrant"]]["rating"]) == row["rating"]
    ids = [r["id"] for r in again.state.games.values()]
    assert len(ids) == len(set(ids))
    # Every pool seed pair is complete and side-swapped.
    for key in keys:
        for seeds in again.state.by_pairing[key].values():
            assert all(sorted(sides) == [0, 1] for sides in seeds.values())


def test_worker_switches_runtimes_without_resetting_torch_threads(monkeypatch):
    import torch
    from drmc_rl.pool.worker import Runtimes
    calls = []

    class Runtime:
        def __init__(self, config):
            torch.set_num_interop_threads(1)
            calls.append(config["rollout_backend"])

        def close(self):
            pass
    monkeypatch.setattr("tools.trainer_planning_arena.ArenaRuntime", Runtime)
    monkeypatch.setattr(torch, "set_num_interop_threads",
                        lambda n: (_ for _ in ()).throw(RuntimeError("twice")) if calls else None)
    runtimes = Runtimes(SimpleNamespace(device="cpu", threads=1, native_library="x", cache="/tmp", planner_workers=None))
    for backend in ("events", "frames", "events"):
        spec = dict(runtime=dict(rollout_backend=backend, anchor_checkpoint="sha256:x"))
        runtimes.get(spec, {"sha256:x": "anchor.pt"})
    assert calls == ["events", "frames", "events"]


def test_report_shows_integer_elo_and_likelihood_of_superiority(tmp_path):
    from statistics import NormalDist
    from drmc_rl.pool.report import PAGE, los_text
    state, keys = setup_state(tmp_path, entrants=("anchor", "a", "b"), paces=("normal", "fast"))
    rng = random.Random(7)
    for key in keys:
        for e, p in (("a", 0.7), ("b", 0.62)):
            for seed in BANK[:120]:
                rows = rows_for(key, e, "anchor", [seed])
                for row in rows:
                    won = float(rng.random() < p)
                    row["score"] = won if row["a"] == e else 1 - won
                state.add_games(rows)
        state.add_games(rows_for(key, "a", "b", BANK[:20], score_a=1.0))
    c = coordinator(tmp_path)
    report = build_report(c)
    table = report["condition_sets"][0]["pooled"]
    assert [r["entrant"] for r in table] == ["a", "b", "anchor"]
    for r in table:
        assert type(r["rating"]) is int and all(type(x) is int for x in r["ci95"])
    assert table[-1]["los"] is None and los_text(table[-1]["los"]) == "–"
    # LOS uses the covariance of the two estimates, averaged over the set's conditions.
    fits = c.all_fits()
    view = c.ratings_for("main")
    var = sum(fits[k].difference_se("a", "b") ** 2 for k in keys) / len(keys) ** 2
    expected = NormalDist().cdf((view["a"]["rating"] - view["b"]["rating"]) / math.sqrt(var))
    assert table[0]["los"] == pytest.approx(expected, abs=1e-4)
    assert fits[keys[0]].difference_se("a", "b") < math.hypot(fits[keys[0]].ratings["a"].se, fits[keys[0]].ratings["b"].se)
    assert table[1]["los"] > 0.99                 # b vs the anchor: clearly stronger
    assert los_text(0.934) == "93%"
    # Per-pace drill-down: each pace table sorted by its own rating, with its own LOS.
    paces = report["condition_sets"][0]["paces"]
    assert [p["pace"] for p in paces] == ["normal", "fast"]
    for p in paces:
        assert [r["rating"] for r in p["ratings"]] == sorted((r["rating"] for r in p["ratings"]), reverse=True)
        assert p["ratings"][-1]["los"] is None and p["ratings"][0]["los"] is not None
    text = summary_text(report)
    line = next(l for l in text.splitlines() if " a " in f" {l.split()[3] if len(l.split()) > 3 else ''} ")
    assert "." not in line.split()[0] and "%" in line
    assert "report.json" in PAGE and "setInterval(load, 60000)" in PAGE and "prefers-color-scheme:dark" in PAGE


def test_lineages_share_budget_collapse_in_reports_and_conclude(tmp_path):
    state, (key,) = setup_state(tmp_path, entrants=("anchor", "a"))
    steps = [0, 25_000_000, 50_000_000, 75_000_000, 100_000_000]
    for i, step in enumerate(steps):
        state.record("entrant", entrant(f"run-f{i}", lineage=dict(run="run", step=step, parent="anchor")))
    rng = random.Random(5)
    truth = [0.1, 0.3, 0.6, 0.4, 0.5]
    for i, t in enumerate(truth[:-1]):          # the newest snapshot is unplayed
        p = 1 / (1 + math.exp(-t))
        for seed in BANK[:150]:
            rows = rows_for(key, f"run-f{i}", "anchor", [seed])
            for row in rows:
                won = float(rng.random() < p)
                row["score"] = won if row["a"] == f"run-f{i}" else 1 - won
            state.add_games(rows)
    assert state.snapshot_role("run-f4") == "newest" and state.snapshot_role("run-f1") == "older"
    c = coordinator(tmp_path)
    available_everything(c)
    # The newest snapshot carries the lineage's new-entrant priority; older ones are maintenance only.
    batch = c.lease(worker())["batch"]
    assert "run-f4" in (batch["a"], batch["b"]) and "(newest)" in batch["why"]
    # The pooled table shows one row for the run, labelled with frames, with its trajectory.
    report = build_report(c)
    rows = report["condition_sets"][0]["pooled"]
    run_rows = [r for r in rows if r.get("lineage") == "run"]
    assert len(run_rows) == 1 and run_rows[0]["label"] == "run · 75M"      # newest rated snapshot
    assert [p["frames"] for p in run_rows[0]["trajectory"]] == ["0", "25M", "50M", "75M"]
    # Per-pace drill-down: the run's row in each pace table carries that pace's snapshots,
    # each with LOS vs the next snapshot (covariance-aware), the last one "–".
    from statistics import NormalDist
    from drmc_rl.pool.report import PAGE, los_text
    pace = report["condition_sets"][0]["paces"][0]
    t = next(r for r in pace["ratings"] if r.get("lineage") == "run")["trajectory"]
    assert [p["entrant"] for p in t] == ["run-f0", "run-f1", "run-f2", "run-f3"] and t[-1]["los"] is None
    f = c.all_fits()[key]
    d = f.ratings["run-f2"].rating - f.ratings["run-f3"].rating
    assert t[2]["los"] == pytest.approx(NormalDist().cdf(d / f.difference_se("run-f2", "run-f3")), abs=1e-4)
    assert los_text(t[-1]["los"]) == "–" and all(type(p["rating"]) is int for p in t)
    assert "data-toggle" in PAGE and "expanded" in PAGE and 'tr class="sub' in PAGE
    assert not {r["entrant"] for r in rows} & {"run-f0", "run-f1", "run-f2"}
    # A 50M-mark stop rule ignores the 25M/75M snapshots; a panel job with step_every does too.
    rule = stop_rule(c, run="run", min_games=128, step_every=50_000_000)
    assert [s["entrant"] for s in rule["snapshots"]] == ["run-f0", "run-f2", "run-f4"]
    c.register(dict(type="job", id="panel", games=64, conditions=["set:main"], entrants=["run-*"],
                    step_every=50_000_000, priority=60))
    assert {a for _, a, b in c.scheduler.job_items(c.state.jobs["panel"])} | \
        {b for _, a, b in c.scheduler.job_items(c.state.jobs["panel"])} == {"anchor", "run-f0", "run-f2", "run-f4"}
    # Concluding keeps the best and final snapshots active and retires the rest (still rated).
    out = c.register(dict(type="conclude", run="run", reason="done"))
    assert out["best"] == "run-f2" and out["final"] == "run-f4"
    status = {e: c.state.entrants[e]["status"] for e in c.state.lineage_members("run")}
    assert status == {"run-f0": "retired", "run-f1": "retired", "run-f2": "active", "run-f3": "retired",
                      "run-f4": "active"}
    rows = build_report(c)["condition_sets"][0]["pooled"]
    shown = [r for r in rows if r.get("lineage") == "run"][0]
    assert shown["entrant"] == "run-f2" and any(p["retired"] for p in shown["trajectory"])
    # Replay reproduces the concluded lineage.
    assert PoolState(tmp_path).lineages["run"]["status"] == "concluded"


def test_pool_stop_rule_concludes_an_opted_in_lineage(tmp_path):
    state, (key,) = setup_state(tmp_path, entrants=("anchor",))
    for i, t in enumerate([0.8, 0.2, 0.1]):
        e = f"auto-f{i}"
        state.record("entrant", entrant(e, lineage=dict(run="auto", step=50_000_000 * i, parent="anchor")))
        p = 1 / (1 + math.exp(-t))
        rng = random.Random(i)
        for seed in BANK[:150]:
            rows = rows_for(key, e, "anchor", [seed])
            for row in rows:
                won = float(rng.random() < p)
                row["score"] = won if row["a"] == e else 1 - won
            state.add_games(rows)
    state.record("lineage", dict(run="auto", stop_rule=dict(set="main", min_games=128, auto=True)))
    c = coordinator(tmp_path)
    assert c.state.lineages["auto"]["status"] == "concluded" and c.state.lineages["auto"]["best"] == "auto-f0"


def test_seed_sets_rotate_by_share_and_feed_views_memorization_and_weights(tmp_path):
    state, keys = setup_state(tmp_path, entrants=("anchor", "a", "b"), paces=("normal", "frame_perfect"))
    c = coordinator(tmp_path, settings=dict(max_inflight_per_pairing=100, background_min_share=0))
    available_everything(c)
    strengths = dict(anchor=0.0, a=0.6, b=-0.4)
    sets_played = []
    for _ in range(48):
        lease = c.lease(worker())
        sets_played.append(c.set_of(lease["batch"]["seeds"][0]))
        submit(c, lease, strengths)
    # Background pairs follow the 50/25/25 shares, always whole side-swapped seed pairs.
    counts = {k: sets_played.count(k) for k in ("mixture", "reserve", "uniform")}
    assert counts["mixture"] >= counts["reserve"] >= 1 and counts["mixture"] >= counts["uniform"] >= 1
    report = build_report(c)
    s0 = report["condition_sets"][0]
    assert s0["weights"] == {"normal": 1.0, "frame_perfect": 3.0}
    # The default ranking is the pace-weighted pooled view, with equal-weight and seed-set views alongside.
    fits = c.all_fits()
    a_w = (fits[keys[0]].ratings["a"].rating * 1 + fits[keys[1]].ratings["a"].rating * 3) / 4
    row = next(r for r in s0["pooled"] if r["entrant"] == "a")
    assert row["rating"] == round(a_w)
    a_e = next(r for r in s0["pooled_equal"] if r["entrant"] == "a")["rating"]
    assert a_e == round((fits[keys[0]].ratings["a"].rating + fits[keys[1]].ratings["a"].rating) / 2)
    assert {r["entrant"] for r in s0["views"]["real_play"]} >= {"anchor"}
    assert s0["pooled"][0]["los"] is not None and s0["pooled"][-1]["los"] is None
    # Memorization: equal strength on seen and reserve seeds, so no flag.
    mem = {m["entrant"]: m for m in report["memorization"]}
    assert "a" in mem and mem["a"]["reserve_seeds"] > 0 and mem["a"]["seen_seeds"] > 0
    assert not any(m.get("flagged") for m in report["memorization"])
    assert "pace-weighted" in summary_text(report)
    # The pool stop rule defaults to the confirmed pace weights.
    for i in range(2):
        c.register(dict(type="entrant", **entrant(f"r-f{i}", lineage=dict(run="r", step=i, parent="anchor"))))
    assert stop_rule(c, run="r", min_games=0)["weighting"] == "pace"


def test_memorization_is_flagged_when_seen_seeds_score_higher(tmp_path):
    from drmc_rl.pool.report import memorization_rows
    state, (key,) = setup_state(tmp_path, entrants=("anchor", "m"))
    for seed in BANK[:150]:
        state.add_games(rows_for(key, "m", "anchor", [seed], score_a=float(seed % 2)))
    for seed in UNIFORM[:150]:
        state.add_games(rows_for(key, "m", "anchor", [seed], score_a=float(seed % 5 != 0)))
    c = coordinator(tmp_path)
    m = {r["entrant"]: r for r in memorization_rows(c)}["m"]
    assert m["gap_points"] > 20 and m["flagged"] and m["detectable_points"] > 0


def test_style_counters_follow_the_big_clear_scorer_and_reach_the_report(tmp_path):
    import numpy as np
    from drmc_rl.pool.style import HUMAN, game_style, metrics
    from drmc_rl.pool.report import PAGE

    def bottle(cells):
        g = np.full(128, 0xFF, np.uint8)
        for (r, c), tile in cells.items():
            g[r * 8 + c] = tile
        return list(bytes(g))
    single = dict(board=bottle({(15, 0): 0x81, (15, 1): 0x81, (15, 2): 0x81}), pill=(0, 2),
                  placement=dict(action=15 * 8 + 3))                       # plain 4-line clear: score 0
    combo = dict(board=bottle({(15, 0): 0x80, (14, 0): 0x81, (13, 0): 0x81, (12, 0): 0x81, (9, 0): 0x80,
                               (8, 0): 0x80}), pill=(1, 0), placement=dict(action=128 + 10 * 8))  # 2 rounds
    nothing = dict(board=bottle({}), pill=(0, 0), placement=dict(action=15 * 8))
    moves = [dict(combo, side=0), dict(single, side=1), dict(nothing, side=0), dict(nothing, side=1)]
    a, b = game_style(dict(side=0), moves)
    assert (a["placements"], a["clears"], a["combos"], a["chains"], a["lines"], a["garbage"]) == (2, 1, 1, 1, 2, 2)
    assert (b["clears"], b["combos"], b["garbage"], b["best"]["score"]) == (1, 0, 0, 0.0)
    m = metrics(dict(a, games=1, best=a["best"]))
    assert m["combos"] == 50.0 and m["chains"] == 50.0 and m["lines_per_clear"] == 2.0 and m["few_games"]
    # Counters travel with the game rows into a per-set section of the report.
    state, (key,) = setup_state(tmp_path, entrants=("anchor", "a"))
    rows = rows_for(key, "a", "anchor", BANK[:2])
    for row in rows:
        row["style"] = [a, b]
    state.add_games(rows)
    report = build_report(coordinator(tmp_path))
    section = report["style"]["sets"][0]
    got = {r["entrant"]: r for r in section["entrants"]}
    assert got["a"]["games"] == 4 and got["a"]["combos"] == 50.0 and got["anchor"]["best"]["score"] == 0.0
    assert report["style"]["human"] == HUMAN and "normal" in section["paces"]
    assert "Combos &amp; style" in PAGE and "data-sort" in PAGE


def test_lineage_roles_resolve_neighbours_and_report_recent_games(tmp_path):
    state, keys = setup_state(tmp_path, entrants=("anchor",), paces=("normal", "fast"))
    truth = {0: -0.4, 1: 0.5, 2: 0.52, 3: 0.2}      # f1 and f2 are nearly equal; f0 is clearly weaker
    for i, t in truth.items():
        e = f"ln-f{i}"
        state.record("entrant", entrant(e, lineage=dict(run="ln", step=25_000_000 * i, parent="anchor")))
        p = 1 / (1 + math.exp(-t))
        rng = random.Random(i)
        for key in keys:
            for seed in BANK[:(40 if i == 2 else 300)]:
                rows = rows_for(key, e, "anchor", [seed])
                for row in rows:
                    won = float(rng.random() < p)
                    row["score"] = won if row["a"] == e else 1 - won
                    row["time"] = "2000-01-01T00:00:00Z"
                state.add_games(rows)
    c = coordinator(tmp_path)
    c.all_fits()
    roles = c.lineage_roles()
    assert roles["ln-f3"][0] == "newest"
    best = next(e for e, r in roles.items() if r[0] == "best")
    assert best in ("ln-f1", "ln-f2")
    other = "ln-f2" if best == "ln-f1" else "ln-f1"
    # f1 vs f2 is undecided and f2 has few games: the non-best one keeps resolving against the best.
    assert roles[other][0] == "resolving" and best in roles[other][2]
    # f0 is decisively weaker than its neighbour f1 and the best: maintenance.
    assert roles["ln-f0"] == ("maintenance", "resolved", [])
    available_everything(c)
    whys = [c.lease(worker(f"w{i}"))["batch"]["why"] for i in range(12)]
    assert any("(resolving)" in w for w in whys) and not any("(maintenance)" in w for w in whys[:3])
    # The per-snapshot game cap ends the boost even when unresolved.
    c.settings["snapshot_game_cap"] = 100
    c._roles = None
    assert c.lineage_roles()[other][0] == "maintenance"
    c.settings["snapshot_game_cap"] = 2000
    c._roles = None
    # Last-hour counts come from journal timestamps; old games do not count, fresh ones do.
    report = build_report(c)
    row = next(r for r in report["condition_sets"][0]["pooled"] if r.get("lineage") == "ln")
    assert row["recent"] == 0 and all(t["recent"] == 0 for t in row["trajectory"])
    assert {t["entrant"]: t["role"] for t in row["trajectory"]}["ln-f0"] == "maintenance (resolved)"
    fresh = rows_for(keys[0], "ln-f3", "anchor", [BANK[350]])
    for r in fresh:
        r["time"] = __import__("drmc_rl.pool.store", fromlist=["now_iso"]).now_iso()
    c.state.add_games(fresh)
    report = build_report(c)
    row = next(r for r in report["condition_sets"][0]["pooled"] if r.get("lineage") == "ln")
    assert {t["entrant"]: t["recent"] for t in row["trajectory"]}["ln-f3"] == 2
    pace = report["condition_sets"][0]["paces"][0]["ratings"]
    assert next(r for r in pace if r.get("lineage") == "ln")["trajectory"][-1]["recent"] == 2
