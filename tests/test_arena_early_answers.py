import json
import sys
from types import SimpleNamespace

from drmc_rl.arena.identity import mirrored_games, static_identity
from drmc_rl.execution.pace import resolve_pace
import tools.trainer_planning_arena as arena


def _config(tmp_path, variants, schedule, **extra):
    core, other = tmp_path / "core.pt", tmp_path / "other.pt"
    core.write_bytes(b"core weights")
    other.write_bytes(b"other weights")
    for id, params in variants.items():
        params.setdefault("name", id)
        params["checkpoint"] = str(other if params.pop("other", False) else core)
    return dict(checkpoint=str(core), device="cpu", variants=variants, schedule=schedule,
                output=str(tmp_path / "out"), working_db=str(tmp_path / "arena.sqlite"), pairs=8,
                reactive_compute_frames=4, preparation_compute_frames=6, **extra)


def test_static_signature_uses_the_charged_delay_and_model_bytes(tmp_path):
    config = _config(tmp_path, dict(
        base=dict(delay=4), c5=dict(delay=5), c6pin=dict(delay=6, compute_input_frames=4),
        twin=dict(delay=4, name="renamed"), other=dict(delay=4, other=True),
        settled=dict(delay=4, decision_point="settled"), settled8=dict(delay=8, decision_point="settled"),
    ), [])
    top_humans = resolve_pace("top_humans").reaction_frames
    assert top_humans == 6
    match = lambda a, b, pace: dict(a=a, b=b, pace=pace)
    assert static_identity(config, match("twin", "base", "frame_perfect")) == "identical"
    assert static_identity(config, match("c5", "base", "top_humans")) == "probe"
    assert static_identity(config, match("c5", "base", "frame_perfect")) is None
    assert static_identity(config, match("c6pin", "base", "top_humans")) == "probe"
    assert static_identity(config, match("other", "base", "top_humans")) is None
    assert static_identity(config, match("settled", "base", "top_humans")) is None
    # Early decision points use the raw delay for their own timing.
    assert static_identity(config, match("settled8", "settled", "normal")) is None


def _mirror_rows(scores=(1.0, 0.0)):
    rows = [dict(seed=5, side=s, index=s, score=scores[s], frames=900, reason="topout") for s in (0, 1)]
    journals = {0: [dict(frame=3, side=0, placement=dict(action=17))],
                1: [dict(frame=3, side=0, placement=dict(action=17))]}
    return rows, journals


def test_mirrored_games_require_byte_identical_journals_and_opposite_scores():
    rows, journals = _mirror_rows()
    assert mirrored_games(rows, journals)
    journals[1][0]["placement"]["action"] = 18
    assert not mirrored_games(rows, journals)
    rows, journals = _mirror_rows((1.0, 1.0))
    assert not mirrored_games(rows, journals)
    assert not mirrored_games(rows[:1], journals)


def _fake_rollout(identical_pairs):
    """Identical entrants replay one physical game per seed; others favour A by seed."""
    def rollout(config, match, jobs, policy, planner, preparer, *, policies=None):
        output = []
        for seed, side, index in jobs:
            same = (match["a"], match["b"]) in identical_pairs
            score = (1.0 if side == 0 else 0.0) if same else (1.0 if seed % 5 else 0.0)
            action = seed % 11 if same else seed % 11 + side
            row = dict(seed=seed, side=side, index=index, score=score,
                       winner="a" if score == 1 else "b", reason="topout", frames=600 + seed % 7,
                       a_stats={"decisions": 3}, b_stats={"decisions": 3})
            output.append((row, [dict(frame=5, side=0, placement=dict(action=action))], []))
        return output, 0.01
    return rollout


def _run(monkeypatch, tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    stub = SimpleNamespace(aux_dim=0, aux_spec=None, close=lambda: None)
    monkeypatch.setattr(arena, "PlainPolicy", lambda *a, **k: stub)
    monkeypatch.setattr(arena, "NativeReachabilityRunner", lambda: stub)
    monkeypatch.setattr(arena, "run_batch", _fake_rollout({("c5", "base"), ("twin", "base")}))
    monkeypatch.setattr(sys, "argv", ["arena", "--config", str(path)])
    monkeypatch.setattr(arena.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(arena.torch, "set_num_threads", lambda n: None)
    arena.main()
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    games = [json.loads(line) for line in (tmp_path / "out" / "games.jsonl").read_text().splitlines()]
    return {t["id"]: t for t in results["tournaments"]}, games


def test_arena_skips_identical_entrants_and_stops_decided_comparisons(monkeypatch, tmp_path):
    schedule = [
        dict(id="twin", a="twin", b="base", games=64, level=14, pace="frame_perfect", seed=1),
        dict(id="probe", a="c5", b="base", games=64, level=14, pace="top_humans", seed=2),
        dict(id="strong", a="c8", b="base", games=256, level=14, pace="frame_perfect", seed=3,
             sequential=dict(question="threshold", threshold=0.5)),
        dict(id="plain", a="c8", b="base", games=16, level=14, pace="frame_perfect", seed=4),
    ]
    config = _config(tmp_path, dict(base=dict(delay=4), twin=dict(delay=4), c5=dict(delay=5),
                                    c8=dict(delay=8)), schedule, skip_identical=True, identity_probe_games=8)
    tournaments, games = _run(monkeypatch, tmp_path, config)
    twin, probe, strong, plain = (tournaments[k] for k in ("twin", "probe", "strong", "plain"))
    assert twin["played"] == 0 and twin["stopping"]["decision"] == "identical"
    assert twin["stopping"]["score"] == 0.5 and twin["status"] == "Decided: identical"
    assert probe["played"] == 8 and probe["stopping"]["decision"] == "identical"
    assert strong["stopping"]["decision"] == "pass" and strong["stopping"]["early"]
    assert strong["played"] < 256 and strong["played"] % 8 == 0
    assert plain["played"] == 16 and plain["status"] == "Complete" and "stopping" not in plain
    assert len(games) == 8 + strong["played"] + 16

    # A resumed worker recomputes every verdict and plays nothing more.
    tournaments, games_again = _run(monkeypatch, tmp_path, config)
    assert len(games_again) == len(games)
    assert tournaments["probe"]["stopping"]["decision"] == "identical"
    assert tournaments["strong"]["stopping"]["decision"] == "pass"


def test_a_probe_that_finds_different_play_continues_the_comparison(monkeypatch, tmp_path):
    schedule = [dict(id="c6", a="c6", b="base", games=32, level=14, pace="top_humans", seed=5)]
    config = _config(tmp_path, dict(base=dict(delay=4), c6=dict(delay=6)), schedule,
                     skip_identical=True, identity_probe_games=8)
    tournaments, games = _run(monkeypatch, tmp_path, config)
    assert tournaments["c6"]["played"] == 32 and "stopping" not in tournaments["c6"]
