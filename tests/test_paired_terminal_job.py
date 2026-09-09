from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drmc_rl.search.pill_belief import PillReserveBelief
from tests.test_terminal_rollout import FirstLegal
import tools.build_paired_terminal_quality as module


def fixture_job(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"one frozen checkpoint")
    bank = tmp_path / "source.jsonl"
    bank.write_text("source identity")
    rows = [dict(id=f"root-{i}", game_id=f"game-{i}", level=14, speed=2, root_side=0,
                 natural_outcome_available=True, outcome="win", rollout_policy="frozen-public-core-argmax",
                 reserve_belief=PillReserveBelief().to_dict()) for i in range(3)]
    monkeypatch.setattr(module, "load_source_rows", lambda _: rows)
    monkeypatch.setattr(module, "state_from_payload", lambda row: SimpleNamespace(
        source_id=row["id"], legal_actions_by_side=((10, 20), (10, 20)),
        privileged=SimpleNamespace(need_action=(True, True))))
    monkeypatch.setattr(module, "reserve_hypotheses", lambda _: ((bytes(128), 1.),))
    monkeypatch.setattr(module, "PublicPolicyContinuation", lambda *args, **kwargs: FirstLegal())
    return dict(output=str(tmp_path / "targets"), state_bank=str(bank), members={"a": str(checkpoint)},
                continuations=[dict(actor="a", opponent="a", weight=1.)], reference="a", seed=42,
                states=3, root_batch_size=3, device="cpu", threads=1)


def outcome(task):
    # Make different source roots disagree to expose a task-ID collision.
    value = int(task.state.source_id[-1])
    return dict(id=task.id, weight=task.weight, outcome=3 if task.action == 10 else 1 if value % 2 else 2)


def test_multi_root_job_commits_full_roots_and_resumes_only_unfinished_sources(tmp_path, monkeypatch):
    config = fixture_job(tmp_path, monkeypatch)
    called, interrupted = [], False

    def scheduler(tasks, actors, *, on_result, **kwargs):
        nonlocal interrupted
        called.append([t.state.source_id for t in tasks])
        for i, task in enumerate(tasks):
            if not interrupted and i == 3:
                interrupted = True
                raise RuntimeError("simulated scheduler interruption")
            on_result(outcome(task))

    monkeypatch.setattr(module, "rollout_tasks", scheduler)
    with pytest.raises(RuntimeError, match="simulated scheduler"):
        module.run(config)
    output = Path(config["output"])
    committed = list((output / "roots").glob("*.json"))
    assert len(committed) == 1
    before = committed[0].read_bytes()
    first_root = json.loads(before)["target"]["source_id"]
    result = module.run(config)
    assert result["states"] == 3 and result["rollouts"] == 6
    assert first_root not in called[1] and committed[0].read_bytes() == before
    targets = [json.loads(s) for s in (output/"targets.jsonl").read_text().splitlines()]
    assert len({r["source_id"] for r in targets}) == 3
    for target in targets:
        expected = [1., 0., 0.] if int(target["source_id"][-1]) % 2 else [0., 0., 1.]
        assert target["candidates"][1]["wdl"] == expected
    assert module.run(config)["states"] == 3 and len(called) == 2
    with pytest.raises(ValueError, match="resume changed"):
        module.run({**config, "max_events": 99})


def test_multi_root_job_keeps_censored_labels_unknown_after_resume(tmp_path, monkeypatch):
    config = fixture_job(tmp_path, monkeypatch)

    def scheduler(tasks, actors, *, on_result, **kwargs):
        for task in reversed(tasks):
            result = outcome(task)
            if task.state.source_id == "root-1" and task.action == 20:
                result["outcome"] = None
            on_result(result)

    monkeypatch.setattr(module, "rollout_tasks", scheduler)
    result = module.run(config)
    assert result["censored_rollouts"] == 1
    assert module.run(config)["censored_rollouts"] == 1
    root = json.loads(module.root_path(Path(config["output"]), "root-1").read_text())
    assert root["target"]["policy_target"] is None
    assert root["target"]["candidates"][1]["wdl"] is None
