from __future__ import annotations

import json
from pathlib import Path

from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.teachers.counterfactual import CounterfactualTeacher
from drmc_rl.teachers.counterfactual_release import (
    ReleaseSettings,
    build_release,
    select_states,
    sha256_file,
)
from tests.test_counterfactual_teacher import Model, State


def _bank(path: Path, rows: int = 12) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "id": f"state-{index:03d}",
                        "phase": "both",
                        "score": index % 3,
                        "candidate_bin": "wide" if index % 2 else "narrow",
                        "tactical": "race" if index % 3 else "defense",
                    }
                )
                + "\n"
            )


def _settings(path: Path, *, max_nodes: int = 1000) -> ReleaseSettings:
    return ReleaseSettings(
        input_sha256=sha256_file(path),
        adapter="tests.fake:factory",
        root_side=0,
        search={"depth_events": 3, "own_beam": 2, "max_nodes": max_nodes},
        seed=7,
        shard_index=0,
        num_shards=1,
        stratum_fields=("candidate_bin", "tactical"),
        per_stratum=2,
        max_states=8,
        chunk_size=3,
        corpus_release="test-corpus-sha256:abc",
        continuation_mixture="test-mixture-v1",
        native_revision="native-test",
        planner_revision="planner-test",
    )


def _decode(payload: dict[str, object]) -> State:
    return State(str(payload["phase"]), int(payload["score"]))


def test_stratified_selection_and_hash_shards_are_deterministic(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _bank(bank, 40)
    common = dict(
        path=bank,
        seed=11,
        num_shards=2,
        stratum_fields=("candidate_bin", "tactical"),
        per_stratum=3,
        max_states=None,
    )
    left = select_states(shard_index=0, **common)
    right = select_states(shard_index=1, **common)
    assert {item.identity for item in left}.isdisjoint(item.identity for item in right)
    assert left == select_states(shard_index=0, **common)
    assert {item.stratum for item in left + right}


def test_release_is_content_addressed_and_resume_verifies_parts(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    output = tmp_path / "release"
    _bank(bank)
    teacher = CounterfactualTeacher(Model(), config=SearchConfig(depth_events=3, own_beam=2))
    settings = _settings(bank)
    manifest_path = build_release(
        input_path=bank,
        output_dir=output,
        adapter_spec=settings.adapter,
        teacher=teacher,
        decode=_decode,
        settings=settings,
        resume=False,
    )
    first = json.loads(manifest_path.read_text())
    assert first["schema"] == "drmc-counterfactual-release-v1"
    assert first["selected_states"] > 0
    assert first["budget_exhausted"] == 0
    assert all((output / part["file"]).is_file() for part in first["parts"])

    second_path = build_release(
        input_path=bank,
        output_dir=output,
        adapter_spec=settings.adapter,
        teacher=teacher,
        decode=_decode,
        settings=settings,
        resume=True,
    )
    second = json.loads(second_path.read_text())
    assert second["release_sha256"] == first["release_sha256"]


def test_budget_exhaustion_leaves_no_completed_release(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    output = tmp_path / "release"
    _bank(bank, 2)
    settings = _settings(bank, max_nodes=1)
    teacher = CounterfactualTeacher(
        Model(), config=SearchConfig(depth_events=3, own_beam=2, max_nodes=1)
    )
    try:
        build_release(
            input_path=bank,
            output_dir=output,
            adapter_spec=settings.adapter,
            teacher=teacher,
            decode=_decode,
            settings=settings,
            resume=False,
        )
    except RuntimeError as error:
        assert "budget exhausted" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected budget exhaustion")
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("*.complete.json"))
