from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import SearchConfig, WDL
from drmc_rl.teachers.counterfactual import CounterfactualTeacher
from drmc_rl.teachers.counterfactual_release import ReleaseSettings, build_release, sha256_file
from drmc_rl.teachers.release_analysis import compare_beam_sweep, compare_releases, load_release


@dataclass(frozen=True)
class State:
    action: int | None = None


class Model:
    def __init__(self, first: WDL, second: WDL):
        self.values = (first, second)

    def key(self, state):
        return state.action

    def boundary(self, state):
        return DecisionBoundary.P1 if state.action is None else DecisionBoundary.TERMINAL

    def legal_actions(self, state, side):
        del state, side
        return (0, 1)

    def prior(self, state, side, actions):
        del state, side
        return [1.0] * len(actions)

    def apply_actions(self, state, action_p1, action_p2):
        del state, action_p2
        return State(int(action_p1))

    def advance(self, state):
        return state

    def chance_outcomes(self, state):
        del state
        return ()

    def terminal_value(self, state, root_side):
        del root_side
        return None if state.action is None else self.values[state.action]

    def evaluate(self, state, root_side):
        del state, root_side
        return WDL(0.4, 0.2, 0.4)


def _settings(bank: Path, beam: int) -> ReleaseSettings:
    return ReleaseSettings(
        input_sha256=sha256_file(bank),
        adapter="test:factory",
        root_side=0,
        search={
            "depth_events": 1,
            "opponent_beam": beam,
            "own_beam": 2,
            "chance_beam": 9,
            "max_nodes": 100,
        },
        seed=1,
        shard_index=0,
        num_shards=1,
        stratum_fields=(),
        per_stratum=None,
        max_states=None,
        chunk_size=2,
        corpus_release="test",
        continuation_mixture="test-mixture",
        native_revision="native",
        planner_revision="planner",
        mixture_manifest_sha256="a" * 64,
        wdl_calibration_sha256="b" * 64,
        chance_model="nes-reserve-seed-belief-v1",
        information_scope="privileged-test",
    )


def _release(
    tmp_path: Path,
    name: str,
    model: Model,
    beam: int,
    *,
    mutate_settings=None,
) -> Path:
    bank = tmp_path / "bank.jsonl"
    if not bank.exists():
        bank.write_text(
            "\n".join(
                json.dumps({"id": f"state-{index}", "root_side": 0, "score": index})
                for index in range(3)
            )
            + "\n"
        )
    settings = _settings(bank, beam)
    if mutate_settings is not None:
        settings = mutate_settings(settings)
    return build_release(
        input_path=bank,
        output_dir=tmp_path / name,
        adapter_spec=settings.adapter,
        teacher=CounterfactualTeacher(
            model, config=SearchConfig(depth_events=1, own_beam=2)
        ),
        decode=lambda _payload: State(),
        settings=settings,
        resume=False,
    )


def _model(offset: float = 0.0) -> Model:
    return Model(
        WDL(0.8 - offset, 0.1, 0.1 + offset),
        WDL(0.3 + offset, 0.2, 0.5 - offset),
    )


def test_release_comparison_aligns_actions_and_reports_beam_sensitivity(
    tmp_path: Path,
) -> None:
    reference_path = _release(tmp_path, "beam8", _model(), 8)
    candidate_path = _release(tmp_path, "beam4", _model(0.05), 4)
    reference = load_release([reference_path])
    candidate = load_release([candidate_path])
    comparison = compare_releases(reference, candidate)
    assert comparison["aggregate"]["states"] == 3
    assert comparison["aggregate"]["top1_agreement"] == 1.0
    assert comparison["aggregate"]["max_win_delta"]["max"] > 0
    assert comparison["reference"]["chance_model"] == "nes-reserve-seed-belief-v1"
    assert comparison["settings_compatible"] is True

    sweep = compare_beam_sweep({4: candidate, 8: reference})
    assert sweep["reference_beam"] == 8
    assert set(sweep["comparisons"]) == {"4"}


def test_release_comparison_rejects_non_beam_provenance_drift(tmp_path: Path) -> None:
    reference = load_release([_release(tmp_path, "beam8", _model(), 8)])
    candidate = load_release(
        [
            _release(
                tmp_path,
                "beam4-drift",
                _model(0.05),
                4,
                mutate_settings=lambda settings: replace(
                    settings, wdl_calibration_sha256="c" * 64
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="other than opponent_beam"):
        compare_releases(reference, candidate)


def test_beam_sweep_rejects_mislabeled_release(tmp_path: Path) -> None:
    reference = load_release([_release(tmp_path, "beam8", _model(), 8)])
    candidate = load_release([_release(tmp_path, "beam4", _model(0.05), 4)])
    with pytest.raises(ValueError, match="does not match release setting"):
        compare_beam_sweep({4: reference, 8: candidate})
