from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.search.native_pair import capture_native_state
from drmc_rl.search.pill_belief import reserve_for_seed
from drmc_rl.teachers.terminal_rollout import RolloutTask, aggregate_outcomes, rollout_tasks
from tools.audit_terminal_quality import quality_summary


class FirstLegal:
    def infer_batch(self, requests):
        return [({min(state.legal_actions_by_side[side]): 1.}, 0.) for state, side in requests]


@pytest.mark.skipif(not is_library_present(), reason="native pool library missing")
@pytest.mark.parametrize("native_workers", [1, 2])
@pytest.mark.parametrize("level,speed,root_side,seed", [
    (0, 2, 0, (3, 7)), (0, 0, 1, (19, 22)),
    (14, 0, 0, (11, 193)), (20, 2, 1, (43, 177)),
])
def test_reveal_override_rollouts_match_strict_natural_games_and_refill_slots(
    level, speed, root_side, seed, native_workers
):
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        runner.reset(None, [build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
            rng_override=True, rng_state=seed, frame_counter_base=21)])
        initial = capture_native_state(runner, level=level, speed_setting=speed)
        root_action = max(initial.legal_actions_by_side[root_side])
        state = initial
        outcome = None
        for event in range(512):
            actions = np.full(2, -2, np.int32)
            for side, need in enumerate(state.privileged.need_action):
                if need:
                    legal = state.legal_actions_by_side[side]
                    actions[side] = root_action if event == 0 and side == root_side else (min(legal) if legal else -1)
            runner.step_strict(actions)
            state = capture_native_state(runner, level=level, speed_setting=speed)
            if state.privileged.terminal_outcome[root_side]:
                outcome = state.privileged.terminal_outcome[root_side]
                break
        assert outcome in (1, 2, 3)
        task = RolloutTask(0, initial, root_side, root_action, reserve_for_seed(*seed).tobytes(), 1.)
        results = rollout_tasks(
            [replace(task, id=i) for i in range(3)],
            FirstLegal(),
            batch_size=2,
            max_events=2048,
            native_workers=native_workers,
        )
        assert {r["id"] for r in results} == {0, 1, 2}
        assert all(r["outcome"] == outcome for r in results)
        assert all(r["reveals"] > 0 for r in results)
        named = rollout_tasks(
            [replace(task, continuation_id="own", opponent_id="opponent")],
            {"own": FirstLegal(), "opponent": FirstLegal()},
            batch_size=1,
            max_events=2048,
            native_workers=native_workers,
        )
        assert named[0]["outcome"] == outcome
        assert named[0]["continuation_id"] == "own" and named[0]["opponent_id"] == "opponent"
        incomplete = rollout_tasks(
            [task], FirstLegal(), batch_size=1, max_events=1, native_workers=native_workers
        )
        assert incomplete[0]["outcome"] is None
    finally:
        runner.close()


def test_aggregation_preserves_draw_mass_and_rejects_incomplete_coverage():
    tasks = [SimpleNamespace(id=0, weight=.25), SimpleNamespace(id=1, weight=.75)]
    results = [{"id": 1, "outcome": 3, "weight": .75}, {"id": 0, "outcome": 1, "weight": .25}]
    assert aggregate_outcomes(tasks, results) == [.25, .75, 0.]
    with pytest.raises(ValueError, match="coverage"):
        aggregate_outcomes(tasks, results[:1])
    results[0]["outcome"] = None
    assert aggregate_outcomes(tasks, results) is None


def test_candidate_comparison_excludes_any_state_with_incomplete_candidate():
    complete = {"game_id": "a", "public_action": 1, "v3_action": 2,
                "candidates": [{"action": 1, "wdl": [1., 0., 0.]},
                               {"action": 2, "wdl": [0., 1., 0.]}]}
    incomplete = {"game_id": "b", "public_action": 1, "v3_action": 2,
                  "candidates": [{"action": 1, "wdl": None},
                                 {"action": 2, "wdl": [0., 0., 1.]}]}
    summary = quality_summary([complete, incomplete], seed=0)
    assert summary["complete_states"] == 1
    assert summary["incomplete_candidates"] == 1
    assert summary["mean"]["public_action"] == 1.
    assert summary["mean"]["v3_action"] == .5
