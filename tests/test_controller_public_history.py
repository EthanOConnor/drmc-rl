"""Public history must survive native batching without changing play."""
import ctypes as C
from dataclasses import replace
import os

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_vs_pool import build_vs_reset_spec
from drmc_rl.envs.backends.vs_frames import EventVsPool, FrameState, FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.game.pair_state import PairEventKind
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, context_from_info
from drmc_rl.human.backend import plan_candidates
from drmc_rl.human.controller_context import controller_policy_inputs, live_controller_state
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import MemoPolicy
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import run_batch


LIB = os.environ.get("DRMC_FRAME_LIBRARY")


def test_event_skipping_preserves_every_retained_public_event_and_age():
    with FrameVsPool(lib_path=LIB) as reference, EventVsPool(lib_path=LIB) as fast:
        reference.reset([17291])
        fast.reset([17291])
        previous_events = ()
        for _ in range(24):
            progress = fast.advance(20000)
            target = int(fast.states[0].frame)
            while reference.states[0].frame < target:
                reference.step()
            assert bytes(reference.states) == bytes(fast.states)
            for side in (0, 1):
                expected, actual = reference.public_state(side), fast.public_state(side)
                assert expected == actual
                assert len(actual.recent_events) <= 32
                assert all(e.frame_id <= actual.frame_id for e in actual.recent_events)
            previous_events = actual.recent_events
            if fast.states[0].terminal:
                break
            for side, event in enumerate(progress):
                if event.needs_action:
                    fast.install(side)
        assert any(e.kind == PairEventKind.LOCK for e in previous_events)
        assert sum(e.kind == PairEventKind.TERMINAL for e in previous_events) == 2
        final = fast.public_state(0)
        fast.advance(30000)
        assert fast.public_state(0) == final
        fast.reset([17291])
        assert all(e.kind == PairEventKind.SPAWN for e in fast.public_state(0).recent_events)


def test_exact_clear_effects_and_released_volley_survive_bulk_step():
    # Two independent horizontal clears generate a real two-piece attack.
    board = np.full((2, 128), 255, np.uint8)
    board[0, 120:124] = 0xD0
    board[0, 112:116] = 0xD1
    board[0, 127] = board[1, 127] = 0xD2
    spec = build_vs_reset_spec(level=(14, 14), speed_setting=(2, 2),
                              rng_override=True, rng_state=(11, 27))
    spec.checkpoint_enabled = 1
    for side in (0, 1):
        spec.checkpoint_board[side][:] = board[side]
        spec.checkpoint_falling_colors[side][:] = [1, 2]
        spec.checkpoint_preview_colors[side][:] = [0, 1]
        spec.checkpoint_pill_counter_total[side] = 1
    with FrameVsPool(lib_path=LIB) as pool:
        pool._check(pool.lib.drm_vspool_frame_reset(
            pool.handle, None, C.byref(spec), pool.states, C.sizeof(FrameState)))
        collected = []
        # Frame effects are collected without reading public history every tick.
        for _ in range(8):
            pool.step([4, 4], count=150)
            for event in pool.public_state(0).recent_events:
                if event not in collected:
                    collected.append(event)
            if any(e.kind == PairEventKind.VOLLEY for e in collected):
                break
        clears = [e for e in collected if e.kind == PairEventKind.CLEAR and e.side == 0]
        assert sum(e.public_payload["viruses_cleared"] for e in clears) == 8
        assert sum(e.public_payload["tiles_cleared"] for e in clears) == 8
        assert sum(e.public_payload["lines_cleared"] for e in clears) == 2
        volleys = [e for e in collected if e.kind == PairEventKind.VOLLEY]
        assert len(volleys) == 1 and volleys[0].side == 1
        payload = volleys[0].public_payload
        assert payload["garbage_size"] == 2 and payload["sender"] == 0
        assert len(payload["columns"]) == len(payload["colors"]) == 2
        assert 16 <= payload["salt_frames"] <= 256
        assert pool.states[0].garbage_sent_total == 2


class ContextPolicy:
    aux_spec = PUBLIC_CONTEXT_SCHEMA

    def __init__(self):
        self.inputs = []

    def score(self, observations, infos):
        vectors = np.stack([context_from_info(i) for i in infos])
        self.inputs.extend((o.copy(), i["public_pair_state"], v.copy())
                           for o, i, v in zip(observations, infos, vectors))
        masks = np.stack([i["placements/feasible_mask"].reshape(512) for i in infos])
        costs = np.stack([i["placements/cost_to_lock"].reshape(512) for i in infos])
        return np.broadcast_to(np.arange(512), masks.shape), masks, -costs.astype(np.float32)


@pytest.mark.parametrize("pace_name", ["sloth", "top_humans"])
def test_live_context_reaches_both_runners_with_identical_decisions(pace_name):
    config = {"native_library": LIB, "variants": {"a": {"delay": 4}, "b": {"delay": 4}},
              "max_game_frames": 1800, "replay_games": 0}
    match = {"a": "a", "b": "b", "games": 2, "level": 14, "pace": pace_name}
    jobs = [(17291, 0, 0)]
    reference, fast = ContextPolicy(), ContextPolicy()
    planner, parallel = NativeReachabilityRunner(), ParallelPlanning(2)
    try:
        expected, _ = run_batch(config, match, jobs, reference, planner, None)
        actual, _ = run_event_batch(config, match, jobs, fast, parallel, None)
        assert expected[0][1] == actual[0][1]
        assert len(reference.inputs) == len(fast.inputs) > 2
        # Different global batch scheduling is allowed; the same public request
        # must still have byte-identical model input and motor conditioning.
        def indexed(inputs):
            return {(p.frame_id, p.viewer_side): (o, p, v) for o, p, v in inputs}
        left, right = indexed(reference.inputs), indexed(fast.inputs)
        assert left.keys() == right.keys()
        for key, (obs, public, vector) in left.items():
            assert public == right[key][1]
            np.testing.assert_array_equal(obs, right[key][0])
            np.testing.assert_array_equal(vector, right[key][2])
    finally:
        planner.close()
        parallel.close()


def test_memo_identity_includes_actual_history_and_motor_context():
    with FrameVsPool(lib_path=LIB) as pool:
        pool.reset([17291])
        while not pool.states[0].falling:
            pool.step()
        state = pool.semantic(0, public_context=True)
        actor, planner, pace = ContextPolicy(), NativeReachabilityRunner(), resolve_pace("normal")
        try:
            candidate = plan_candidates(planner, state, 22, pace)
            obs, infos = controller_policy_inputs(actor, candidate, state, pace, 22, 4)
            memo = MemoPolicy(actor)
            memo.score(obs, infos)
            memo.score(obs, infos)
            assert len(actor.inputs) == 1
            changed = {**infos[0], "public_execution": replace(infos[0]["public_execution"], compute_frames=5)}
            memo.score(obs, [changed])
            assert len(actor.inputs) == 2
            changed = {**infos[0], "public_pair_state": replace(infos[0]["public_pair_state"], recent_events=())}
            assert infos[0]["public_pair_state"].recent_events
            memo.score(obs, [changed])
            assert len(actor.inputs) == 3
        finally:
            planner.close()


@pytest.mark.parametrize("speed, speed_ups, period", [(0, 0, 40), (2, 49, 1)])
def test_motor_context_uses_the_actual_gravity_period(speed, speed_ups, period):
    with FrameVsPool(lib_path=LIB) as pool:
        pool.reset([17291])
        while not pool.states[0].falling:
            pool.step()
        state = pool.semantic(0, public_context=True)
        state.update(speed=speed, speed_ups=speed_ups)
        actor, planner, pace = ContextPolicy(), NativeReachabilityRunner(), resolve_pace("normal")
        try:
            candidate = plan_candidates(planner, state, 0, pace)
            _, infos = controller_policy_inputs(actor, candidate, state, pace, 0, 0)
            assert infos[0]["public_execution"].gravity_frames == period
        finally:
            planner.close()


def test_live_wire_context_matches_native_features_and_rejects_inconsistent_inputs():
    from copy import deepcopy
    from drmc_rl.game.public_context import encode_public_context

    with FrameVsPool(lib_path=LIB) as pool:
        pool.reset([17291])
        while not pool.states[1].falling:
            pool.step()
        expected = pool.public_state(1)
        wire = expected.to_dict()
        for side in wire["sides"]:
            del side["board_b64"]
        wire.update(schema="public-controller-history-v1", compute_frames=4)
        state = pool.semantic(1) | {"public_live_context": wire}
        actual = live_controller_state(state)["public_pair_state"]
        np.testing.assert_array_equal(encode_public_context(actual, 1), encode_public_context(expected, 1))
        changed = deepcopy(state)
        changed["public_live_context"]["sides"][1]["active"]["row_top"] += 1
        with pytest.raises(ValueError, match="pose disagree"):
            live_controller_state(changed)
        changed = deepcopy(state)
        changed["public_live_context"]["recent_events"][0]["frame_id"] = expected.frame_id + 1
        with pytest.raises(ValueError, match="causal"):
            live_controller_state(changed)
        changed = deepcopy(state)
        changed["public_live_context"]["pending_attack"] = 4
        with pytest.raises(ValueError, match="forbidden"):
            live_controller_state(changed)
