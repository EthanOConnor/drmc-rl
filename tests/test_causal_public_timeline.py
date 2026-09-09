from dataclasses import replace

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, context_from_info
from drmc_rl.search.native_pair import (
    CAUSAL_PUBLIC_SCHEMA,
    NativePairSearchModel,
    capture_native_state,
    state_from_payload,
    state_to_payload,
)
from drmc_rl.search.public_policy import PublicPolicyContinuation, policy_request

pytestmark = pytest.mark.skipif(not is_library_present(), reason="native library missing")


def test_unobserved_opponent_commitment_cannot_change_public_input():
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        runner.reset(
            None,
            [
                build_vs_reset_spec(
                    level=(14, 14), speed_setting=(2, 2), rng_state=(0x37, 0x91), rng_override=True
                )
            ],
        )
        root = capture_native_state(runner, level=14, causal_public=True)
        model = NativePairSearchModel(runner)
        actions = sorted(
            zip(root.legal_actions_by_side[1], root.action_costs_by_side[1]), key=lambda x: x[1]
        )
        children = [model.apply_actions(root, None, actions[i][0]) for i in (0, -1)]
        assert children[0].privileged.pair_clocks != children[1].privileged.pair_clocks
        assert children[0].privileged.engine_checkpoint != children[1].privileged.engine_checkpoint
        assert children[0].privileged.public == children[1].privileged.public
        assert children[0].privileged.public.sides[1].board == root.privileged.public.sides[1].board
        assert children[0].privileged.public.observable_clock_delta_frames is None
        inputs = [
            policy_request(
                s.privileged.public,
                0,
                s.legal_actions_by_side[0],
                s.action_costs_by_side[0],
                context_schema=PUBLIC_CONTEXT_SCHEMA,
            )
            for s in children
        ]
        np.testing.assert_array_equal(inputs[0][0], inputs[1][0])
        np.testing.assert_array_equal(
            context_from_info(inputs[0][1]), context_from_info(inputs[1][1])
        )
        restored = state_from_payload(state_to_payload(children[0]))
        assert restored.public_observation_schema == CAUSAL_PUBLIC_SCHEMA
        assert restored.privileged.public == children[0].privileged.public
        # Exact physics remains different, so the teacher's private cache must
        # not merge these states even though the public actor must merge them.
        assert model.key(children[0]) != model.key(children[1])
        assert PublicPolicyContinuation._request_key(
            children[0], 0
        ) == PublicPolicyContinuation._request_key(children[1], 0)
        legacy = replace(restored, public_observation_schema="legacy-warp-buffer-v1")
        with pytest.raises(ValueError, match="future locks"):
            PublicPolicyContinuation._request_key(legacy, 0)
        with pytest.raises(ValueError, match="legacy"):
            capture_native_state(runner, previous=legacy, causal_public=True)
    finally:
        runner.close()


def test_public_view_updates_when_the_warped_timeline_catches_up():
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        runner.reset(
            None,
            [
                build_vs_reset_spec(
                    level=(14, 14), speed_setting=(2, 2), rng_state=(19, 72), rng_override=True
                )
            ],
        )
        state = capture_native_state(runner, level=14, causal_public=True)
        model = NativePairSearchModel(runner)
        initial = state.privileged.public.sides[1].board
        for _ in range(40):
            actions = [
                min(state.legal_actions_by_side[s], default=-1)
                if state.privileged.need_action[s]
                else None
                for s in (0, 1)
            ]
            before = state
            state = model.apply_actions(state, *actions)
            assert state.privileged.public.frame_id >= before.privileged.public.frame_id
            assert state.public_observation_schema == CAUSAL_PUBLIC_SCHEMA
            for side in (0, 1):
                if state.privileged.need_action[side] or state.privileged.pair_clocks[side] == min(
                    state.privileged.pair_clocks
                ):
                    assert state.privileged.public.sides[side].board == bytes(
                        runner.buffers.board_bytes[side]
                    )
                    assert state.privileged.public.sides[side].pill == tuple(
                        runner.buffers.pill_colors[side]
                    )
            if state.privileged.public.sides[1].board != initial:
                break
        else:
            pytest.fail("public bottle never refreshed after the actual lock")
    finally:
        runner.close()


def test_full_reserve_history_cannot_fall_back_after_cache_eviction():
    from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
    from drmc_rl.search.pill_belief import PillReserveBelief

    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        runner.reset(
            None,
            [
                build_vs_reset_spec(
                    level=(14, 14), speed_setting=(2, 2), rng_state=(19, 72), rng_override=True
                )
            ],
        )
        root = capture_native_state(runner, level=14, causal_public=True)
        belief = PillReserveBelief.from_initial_board(level=14, board=runner.buffers.board_bytes[0])
        model = BeliefNativePairSearchModel(runner, belief_cache_size=1)
        model.register_belief(root, belief)
        with pytest.raises(ValueError, match="different public reserve histories"):
            model.register_belief(root, belief.condition(0, 0))
        child = replace(
            root, privileged=replace(root.privileged, engine_checkpoint=b"another branch")
        )
        model.register_belief(child, belief)
        with pytest.raises(ValueError, match="evicted"):
            model.belief(root)
    finally:
        runner.close()
