"""The public event timeline is causal and independent of host polling."""
import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import DrMarioPoolError, is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.search.native_pair import (
    EVENT_PUBLIC_SCHEMA, NativePairSearchModel, capture_native_state,
    state_from_payload, state_to_payload,
)
from drmc_rl.search.pill_belief import reserve_for_seed, pill_id_to_raw_pair
from drmc_rl.search.public_policy import PublicPolicyContinuation

pytestmark = pytest.mark.skipif(not is_library_present(), reason="native library missing")


def reset(runner, level=14, speed=2, seed=(19, 72)):
    runner.reset(None, [build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
        rng_override=True, rng_state=seed, frame_counter_base=21)] * runner.num_pairs)


@pytest.mark.parametrize("level,speed,seed", [
    (0, 2, (3, 7)), (14, 0, (11, 193)), (14, 2, (19, 22)), (20, 2, (43, 177)),
])
@pytest.mark.parametrize("root_side", [0, 1])
def test_natural_games_match_with_dense_sparse_and_no_reveal_polls(level, speed, seed, root_side):
    boundary, direct = DrMarioVsPoolRunner(num_pairs=1), DrMarioVsPoolRunner(num_pairs=1)
    try:
        for runner in (boundary, direct):
            reset(runner, level, speed, seed)
        dense = sparse = capture_native_state(boundary, level=level, speed_setting=speed, event_public=True)
        direct_state = capture_native_state(direct, level=level, speed_setting=speed, event_public=True)
        assert direct_state.public_observation_schema == EVENT_PUBLIC_SCHEMA
        reserve = reserve_for_seed(*seed)
        random = np.random.default_rng(5 + root_side)
        reveals = decisions = 0
        forced = False
        for _ in range(512):
            actions = np.full(2, -2, dtype=np.int32)
            for side, need in enumerate(dense.privileged.need_action):
                if need:
                    legal = dense.legal_actions_by_side[side]
                    actions[side] = int(random.choice(legal)) if legal else -1
                    if side == root_side and not forced:
                        actions[side] = max(legal)
                        forced = True
                    decisions += bool(legal)
            direct.step_strict(actions)
            boundary.step_search(actions)
            dense = capture_native_state(boundary, previous=dense, level=level, speed_setting=speed)
            while (reveal := boundary.search_reveal_info(0)) is not None:
                side, index = reveal
                boundary.search_reveal(0, side, pill_id_to_raw_pair(reserve[index]))
                dense = capture_native_state(boundary, previous=dense, level=level, speed_setting=speed)
                reveals += 1
            sparse = capture_native_state(boundary, previous=sparse, level=level, speed_setting=speed)
            direct_state = capture_native_state(direct, previous=direct_state, level=level, speed_setting=speed)
            assert dense.privileged.public == sparse.privileged.public == direct_state.privileged.public
            assert dense.legal_actions_by_side == direct_state.legal_actions_by_side
            assert dense.action_costs_by_side == direct_state.action_costs_by_side
            assert dense.privileged.pair_clocks == direct_state.privileged.pair_clocks
            assert dense.privileged.terminal_outcome == direct_state.privileged.terminal_outcome
            for key in ("board_bytes", "pill_colors", "preview_colors", "spawn_id", "garbage_pending"):
                np.testing.assert_array_equal(getattr(boundary.buffers, key), getattr(direct.buffers, key))
            for side, need in enumerate(dense.privileged.need_action):
                if need and dense.legal_actions_by_side[side]:
                    assert PublicPolicyContinuation._request_key(dense, side) == PublicPolicyContinuation._request_key(direct_state, side)
                    assert dense.privileged.public.sides[side].board == bytes(boundary.buffers.board_bytes[side])
            if any(dense.privileged.terminal_outcome):
                break
        else:
            pytest.fail("natural game did not finish")
        assert reveals > 0 and decisions > 2
        assert not boundary.buffers.truncated[0] and not direct.buffers.truncated[0]
    finally:
        boundary.close()
        direct.close()


@pytest.mark.parametrize("acting_side", [0, 1])
def test_hidden_commitment_and_its_duration_do_not_change_public_inputs(acting_side):
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        reset(runner)
        root = capture_native_state(runner, level=14, event_public=True)
        model = NativePairSearchModel(runner)
        other = 1 - acting_side
        choices = sorted(zip(root.action_costs_by_side[other], root.legal_actions_by_side[other]))
        children = []
        for _, action in (choices[0], choices[-1]):
            actions = [None, None]
            actions[other] = action
            children.append(model.apply_actions(root, *actions))
        assert children[0].privileged.pair_clocks != children[1].privileged.pair_clocks
        assert children[0].privileged.public == children[1].privileged.public
        assert children[0].privileged.public.sides[other].board == root.privileged.public.sides[other].board
        assert PublicPolicyContinuation._request_key(children[0], acting_side) == PublicPolicyContinuation._request_key(children[1], acting_side)
        decoded = state_from_payload(state_to_payload(children[0]))
        assert decoded == children[0]
        runner.restore(0, decoded.privileged.engine_checkpoint)
        before = runner.snapshot(0)
        for _ in range(3):
            runner.settled_public()
            assert runner.snapshot(0) == before
        repeated = model.apply_actions(root, *([None, choices[0][1]] if acting_side == 0 else [choices[0][1], None]))
        assert repeated == children[0]
    finally:
        runner.close()


def test_partial_reset_preserves_other_timeline_and_v1_cannot_be_relabelled():
    runner = DrMarioVsPoolRunner(num_pairs=2)
    try:
        reset(runner)
        runner.step_strict(np.array([-1, -2, -1, -2], np.int32))
        before = runner.snapshot(1), bytes(runner.settled_public(1))
        specs = [build_vs_reset_spec(level=(14, 14), rng_override=True, rng_state=(3, 7))] * 2
        runner.reset(np.array([1, 0], np.uint8), specs)
        assert (runner.snapshot(1), bytes(runner.settled_public(1))) == before
        # The old public contract is retained deliberately, even on a new native library.
        v1 = capture_native_state(runner, level=14, causal_public=True)
        with pytest.raises(ValueError, match="cannot be relabeled"):
            capture_native_state(runner, previous=v1, event_public=True)
        runner.step(np.full(4, -2, np.int32), None, None)
        with pytest.raises(DrMarioPoolError, match="history unavailable"):
            runner.settled_public()
    finally:
        runner.close()


def test_actual_causal_policy_scores_v2_and_rejects_missing_history(tmp_path):
    import torch
    from dataclasses import replace
    from tests.test_quality_supervision import checkpoint

    torch.set_num_threads(1)
    parent, _ = checkpoint()
    parent["cfg"]["env"] = {"public_observations": True}
    path = tmp_path / "parent.pt"
    torch.save(parent, path)
    actor = PublicPolicyContinuation(path, device="cpu", cache_size=8)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        reset(runner)
        state = capture_native_state(runner, level=14, event_public=True)
        answers = actor.infer_batch([(state, 0), (state, 1)])
        for side, (probabilities, value) in enumerate(answers):
            assert set(probabilities) == set(state.legal_actions_by_side[side])
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert np.isfinite(value)
        with pytest.raises(ValueError, match="causal"):
            actor.infer_batch([(replace(state, public_observation_schema="legacy-warp-buffer-v1"), 0)])
    finally:
        runner.close()
