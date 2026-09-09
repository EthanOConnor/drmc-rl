from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.search.native_pair import LEGACY_PUBLIC_SCHEMA, capture_native_state, state_from_payload
from drmc_rl.search.pill_belief import reserve_for_seed
from drmc_rl.search.public_policy import PublicPolicyContinuation, policy_request
from drmc_rl.teachers.terminal_rollout import RolloutTask, rollout_tasks
from tests.test_quality_supervision import checkpoint, data
from tests.test_terminal_rollout import FirstLegal


def changed_public(state, *, side=0, **changes):
    public = state.privileged.public
    sides = list(public.sides)
    sides[side] = replace(sides[side], **changes)
    return replace(state, privileged=replace(state.privileged, public=replace(public, sides=tuple(sides))))


def packed(continuation, state, side=0):
    obs, info = policy_request(state.privileged.public, side,
                               state.legal_actions_by_side[side], state.action_costs_by_side[side])
    info["vs/observation_timeline"] = state.public_observation_schema
    inputs, aux, _, _ = continuation.policy.model_inputs(obs[None], [info])
    return [tensor.cpu().numpy() for tensor in inputs] + ([] if aux is None else [aux.cpu().numpy()])


def test_bounded_cache_deduplicates_and_matches_actual_public_model_inputs(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    parent, _ = checkpoint()
    path = tmp_path / "parent.pt"
    torch.save(parent, path)
    actor = PublicPolicyContinuation(path, device="cpu", cache_size=2)
    state = state_from_payload(data()[0])
    cold = actor.infer_batch([(state, 0)] * 3)
    assert actor.last_inference_batch_rows == (1,)
    assert actor.last_batch_duplicates == 2
    assert cold[0] == cold[1] == cold[2]
    cold[0][0].clear()
    assert actor.infer_batch([(state, 0)])[0] == cold[1]
    assert actor.last_cache_hits == 1 and actor.last_inference_batch_rows == ()
    # Neither private game state nor unused legacy clock/age/preview fields
    # change the actual network inputs. Validate tensors as well as cache keys.
    public = state.privileged.public
    clock = replace(state, privileged=replace(state.privileged,
                                              public=replace(public, frame_id=public.frame_id + 7)))
    variants = [clock, changed_public(state, side=1, preview=(0, 0)),
                changed_public(state, state_age_frames=13, viruses_remaining=3),
                replace(state, privileged=replace(state.privileged, pending_attacks=(9, 7),
                                                   engine_checkpoint=b"different hidden state"))]
    original = packed(actor, state)
    for variant in variants:
        assert actor._policy_key(state, 0) == actor._policy_key(variant, 0)
        for expected, actual in zip(original, packed(actor, variant), strict=True):
            np.testing.assert_array_equal(expected, actual)
        assert actor.infer_batch([(variant, 0)])[0] == cold[1]
        assert actor.last_cache_hits == 1
    altered_board = bytes([0x80]) + public.sides[0].board[1:]
    changed_inputs = [changed_public(state, board=altered_board),
                      changed_public(state, side=1, board=altered_board),
                      changed_public(state, pill=(1, 1)),
                      changed_public(state, preview=(0, 2)),
                      changed_public(state, side=1, pill=(2, 2)),
                      replace(state, action_costs_by_side=((4, 5, 7), state.action_costs_by_side[1]))]
    for variant in changed_inputs:
        assert actor._policy_key(state, 0) != actor._policy_key(variant, 0)
    # A batch larger than capacity still returns every result in input order.
    result = actor.infer_batch([(s, 0) for s in changed_inputs] + [(changed_inputs[0], 0)])
    assert len(result) == 7 and result[0] == result[-1]
    assert actor.last_inference_batch_rows == (6,)
    assert len(actor._batch_cache) == 2
    actor.infer_batch([(state, 0)])
    assert actor.last_inference_batch_rows == (1,)
    # A cached public input cannot bypass causal/acting-boundary validation.
    with pytest.raises(ValueError, match="causal"):
        actor.infer_batch([(replace(state, public_observation_schema=LEGACY_PUBLIC_SCHEMA), 0)])
    with pytest.raises(ValueError, match="acting"):
        actor.infer_batch([(replace(state, privileged=replace(state.privileged, need_action=(False, True))), 0)])


def test_public_history_keys_keep_age_and_clock_context():
    actor = PublicPolicyContinuation.__new__(PublicPolicyContinuation)
    actor.policy = SimpleNamespace(aux_spec=PUBLIC_CONTEXT_SCHEMA)
    state = state_from_payload(data()[0])
    key = actor._policy_key(state, 0)
    public = state.privileged.public
    assert actor._policy_key(changed_public(state, state_age_frames=7), 0) != key
    assert actor._policy_key(changed_public(state, side=1, preview=(0, 0)), 0) != key
    clock = replace(state, privileged=replace(state.privileged,
                                              public=replace(public, frame_id=public.frame_id + 7)))
    assert actor._policy_key(clock, 0) != key
    hidden = replace(state, privileged=replace(state.privileged, pending_attacks=(7, 8),
                                                engine_checkpoint=b"private"))
    assert actor._policy_key(hidden, 0) == key


@pytest.mark.skipif(not is_library_present(), reason="native pool library missing")
def test_cached_public_panel_keeps_native_outcomes_and_counts_actual_neural_work():
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        seed = (3, 7)
        runner.reset(None, [build_vs_reset_spec(level=(0, 0), speed_setting=(2, 2),
                                               rng_override=True, rng_state=seed, frame_counter_base=21)])
        state = capture_native_state(runner, level=0, causal_public=True)
        task = RolloutTask(0, state, 0, max(state.legal_actions_by_side[0]),
                           reserve_for_seed(*seed).tobytes(), 1.)
        tasks = [replace(task, id=i) for i in range(4)]
        uncached = rollout_tasks(tasks, FirstLegal(), batch_size=4, native_workers=2)
        actor = PublicPolicyContinuation.__new__(PublicPolicyContinuation)
        actor.policy = SimpleNamespace(aux_spec="zero_v1_vs")
        actor.cache_size = 32
        actor._batch_cache = OrderedDict()
        actor._infer_uncached = FirstLegal().infer_batch
        measured = {}
        cached = rollout_tasks(tasks, actor, batch_size=4, native_workers=2, metrics=measured)
        assert cached == uncached
        assert measured["neural_policy_rows"] < measured["policy_decisions"]
        assert measured["policy_decisions"] == (measured["neural_policy_rows"]
                                                + measured["policy_cache_hits"]
                                                + measured["within_batch_duplicates"])
        assert measured["neural_policy_rows"] == sum(size * n for size, n in measured["inference_batch_rows"].items())
    finally:
        runner.close()
