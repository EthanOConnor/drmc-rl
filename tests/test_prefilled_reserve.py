"""Complete-reserve execution must retain every public decision and outcome."""
from dataclasses import replace

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.search.native_pair import capture_native_state
from drmc_rl.search.pill_belief import pill_id_to_raw_pair, reserve_for_seed
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.teachers.terminal_rollout import RolloutTask, rollout_tasks

pytestmark = pytest.mark.skipif(not is_library_present(), reason="native library missing")


class TracedActor:
    def __init__(self):
        self.trace = []

    def infer_batch(self, requests):
        answers = []
        for state, side in requests:
            self.trace.append((PublicPolicyContinuation._request_key(state, side),
                               state.privileged.pair_clocks, state.privileged.terminal_outcome))
            answers.append(({min(state.legal_actions_by_side[side]): 1.}, 0.))
        return answers


def compare(task):
    boundary, bulk = TracedActor(), TracedActor()
    reference_metrics, bulk_metrics = {}, {}
    reference = rollout_tasks([task], boundary, batch_size=1, metrics=reference_metrics)
    result = rollout_tasks([task], bulk, batch_size=1, reserve_execution="prefilled", metrics=bulk_metrics)
    assert boundary.trace == bulk.trace
    assert result[0]["outcome"] == reference[0]["outcome"] in (1, 2, 3)
    assert result[0]["events"] <= reference[0]["events"]
    assert reference[0]["reveals"] == reference[0]["boundary_reveal_calls"]
    assert result[0]["reveals"] is None and result[0]["boundary_reveal_calls"] == 0
    assert bulk_metrics["reserve_install_calls"] == 1
    assert bulk_metrics["policy_decisions"] == reference_metrics["policy_decisions"]
    # Distinct reused slots must not read stale output buffers after restore.
    many = rollout_tasks([replace(task, id=i) for i in range(3)], TracedActor(),
                         batch_size=2, native_workers=2, reserve_execution="prefilled")
    assert [r["outcome"] for r in many] == [reference[0]["outcome"]] * 3


@pytest.mark.parametrize("level,speed,seed", [(0, 2, (3, 7)), (14, 0, (11, 193)),
                                             (14, 2, (19, 22)), (20, 2, (43, 177))])
@pytest.mark.parametrize("root_side", [0, 1])
def test_selected_reserves_preserve_every_public_input_and_natural_outcome(level, speed, seed, root_side):
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        runner.reset(None, [build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
                                               rng_override=True, rng_state=seed, frame_counter_base=21)])
        initial = capture_native_state(runner, level=level, speed_setting=speed, event_public=True)
        actual = reserve_for_seed(*seed)
        altered = actual.copy()
        altered[4:] = (altered[4:] + 1) % 9
        for reserve in (actual, altered):
            compare(RolloutTask(0, initial, root_side, max(initial.legal_actions_by_side[root_side]),
                                reserve.tobytes(), 1.))
    finally:
        runner.close()


def test_bulk_resolves_an_earlier_reveal_before_asking_for_actions():
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        seed = (19, 22)
        runner.reset(None, [build_vs_reset_spec(level=(14, 14), rng_override=True, rng_state=seed)])
        state = capture_native_state(runner, level=14, event_public=True)
        reserve = reserve_for_seed(*seed)
        for _ in range(32):
            reveal = runner.search_reveal_info(0)
            if reveal is not None and any(state.privileged.need_action):
                side = next(i for i in (0, 1) if state.privileged.need_action[i])
                compare(RolloutTask(0, state, side, min(state.legal_actions_by_side[side]), reserve.tobytes(), 1.))
                break
            if reveal is not None:
                side, index = reveal
                runner.search_reveal(0, side, pill_id_to_raw_pair(reserve[index]))
            else:
                runner.step_search(np.array([min(state.legal_actions_by_side[i], default=-1)
                    if state.privileged.need_action[i] else -2 for i in (0, 1)], np.int32))
            state = capture_native_state(runner, level=14, previous=state)
        else:
            pytest.fail("no parked input with an earlier reveal")
    finally:
        runner.close()


def test_v1_is_rejected_and_private_install_does_not_change_other_pair_or_outputs():
    runner = DrMarioVsPoolRunner(num_pairs=2)
    try:
        runner.reset(None, [build_vs_reset_spec(level=(14, 14), rng_override=True, rng_state=(3, 7))] * 2)
        initial = capture_native_state(runner, level=14, causal_public=True)
        task = RolloutTask(0, initial, 0, min(initial.legal_actions_by_side[0]), reserve_for_seed(3, 7).tobytes(), 1.)
        # Use a one-pair root for the high-level contract check below.
        before = runner.snapshot(1)
        outputs = {k: getattr(runner.buffers, k).copy() for k in
                   ("board_bytes", "pill_colors", "preview_colors", "spawn_id", "side_frames")}
        colors = np.asarray([pill_id_to_raw_pair(x) for x in task.reserve], np.uint8)
        original = runner.snapshot(0)
        runner.search_set_reserve(0, colors)
        assert runner.snapshot(0) == original and runner.snapshot(1) == before
        for key, value in outputs.items():
            np.testing.assert_array_equal(value, getattr(runner.buffers, key))
        for invalid in (colors[:127], colors.astype(float), np.full((128, 2), -1), np.full((128, 2), 256)):
            with pytest.raises(ValueError, match="reserve"):
                runner.search_set_reserve(0, invalid)
            assert runner.snapshot(0) == original
    finally:
        runner.close()
    one = DrMarioVsPoolRunner(num_pairs=1)
    try:
        one.reset(None, [build_vs_reset_spec(level=(14, 14))])
        root = capture_native_state(one, level=14, causal_public=True)
        task = replace(task, state=root, action=min(root.legal_actions_by_side[0]))
        with pytest.raises(ValueError, match="V2 event timeline"):
            rollout_tasks([task], TracedActor(), reserve_execution="prefilled")
    finally:
        one.close()
