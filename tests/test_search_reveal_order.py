import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import DrMarioPoolError, is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.search.native_pair import capture_native_state
from drmc_rl.search.pill_belief import pill_id_to_raw_pair, reserve_for_seed


@pytest.mark.skipif(not is_library_present(), reason="native library unavailable")
@pytest.mark.parametrize(
    "level,speed,seed,root_side",
    [(0, 2, (3, 7), 0), (14, 0, (11, 193), 1), (14, 2, (19, 22), 0), (20, 2, (43, 177), 1)],
)
def test_chance_reveal_obeys_strict_clock_order_and_cannot_bypass_parked_input(
    level, speed, seed, root_side
):
    runner = DrMarioVsPoolRunner(num_pairs=1)
    reveals, blocked_attempts = 0, 0
    try:
        runner.reset(
            None,
            [
                build_vs_reset_spec(
                    level=(level, level),
                    speed_setting=(speed, speed),
                    rng_override=True,
                    rng_state=seed,
                    frame_counter_base=21,
                )
            ],
        )
        state = capture_native_state(runner, level=level, speed_setting=speed, causal_public=True)
        reserve = reserve_for_seed(*seed)
        forced = False
        for _ in range(1024):
            reveal = runner.search_reveal_info(0)
            clocks = state.privileged.pair_clocks
            if reveal is not None:
                side, index = reveal
                assert (
                    clocks[side] <= clocks[1 - side]
                    if side == 0
                    else clocks[side] < clocks[1 - side]
                )
                runner.search_reveal(0, side, pill_id_to_raw_pair(reserve[index]))
                reveals += 1
            else:
                before = runner.snapshot(0)
                # The mutating API must reject non-runnable chance requests too,
                # not merely rely on cooperative use of reveal_info.
                for side in (0, 1):
                    with pytest.raises(DrMarioPoolError, match="rc=-2"):
                        runner.search_reveal(0, side, (0, 0))
                    assert runner.snapshot(0) == before
                    blocked_attempts += 1
                actions = np.full(2, -2, np.int32)
                for side, needed in enumerate(state.privileged.need_action):
                    if needed:
                        legal = state.legal_actions_by_side[side]
                        actions[side] = (
                            max(legal)
                            if not forced and side == root_side
                            else min(legal)
                            if legal
                            else -1
                        )
                        if side == root_side:
                            forced = True
                runner.step_search(actions)
            state = capture_native_state(runner, level=level, speed_setting=speed, previous=state)
            if any(state.privileged.terminal_outcome):
                break
        assert state.privileged.terminal_outcome[root_side] in (1, 2, 3)
        assert reveals > 0 and blocked_attempts > 0
    finally:
        runner.close()
