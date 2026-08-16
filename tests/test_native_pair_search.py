from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present

pytestmark = pytest.mark.skipif(not is_library_present(), reason="native pool library missing")


def test_native_pair_state_round_trip_and_branch_isolation() -> None:
    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
    from drmc_rl.search.native_pair import (
        NativePairSearchModel,
        capture_native_state,
        state_from_payload,
        state_to_payload,
    )

    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        if runner._snapshot_fn is None:  # pragma: no cover - stale local build
            pytest.skip("native VS pool predates snapshot/restore ABI")
        runner.reset(
            None,
            [
                build_vs_reset_spec(
                    level=(5, 5),
                    speed_setting=(2, 2),
                    rng_state=(0x37, 0x91),
                    rng_override=True,
                )
            ],
        )
        root = capture_native_state(runner)
        decoded = state_from_payload(state_to_payload(root))
        assert decoded.privileged.stable_hash() == root.privileged.stable_hash()
        model = NativePairSearchModel(runner)
        actions0 = root.legal_actions_by_side[0]
        actions1 = root.legal_actions_by_side[1]
        child_a = model.apply_actions(root, actions0[0], actions1[0])
        child_b = model.apply_actions(root, actions0[-1], actions1[-1])
        assert model.key(root) != model.key(child_a)
        assert model.key(child_a) != model.key(child_b)

        # Repeating a branch from the same root is byte-identical even after a
        # different branch mutated the runner in between.
        repeated = model.apply_actions(root, actions0[0], actions1[0])
        assert repeated.privileged.engine_checkpoint == child_a.privileged.engine_checkpoint
        assert np.isfinite(model.evaluate(child_a, 0).utility)
    finally:
        runner.close()
