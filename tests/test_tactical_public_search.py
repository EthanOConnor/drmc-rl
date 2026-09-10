from dataclasses import replace
from types import SimpleNamespace

import pytest

from drmc_rl.game.pair_state import DecisionBoundary, PublicPairState, VisibleSideState
from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.search.native_pair import NativePairSearchModel, public_tactical_reasons


def test_tactical_predicate_reads_only_visible_bottle_and_remaining_viruses():
    quiet = VisibleSideState(bytes([255] * 128), (0, 1), (2, 0), None, 20)
    visible = PublicPairState(100, 0, (quiet, quiet), DecisionBoundary.BOTH)
    assert public_tactical_reasons(visible) == ()
    near_top = replace(quiet, board=bytes([255] * 31 + [0xD0] + [255] * 96))
    endgame = replace(quiet, viruses_remaining=4, state_age_frames=40)
    view = replace(visible, sides=(near_top, endgame))
    expected = ("p1_top_four_rows", "p2_last_four_viruses")
    assert public_tactical_reasons(view) == expected
    assert public_tactical_reasons(replace(view, viewer_side=1)) == expected
    assert public_tactical_reasons(replace(visible, sides=(replace(quiet, viruses_remaining=None), quiet))) == ()

    # The native adapter may carry hidden engine bytes, but the predicate has
    # access only to this public object; attempts to read anything else fail.
    class Hidden:
        public = view

        def __getattr__(self, name):
            raise AssertionError("read hidden field " + name)

    state = SimpleNamespace(privileged=Hidden(), public_observation_schema="causal-settled-pair-v2")
    model = object.__new__(NativePairSearchModel)
    assert model.tactical_reasons(state) == expected
    state.public_observation_schema = "legacy-warp-buffer-v1"
    with pytest.raises(ValueError, match="causal"):
        model.tactical_reasons(state)


@pytest.mark.parametrize("value", [-1, 9, 1.5, True])
def test_tactical_extension_budget_requires_a_bounded_integer(value):
    with pytest.raises(ValueError, match="tactical"):
        SearchConfig(tactical_extension_events=value)
