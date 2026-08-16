import pytest

from drmc_rl.game.pair_state import (
    DecisionBoundary,
    PairEvent,
    PairEventKind,
    PublicPairState,
    VisibleSideState,
)


def side(fill: int) -> VisibleSideState:
    return VisibleSideState(
        board=bytes([fill] * 128),
        pill=(0, 1),
        preview=(2, 0),
        active=None,
        viruses_remaining=20,
    )


def test_public_state_round_trip_and_hash() -> None:
    state = PublicPairState(
        frame_id=42,
        viewer_side=0,
        sides=(side(0xFF), side(0xD0)),
        decision_boundary=DecisionBoundary.P1,
        recent_events=(PairEvent(PairEventKind.SPAWN, 42, 0, {"visible": True}),),
    )
    restored = PublicPairState.from_dict(state.to_dict())
    assert restored == state
    assert restored.stable_hash() == state.stable_hash()


def test_hidden_state_is_rejected() -> None:
    value = PublicPairState(
        frame_id=1,
        viewer_side=0,
        sides=(side(0xFF), side(0xFF)),
        decision_boundary=DecisionBoundary.P1,
    ).to_dict()
    value["rng_state"] = [1, 2]
    with pytest.raises(ValueError, match="forbidden"):
        PublicPairState.from_dict(value)


def test_internal_pending_attack_is_rejected() -> None:
    value = PublicPairState(
        frame_id=1,
        viewer_side=0,
        sides=(side(0xFF), side(0xFF)),
        decision_boundary=DecisionBoundary.P1,
    ).to_dict()
    value["garbage_pending"] = 3
    with pytest.raises(ValueError, match="forbidden"):
        PublicPairState.from_dict(value)
