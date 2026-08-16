import pytest

from drmc_rl.game.pair_state import (
    DecisionBoundary,
    PairEvent,
    PairEventKind,
    PrivilegedPairState,
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
    assert state.stable_hash() == "e0b7a2f9fd82c92f5003f12ba24c94008ef5a0506d1a9cca424bdcf8fb6fe752"


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


def test_privileged_state_round_trip_preserves_native_checkpoint() -> None:
    public = PublicPairState(
        frame_id=91,
        viewer_side=1,
        sides=(side(0xFF), side(0xD0)),
        decision_boundary=DecisionBoundary.BOTH,
    )
    state = PrivilegedPairState(
        public=public,
        pair_clocks=(91, 89),
        need_action=(True, True),
        pending_attacks=(2, 0),
        native_phases=("pill_falling", "pill_falling"),
        committed_actions=(None, 17),
        engine_checkpoint=b"DRMVSP2\x00checkpoint",
    )
    restored = PrivilegedPairState.from_dict(state.to_dict())
    assert restored == state
    assert restored.stable_hash() == state.stable_hash()
    assert "engine_checkpoint_b64" not in restored.public_view().to_dict()


def test_privileged_state_rejects_empty_or_invalid_checkpoint() -> None:
    public = PublicPairState(
        frame_id=1,
        viewer_side=0,
        sides=(side(0xFF), side(0xFF)),
        decision_boundary=DecisionBoundary.P1,
    )
    with pytest.raises(ValueError, match="restorable engine checkpoint"):
        PrivilegedPairState(
            public=public,
            pair_clocks=(1, 1),
            need_action=(True, False),
            pending_attacks=(0, 0),
            native_phases=("pill_falling", "resolving"),
            committed_actions=(None, None),
            engine_checkpoint=b"",
        )
