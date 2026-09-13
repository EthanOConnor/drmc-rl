from dataclasses import replace

import numpy as np
import pytest

from drmc_rl.game.pair_state import (
    DecisionBoundary,
    FallingPillView,
    PairEvent,
    PairEventKind,
    PrivilegedPairState,
    PublicPairState,
    VisibleSideState,
)
from drmc_rl.game.public_context import (
    PUBLIC_CONTEXT_DIM,
    PUBLIC_CONTEXT_SCHEMA,
    PublicExecutionContext,
    context_from_info,
    encode_public_context,
)
from drmc_rl.search.public_policy import policy_request


def test_progress_decodes_bcd_and_counts_actual_gravity_changes():
    from drmc_rl.game.public_context import progress_features
    from drmc_rl.planning.fast_reach import compute_speed_threshold

    for count in (1, 9, 10, 99, 100, 999, 1000, 9999):
        bcd = int(str(count), 16)
        feature = progress_features(14, bcd, 2, 0)
        assert feature[0] == pytest.approx(.7)
        assert feature[1] == pytest.approx(np.log1p(count) / np.log1p(1000))
    for speed in range(3):
        for ups in range(50):
            for count in (9, 10, 19):
                features = progress_features(20, int(str(count), 16), speed, ups, countdown=True)
                # Independently advance future spawns, following the ROM's
                # every-tenth-spawn increment and saturated speed-up counter.
                future_ups, expected = ups, 0
                for spawns in range(1, 501):
                    if (count + spawns) % 10 == 0:
                        future_ups = min(49, future_ups + 1)
                    if compute_speed_threshold(speed, future_ups) < compute_speed_threshold(speed, ups):
                        expected = spawns
                        break
                assert features[2] == bool(expected)
                assert features[3] == pytest.approx(expected / 100)
    with pytest.raises(ValueError, match="BCD"):
        progress_features(14, 0x1A, 2, 0)


def test_progress_preserves_existing_timer_and_requires_observed_counters():
    from drmc_rl.game.public_context import PROGRESS_CONTEXT_SCHEMA, CONTEXT_FEATURE_NAMES
    public = public_state()
    _, info = policy_request(public, 0, [0], [1], context_schema=PUBLIC_CONTEXT_SCHEMA)
    original = context_from_info(info)
    info["public_context_schema"] = PROGRESS_CONTEXT_SCHEMA
    with pytest.raises(KeyError):
        context_from_info(info)
    info["public_progress"] = dict(level=14, pill_counter_bcd=0x19, speed=2, speed_ups=1)
    extended = context_from_info(info)
    np.testing.assert_array_equal(extended[:PUBLIC_CONTEXT_DIM], original)
    info["public_pair_state"] = replace(public, frame_id=180)
    assert context_from_info(info)[CONTEXT_FEATURE_NAMES.index("game_age")] > extended[CONTEXT_FEATURE_NAMES.index("game_age")]


def public_state():
    board = bytearray([255] * 128)
    board[120], board[121] = 0x60, 0x70
    sides = tuple(
        VisibleSideState(
            board=bytes(board),
            pill=(0, 0),
            preview=(1, 2),
            active=FallingPillView(3, 2, 0, (0, 0), True, 7),
            viruses_remaining=20,
            animation_phase="falling",
            state_age_frames=9,
        )
        for _ in range(2)
    )
    return PublicPairState(
        frame_id=90,
        viewer_side=0,
        sides=sides,
        decision_boundary=DecisionBoundary.P1,
        observable_clock_delta_frames=12,
        recent_events=(PairEvent(PairEventKind.CLEAR, 87, 1, {"tiles_cleared": 8}),),
    )


def test_public_context_is_deterministic_and_represents_visible_changes():
    state = public_state()
    motor = PublicExecutionContext(22, 4, 8, 1, 20, 3, 22, 4)
    vector = encode_public_context(state, 0, motor)
    assert vector.shape == (PUBLIC_CONTEXT_DIM,) and np.isfinite(vector).all()
    np.testing.assert_array_equal(vector, encode_public_context(state, 0, motor))
    variants = [
        replace(state, sides=(state.sides[0], replace(state.sides[1], preview=(2, 2)))),
        replace(state, sides=(state.sides[0], replace(state.sides[1], animation_phase="clearing"))),
        replace(state, observable_clock_delta_frames=None),
        replace(state, recent_events=()),
    ]
    for variant in variants:
        assert not np.array_equal(vector, encode_public_context(variant, 0, motor))
    assert not np.array_equal(
        vector, encode_public_context(state, 0, replace(motor, motion_interval=24))
    )
    with pytest.raises(ValueError, match="viewer"):
        encode_public_context(state, 1, motor)


def test_new_policy_boundary_keeps_all_legal_actions_and_horizontal_bonds():
    state = public_state()
    # Deliberately include all rotations to catch historical deduplication.
    legal, cost = tuple(range(512)), tuple(range(512))
    obs, info = policy_request(state, 0, legal, cost, context_schema=PUBLIC_CONTEXT_SCHEMA)
    legacy, _ = policy_request(state, 0, legal, cost)
    assert info["placements/feasible_mask"].sum() == 512
    assert obs[6:8].sum() > 0 and obs[14:16].sum() > 0
    assert legacy[6:8].sum() == 0 and legacy[14:16].sum() == 0
    np.testing.assert_array_equal(context_from_info(info), encode_public_context(state, 0))
    with pytest.raises(ValueError, match="versioned"):
        context_from_info({})


def test_hidden_teacher_changes_never_enter_context_even_as_unknown_info_fields():
    state = public_state()
    private = PrivilegedPairState(
        state, (90, 90), (True, False), (0, 4), ("decision", "resolving"), (None, 42), b"opaque"
    )
    changed = replace(
        private,
        pending_attacks=(99, 77),
        engine_checkpoint=b"different seed and future reserve",
        committed_actions=(None, 43),
    )
    obs, info = policy_request(
        private.public_view(), 0, (0, 1), (2, 3), context_schema=PUBLIC_CONTEXT_SCHEMA
    )
    obs2, info2 = policy_request(
        changed.public_view(), 0, (0, 1), (2, 3), context_schema=PUBLIC_CONTEXT_SCHEMA
    )
    info2.update(raw_ram=b"hidden bytes", garbage_pending=99, seed=771)
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(context_from_info(info), context_from_info(info2))
    with pytest.raises(TypeError, match="PublicPairState"):
        encode_public_context(private, 0)


def test_port_swap_preserves_relative_input_meaning():
    p = public_state()
    swapped = replace(
        p,
        viewer_side=1,
        sides=tuple(reversed(p.sides)),
        decision_boundary=DecisionBoundary.P2,
        observable_clock_delta_frames=-p.observable_clock_delta_frames,
        recent_events=tuple(replace(e, side=1 - e.side) for e in p.recent_events),
    )
    np.testing.assert_array_equal(encode_public_context(p, 0), encode_public_context(swapped, 1))


def test_new_checkpoint_cannot_be_mistaken_for_the_legacy_contract():
    import torch
    from tools.eval_policy import _build_net_from_cfg, _make_aux_builder

    cfg = dict(
        candidate_architecture="g5",
        candidate_board_channels=16,
        candidate_d_model=16,
        encoder_blocks=1,
        pill_embed_dim=8,
        candidate_hidden_dim=24,
        candidate_cross_layers=1,
        candidate_interaction_layers=1,
        candidate_transformer_heads=2,
        candidate_patch_kernel=3,
        aux_spec=PUBLIC_CONTEXT_SCHEMA,
    )
    net, dim, _ = _build_net_from_cfg(cfg, 20, "cpu")
    assert dim == PUBLIC_CONTEXT_DIM and net.public_context_schema == PUBLIC_CONTEXT_SCHEMA
    shim = _make_aux_builder(dim, aux_spec=PUBLIC_CONTEXT_SCHEMA)
    obs, info = policy_request(
        public_state(), 0, (0, 1), (2, 3), context_schema=PUBLIC_CONTEXT_SCHEMA
    )
    aux = shim._build_aux_batch(obs[None], [info])
    with torch.no_grad():
        logits, value = net(
            torch.tensor(obs[None]),
            torch.tensor([[0, 0]]),
            torch.tensor([[1, 2]]),
            torch.tensor([[0, 1]]),
            torch.tensor([[2.0, 3.0]]),
            torch.tensor([[True, True]]),
            aux=torch.tensor(aux),
        )
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    old, _, _ = _build_net_from_cfg(cfg | {"aux_spec": "zero_v1_vs"}, 20, "cpu")
    with pytest.raises(RuntimeError):
        old.load_state_dict(net.state_dict())
