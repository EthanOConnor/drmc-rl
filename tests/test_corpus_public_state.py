"""Corpus public-view reconstruction and the human-imitation dataset contract."""

import os

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary, PairEventKind
from drmc_rl.game.public_context import PUBLIC_CONTEXT_DIM, encode_public_context
from drmc_rl.human.corpus_public_state import CorpusGame, place_action, pose_action, raw_pair

EMPTY = b"\xff" * 128


def _board(cells):
    board = bytearray(EMPTY)
    for index, tile in cells.items():
        board[index] = tile
    return bytes(board)


def _row(slot, spawn, lock, field, pill=(0, 1), preview=(2, 2), pose=(3, 15, 0), **extra):
    x, y, rot = pose
    return dict(decision_id=f"{slot}:{spawn}", player_slot=slot, spawn_frame=spawn, lock_frame=lock,
                tau_frames=lock - spawn, field=field, opp_field=EMPTY, pill_left=pill[0], pill_right=pill[1],
                preview_left=preview[0], preview_right=preview[1], lock_x=x, lock_y_top=y, lock_rotation=rot,
                input_frames=0, input_rle_u16_u8=b"", **extra)


def test_raw_colors_map_to_canonical():
    assert raw_pair(0, 1) == (1, 0)  # NES yellow, red -> canonical R=0, Y=1, B=2
    assert raw_pair(2, 2) == (2, 2)


def test_placement_matches_exact_afterstate_geometry():
    from drmc_rl.game.afterstate import resolve_placement

    viruses = _board({15 * 8 + 0: 0xD1, 15 * 8 + 1: 0xD1, 15 * 8 + 2: 0xD1})
    for pose in ((3, 14, 0), (0, 13, 1), (5, 10, 2), (6, 9, 3)):
        action = pose_action(*pose)
        assert action >= 0
        placed = place_action(viruses, (0, 0), action)
        after, _facts = resolve_placement(np.frombuffer(viruses, np.uint8), (0, 0), action)
        from drmc_rl.game import cascade

        assert cascade.resolve_cascade(placed).settled_field == after


def test_public_view_orders_events_and_tracks_the_opponent():
    viruses = _board({15 * 8 + 0: 0xD1, 15 * 8 + 1: 0xD1, 15 * 8 + 2: 0xD1})
    rows = [
        _row(1, 100, 130, viruses, pill=(1, 1), pose=(3, 14, 0)),  # red pill completes a four-line
        _row(2, 100, 140, EMPTY, pose=(4, 15, 0)),
        _row(2, 160, 190, _board({15 * 8 + 4: 0x61, 15 * 8 + 5: 0x70}), pose=(4, 14, 0)),
        _row(1, 150, 180, _board({}), pose=(0, 15, 0)),
    ]
    game = CorpusGame(rows)
    p1_second = game.sides[0][1]
    public = game.public_state(p1_second)
    assert public.viewer_side == 0 and public.decision_boundary == DecisionBoundary.P1
    own, opponent = public.sides
    assert own.active.column == 3 and own.active.age_frames == 0 and own.animation_phase == "falling"
    # Player 2 locked at 140 and spawns next at 160: at 150 it is settling with the locked pill current.
    assert opponent.active is None and opponent.pill == raw_pair(0, 1)
    assert opponent.animation_phase == "settling"
    kinds = [(e.kind, e.side) for e in public.recent_events]
    assert kinds[0] == (PairEventKind.SPAWN, 0) and kinds[-1] == (PairEventKind.SPAWN, 0)
    assert (PairEventKind.CLEAR, 0) in kinds
    assert all(a.frame_id <= b.frame_id for a, b in zip(public.recent_events, public.recent_events[1:]))
    context = encode_public_context(public, 0, None)
    assert context.shape == (PUBLIC_CONTEXT_DIM,) and np.isfinite(context).all()
    # Player 2's spawn at 160 sees player 1 falling with age 10.
    view = game.public_state(game.sides[1][1])
    assert view.sides[0].active is not None and view.sides[0].active.age_frames == 10


def test_same_frame_spawns_are_a_joint_boundary():
    rows = [_row(1, 50, 80, EMPTY), _row(2, 50, 90, EMPTY)]
    game = CorpusGame(rows)
    assert game.public_state(game.sides[0][0]).decision_boundary == DecisionBoundary.BOTH


def test_imitation_targets_share_mass_over_identical_afterstates():
    torch = pytest.importorskip("torch")
    from tools.imitate_afterstate_core import _targets

    after = torch.zeros((1, 4, 128), dtype=torch.uint8)
    after[0, 1, 5] = 1
    after[0, 3, 5] = 1
    valid = torch.tensor([[True, True, True, False]])
    target = _targets(after, valid, torch.tensor([1]))
    assert torch.allclose(target, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    target = _targets(after, valid, torch.tensor([0]))
    assert torch.allclose(target, torch.tensor([[0.5, 0.0, 0.5, 0.0]]))


def test_reconstruction_matches_native_public_context():
    library = os.environ.get("DRMARIO_POOL_LIB")
    if not library or not os.environ.get("DRMARIO_REACH_LIB"):
        pytest.skip("native pool and planner libraries are required")
    from tools.validate_corpus_public_state import compare, play

    frames, decisions = play(20260924, native_library=library)
    report = compare(frames, decisions)
    groups = report["groups"]
    assert report["decisions"] > 20
    assert groups["own"]["exact_rate"] == 1.0
    assert groups["opponent.pill_preview"]["exact_rate"] > 0.95
    assert groups["opponent.phase"]["exact_rate"] > 0.95
    assert groups["event.kind_side_known"]["exact_rate"] > 0.75
