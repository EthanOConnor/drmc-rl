"""Opt-in arena timing-contract knobs: compute-input pinning and pre-spawn decisions."""
from collections import Counter
import os

import numpy as np
import pytest

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.game.pair_state import (
    DecisionBoundary, PairEvent, PairEventKind, PublicPairState, VisibleSideState,
)
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.human.anticipation import score_public_inputs
from drmc_rl.human.controller_context import controller_policy_inputs
from drmc_rl.human.early_decision import (
    PREVIEWS, SETTLED_PHASE, early_public_view, early_start_delay, garbage_safe, lead_bucket,
    marginal_action, network_execution_frames, predicted_board, validate_timing_params,
)
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import run_batch

LIBRARY = os.environ.get("DRMC_FRAME_LIBRARY")


class RecordingPublicPolicy:
    """Deterministic public-context actor that records the execution inputs it saw."""
    aux_spec = PUBLIC_CONTEXT_SCHEMA

    def __init__(self):
        self.seen = []

    def score(self, observations, infos):
        masks = np.stack([i["placements/feasible_mask"].reshape(512) for i in infos]).astype(bool)
        costs = np.stack([i["placements/cost_to_lock"].reshape(512) for i in infos])
        self.seen.extend((i["public_execution"], i["public_pair_state"]) for i in infos)
        logits = -costs.astype(np.float32) - np.arange(512, dtype=np.float32)[None] * .0001
        return np.broadcast_to(np.arange(512), masks.shape), masks, logits


def test_default_execution_inputs_are_unchanged_and_pinning_is_explicit():
    fast, perfect = resolve_pace("fast"), resolve_pace("frame_perfect")
    assert network_execution_frames({"delay": 4}, perfect, 4) == (4, 4)
    assert network_execution_frames({"delay": 8}, perfect, 8) == (8, 8)
    assert network_execution_frames({"delay": 8}, fast, 12) == (12, 8)
    pinned = {"delay": 8, "compute_input_frames": 4}
    assert network_execution_frames(pinned, perfect, 8) == (4, 4)
    assert network_execution_frames(pinned, fast, 12) == (12, 4)
    assert network_execution_frames({"delay": 4}, perfect, 1, early=True) == (1, 4)
    contract = {"delay": 4, "early_delay_input": "contract"}
    assert network_execution_frames(contract, perfect, 1, early=True) == (4, 4)
    assert validate_timing_params({"delay": 4}) == "spawn"
    for bad in ({"decision_point": "later"}, {"compute_input_frames": 4.0},
                {"early_delay_input": "guess"}, {"preview_input": "hidden"},
                {"early_preview": "oracle"}):
        with pytest.raises(ValueError):
            validate_timing_params({"delay": 4, **bad})


def test_reaction_floor_is_counted_from_spawn():
    assert early_start_delay(100, 103, 4, 0) == 1
    assert early_start_delay(100, 103, 4, 2) == 2
    assert early_start_delay(100, 115, 4, 0) == 0
    assert early_start_delay(100, 115, 4, 6) == 6
    assert early_start_delay(100, 100, 4, 0) == 4
    with pytest.raises(ValueError):
        early_start_delay(101, 100, 4, 0)


def test_marginal_action_averages_preview_policies():
    scores = np.full((9, 512), -np.inf, np.float32)
    scores[:, [3, 7]] = 0.0
    scores[0, 3] = 10.0          # one confident preview
    scores[1:, 7] = 1.0          # most previews mildly prefer another action
    assert marginal_action(scores) == 7
    scores[1:, 7] = 0.1
    assert marginal_action(scores) == 3
    broken = scores.copy()
    broken[2, 3] = -np.inf
    with pytest.raises(ValueError):
        marginal_action(broken)


def test_native_lock_prediction_and_settled_phase_match_the_spawn():
    """The cascade port predicts the spawn bottle; the settled phase precedes spawn."""
    checked = leads = 0
    with FrameVsPool(1, lib_path=LIBRARY) as pool:
        pool.reset([1234], level=14)
        was, lock, settled = False, None, None
        for _ in range(1500):
            s = pool.states[0]
            if was and not s.falling:
                lock, settled = (s.frame, predicted_board(s.board, "lock"), bytes(s.preview)), None
            if lock and settled is None and not s.falling and s.phase == SETTLED_PHASE:
                settled = (s.frame, predicted_board(s.board, "settled"))
            if s.falling and not was and lock:
                if pool.states[1].garbage_sent_total == 0:
                    assert bytes(s.board) == lock[1] == settled[1]
                    assert bytes(s.pill) == lock[2]
                    assert s.frame - settled[0] == 3 and lock[0] < settled[0]
                    checked += 1
                    leads += s.frame - lock[0]
            was = s.falling
            pool.step([0x04, 0x04])
    assert checked >= 5


def test_early_view_freezes_request_context_and_replaces_only_own_side():
    with FrameVsPool(1, lib_path=LIBRARY) as pool:
        pool.reset([17291], level=14)
        for _ in range(40):
            pool.step([0, 0])
        public = pool.public_state(1)
        falling = pool.states[1].semantic(pool.states[0])["falling"]
        board = bytes(pool.states[1].board)
        view = early_public_view(public, board=board, pill=(2, 1), preview=(0, 0), falling=falling)
    assert view.frame_id == public.frame_id and view.recent_events == public.recent_events
    assert view.sides[0] == public.sides[0]
    own = view.sides[1]
    assert own.pill == (2, 1) and own.preview == (0, 0) and own.active.age_frames == 0
    assert own.animation_phase == "falling" and own.board == board
    assert len(PREVIEWS) == 9 and len(set(PREVIEWS)) == 9


def _run(variants, pace, jobs=((19071, 0, 0), (19071, 1, 1)), frames=4000):
    config = {"native_library": LIBRARY, "variants": variants, "max_game_frames": frames, "replay_games": 0}
    match = {"a": "a", "b": "b", "games": len(jobs), "level": 14, "pace": pace}
    actor, planner = RecordingPublicPolicy(), NativeReachabilityRunner()
    try:
        rows, _ = run_batch(config, match, list(jobs), actor, planner, None)
    finally:
        planner.close()
    return rows, actor


@pytest.mark.parametrize("point,pace,expected", [
    ("settled", "frame_perfect", 1), ("settled", "super_human", 2),
    ("lock", "frame_perfect", 0), ("lock", "top_humans", 6),
])
def test_pre_spawn_decisions_execute_validated_scripts(point, pace, expected):
    rows, actor = _run({"a": {"delay": 4, "decision_point": point}, "b": {"delay": 4}}, pace)
    early = [m for _, moves, _ in rows for m in moves if "timing" in m]
    assert early and all(m["delay"] == expected for m in early)
    assert all(m["timing"]["delay_input"] == expected and m["timing"]["compute_input"] == 4 for m in early)
    spawn = [m for _, moves, _ in rows for m in moves if "timing" not in m]
    assert all(m["delay"] == max(4, resolve_pace(pace).reaction_frames) for m in spawn)
    assert sum(r["a_stats"].get("early_accepted", 0) for r, _, _ in rows) == len(early)
    assert not any(r["b_stats"].get("early_requests", 0) for r, _, _ in rows)
    assert any(e.decision_delay_frames == expected for e, _ in actor.seen)
    # Nine marginal preview rows per accepted early decision, one per spawn decision.
    assert len(actor.seen) == 9 * len(early) + len(spawn)


def test_pinned_compute_input_charges_the_real_delay():
    rows, actor = _run({"a": {"delay": 8, "compute_input_frames": 4}, "b": {"delay": 4}}, "frame_perfect",
                       jobs=((19071, 0, 0),), frames=1500)
    moves = rows[0][1]
    assert {m["delay"] for m in moves} == {4, 8}
    assert {(e.decision_delay_frames, e.compute_frames) for e, _ in actor.seen} == {(4, 4)}


def test_event_runner_matches_frame_runner_with_pinned_inputs_and_rejects_early_points():
    variants = {"a": {"delay": 6, "compute_input_frames": 4}, "b": {"delay": 5}}
    config = {"native_library": LIBRARY, "variants": variants, "max_game_frames": 3000, "replay_games": 0}
    match = {"a": "a", "b": "b", "games": 2, "level": 14, "pace": "super_human"}
    jobs = [(17291, 0, 0), (17291, 1, 1)]
    planner, parallel = NativeReachabilityRunner(), ParallelPlanning(2)
    try:
        reference, _ = run_batch(config, match, jobs, RecordingPublicPolicy(), planner, None)
        batched, _ = run_event_batch(config, match, jobs, RecordingPublicPolicy(), parallel, None)
        for (expected, moves, _), (actual, event_moves, _) in zip(reference, batched):
            assert event_moves == moves and actual["score"] == expected["score"]
        early = {**config, "variants": {"a": {"delay": 4, "decision_point": "settled"}, "b": {"delay": 4}}}
        with pytest.raises(ValueError, match="frame runner"):
            run_event_batch(early, match, jobs, RecordingPublicPolicy(), parallel, None)
    finally:
        parallel.close()
        planner.close()


def test_spawn_preview_marginal_diagnostic_scores_nine_previews_without_moving_the_clock():
    rows, actor = _run({"a": {"delay": 4, "preview_input": "marginal"}, "b": {"delay": 4}}, "frame_perfect",
                       jobs=((19071, 0, 0),), frames=1500)
    moves = rows[0][1]
    ours = [m for m in moves if m["side"] == 0]
    assert ours and all(m["delay"] == 4 and "timing" not in m for m in ours)
    assert len(actor.seen) == 9 * len(ours) + len(moves) - len(ours)
    previews = {p.sides[p.viewer_side].preview for _, p in actor.seen}
    assert set(PREVIEWS) <= previews


def _public(events, *, opponent_phase="falling", opponent_board=bytes([0xFF]) * 128):
    sides = [VisibleSideState(board=bytes([0xFF]) * 128, pill=(0, 0), preview=(0, 0), active=None,
                              animation_phase="resolving"),
             VisibleSideState(board=opponent_board, pill=(0, 0), preview=(0, 0), active=None,
                              animation_phase=opponent_phase)]
    return PublicPairState(frame_id=1000, viewer_side=0, sides=tuple(sides),
                           decision_boundary=DecisionBoundary.ADVANCE, recent_events=tuple(events))


def _event(kind, frame, side, **payload):
    return PairEvent(PairEventKind(kind), frame, side, payload)


def test_public_garbage_rule_uses_combos_volleys_and_visible_cascades():
    own_spawn = _event("spawn", 900, 0, column=3, row_top=0, rotation=0)
    combo = [_event("lock", 880, 1, column=3, row_top=5, rotation=0),
             _event("clear", 890, 1, tiles_cleared=8, viruses_cleared=2, lines_cleared=2)]
    assert garbage_safe(_public([own_spawn]))
    assert not garbage_safe(_public([*combo, own_spawn]))
    stored = _event("spawn", 896, 1, column=3, row_top=0, rotation=0)       # combo stored at 893
    released = _event("volley", 905, 0, garbage_size=2, columns=[0, 4], colors=[0, 1], salt_frames=16, sender=1)
    assert not garbage_safe(_public([*combo, own_spawn, stored]))
    assert garbage_safe(_public([*combo, stored, own_spawn, released]))
    # A volley before the combo was stored released an older attack, not this one.
    early_volley = _event("volley", 892, 0, garbage_size=2, columns=[0, 4], colors=[0, 1], salt_frames=16, sender=1)
    assert not garbage_safe(_public([*combo, early_volley, stored]))
    # A cascade whose start fell out of the retained history is assumed to be a combo.
    assert not garbage_safe(_public([combo[1], own_spawn]))
    single = [_event("lock", 880, 1, column=3, row_top=5, rotation=0),
              _event("clear", 890, 1, tiles_cleared=4, viruses_cleared=1, lines_cleared=1)]
    assert garbage_safe(_public([*single, own_spawn]))
    # One line so far while the opponent is still resolving a visible second match.
    board = bytearray([0xFF]) * 128
    board[15*8:15*8+4] = bytes([0x81] * 4)
    assert not garbage_safe(_public([*single, own_spawn], opponent_phase="settling", opponent_board=bytes(board)))
    assert garbage_safe(_public([*single, own_spawn], opponent_phase="settling"))
    # Our later attack check without a volley proves a truncated cascade was not a combo.
    later_check = _event("spawn", 950, 0, column=3, row_top=0, rotation=0)
    assert garbage_safe(_public([combo[1], stored, later_check]))
    # A lock with own clears lengthens our window; any live opponent is then unsafe.
    assert not garbage_safe(_public([own_spawn]), own_clears=True)
    assert not garbage_safe(_public([own_spawn], opponent_phase="spawn"), own_clears=True)
    assert garbage_safe(_public([own_spawn]), own_clears=True, commit=True)


def test_lead_buckets():
    assert [lead_bucket(f) for f in (0, 3, 4, 40, 127, 128, 999)] == [
        "lt4", "lt4", "lt8", "lt64", "lt128", "ge128", "ge128"]


def _stats(rows):
    total = Counter()
    for row, _, _ in rows:
        total.update(row["a_stats"])
    return total


def test_lock_mismatches_are_classified():
    rows, _ = _run({"a": {"delay": 4, "decision_point": "lock"}, "b": {"delay": 4}}, "frame_perfect",
                   jobs=((19071, 0, 0), (19071, 1, 1), (17291, 0, 2), (17291, 1, 3)), frames=6000)
    stats = _stats(rows)
    reasons = {k: v for k, v in stats.items() if k.startswith("early_mismatch_")}
    assert sum(reasons.values()) == stats["early_mismatch"]
    assert set(reasons) <= {"early_mismatch_lock_garbage", "early_mismatch_lock_cascade", "early_mismatch_lock_pill"}
    assert stats["early_mismatch_lock_cascade"] == 0 and stats["early_mismatch_lock_pill"] == 0


@pytest.mark.parametrize("preview", ["marginal", "repeat", "branches"])
def test_commit_safe_predicts_the_committed_lock_and_falls_back_in_order(preview):
    rows, actor = _run({"a": {"delay": 4, "decision_point": "commit_safe", "early_preview": preview},
                        "b": {"delay": 4}}, "frame_perfect", frames=4000)
    stats = _stats(rows)
    assert stats["early_accepted_commit"] > 0
    assert not any(k.endswith(("_lock_pose", "_cascade", "_pill")) for k in stats if k.startswith("early_mismatch"))
    early = [m for _, moves, _ in rows for m in moves if "timing" in m]
    commits = [m for m in early if m["timing"]["kind"] == "commit"]
    assert all(m["delay"] == 0 and m["timing"]["preview"] == preview for m in commits)
    assert min(m["timing"]["lead_frames"] for m in commits) > 3
    assert {m["timing"]["kind"] for m in early} <= {"commit", "lock", "settled"}
    rows_per = {"marginal": 9, "repeat": 1, "branches": 9}[preview]
    spawn = [m for _, moves, _ in rows for m in moves if "timing" not in m]
    assert len(actor.seen) == rows_per * len(early) + len(spawn)


def test_lock_safe_uses_lock_or_settled_requests_without_mismatches():
    rows, _ = _run({"a": {"delay": 4, "decision_point": "lock_safe"}, "b": {"delay": 4}}, "super_human")
    stats = _stats(rows)
    assert stats["early_accepted_lock"] > 0
    assert stats["early_accepted"] == stats["early_accepted_lock"] + stats["early_accepted_settled"]
    assert stats["early_mismatch"] == 0


class PreviewSensitivePolicy(RecordingPublicPolicy):
    """Scores that depend on the observed preview, so the marginal matters."""

    def score(self, observations, infos):
        actions, masks, logits = super().score(observations, infos)
        previews = np.asarray([3 * i["public_pair_state"].sides[i["public_pair_state"].viewer_side].preview[0]
                               + i["public_pair_state"].sides[i["public_pair_state"].viewer_side].preview[1]
                               for i in infos], dtype=np.float32)
        return actions, masks, logits + np.sin(previews[:, None] * np.arange(512)[None] * .37) * 3


@pytest.mark.parametrize("mode", ["repeat", "marginal"])
def test_backend_early_preview_matches_the_arena_on_the_hosts_wire_view(mode):
    """The browser sends the request-frame view with its own side predicted; the
    backend's early preview mode must choose what the arena's does at spawn."""
    from drmc_rl.human.backend import early_preview_scores

    pace = resolve_pace("frame_perfect")
    with FrameVsPool(1, lib_path=LIBRARY) as pool:
        pool.reset([4321], level=14)
        was, request, checked = False, None, 0
        planner = NativeReachabilityRunner()
        try:
            for _ in range(2500):
                s = pool.states[1]
                if was and not s.falling:
                    request = (s.frame, pool.public_state(1), predicted_board(s.board, "lock"))
                if s.falling and not was and request and pool.states[0].garbage_sent_total == 0:
                    frame, public, board = request
                    state = pool.semantic(1, public_context=True)
                    assert bytes(s.board) == board
                    delay = early_start_delay(frame, s.frame, 4, pace.reaction_frames)
                    from drmc_rl.human.backend import plan_candidates
                    candidate = plan_candidates(planner, state, delay, pace)
                    policy = PreviewSensitivePolicy()
                    rows = []
                    # As tools/trainer_planning_arena.py scores an early request.
                    previews = [tuple(state["pill"])] if mode == "repeat" else PREVIEWS
                    for preview in previews:
                        view = early_public_view(public, board=board, pill=state["pill"], preview=preview,
                                                 falling=state["falling"])
                        rows.append(score_public_inputs(policy, *controller_policy_inputs(
                            policy, candidate, state, pace, delay, 4, public=view, decision_delay_frames=delay)))
                    expected = (int(rows[0][0].argmax()) if mode == "repeat"
                                else marginal_action(np.concatenate(rows)))
                    # The host's wire view: frozen at the request frame, own side predicted.
                    wire = public.to_dict()
                    for side in wire["sides"]:
                        del side["board_b64"]
                    own = wire["sides"][1]
                    own.update(pill=list(state["pill"]), preview=[0, 0], animation_phase="falling",
                               state_age_frames=0, viruses_remaining=sum((t & 0xF0) == 0xD0 for t in board),
                               active=dict(column=state["falling"]["x"], row_top=state["falling"]["y"],
                                           rotation=state["falling"]["rotation"], colors=list(state["pill"]),
                                           controllable=True, age_frames=0))
                    wire.update(schema="public-controller-history-v1", compute_frames=4)
                    from drmc_rl.game.observation import board_bytes_to_semantic_planes
                    host = {**pool.semantic(1), "preview": [0, 0], "public_live_context": wire,
                            "opponent_pill": list(public.sides[0].pill),
                            "opponent_board_planes": board_bytes_to_semantic_planes(public.sides[0].board)}
                    action, _ = early_preview_scores(policy, candidate, host, pace, delay, 4, mode)
                    assert action == expected
                    checked += 1
                was = s.falling
                pool.step([0x04, 0x04])
        finally:
            planner.close()
    assert checked >= 5
