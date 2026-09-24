"""Opt-in timing-contract experiments for the trainer planning arena.

The default contract requests a decision on the first falling frame of a new
pill and starts execution ``max(compute, reaction)`` frames later. These helpers
support two measured alternatives without changing that default:

* ``compute_input_frames`` pins the network's execution-context inputs to a
  compute tier other than the one actually charged, separating "waiting
  longer" from "input off-distribution".
* ``decision_point`` = ``settled`` or ``lock`` requests the decision before the
  pill spawns. The public view is frozen at the request frame; the own side is
  replaced by the deterministic spawn view (settled board, next pill at the
  spawn pose). The after-next preview is not yet public, so the policy is
  marginalized over all nine color pairs. The prediction is validated at spawn;
  a mismatch (for example incoming garbage) falls back to a fresh spawn request.
* ``lock_safe`` requests at the own lock only when public information rules out
  incoming garbage before the next spawn (``garbage_safe``), otherwise at the
  settled phase. ``commit_safe`` requests the next pill when the current pill's
  script starts executing, predicting lock and cascade from the committed
  placement under the same public garbage rule, and otherwise falls back to
  ``lock_safe``. Every mismatch at spawn is classified.
* ``early_preview`` chooses how an early decision handles the unknown
  after-next preview: ``marginal`` (average the nine policies), ``repeat`` (one
  pass with the next pill's colors as the preview) or ``branches`` (score all
  nine, execute the branch matching the preview revealed at spawn).
* ``preview_input`` = ``marginal`` is a spawn-time diagnostic that applies the
  same preview marginalization while the preview is visible (frame runner only).

Reaction floors stay counted from spawn: execution starts at
``spawn + max(reaction, request + compute - spawn, 0)``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.game.pair_state import (
    DecisionBoundary, FallingPillView, PairEventKind, PublicPairState, VisibleSideState,
)

DECISION_POINTS = ("spawn", "settled", "lock", "lock_safe", "commit_safe")
EARLY_PREVIEWS = ("marginal", "repeat", "branches")
# The attack check (combo storage and garbage release) runs this many frames
# before the side's next first-falling (SPAWN event) frame.
ATTACK_CHECK_LEAD = 3
LEAD_BUCKETS = (4, 8, 16, 32, 64, 128)
# Native VS phase entered after the attack check: the bottle, including any
# garbage that has landed, is final and the next pill has not spawned yet.
SETTLED_PHASE = 5
PREVIEWS = tuple((a, b) for a in range(3) for b in range(3))
VIRUS = 0xD0


def validate_timing_params(params):
    """Reject malformed opt-in knobs; absent knobs keep the default contract."""
    point = params.get("decision_point", "spawn")
    if point not in DECISION_POINTS:
        raise ValueError(f"decision_point must be one of {DECISION_POINTS}")
    pinned = params.get("compute_input_frames")
    if pinned is not None and (type(pinned) is not int or not 0 <= pinned <= 60):
        raise ValueError("compute_input_frames must be an integer frame count")
    if params.get("early_delay_input", "actual") not in ("actual", "contract"):
        raise ValueError("early_delay_input must be actual or contract")
    if params.get("preview_input", "visible") not in ("visible", "marginal"):
        raise ValueError("preview_input must be visible or marginal")
    if params.get("early_preview", "marginal") not in EARLY_PREVIEWS:
        raise ValueError(f"early_preview must be one of {EARLY_PREVIEWS}")
    return point


def network_execution_frames(params, pace, delay, *, early=False):
    """(decision_delay_frames, compute_frames) presented to the network.

    Default: the charged delay and the variant's compute tier. With
    ``compute_input_frames`` the network sees that tier and its spawn-contract
    delay ``max(tier, reaction)`` while candidates use the charged delay. An
    early decision reports its actual frames from spawn to first input unless
    ``early_delay_input`` is ``contract``.
    """
    compute = int(params["delay"])
    pinned = params.get("compute_input_frames")
    compute_input = compute if pinned is None else int(pinned)
    if early:
        if params.get("early_delay_input", "actual") == "actual":
            return int(delay), compute_input
    elif pinned is None:
        return int(delay), compute_input
    return max(compute_input, pace.reaction_frames), compute_input


def early_start_delay(request_frame, spawn_frame, compute_frames, reaction_frames):
    """Frames from spawn to the first executed input for a pre-spawn request."""
    if spawn_frame < request_frame:
        raise ValueError("an early request cannot follow its spawn")
    return max(int(reaction_frames), int(request_frame) + int(compute_frames) - int(spawn_frame), 0)


def predicted_board(board, point):
    """The own bottle the pill will spawn into, as known at the request frame."""
    if point == "lock":
        return resolve_cascade(bytes(board)).settled_field
    if point == "settled":
        return bytes(board)
    raise ValueError("only pre-spawn decision points predict a board")


def early_public_view(public, *, board, pill, preview, falling):
    """The request-time public pair with the own side at its predicted spawn."""
    if type(public) is not PublicPairState:
        raise TypeError("early views require the request-time PublicPairState")
    viewer = public.viewer_side
    board = bytes(board)
    own = VisibleSideState(
        board=board, pill=tuple(pill), preview=tuple(preview),
        active=FallingPillView(int(falling["x"]), int(falling["y"]), int(falling["rotation"]),
                               tuple(pill), True, 0),
        viruses_remaining=sum((tile & 0xF0) == VIRUS for tile in board),
        animation_phase="falling", state_age_frames=0,
    )
    opponent = public.sides[1 - viewer]
    sides = (own, opponent) if viewer == 0 else (opponent, own)
    own_boundary = DecisionBoundary.P1 if viewer == 0 else DecisionBoundary.P2
    both = opponent.active is not None and opponent.active.age_frames == 0
    return PublicPairState(
        frame_id=public.frame_id, viewer_side=viewer, sides=sides,
        decision_boundary=DecisionBoundary.BOTH if both else own_boundary,
        recent_events=public.recent_events, observable_clock_delta_frames=0,
        own_controller_state=dict(falling),
    )


def with_own_preview(public, preview):
    """Replace only the viewer's preview (spawn-time preview-marginal diagnostic)."""
    sides = list(public.sides)
    sides[public.viewer_side] = replace(sides[public.viewer_side], preview=tuple(preview))
    return replace(public, sides=tuple(sides))


def marginal_action(scores):
    """Argmax of the preview-marginal policy (uniform over the nine previews)."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if scores.ndim != 2 or not finite.any(axis=1).all() or (finite != finite[:1]).any():
        raise ValueError("marginal scoring requires one identical nonempty frontier per preview")
    shifted = np.where(finite, scores - np.where(finite, scores, -np.inf).max(axis=1, keepdims=True), -np.inf)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return int(probabilities.mean(axis=0).argmax())


def committed_lock_board(board, raw_pill, action):
    """The own bottle right after the committed placement locks (pill baked in)."""
    from drmc_rl.human.expressive_sequences import locked_field
    return bytes(locked_field(np.frombuffer(bytes(board), np.uint8), [int(c) & 3 for c in raw_pill],
                              int(action)).reshape(-1))


def _cascade_lines_upper(board):
    return resolve_cascade(bytes(board)).cells_cleared // 4


def garbage_safe(public, *, own_clears=False, commit=False):
    """True when public information rules out garbage before the viewer's next attack check."""
    return garbage_risk(public, own_clears=own_clears, commit=commit) is None


def garbage_risk(public, *, own_clears=False, commit=False):
    """Why garbage may land before the viewer's next attack check, or None when it cannot.

    Uses public events and boards only (no frame-counter columns, no hidden
    attack state). The ROM stores a combo of two or more lines at the sender's
    attack check, which precedes its next spawn by ``ATTACK_CHECK_LEAD`` frames,
    and releases it at the receiver's next attack check. Unsafe when (i) an
    opponent cascade of two or more lines is not stored yet, or was stored after
    our last attack check; (ii) the opponent is resolving a cascade that can
    still reach two lines; or (iii) at a lock with own clears, whose long
    animation leaves the opponent time to store a new combo first. A commit
    request skips (iii); spawn validation catches what it misses. A cascade whose
    start is missing from the retained history is assumed to be a combo.
    """
    viewer, opponent = public.viewer_side, 1 - public.viewer_side
    cascades, current = [], None
    for event in public.recent_events:
        if event.side != opponent and not (
                event.side == viewer and event.kind in (PairEventKind.VOLLEY, PairEventKind.SPAWN)):
            continue
        if event.kind == PairEventKind.LOCK:
            current = dict(lines=0, stored=None)
            cascades.append(current)
        elif event.kind == PairEventKind.CLEAR:
            if current is None:   # the cascade began before the retained history: assume a combo
                current = dict(lines=2, stored=None)
                cascades.append(current)
            current["lines"] += int(event.public_payload.get("lines_cleared", 0))
        elif event.kind == PairEventKind.SPAWN and event.side == opponent:
            if current is not None and current["stored"] is None:
                current["stored"] = event.frame_id - ATTACK_CHECK_LEAD
        elif event.kind == PairEventKind.SPAWN or event.kind == PairEventKind.VOLLEY:
            # Our attack check released (volley) or proved empty every combo stored before it.
            check = event.frame_id - (ATTACK_CHECK_LEAD if event.kind == PairEventKind.SPAWN else 0)
            for cascade in cascades:
                if cascade["stored"] is not None and cascade["stored"] < check:
                    cascade["lines"] = 0
    if any(c["lines"] >= 2 for c in cascades):
        return "stored_combo"
    side = public.sides[opponent]
    if side.animation_phase not in ("falling", "spawn", "terminal"):
        so_far = cascades[-1]["lines"] if cascades and cascades[-1]["stored"] is None else 0
        if so_far + _cascade_lines_upper(side.board) >= 2:
            return "opponent_cascade"
    if own_clears and not commit and side.animation_phase != "terminal":
        return "own_clears"
    return None


def lead_bucket(frames):
    return next((f"lt{b}" for b in LEAD_BUCKETS if frames < b), f"ge{LEAD_BUCKETS[-1]}")


class EarlyRequests:
    """Per-side pre-spawn request state for the frame runner.

    A request freezes the public view at its frame together with the predicted
    spawn bottle, the next pill and the incoming-garbage count. ``resolve`` is
    called on the first falling frame and returns the request only if every
    prediction holds; otherwise the caller makes a fresh spawn decision.
    """

    def __init__(self, sides):
        self.request = [None] * sides     # dict, or "await" for the settled phase
        self.commit = [None] * sides      # scheduled commit capture

    def on_lock(self, side, point, frame, current, opponent, pool, stats):
        if point == "spawn":
            return
        request = self.request[side]
        self.commit[side] = None
        if isinstance(request, dict) and request["kind"] == "commit":
            if bytes(current.board) != request["lock_board"]:
                request["lock_miss"] = True
            return
        self.request[side] = "await"
        if point == "lock":
            self._capture(side, "lock", frame, current, opponent, pool, stats)
        elif point in ("lock_safe", "commit_safe"):
            public = pool.public_state(side)
            own_clears = bool(resolve_cascade(bytes(current.board)).steps)
            risk = garbage_risk(public, own_clears=own_clears)
            if risk is None:
                self._capture(side, "lock", frame, current, opponent, pool, stats, public=public)
            else:
                stats["early_lock_unsafe"] += 1
                stats[f"early_lock_unsafe_{risk}"] += 1

    def on_frame(self, side, point, frame, current, opponent, pool, stats):
        if point == "spawn" or current.terminal:
            return
        plan = self.commit[side]
        if plan is not None and frame >= plan["frame"] and current.falling:
            self.commit[side] = None
            public = pool.public_state(side)
            risk = garbage_risk(public, commit=True)
            if risk is None:
                self.request[side] = dict(kind="commit", frame=frame, public=public,
                    board=resolve_cascade(plan["lock_board"]).settled_field, lock_board=plan["lock_board"],
                    pill=plan["pill"], incoming=int(opponent.garbage_sent_total))
                stats["early_requests"] += 1
            else:
                stats["early_commit_unsafe"] += 1
                stats[f"early_commit_unsafe_{risk}"] += 1
        if self.request[side] == "await" and not current.falling and current.phase == SETTLED_PHASE:
            self._capture(side, "settled", frame, current, opponent, pool, stats)

    def on_commit(self, side, point, start_frame, current, action):
        """Schedule the next pill's request for when the current script starts."""
        if point == "commit_safe":
            self.commit[side] = dict(frame=start_frame, pill=bytes(current.preview),
                                     lock_board=committed_lock_board(current.board, current.pill, action))

    def _capture(self, side, kind, frame, current, opponent, pool, stats, public=None):
        board = predicted_board(current.board, kind)
        self.request[side] = dict(kind=kind, frame=frame, public=public or pool.public_state(side),
                                  board=board, pill=bytes(current.preview),
                                  incoming=int(opponent.garbage_sent_total))
        stats["early_requests"] += 1

    def resolve(self, side, point, frame, current, opponent, stats):
        """Validated request for this spawn, or None (with the reason counted)."""
        request, self.request[side] = self.request[side], None
        self.commit[side] = None
        if point == "spawn":
            return None
        if not isinstance(request, dict):
            stats["early_unavailable"] += 1
            return None
        reason = ("garbage" if int(opponent.garbage_sent_total) != request["incoming"] else
                  "pill" if bytes(current.pill) != request["pill"] else
                  "lock_pose" if request.get("lock_miss") else
                  "cascade" if bytes(current.board) != request["board"] else None)
        if reason is not None:
            stats["early_mismatch"] += 1
            stats[f"early_mismatch_{request['kind']}_{reason}"] += 1
            return None
        lead = frame - request["frame"]
        stats["early_accepted"] += 1
        stats[f"early_accepted_{request['kind']}"] += 1
        stats["early_lead_frames"] += lead
        stats[f"early_lead_{request['kind']}_frames"] += lead
        stats[f"early_lead_{request['kind']}_{lead_bucket(lead)}"] += 1
        return request
