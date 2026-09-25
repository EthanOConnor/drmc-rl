"""Causal public pair views reconstructed from human-corpus placement rows.

A corpus release stores, per placement, the acting side's bottle at spawn, its
pill and preview, spawn and lock frames, the lock pose and the controller
bytes from spawn to lock. Both players' rows share one console clock, so one
game's rows determine what a live ``public_pair_context_v3`` actor would have
seen at each human spawn, up to post-lock animation timing:

* own side: exact (bottle, pill, preview, spawn pose, virus count);
* opponent while falling: exact bottle, pill and preview; the pose is stepped
  from the recorded inputs under the audited FBNeo input contract
  (``tools/audit_execution_replay.py``), else interpolated;
* opponent after its lock: the locked pill stays current (as in the ROM);
  the bottle is the placed pill until the first clear round, then the settled
  afterstate, then the next spawn's bottle once garbage has been released;
  the animation phase follows the ROM's post-lock action sequence;
* events: spawns and locks at their recorded frames, one clear per cascade
  round and garbage volleys (seen as new single tiles in the receiver's next
  bottle) at modelled frames.

Modelled frames (clear rounds, volley release, clearing/resolving phases) are
calibrated against the native engine (``tools.validate_corpus_public_state``);
their residual error only shifts log-scaled event ages by a few frames.
Nothing here reads hidden state: the attack a combo stores is inferred from
what the receiver visibly gets, never from the opponent's RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from drmc_rl.game import cascade as _cascade
from drmc_rl.game.pair_state import (
    DecisionBoundary,
    FallingPillView,
    PairEvent,
    PairEventKind,
    PublicPairState,
    VisibleSideState,
)

RAW_TO_CANON = (1, 0, 2)
_CANON_TO_RAW = (1, 0, 2)
# Macro action geometry (drmc_rl.game.afterstate): second half right, below, left, above.
_SECOND = ((0, 1), (1, 0), (0, -1), (-1, 0))
_TILE_TYPES = ((0x60, 0x70), (0x40, 0x50), (0x70, 0x60), (0x50, 0x40))
# Native reset puts the first spawn two frames after frame 0.
FIRST_SPAWN_FRAME = 2
# ROM post-lock timing, fitted on the native engine (19f292c) against the
# lowest row the locked pill occupies (top-origin): the first match round is
# marked max(7, row + 3) frames after lock (+-1), a lock without clears spawns
# the next pill max(13, row + 9) frames later (+-1) with CheckAttack four
# frames before that, and each further cascade round adds about 20 frames.
CLEAR_MIN, CLEAR_ROW = 7, 3
SETTLE_MIN, SETTLE_ROW = 13, 9
ROUND_FRAMES = 20
HISTORY_EVENTS = 64


def raw_pair(left: int, right: int) -> tuple[int, int]:
    return RAW_TO_CANON[int(left) & 3], RAW_TO_CANON[int(right) & 3]


_POSE_TO_ACTION = None


def pose_action(x: int, y: int, rotation: int) -> int:
    """Macro action of a native top-origin lock pose, or -1."""
    global _POSE_TO_ACTION
    if _POSE_TO_ACTION is None:
        from drmc_rl.human.backend import ACTION_TO_POSE

        table = np.full(512, -1, dtype=np.int64)
        for action, pose in enumerate(ACTION_TO_POSE):
            if pose >= 0 and table[pose] < 0:
                table[pose] = action
        _POSE_TO_ACTION = table
    if not (0 <= x < 8 and 0 <= y < 16 and 0 <= rotation < 4):
        return -1
    return int(_POSE_TO_ACTION[int(rotation) * 128 + int(y) * 8 + int(x)])


def place_action(field: bytes, pill_canon: Sequence[int], action: int) -> bytes:
    """Write a locked pill (macro action geometry) into a 128-byte NES bottle."""
    if not 0 <= int(action) < 512:
        raise ValueError("invalid placement action")
    orientation, cell = divmod(int(action), 128)
    row, col = divmod(cell, 8)
    dr, dc = _SECOND[orientation]
    board = bytearray(field)
    for (r, c), tile, color in zip(((row, col), (row + dr, col + dc)), _TILE_TYPES[orientation], pill_canon):
        if not (0 <= r < 16 and 0 <= c < 8) or board[r * 8 + c] != 0xFF:
            raise ValueError("placement outside the empty bottle cells")
        board[r * 8 + c] = tile | _CANON_TO_RAW[int(color)]
    return bytes(board)


def count_viruses(board: bytes) -> int:
    return sum(1 for tile in board if tile & 0xF0 == 0xD0)


@dataclass
class Placement:
    """One corpus placement with its derived post-lock facts."""

    row: Mapping
    side: int
    spawn: int
    lock: int | None
    board: bytes
    pill: tuple[int, int]
    preview: tuple[int, int]
    pose: tuple[int, int, int] | None
    action: int = -1
    placed: bytes | None = None
    settled: bytes | None = None
    rounds: list[tuple[int, int, int]] = field(default_factory=list)  # (tiles, viruses, lines)
    next_spawn: int | None = None
    next_board: bytes | None = None
    volley: int = 0  # garbage tiles visibly received during this post-lock


def _lines_in_round(cells) -> int:
    rows, cols = {}, {}
    for cell in cells:
        rows.setdefault(cell.row, []).append(cell.col)
        cols.setdefault(cell.col, []).append(cell.row)
    lines = 0
    for group in (*rows.values(), *cols.values()):
        run, prev = 0, None
        for value in sorted(group):
            run = run + 1 if prev is not None and value == prev + 1 else 1
            if run == _cascade.MIN_CHAIN:
                lines += 1
            prev = value
    return max(1, lines)


def _received_garbage(settled: bytes, following: bytes) -> int:
    """New single tiles in the next bottle that the own cascade did not produce."""
    new = [i for i in range(128) if settled[i] == 0xFF and following[i] != 0xFF]
    if not 2 <= len(new) <= 4 or any(following[i] & 0xF0 != 0x80 for i in new):
        return 0
    return len(new)


class CorpusGame:
    """All placements of one corpus game and the public views they imply."""

    def __init__(self, rows: Iterable[Mapping], *, pose_at=None):
        by_side: dict[int, list[Placement]] = {0: [], 1: []}
        for row in rows:
            slot = int(row["player_slot"])
            if slot not in (1, 2):
                raise ValueError("corpus player_slot must be 1 or 2")
            side = slot - 1
            lock = int(row["lock_frame"]) if row.get("lock_frame") is not None else None
            x, y, rot = row.get("lock_x"), row.get("lock_y_top"), row.get("lock_rotation")
            pose = None
            if lock is not None and x is not None and 0 <= int(x) < 8 and 0 <= int(y) < 16:
                pose = (int(x), int(y), int(rot) & 3)
            by_side[side].append(Placement(
                row=row, side=side, spawn=int(row["spawn_frame"]), lock=lock,
                board=bytes(row["field"]), pill=raw_pair(row["pill_left"], row["pill_right"]),
                preview=raw_pair(row["preview_left"], row["preview_right"]), pose=pose,
            ))
        self.sides = {s: sorted(v, key=lambda p: p.spawn) for s, v in by_side.items()}
        spawns = [p.spawn for v in self.sides.values() for p in v]
        if not spawns:
            raise ValueError("empty corpus game")
        self.origin = min(spawns) - FIRST_SPAWN_FRAME
        self.pose_at = pose_at or corpus_pose_at
        for placements in self.sides.values():
            for i, p in enumerate(placements):
                if i + 1 < len(placements):
                    p.next_spawn = placements[i + 1].spawn
                    p.next_board = placements[i + 1].board
                if p.pose is None:
                    continue
                p.action = pose_action(*p.pose)
                try:
                    p.placed = place_action(p.board, p.pill, p.action)
                except ValueError:
                    p.pose, p.action = None, -1
                    continue
                result = _cascade.resolve_cascade(p.placed)
                p.settled = result.settled_field
                p.rounds = [(len(s.cleared), sum(c.is_virus for c in s.cleared), _lines_in_round(s.cleared))
                            for s in result.steps]
                if p.next_board is not None:
                    p.volley = _received_garbage(p.settled, p.next_board)
        self.events = self._events()

    # -- modelled post-lock frames --------------------------------------
    @staticmethod
    def _bottom(p: Placement) -> int:
        _x, y, rot = p.pose
        return y + (1 if rot & 1 else 0)

    def clear_frames(self, p: Placement) -> list[int]:
        first = p.lock + max(CLEAR_MIN, self._bottom(p) + CLEAR_ROW)
        frames = [first + ROUND_FRAMES * k for k in range(len(p.rounds))]
        if p.next_spawn is not None:
            frames = [min(f, p.next_spawn - 9) for f in frames]
        return [max(p.lock + 1, f) for f in frames]

    def check_attack_frame(self, p: Placement) -> int:
        frame = p.lock + max(SETTLE_MIN, self._bottom(p) + SETTLE_ROW) - 4 + ROUND_FRAMES * len(p.rounds)
        if p.next_spawn is not None:
            frame = min(frame, p.next_spawn - 4)
        return max(p.lock + 1, frame)

    def _events(self) -> list[PairEvent]:
        events = []
        for side, placements in self.sides.items():
            for p in placements:
                events.append((p.spawn, 0, side, PairEvent(PairEventKind.SPAWN, p.spawn - self.origin, side,
                                                            dict(column=3, row_top=0, rotation=0))))
                if p.lock is None or p.pose is None:
                    continue
                x, y, rot = p.pose
                events.append((p.lock, 1, side, PairEvent(PairEventKind.LOCK, p.lock - self.origin, side,
                                                           dict(column=x, row_top=y, rotation=rot))))
                for frame, (tiles, viruses, lines) in zip(self.clear_frames(p), p.rounds):
                    events.append((frame, 2, side, PairEvent(PairEventKind.CLEAR, frame - self.origin, side,
                                   dict(tiles_cleared=tiles, viruses_cleared=viruses, lines_cleared=lines))))
                if p.volley:
                    frame = self.check_attack_frame(p)
                    events.append((frame, 3, side, PairEvent(PairEventKind.VOLLEY, frame - self.origin, side,
                                   dict(garbage_size=p.volley, sender=1 - side))))
        events.sort(key=lambda e: (e[0], e[2], e[1]))
        return [e for *_, e in events]

    # -- views ------------------------------------------------------------
    def _opponent_view(self, side: int, frame: int) -> tuple[VisibleSideState, bool]:
        placements = self.sides[1 - side]
        current = None
        for p in placements:
            if p.spawn <= frame:
                current = p
            else:
                break
        if current is None:
            first = placements[0] if placements else None
            if first is None:
                board, pill, preview = bytes(self.sides[side][0].row["opp_field"]), (0, 0), (0, 0)
            else:
                board, pill, preview = first.board, first.pill, first.preview
            return VisibleSideState(board=board, pill=pill, preview=preview, active=None,
                                    viruses_remaining=count_viruses(board), animation_phase="spawn"), False
        p = current
        falling = p.lock is None or frame < p.lock
        if falling:
            age = frame - p.spawn
            pose = self.pose_at(p, age)
            active = FallingPillView(pose[0], pose[1], pose[2], p.pill, True, age)
            return VisibleSideState(board=p.board, pill=p.pill, preview=p.preview, active=active,
                                    viruses_remaining=count_viruses(p.board), animation_phase="falling"), age == 0
        board = p.placed if p.placed is not None else p.board
        clears = self.clear_frames(p) if p.placed is not None else []
        if clears and frame >= clears[0]:
            board = p.settled
        attack = self.check_attack_frame(p) if p.lock is not None and p.pose is not None else None
        if p.volley and attack is not None and frame > attack and p.next_board is not None:
            board = p.next_board
        phase = "settling"
        if p.next_spawn is not None:
            to_spawn = p.next_spawn - frame
            if 1 <= to_spawn <= 3:
                phase = "spawn"
            elif to_spawn == 4:
                phase = "resolving"
            elif 6 <= to_spawn <= 8:
                phase = "clearing"
        if phase == "settling":
            if any(c - 2 <= frame <= c for c in clears):
                phase = "clearing"
            elif p.volley and attack is not None and frame == attack:
                phase = "resolving"
        pill, preview = p.pill, p.preview
        if p.next_spawn is not None and p.next_spawn - frame == 1:
            following = placements[placements.index(p) + 1]
            pill, preview = following.pill, following.preview  # the ROM loads the next pill one frame early
        return VisibleSideState(board=board, pill=pill, preview=preview, active=None,
                                viruses_remaining=count_viruses(board), animation_phase=phase), False

    def public_state(self, placement: Placement) -> PublicPairState:
        """The live actor's causal view at this placement's spawn frame."""
        side, frame = placement.side, placement.spawn
        own = VisibleSideState(
            board=placement.board, pill=placement.pill, preview=placement.preview,
            active=FallingPillView(3, 0, 0, placement.pill, True, 0),
            viruses_remaining=count_viruses(placement.board), animation_phase="falling",
        )
        opponent, opponent_deciding = self._opponent_view(side, frame)
        sides = (own, opponent) if side == 0 else (opponent, own)
        now = frame - self.origin
        history = [e for e in self.events if e.frame_id <= now][-HISTORY_EVENTS:]
        mine = DecisionBoundary.P1 if side == 0 else DecisionBoundary.P2
        return PublicPairState(
            frame_id=now, viewer_side=side, sides=sides,
            decision_boundary=DecisionBoundary.BOTH if opponent_deciding else mine,
            recent_events=tuple(history), observable_clock_delta_frames=0,
        )


def _direction(mask: int) -> int:
    return 1 if mask & 2 else 2 if mask & 1 else 0


def _rotation(mask: int) -> int:
    return 1 if mask & 0x80 else 2 if mask & 0x40 else 0


def spawn_frame_state(row: Mapping):
    """Planner spawn state under the audited FBNeo input contract (prior held byte, parity xor 1)."""
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation

    prior = row.get("held_before_spawn")
    initial = int(row["held_at_spawn"] if prior is None else prior)
    return FrameState(
        x=3, y=0, rot=0, speed_counter=int(row["speed_counter"]),
        hor_velocity=int(row["horizontal_velocity"]) & 15,
        hold_dir=HoldDir(_direction(initial)), rot_hold=Rotation(_rotation(initial)),
        frame_parity=(int(row["frame_counter"]) & 1) ^ 1,
    )


def corpus_pose_at(p: Placement, age: int) -> tuple[int, int, int]:
    """Falling pose ``age`` frames after spawn, stepped from recorded inputs."""
    from drmc_rl.data.human_corpus import decode_input_rle
    from drmc_rl.planning.fast_reach import compute_speed_threshold, simulate_frame

    row = p.row
    if age <= 0:
        return 3, 0, 0
    tau = int(row.get("tau_frames") or 0)
    try:
        raw = decode_input_rle(row["input_rle_u16_u8"], row["input_frames"]) if row.get("input_frames") else b""
    except ValueError:
        raw = b""
    if raw and len(raw) == tau + 1 and not any(m & 3 == 3 or m & 0xC0 == 0xC0 for m in raw):
        board = np.frombuffer(p.board, dtype=np.uint8).reshape(16, 8)
        columns = np.zeros(8, dtype=np.uint16)
        for y in range(16):
            columns |= (board[y] != 0xFF).astype(np.uint16) << y
        state = spawn_frame_state(row)
        threshold = compute_speed_threshold(int(row["speed"]), int(row["speed_ups"]))
        for mask in raw[:-1][:age]:
            action = _direction(mask) * 6 + (3 if mask & 4 else 0) + _rotation(mask)
            following = simulate_frame(columns, state, action, speed_threshold=threshold)
            if following.locked:
                break
            state = following
        return int(state.x), int(state.y), int(state.rot)
    if p.pose is None or tau <= 0:
        return 3, 0, 0
    x, y, rot = p.pose
    return x, int(round(y * min(1.0, age / tau))), rot


__all__ = [
    "CorpusGame",
    "Placement",
    "corpus_pose_at",
    "place_action",
    "pose_action",
    "raw_pair",
    "spawn_frame_state",
]
