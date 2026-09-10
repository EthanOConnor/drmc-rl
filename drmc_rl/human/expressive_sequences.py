"""Verified human constructions for an auxiliary, public-input plan proposer.

Geometry describes observed clears, never value, preference, or community motif
names. Only uninterrupted, exactly reproduced own-board transitions are used.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.game.replay_events import decode_field, parse_quark_events

GOALS = ("horizontal_clear", "crossing_clear", "large_clear", "cascade")
SCHEMA = "drmc-expressive-sequences-v1"


def locked_field(board, pill, action):
    """Bake an ordered raw-color pill at a canonical first-half macro action."""
    field = np.asarray(board, dtype=np.uint8).reshape(16, 8).copy()
    orient, cell = divmod(int(action), 128)
    if not 0 <= orient < 4:
        raise ValueError("invalid action")
    row, col = divmod(cell, 8)
    dr, dc = ((0, 1), (1, 0), (0, -1), (-1, 0))[orient]
    positions = ((row, col), (row+dr, col+dc))
    if all(0 <= r+1 < 16 and 0 <= c < 8 and field[r+1,c] == 0xFF for r,c in positions):
        raise ValueError("recorded pill has not reached a lock position")
    types = ((0x60, 0x70), (0x40, 0x50), (0x70, 0x60), (0x50, 0x40))[orient]
    for (r, c), color, kind in zip(positions, pill, types, strict=True):
        if not 0 <= r < 16 or not 0 <= c < 8 or field[r, c] != 0xFF or not 0 <= color < 3:
            raise ValueError("pill overlaps the board or its boundary")
        field[r, c] = kind | int(color)
    return field


def observed_goals(result):
    horizontal, crossing, large = False, False, False
    for step in result.steps:
        cells = {(c.row, c.col, c.color) for c in step.cleared}
        axes = []
        for dr, dc in ((0, 1), (1, 0)):
            marked = set()
            for r, c, color in cells:
                run = {(r+i*dr, c+i*dc, color) for i in range(4)}
                if run <= cells:
                    marked |= run
            axes.append(marked)
        horizontal |= bool(axes[0])
        crossing |= bool(axes[0] & axes[1])
        large |= len(cells) >= 8
    return np.asarray((horizontal, crossing, large, len(result.steps) >= 2), dtype=np.uint8)


def replay_sequences(raw):
    """Return contiguous verified segments, with no inferred missing placements.

    A recorded lock is accepted only if its entire resolved bottle equals the
    next observed spawn. No repair by fitting an arbitrary action to a payoff.
    Exact controller/motor reachability is a separate requirement at execution.
    """
    from tools.annotate_replay_events import POSE_TO_ACTION, pair_moves

    counters = Counter()
    events = parse_quark_events(raw)
    moves = pair_moves(events, counters)
    spawn_frames = {p:sorted(int(e['f']) for e in events['spawn'] if int(e['p']) == p) for p in (1,2)}
    garbage_frames = {p:sorted(int(e['f']) for e in events['grb'] if int(e['grb']) == p) for p in (1,2)}
    groups = defaultdict(list)
    for move in moves:
        groups[(move["ctx"]["init_f"], move["p"])].append(move)
    segments = []
    for (init_f, player), entries in groups.items():
        entries.sort(key=lambda m: int(m["spawn"]["f"]))
        segment = []
        for current, following in zip(entries, entries[1:]):
            spawn, lock, after = current["spawn"], current["lock"], following["spawn"]
            counters["examined"] += 1
            reason = None
            lo, hi = int(spawn["f"]), int(after["f"])
            if bisect_left(garbage_frames[player],hi) > bisect_left(garbage_frames[player],lo):
                reason = "garbage"
            elif bisect_left(spawn_frames[player],hi)-bisect_left(spawn_frames[player],lo) != 1:
                reason = "missing_lock"
            elif not lo <= int(lock["f"]) < hi or spawn["prev"] != after["pill"]:
                reason = "timeline_or_preview"
            board = decode_field(spawn["field"]).copy()
            board[board >= 0xF0] = 0xFF
            expected = decode_field(after["field"]).copy()
            expected[expected >= 0xF0] = 0xFF
            try:
                x, y, rotation = int(lock["x"]), 15-int(lock["y"]), int(lock["rot"])
                if not 0 <= x < 8 or not 0 <= y < 16 or not 0 <= rotation < 4:
                    raise ValueError("invalid lock pose")
                action = int(POSE_TO_ACTION[rotation*128+y*8+x])
                if resolve_cascade(board).settled_field != board.tobytes():
                    reason = reason or "unsettled_spawn"
                result = resolve_cascade(locked_field(board, spawn["pill"], action))
                if result.settled_field != expected.tobytes():
                    reason = reason or "board_mismatch"
            except (ValueError, IndexError):
                reason = reason or "invalid_lock"
            if reason:
                counters[reason] += 1
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            counters["verified"] += 1
            segment.append(dict(board=board, pill=np.asarray(spawn["pill"], np.uint8),
                preview=np.asarray(spawn["prev"], np.uint8), action=action,
                goals=observed_goals(result), frame=lo, lock_frame=int(lock["f"]),
                game=init_f, player=player, level=int(current["ctx"]["level"]), speed=int(spawn["spd"])))
        if segment:
            segments.append(segment)
    return segments, counters


def construction_windows(segment):
    """A goal first pays off after 2–6 moves; setup moves retain one intent."""
    for end, step in enumerate(segment):
        for goal in np.flatnonzero(step["goals"]):
            for length in range(2, min(6, end+1)+1):
                start = end-length+1
                if not any(segment[i]["goals"][goal] for i in range(start, end)):
                    yield start, length, int(goal)
