"""Admissible per-step frame bounds for exact search, measured from the engine.

A decision step's tau = movement frames to lock + post-lock frames to the
next decision. The global minimum over all reachable situations is realized
on an extremal board: full support directly under the spawn makes the first
gravity/down drop fail as early as the rules allow, and a flat junk row with
alternating colors guarantees no clears (no animation, no settle), so the
post-lock path is the bare next-spawn sequence.

Two flavors per (speed_setting, speed_ups), each minimized over frame parity:
- continuing: support one row below the spawn row, so the next pill can
  spawn (lock at row 14; spawn row stays free) — a lower bound for any step
  that is followed by another decision.
- terminal: support directly below the spawn cells (lock at row 15 tops the
  bottle out) — a lower bound for the episode's final step, which skips the
  next-spawn wait.

speed_counter is 0 at every real spawn (reset on each successful drop and at
lock), so measuring from a fresh checkpoint matches play exactly.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from drmc_rl.seedlab import rng as slrng

TILE_EMPTY = 0xFF
TILE_SINGLE = 0x80
TILE_VIRUS = 0xD0
SPEEDUPS_MAX = 0x31
SPAWN_COL = 3  # PILL_START_COL; left half of the horizontal spawn


def _support_board(*, rows_below_spawn: int) -> bytes:
    """Grounded checkerboard block whose top is `rows_below_spawn` rows under
    the spawn row (0 = blocks the spawn row itself, terminal flavor; 1 leaves
    it free, continuing flavor). Grounded so nothing settles post-lock;
    checkerboard colors so no clear can form."""

    board = bytearray([TILE_EMPTY] * 128)
    top_idx = 1 + int(rows_below_spawn)  # engine board row index (0 = top)
    for row_idx in range(top_idx, 16):
        for col in range(8):
            board[row_idx * 8 + col] = TILE_SINGLE | ((row_idx + col) & 1)
    return bytes(board)


def _clear_terminal_board(pill_left_raw_color: int) -> bytes:
    """Three viruses under the spawn's left half: locking at the spawn row
    completes a vertical 4-run and stage-clears — the fastest possible final
    step (viruses never settle, so they can float)."""

    board = bytearray([TILE_EMPTY] * 128)
    for row_idx in (1, 2, 3):  # game rows 14, 13, 12
        board[row_idx * 8 + SPAWN_COL] = TILE_VIRUS | (int(pill_left_raw_color) & 0x03)
    return bytes(board)


class StepBounds:
    """Lazy per-(speed_ups) cache of measured minimal step frames."""

    def __init__(self, engine, *, speed_setting: int, seed: int = 0x8988) -> None:
        self.engine = engine
        self.speed_setting = int(speed_setting)
        self.seed = int(seed)
        self._continuing: Dict[int, int] = {}
        self._terminal: Dict[int, int] = {}

    def continuing(self, speed_ups: int) -> int:
        return self._measure(speed_ups, terminal=False)

    def terminal(self, speed_ups: int) -> int:
        return self._measure(speed_ups, terminal=True)

    def _measure(self, speed_ups: int, *, terminal: bool) -> int:
        ups = max(0, min(SPEEDUPS_MAX, int(speed_ups)))
        cache = self._terminal if terminal else self._continuing
        hit = cache.get(ups)
        if hit is not None:
            return hit

        boards = [_support_board(rows_below_spawn=0 if terminal else 1)]
        if terminal:
            # The final step can also end via an immediate virus clear, which
            # is faster than the topout path; bound by the minimum of both.
            pills, _r0, _r1 = slrng.generate_pill_reserve(*slrng.seed_to_bytes(self.seed))
            left_raw = slrng.pill_colors_raw(pills[0])[0]
            boards.append(_clear_terminal_board(left_raw))

        best = None
        for board in boards:
            for parity in (0, 1):
                tau = self._measure_one(board, ups, parity)
                if tau is not None:
                    best = tau if best is None else min(best, tau)
        if best is None:
            best = 1  # measurement failed: fall back to the trivial bound
        cache[ups] = int(best)
        return int(best)

    def _measure_one(self, board: bytes, ups: int, parity: int):
        from drmc_rl.envs.backends.drmario_pool import build_reset_spec

        eng = self.engine
        b = eng.runner.buffers
        spec = build_reset_spec(
            level=0,
            speed_setting=self.speed_setting,
            rng_state=slrng.seed_to_bytes(self.seed),
            rng_override=True,
            checkpoint_enabled=True,
            checkpoint_board=np.frombuffer(board, dtype=np.uint8),
            checkpoint_pill_counter=0,
            checkpoint_speed_ups=ups,
            checkpoint_frame_parity=parity,
        )
        mask = np.zeros(eng.num_envs, dtype=np.uint8)
        mask[0] = 1
        eng.restore([spec] + [eng._noop_spec] * (eng.num_envs - 1), mask)
        costs = b.cost_to_lock[0].reshape(-1)
        feas = b.feasible_mask[0].reshape(-1).astype(bool)
        if not feas.any():
            return None
        action = int(np.flatnonzero(feas)[np.argmin(costs[feas].astype(np.int64))])
        actions = np.zeros(eng.num_envs, dtype=np.int32)
        actions[0] = action
        eng.step(actions)
        if int(b.invalid_action[0]) != -1:
            return None
        return max(1, int(b.tau_frames[0]))
