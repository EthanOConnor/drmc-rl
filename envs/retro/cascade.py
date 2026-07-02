"""Pure-Python (numpy-free-core) port of the NES Dr. Mario post-lock cascade.

Ported from drmario-native GameLogic.cpp (pillPlaced_checkDrop /
pillPlaced_checkHorMatch / pillPlaced_checkVerMatch / pillPlaced_updateField),
which is itself a faithful port of the ROM disassembly ($8CA4/$920B/$9479/$92EB).

Input: a 128-byte NES-format bottle field (row-major, row 0 = top, 8 cols)
with the locked pill already baked in. Tile encoding (drmario_constants.asm):

    high nibble (type): 0x4 top half (partner below), 0x5 bottom half
    (partner above), 0x6 left half (partner right), 0x7 right half
    (partner left), 0x8 single, 0xD virus, 0xF empty/just-emptied
    low nibble (color): low 2 bits — 0 yellow, 1 red, 2 blue

No emulator, ROM, or native library required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np

BOARD_WIDTH = 8
BOARD_HEIGHT = 16
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT

MASK_TYPE = 0xF0
MASK_COLOR = 0x0F

TILE_EMPTY = 0xFF
TILE_TOP = 0x40
TILE_BOTTOM = 0x50
TILE_LEFT = 0x60
TILE_RIGHT = 0x70
TILE_SINGLE = 0x80
TILE_MIDDLE_VER = 0x90  # unused in retail
TILE_MIDDLE_HOR = 0xA0  # unused in retail
TILE_CLEARED = 0xB0
TILE_VIRUS = 0xD0
TILE_JUST_EMPTIED = 0xF0

MIN_CHAIN = 4


@dataclass(frozen=True)
class ClearedCell:
    row: int  # 0 = top
    col: int
    color: int  # 0 yellow, 1 red, 2 blue
    is_virus: bool


@dataclass(frozen=True)
class CascadeStep:
    """One match round: gravity settled, then these cells cleared together."""

    cleared: Tuple[ClearedCell, ...]


@dataclass(frozen=True)
class CascadeResult:
    steps: Tuple[CascadeStep, ...]
    settled_field: bytes  # 128-byte NES-format field after everything settles
    viruses_cleared: int
    cells_cleared: int


def _drop_pass(board: bytearray) -> bool:
    """One gravity pass ($8CA4). Returns True if anything moved."""
    dropped_any = False
    for pos in range(BOARD_SIZE - 1, -1, -1):
        cur = board[pos]
        if cur < TILE_JUST_EMPTIED:
            continue
        board[pos] = TILE_EMPTY  # normalize just-emptied -> empty
        above = pos - BOARD_WIDTH
        if above < 0:
            continue
        top_tile = board[above]
        if top_tile >= TILE_MIDDLE_HOR:
            continue
        top_type = top_tile & MASK_TYPE
        if top_type in (TILE_LEFT, TILE_MIDDLE_HOR):
            continue
        if top_type == TILE_RIGHT:
            # Horizontal pill: we are under the right half. Both halves drop
            # together only if the cell(s) under the left part are empty too.
            top_pos, bottom_pos = above, pos
            top_left, bottom_left = above, pos
            blocked = False
            while True:
                top_left -= 1
                bottom_left -= 1
                if bottom_left < 0 or top_left < 0:
                    break
                if board[bottom_left] < TILE_JUST_EMPTIED:
                    blocked = True
                    break
                tl_type = board[top_left] & MASK_TYPE
                if tl_type == TILE_LEFT or tl_type != TILE_MIDDLE_HOR:
                    break
            if blocked:
                continue
            while True:
                board[bottom_pos] = board[top_pos]
                board[top_pos] = TILE_EMPTY
                if bottom_pos == bottom_left:
                    break
                bottom_pos -= 1
                top_pos -= 1
            dropped_any = True
            continue
        # Vertical half or single: move it down one row.
        board[pos] = board[above]
        board[above] = TILE_EMPTY
        dropped_any = True
    return dropped_any


def _mark_matches(board: bytearray) -> List[ClearedCell]:
    """Horizontal then vertical scans ($920B/$9479); marks TILE_CLEARED."""
    cleared: List[ClearedCell] = []

    def mark(idx: int) -> None:
        t = board[idx]
        t_type = t & MASK_TYPE
        if t_type != TILE_CLEARED:
            cleared.append(
                ClearedCell(
                    row=idx // BOARD_WIDTH,
                    col=idx % BOARD_WIDTH,
                    color=t & 0x03,
                    is_virus=(t_type == TILE_VIRUS),
                )
            )
        board[idx] = TILE_CLEARED | (t & MASK_COLOR)

    # Horizontal
    for row in range(BOARD_HEIGHT):
        col = 0
        while col <= BOARD_WIDTH - MIN_CHAIN:
            start = row * BOARD_WIDTH + col
            tile = board[start]
            if tile >= TILE_JUST_EMPTIED:
                col += 1
                continue
            color = tile & MASK_COLOR
            chain = 1
            while col + chain < BOARD_WIDTH:
                nxt = board[row * BOARD_WIDTH + col + chain]
                if (nxt & MASK_COLOR) != color:
                    break
                chain += 1
            if chain >= MIN_CHAIN:
                for k in range(chain):
                    mark(row * BOARD_WIDTH + col + k)
                col += chain
            else:
                col += 1

    # Vertical
    for col in range(BOARD_WIDTH):
        row = 0
        while row <= BOARD_HEIGHT - MIN_CHAIN:
            start = row * BOARD_WIDTH + col
            tile = board[start]
            if tile >= TILE_JUST_EMPTIED:
                row += 1
                continue
            color = tile & MASK_COLOR
            chain = 1
            while row + chain < BOARD_HEIGHT:
                nxt = board[(row + chain) * BOARD_WIDTH + col]
                if (nxt & MASK_COLOR) != color:
                    break
                chain += 1
            if chain >= MIN_CHAIN:
                for k in range(chain):
                    mark((row + k) * BOARD_WIDTH + col)
                row += chain
            else:
                row += 1

    return cleared


def _update_field(board: bytearray) -> None:
    """updateField ($92EB): cleared -> just-emptied, orphan halves -> singles."""
    for pos in range(BOARD_SIZE - 1, -1, -1):
        tile = board[pos]
        t_type = tile & MASK_TYPE

        if t_type == TILE_CLEARED:
            board[pos] = tile | TILE_JUST_EMPTIED
            continue

        if t_type == TILE_TOP:
            below = pos + BOARD_WIDTH
            below_type = (board[below] & MASK_TYPE) if below < BOARD_SIZE else TILE_EMPTY & MASK_TYPE
            if below_type not in (TILE_BOTTOM, TILE_MIDDLE_VER):
                board[pos] = TILE_SINGLE | (tile & MASK_COLOR)
            continue

        if t_type == TILE_BOTTOM:
            above = pos - BOARD_WIDTH
            above_type = (board[above] & MASK_TYPE) if above >= 0 else TILE_EMPTY & MASK_TYPE
            if above_type not in (TILE_TOP, TILE_MIDDLE_VER):
                board[pos] = TILE_SINGLE | (tile & MASK_COLOR)
            continue

        if t_type == TILE_LEFT:
            right = pos + 1
            in_row = right < BOARD_SIZE and (right % BOARD_WIDTH) != 0
            right_type = (board[right] & MASK_TYPE) if in_row else TILE_EMPTY & MASK_TYPE
            if right_type not in (TILE_RIGHT, TILE_MIDDLE_HOR):
                board[pos] = TILE_SINGLE | (tile & MASK_COLOR)
            continue

        if t_type == TILE_RIGHT:
            left = pos - 1
            in_row = left >= 0 and (left % BOARD_WIDTH) != BOARD_WIDTH - 1
            left_type = (board[left] & MASK_TYPE) if in_row else TILE_EMPTY & MASK_TYPE
            if left_type not in (TILE_LEFT, TILE_MIDDLE_HOR):
                board[pos] = TILE_SINGLE | (tile & MASK_COLOR)
            continue


def resolve_cascade(field: Union[bytes, bytearray, np.ndarray]) -> CascadeResult:
    """Run the full post-lock drop/match cascade until the field settles.

    `field` is a 128-byte NES-format bottle (row-major, row 0 = top) with the
    locked pill already written in. Returns per-round cleared cells (each
    round = matches found after gravity fully settles), totals, and the
    settled field.
    """
    if isinstance(field, np.ndarray):
        raw = field.astype(np.uint8).tobytes()
    else:
        raw = bytes(field)
    if len(raw) != BOARD_SIZE:
        raise ValueError(f"field must be {BOARD_SIZE} bytes, got {len(raw)}")
    board = bytearray(raw)

    steps: List[CascadeStep] = []
    while True:
        while _drop_pass(board):
            pass
        cleared = _mark_matches(board)
        if not cleared:
            break
        steps.append(CascadeStep(cleared=tuple(cleared)))
        _update_field(board)

    # Normalize any lingering just-emptied markers to plain empty.
    for pos in range(BOARD_SIZE):
        if board[pos] >= TILE_JUST_EMPTIED:
            board[pos] = TILE_EMPTY

    all_cells = [c for s in steps for c in s.cleared]
    return CascadeResult(
        steps=tuple(steps),
        settled_field=bytes(board),
        viruses_cleared=sum(1 for c in all_cells if c.is_virus),
        cells_cleared=len(all_cells),
    )
