"""Observable first-clear geometry for repertoire diagnostics and curricula.

These are geometric labels, not value bonuses or community motif definitions.
Horizontal capsule orientation alone does not identify a horizontal clear.
"""
from __future__ import annotations

import numpy as np


def clear_geometry(field: np.ndarray) -> dict[str, int]:
    board = np.asarray(field, dtype=np.uint8).reshape(16, 8)
    high, color = board & 0xF0, board & 0x0F
    occupied = (((high >= 0x40) & (high <= 0x80)) | (high == 0xD0)) & (color < 3)
    masks = [np.zeros((16, 8), bool), np.zeros((16, 8), bool)]
    counts, longest = [0, 0], [0, 0]
    for axis in (0, 1):  # horizontal, vertical
        for outer in range(16 if axis == 0 else 8):
            cells = [(outer, i) if axis == 0 else (i, outer)
                     for i in range(8 if axis == 0 else 16)]
            start = 0
            while start < len(cells):
                y, x = cells[start]
                stop = start + 1
                if occupied[y, x]:
                    while stop < len(cells):
                        yy, xx = cells[stop]
                        if not occupied[yy, xx] or color[yy, xx] != color[y, x]:
                            break
                        stop += 1
                    if stop - start >= 4:
                        counts[axis] += 1
                        longest[axis] = max(longest[axis], stop-start)
                        for yy, xx in cells[start:stop]:
                            masks[axis][yy, xx] = True
                start = stop
    return {"horizontal_lines": counts[0], "vertical_lines": counts[1],
            "longest_horizontal": longest[0], "longest_vertical": longest[1],
            "crossing_cells": int((masks[0] & masks[1]).sum()),
            "first_wave_cells": int((masks[0] | masks[1]).sum())}


def placement_geometry(field: np.ndarray, pill, action: int) -> dict[str, int]:
    """Macro action uses first-half anchor and right/down/left/up orientation."""
    board = np.asarray(field, dtype=np.uint8).reshape(16, 8).copy()
    orient, cell = divmod(int(action), 128)
    if not 0 <= orient < 4:
        raise ValueError("invalid placement action")
    y, x = divmod(cell, 8)
    dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[orient]
    for (yy, xx), canonical in zip(((y, x), (y+dy, x+dx)), pill, strict=True):
        if not (0 <= yy < 16 and 0 <= xx < 8) or board[yy, xx] != 0xFF:
            raise ValueError("placement outside the empty bottle cells")
        board[yy, xx] = 0x80 | (1, 0, 2)[int(canonical)]
    return clear_geometry(board)
