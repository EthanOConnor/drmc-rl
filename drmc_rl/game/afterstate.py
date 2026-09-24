"""Exact post-lock afterstates and their public consequences, without the native engine.

Every candidate placement is written into the acting side's bottle and resolved
through the ROM's drop/match cascade (``drmc_rl.game.cascade``). The result is
the settled bottle the move actually produces plus exact facts about what it
did: tiles and viruses cleared, matched lines (the ROM combo counter), cascade
rounds, the garbage that combo stores against the opponent, and whether the
next spawn is blocked.

Inputs are the public model tensors themselves (semantic own planes, canonical
pill colors, packed candidate actions), so arena, trainer, browser (Pyodide)
and early pre-spawn decision points share one deterministic function. Nothing
here reads hidden state: incoming garbage the actor cannot see is not applied.
"""

from __future__ import annotations

import numpy as np

from drmc_rl.game import cascade as _cascade

AFTERSTATE_SCHEMA = "drmc-exact-afterstate-v1"
FACT_NAMES = (
    "viruses_cleared",
    "tiles_cleared",
    "lines",
    "rounds",
    "garbage_sent",
    "garbage_red",
    "garbage_yellow",
    "garbage_blue",
    "spawn_blocked",
    "viruses_after",
    "win",
    "height_after",
    "height_change",
    "occupied_after",
    "changed_cells",
    "viruses_before",
)
FACT_DIM = len(FACT_NAMES)
_FACT_SCALE = np.asarray(
    (4.0, 8.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 20.0, 1.0, 16.0, 8.0, 64.0, 16.0, 20.0),
    dtype=np.float32,
)
# ROM attackSize_min / attackSize_max: a combo of two or more lines stores
# garbage; at most four pieces (the first four line colors) are released.
_ATTACK_MIN, _ATTACK_MAX = 2, 4
# Canonical model colors are R, Y, B; the NES low nibble is Y=0, R=1, B=2.
_CANONICAL_TO_NES = np.asarray((1, 0, 2), dtype=np.uint8)
_NES_TO_CANONICAL = np.asarray((1, 0, 2), dtype=np.int64)
# Macro action = orientation*128 + anchor cell; the second half lies right,
# below, left or above the anchor (drmc_rl.human.repertoire.placement_geometry).
_SECOND = ((0, 1), (1, 0), (0, -1), (-1, 0))
_TILE_TYPES = ((0x60, 0x70), (0x40, 0x50), (0x70, 0x60), (0x50, 0x40))
_SPAWN_CELLS = (3, 4)


def planes_to_fields(planes: np.ndarray) -> np.ndarray:
    """Invert eight semantic planes ``[...,8,16,8]`` into native tile bytes ``[...,128]``."""

    value = np.asarray(planes)
    if value.shape[-3:] != (8, 16, 8):
        raise ValueError(f"semantic planes must end in (8,16,8), got {value.shape}")
    occupied = value[..., :3, :, :].sum(axis=-3) > 0.5
    low = _CANONICAL_TO_NES[value[..., :3, :, :].argmax(axis=-3)]
    high = np.full(occupied.shape, 0x80, dtype=np.uint8)
    virus = value[..., 3, :, :] > 0.5
    for channel, tile in ((4, 0x50), (5, 0x40), (6, 0x70), (7, 0x60)):
        high[(value[..., channel, :, :] > 0.5) & ~virus] = tile
    high[virus] = 0xD0
    fields = np.where(occupied, high | low, np.uint8(0xFF)).astype(np.uint8)
    return fields.reshape(*fields.shape[:-2], 128)


def _mark_lines(board: bytearray) -> list[int]:
    """ROM horizontal-then-vertical scan; returns each matched line's NES color."""

    colors: list[int] = []
    width, height = _cascade.BOARD_WIDTH, _cascade.BOARD_HEIGHT
    for row in range(height):
        col = 0
        while col <= width - _cascade.MIN_CHAIN:
            tile = board[row * width + col]
            if tile >= _cascade.TILE_JUST_EMPTIED:
                col += 1
                continue
            color = tile & _cascade.MASK_COLOR
            chain = 1
            while col + chain < width and (board[row * width + col + chain] & _cascade.MASK_COLOR) == color:
                chain += 1
            if chain >= _cascade.MIN_CHAIN:
                colors.append(color & 0x03)
                for k in range(chain):
                    index = row * width + col + k
                    board[index] = _cascade.TILE_CLEARED | (board[index] & _cascade.MASK_COLOR)
                col += chain
            else:
                col += 1
    for col in range(width):
        row = 0
        while row <= height - _cascade.MIN_CHAIN:
            tile = board[row * width + col]
            if tile >= _cascade.TILE_JUST_EMPTIED:
                row += 1
                continue
            color = tile & _cascade.MASK_COLOR
            chain = 1
            while row + chain < height and (board[(row + chain) * width + col] & _cascade.MASK_COLOR) == color:
                chain += 1
            if chain >= _cascade.MIN_CHAIN:
                colors.append(color & 0x03)
                for k in range(chain):
                    index = (row + k) * width + col
                    board[index] = _cascade.TILE_CLEARED | (board[index] & _cascade.MASK_COLOR)
                row += chain
            else:
                row += 1
    return colors


def _viruses(board) -> int:
    return sum(1 for tile in board if tile & 0xF0 == 0xD0)


def _height(board) -> int:
    for row in range(16):
        if any(board[row * 8 + col] != 0xFF for col in range(8)):
            return 16 - row
    return 0


def resolve_placement(field, pill, action: int) -> tuple[bytes, np.ndarray]:
    """Lock one canonical-color pill at ``action`` and settle the bottle exactly.

    Returns the settled 128-byte field and the raw (unscaled) fact vector.
    """

    root = bytes(np.asarray(field, dtype=np.uint8).reshape(128))
    orientation, cell = divmod(int(action), 128)
    if not 0 <= orientation < 4:
        raise ValueError("invalid placement action")
    row, col = divmod(cell, 8)
    dr, dc = _SECOND[orientation]
    board = bytearray(root)
    for (r, c), tile, color in zip(
        ((row, col), (row + dr, col + dc)), _TILE_TYPES[orientation], pill, strict=True
    ):
        if not (0 <= r < 16 and 0 <= c < 8) or board[r * 8 + c] != 0xFF:
            raise ValueError("placement outside the empty bottle cells")
        board[r * 8 + c] = tile | int(_CANONICAL_TO_NES[int(color)])
    before_viruses = _viruses(root)
    lines: list[int] = []
    rounds = tiles = viruses = 0
    while True:
        while _cascade._drop_pass(board):
            pass
        pre = bytes(board)
        found = _mark_lines(board)
        if not found:
            break
        rounds += 1
        lines.extend(found)
        for before, after in zip(pre, board):
            if after & 0xF0 == _cascade.TILE_CLEARED and before & 0xF0 != _cascade.TILE_CLEARED:
                tiles += 1
                viruses += before & 0xF0 == 0xD0
        _cascade._update_field(board)
    for index in range(128):
        if board[index] >= _cascade.TILE_JUST_EMPTIED:
            board[index] = 0xFF
    after = bytes(board)
    combo = len(lines)
    sent = lines[:_ATTACK_MAX] if combo >= _ATTACK_MIN else []
    garbage = [0, 0, 0]
    for color in sent:
        garbage[int(_NES_TO_CANONICAL[color])] += 1
    after_viruses = before_viruses - viruses
    root_height, after_height = _height(root), _height(after)
    facts = np.asarray(
        (
            viruses,
            tiles,
            combo,
            rounds,
            len(sent),
            *garbage,
            any(after[c] != 0xFF for c in _SPAWN_CELLS),
            after_viruses,
            before_viruses > 0 and after_viruses == 0,
            after_height,
            after_height - root_height,
            sum(tile != 0xFF for tile in after),
            sum(a != b for a, b in zip(root, after)),
            before_viruses,
        ),
        dtype=np.float32,
    )
    return after, facts


def afterstate_batch(
    own_planes: np.ndarray,
    pill: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Settled own afterstates ``[B,K,128]`` (uint8) and scaled facts ``[B,K,F]``.

    Padded candidates receive the root bottle and zero facts. Duplicate
    actions within a row are resolved once.
    """

    fields = planes_to_fields(np.asarray(own_planes)[..., :8, :, :])
    actions = np.asarray(actions)
    mask = np.asarray(mask, dtype=bool)
    pill = np.asarray(pill)
    batch, width = mask.shape
    tiles = np.repeat(fields[:, None, :], width, axis=1)
    facts = np.zeros((batch, width, FACT_DIM), dtype=np.float32)
    for b in range(batch):
        colors = (int(pill[b, 0]), int(pill[b, 1]))
        seen: dict[int, int] = {}
        for k in np.flatnonzero(mask[b]):
            action = int(actions[b, k])
            if action in seen:
                j = seen[action]
                tiles[b, k], facts[b, k] = tiles[b, j], facts[b, j]
                continue
            after, raw = resolve_placement(fields[b], colors, action)
            tiles[b, k] = np.frombuffer(after, dtype=np.uint8)
            facts[b, k] = raw / _FACT_SCALE
            seen[action] = int(k)
    return tiles, facts


__all__ = [
    "AFTERSTATE_SCHEMA",
    "FACT_DIM",
    "FACT_NAMES",
    "afterstate_batch",
    "planes_to_fields",
    "resolve_placement",
]
