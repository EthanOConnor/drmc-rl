"""Semantic bottles and the frozen VS policies' observation encoding.

Physics and afterstate models need the complete capsule bonds. The legacy VS
actors were trained with horizontal bonds hidden on same-color pill turns;
that lossy encoding belongs only at their network input boundary.
"""

from __future__ import annotations

import numpy as np

import drmc_rl.game.specs.ram_to_state as ram_specs


def board_bytes_to_semantic_planes(board_bytes: bytes | np.ndarray) -> np.ndarray:
    """Decode one native bottle into eight planes, preserving every bond."""

    board = (
        np.frombuffer(board_bytes, dtype=np.uint8)
        if isinstance(board_bytes, bytes)
        else np.asarray(board_bytes, dtype=np.uint8)
    ).reshape(16, 8)
    kind = board & 0xF0
    color = board & 0x03
    visible = (board != 0xFF) & (board != 0) & (kind != 0xB0) & (kind != 0xF0)
    return np.stack((
        visible & (color == 1),
        visible & (color == 0),
        visible & (color == 2),
        kind == ram_specs.T_VIRUS,
        kind == ram_specs.T_BOTTOM,
        kind == ram_specs.T_TOP,
        kind == ram_specs.T_RIGHT,
        kind == ram_specs.T_LEFT,
    )).astype(np.float32)


def legacy_vs_policy_boards(
    own: np.ndarray,
    opponent: np.ndarray,
    pill: np.ndarray,
    opponent_pill: np.ndarray,
) -> np.ndarray:
    """Encode both bottles for frozen VS actors without mutating semantics.

    Pill colors are public canonical R/Y/B indices. Feasibility is packed
    separately by the caller; it must never overwrite the semantic bonds.
    """

    boards = np.concatenate((own, opponent), axis=0).astype(np.float32)
    if pill[0] == pill[1]:
        boards[6:8] = 0
    if opponent_pill[0] == opponent_pill[1]:
        boards[14:16] = 0
    return boards
