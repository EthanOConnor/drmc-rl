from __future__ import annotations

import numpy as np

from envs.specs import ram_to_state as r2s


def test_count_tile_removals_counts_virus_and_pill() -> None:
    prev = np.full((16, 8), r2s.FIELD_EMPTY, dtype=np.uint8)
    nxt = np.full((16, 8), r2s.FIELD_EMPTY, dtype=np.uint8)

    # One virus cleared.
    prev[0, 0] = np.uint8(r2s.T_VIRUS | 0x01)
    nxt[0, 0] = np.uint8(r2s.CLEARED_TILE | 0x01)

    # One pill half cleared.
    prev[1, 1] = np.uint8(0x40 | 0x02)  # pill type (top) + color bits
    nxt[1, 1] = np.uint8(r2s.FIELD_JUST_EMPTIED | 0x02)

    total, viruses, nonvirus = r2s.count_tile_removals(prev, nxt)
    assert total == 2
    assert viruses == 1
    assert nonvirus == 1


def test_count_tile_removals_ignores_new_tiles_and_type_changes() -> None:
    prev = np.full((16, 8), r2s.FIELD_EMPTY, dtype=np.uint8)
    nxt = np.full((16, 8), r2s.FIELD_EMPTY, dtype=np.uint8)

    # New pill placed: should not count as a removal.
    nxt[2, 2] = np.uint8(0x60 | 0x01)  # pill type (left) + color bits

    # Pill orientation/type changes in-place: still occupied, so not a removal.
    prev[3, 3] = np.uint8(0x60 | 0x01)
    nxt[3, 3] = np.uint8(0x70 | 0x01)

    total, viruses, nonvirus = r2s.count_tile_removals(prev, nxt)
    assert total == 0
    assert viruses == 0
    assert nonvirus == 0

