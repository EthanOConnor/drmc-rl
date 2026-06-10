from __future__ import annotations

import envs.specs.ram_to_state as ram_specs


def _offsets() -> dict:
    return {
        "bottle": {"base_addr": "0x0000", "stride": 8},
        "falling_pill": {
            "row_addr": "0x0100",
            "col_addr": "0x0101",
            "orient_addr": "0x0102",
            "left_color_addr": "0x0103",
            "right_color_addr": "0x0104",
        },
        "preview_pill": {"left_color_addr": "0x0110", "right_color_addr": "0x0111"},
        "gravity_lock": {"gravity_counter_addr": "0x0120", "lock_counter_addr": "0x0121"},
        "level": {"addr": "0x0130"},
    }


def _put(board: bytearray, row: int, col: int, tile: int) -> None:
    board[row * 8 + col] = int(tile) & 0xFF


def test_bottle_connection_edge_planes_from_tile_codes() -> None:
    prev = ram_specs.get_state_representation()
    ram_specs.set_state_representation("bitplane_bottle_conn")
    try:
        ram = bytearray([ram_specs.FIELD_EMPTY] * 0x800)

        # Horizontal two-piece pill: left half connected right, right half connected left.
        _put(ram, 10, 2, ram_specs.T_LEFT | 0x01)
        _put(ram, 10, 3, ram_specs.T_RIGHT | 0x02)

        # Vertical two-piece pill: top connected down, bottom connected up.
        _put(ram, 5, 4, ram_specs.T_TOP | 0x00)
        _put(ram, 6, 4, ram_specs.T_BOTTOM | 0x01)

        # Non-connected objects should not create edges.
        _put(ram, 7, 7, ram_specs.T_SINGLE | 0x02)
        _put(ram, 8, 0, ram_specs.T_VIRUS | 0x01)
        _put(ram, 9, 0, ram_specs.T_MIDDLE_VER | 0x01)
        _put(ram, 9, 1, ram_specs.T_MIDDLE_HOR | 0x02)

        state = ram_specs.ram_to_state(bytes(ram), _offsets())
        idx = ram_specs.STATE_IDX

        assert state.shape == (8, 16, 8)
        assert state[idx.connected_right, 10, 2] == 1.0
        assert state[idx.connected_left, 10, 3] == 1.0
        assert state[idx.connected_down, 5, 4] == 1.0
        assert state[idx.connected_up, 6, 4] == 1.0

        edge_sum = (
            state[idx.connected_up]
            + state[idx.connected_down]
            + state[idx.connected_left]
            + state[idx.connected_right]
        )
        assert int(edge_sum.sum()) == 4
        assert edge_sum[7, 7] == 0.0
        assert edge_sum[8, 0] == 0.0
        assert edge_sum[9, 0] == 0.0
        assert edge_sum[9, 1] == 0.0
    finally:
        ram_specs.set_state_representation(prev)


def test_bottle_connection_mask_channel_order() -> None:
    prev = ram_specs.get_state_representation()
    ram_specs.set_state_representation("bitplane_bottle_conn_mask")
    try:
        assert ram_specs.STATE_CHANNELS == 12
        assert ram_specs.get_plane_names() == (
            "color_red",
            "color_yellow",
            "color_blue",
            "virus_mask",
            "connected_up",
            "connected_down",
            "connected_left",
            "connected_right",
            "feasible_o0",
            "feasible_o1",
            "feasible_o2",
            "feasible_o3",
        )
        assert ram_specs.STATE_IDX.feasible_mask_channels == (8, 9, 10, 11)
    finally:
        ram_specs.set_state_representation(prev)
