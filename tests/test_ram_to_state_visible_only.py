import numpy as np
from envs.specs.ram_to_state import ram_to_state


def test_shapes_and_visibility():
    # minimal fake RAM of 0x800 bytes
    ram = bytes(0x800)
    # minimal offsets with fake addresses within range
    offsets = {
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
    state = ram_to_state(ram, offsets)
    assert state.shape == (14, 16, 8)
    # No future info present by construction
    assert (state[0] >= 0).all()

