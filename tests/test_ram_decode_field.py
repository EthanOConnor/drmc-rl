import numpy as np
from envs.specs.ram_to_state import ram_to_state


def test_field_encoding_decodes_to_channels():
    # Create RAM and place bytes in P1 bottle
    ram = bytearray(0x800)
    base = 0x0400
    stride = 8

    # Virus red at (row=0,col=0): type=0xD0, color=1
    ram[base + 0 * stride + 0] = 0xD0 | 0x01
    # Fixed pill blue at (row=0,col=1): topHalfPill=0x40, color=2
    ram[base + 0 * stride + 1] = 0x40 | 0x02

    # Minimal offsets (we won't use falling/grav/level here)
    offsets = {
        "bottle": {"base_addr": hex(base), "stride": stride},
        "falling_pill": {
            "row_addr": hex(0x0306),
            "col_addr": hex(0x0305),
            "orient_addr": hex(0x0325),
            "left_color_addr": hex(0x0301),
            "right_color_addr": hex(0x0302),
        },
        "gravity_lock": {
            "gravity_counter_addr": hex(0x0312),
            "lock_counter_addr": hex(0x0307),
        },
        "level": {"addr": hex(0x0316)},
    }

    state = ram_to_state(bytes(ram), offsets)
    assert state.shape == (14, 16, 8)

    # Viruses: channel 0 is red, should mark (0,0)
    assert state[0, 0, 0] == 1.0
    # Fixed pills: channel 5 is blue, should mark (0,1)
    assert state[5, 0, 1] == 1.0

