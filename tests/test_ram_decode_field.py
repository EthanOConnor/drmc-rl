import numpy as np
import pytest

import envs.specs.ram_to_state as ram_specs


@pytest.mark.parametrize("state_repr", ["extended", "bitplane"])
def test_field_encoding_decodes_to_channels(state_repr: str) -> None:
    ram_specs.set_state_representation(state_repr)
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
        "preview_pill": {
            "left_color_addr": hex(0x031A),
            "right_color_addr": hex(0x031B),
            "rotation_addr": hex(0x0322),
        },
    }

    # Upcoming pill preview: left=red(1), right=blue(2), rotation=3
    ram[0x031A] = 0x01
    ram[0x031B] = 0x02
    ram[0x0322] = 0x03

    state = ram_specs.ram_to_state(bytes(ram), offsets)
    assert state.shape == (ram_specs.STATE_CHANNELS, 16, 8)

    if ram_specs.STATE_USE_BITPLANES:
        idx = ram_specs.STATE_IDX
        red_idx, yellow_idx, blue_idx = idx.color_channels
        assert np.isclose(state[red_idx, 0, 0], 1.0)
        assert idx.virus_mask is not None
        assert np.isclose(state[idx.virus_mask, 0, 0], 1.0)
        assert np.isclose(state[blue_idx, 0, 1], 1.0)
        assert idx.locked_mask is not None
        assert np.isclose(state[idx.locked_mask, 0, 1], 1.0)
        preview_mask = ram_specs.get_preview_mask(state)
        assert int(preview_mask.sum()) == 2
    else:
        idx = ram_specs.STATE_IDX
        virus_red_idx = idx.virus_color_channels[0]
        static_blue_idx = idx.static_color_channels[2]
        assert np.isclose(state[virus_red_idx, 0, 0], 1.0)
        assert np.isclose(state[static_blue_idx, 0, 1], 1.0)
        assert idx.preview_first is not None
        assert np.isclose(state[idx.preview_first, 0, 0], 0.5)  # red -> 1 / 2
        assert idx.preview_second is not None
        assert np.isclose(state[idx.preview_second, 0, 0], 1.0)  # blue -> 2 / 2
        assert idx.preview_rotation is not None
        assert np.isclose(state[idx.preview_rotation, 0, 0], 1.0)  # rotation 3 -> 3 / 3
