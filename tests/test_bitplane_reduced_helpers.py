from __future__ import annotations

import numpy as np

import envs.specs.ram_to_state as ram_specs


def test_reduced_bitplane_static_mask_is_derived_by_exclusion() -> None:
    prev = ram_specs.get_state_representation()
    ram_specs.set_state_representation("bitplane_reduced")
    try:
        C, H, W = int(ram_specs.STATE_CHANNELS), 16, 8
        assert C == 6
        frame = np.zeros((C, H, W), dtype=np.float32)

        ch = ram_specs.STATE_IDX

        # Static pill tile: has a color but no entity masks.
        frame[ch.color_channels[0], 10, 2] = 1.0  # red

        # Virus tile: should not be counted as "static pill".
        frame[ch.color_channels[2], 11, 3] = 1.0  # blue
        frame[ch.virus_mask, 11, 3] = 1.0

        # Falling pill tile: should not be counted as "static pill".
        frame[ch.color_channels[1], 12, 4] = 1.0  # yellow
        frame[ch.falling_mask, 12, 4] = 1.0

        # Preview tile: should not be counted as "static pill".
        frame[ch.color_channels[0], 0, 3] = 1.0  # red
        frame[ch.preview_mask, 0, 3] = 1.0

        static = ram_specs.get_static_mask(frame)
        assert bool(static[10, 2])
        assert not bool(static[11, 3])
        assert not bool(static[12, 4])
        assert not bool(static[0, 3])

        static_colors = ram_specs.get_static_color_planes(frame)
        assert static_colors.shape == (3, H, W)
        assert float(static_colors[:, 10, 2].sum()) == 1.0
        assert float(static_colors[:, 11, 3].sum()) == 0.0
        assert float(static_colors[:, 12, 4].sum()) == 0.0
        assert float(static_colors[:, 0, 3].sum()) == 0.0
    finally:
        ram_specs.set_state_representation(prev)

