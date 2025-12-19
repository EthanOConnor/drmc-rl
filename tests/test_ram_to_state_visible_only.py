import numpy as np
import pytest

import envs.specs.ram_to_state as ram_specs


@pytest.mark.parametrize(
    "state_repr",
    [
        "extended",
        "bitplane",
        "bitplane_bottle",
        "bitplane_bottle_mask",
        "bitplane_reduced",
        "bitplane_reduced_mask",
    ],
)
def test_shapes_and_visibility(state_repr: str) -> None:
    prev = ram_specs.get_state_representation()
    ram_specs.set_state_representation(state_repr)
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
    try:
        state = ram_specs.ram_to_state(ram, offsets)
        assert state.shape == (ram_specs.STATE_CHANNELS, 16, 8)
        # No future info present by construction
        assert (state[0] >= 0).all()
        if ram_specs.STATE_USE_BITPLANES:
            idx = ram_specs.STATE_IDX
            if idx.virus_mask is not None:
                assert not state[idx.virus_mask].any()
            if getattr(idx, "locked_mask", None) is not None:
                assert not state[idx.locked_mask].any()
        else:
            idx = ram_specs.STATE_IDX
            assert idx.preview_first is not None
            assert np.allclose(state[idx.preview_first], 0.0)
            assert idx.preview_second is not None
            assert np.allclose(state[idx.preview_second], 0.0)
            assert getattr(idx, "preview_rotation", None) is not None
            assert np.allclose(state[idx.preview_rotation], 0.0)
    finally:
        ram_specs.set_state_representation(prev)
