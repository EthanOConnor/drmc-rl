from __future__ import annotations

import pytest

import envs.specs.ram_to_state as ram_specs


@pytest.mark.parametrize("state_repr", ["extended", "bitplane"])
def test_plane_names_match_channel_count(state_repr: str) -> None:
    prev = ram_specs.get_state_representation()
    try:
        ram_specs.set_state_representation(state_repr)
        names = ram_specs.get_plane_names()
        assert len(names) == int(ram_specs.STATE_CHANNELS)
        assert len(set(names)) == len(names)
    finally:
        ram_specs.set_state_representation(prev)

