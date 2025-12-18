from __future__ import annotations

from envs.retro.drmario_env import DrMarioRetroEnv
from envs.specs import ram_to_state as ram_specs
from envs.state_core import build_state


def _make_env_with_bottle(bottle: bytes) -> DrMarioRetroEnv:
    assert len(bottle) == 16 * 8
    env = DrMarioRetroEnv(obs_mode="state", backend="mock", auto_start=False)
    ram = bytearray([0x00] * 0x800)
    ram[0x0400 : 0x0400 + len(bottle)] = bottle
    env._state_cache = build_state(
        ram_bytes=bytes(ram),
        ram_offsets=env._ram_offsets,
        prev_stack4=None,
        t=0,
        elapsed_frames=0,
        frame_skip=1,
        last_terminal=None,
    )
    env._gameplay_active = True
    return env


def test_count_clearing_tiles_excludes_empty_ff() -> None:
    bottle = bytes([ram_specs.FIELD_EMPTY] * (16 * 8))
    env = _make_env_with_bottle(bottle)
    assert env._count_clearing_tiles() == 0


def test_count_clearing_tiles_counts_cleared_and_just_emptied() -> None:
    bottle = bytearray([ram_specs.FIELD_EMPTY] * (16 * 8))
    # Cleared tiles: 0xB0..0xB2 appear during clear animation (color encoded in low bits).
    bottle[0:4] = bytes([0xB0, 0xB1, 0xB2, 0xB0])
    env = _make_env_with_bottle(bytes(bottle))
    assert env._count_clearing_tiles() == 4

    # Just-emptied tiles: 0xF0..0xF2 appear after clear before settling back to 0xFF.
    bottle2 = bytearray([ram_specs.FIELD_EMPTY] * (16 * 8))
    bottle2[0:4] = bytes([0xF0, 0xF1, 0xF2, 0xF0])
    env2 = _make_env_with_bottle(bytes(bottle2))
    assert env2._count_clearing_tiles() == 4

