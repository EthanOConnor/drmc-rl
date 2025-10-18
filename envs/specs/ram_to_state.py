"""RAM → state-tensor mapper for Dr. Mario (visible info only).

This module converts NES RAM bytes into a 14-channel 16×8 state tensor:
- 3 channels for viruses (R/Y/B)
- 3 channels for fixed pill halves (R/Y/B)
- 3 channels for falling pill halves (R/Y/B)
- 1 channel for orientation (0 vertical / 1 horizontal), broadcast on falling pills area
- 4 scalar planes: gravity counter, lock/settle step proxy, level (normalized), and a spare.

Notes:
- Bottle encoding: high nibble encodes the object type, low nibble encodes color.
  See defines/drmario_constants.asm for masks and type codes.
- Offsets are provided by envs/specs/ram_offsets.json (generated from disassembly).
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def _read_byte(ram: bytes, addr_hex: str) -> int:
    return int(ram[int(addr_hex, 16)])


def ram_to_state(ram: bytes, offsets: Dict, *, H: int = 16, W: int = 8) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.uint8)
    base = int(offsets["bottle"]["base_addr"], 16)
    stride = int(offsets["bottle"]["stride"])
    for r in range(H):
        row = ram[base + r * stride : base + r * stride + W]
        grid[r, :] = np.frombuffer(row, dtype=np.uint8)

    # Channels
    C = 14
    state = np.zeros((C, H, W), dtype=np.float32)

    # Decode type/color from encoded field bytes (hi:type, lo:color)
    type_hi = (grid & 0xF0).astype(np.uint8)
    color_lo = (grid & 0x03).astype(np.uint8)  # yellow=0, red=1, blue=2

    # Type codes (see defines/drmario_constants.asm)
    T_VIRUS = 0xD0
    PILL_TYPES = {0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0}

    # Viruses channels (R/Y/B order)
    state[0, :, :] = ((type_hi == T_VIRUS) & (color_lo == 1)).astype(np.float32)  # red
    state[1, :, :] = ((type_hi == T_VIRUS) & (color_lo == 0)).astype(np.float32)  # yellow
    state[2, :, :] = ((type_hi == T_VIRUS) & (color_lo == 2)).astype(np.float32)  # blue

    # Fixed pill halves channels (R/Y/B order)
    pill_mask = np.isin(type_hi, list(PILL_TYPES))
    state[3, :, :] = (pill_mask & (color_lo == 1)).astype(np.float32)  # red
    state[4, :, :] = (pill_mask & (color_lo == 0)).astype(np.float32)  # yellow
    state[5, :, :] = (pill_mask & (color_lo == 2)).astype(np.float32)  # blue

    # Falling pill
    fr = _read_byte(ram, offsets["falling_pill"]["row_addr"]) if "falling_pill" in offsets else 0
    fc = _read_byte(ram, offsets["falling_pill"]["col_addr"]) if "falling_pill" in offsets else 0
    rotation = (
        _read_byte(ram, offsets["falling_pill"]["orient_addr"]) & 0x03
        if "falling_pill" in offsets
        else 0
    )
    orient = 1 - (rotation & 1)
    lc = _read_byte(ram, offsets["falling_pill"]["left_color_addr"]) if "falling_pill" in offsets else 0
    rc = _read_byte(ram, offsets["falling_pill"]["right_color_addr"]) if "falling_pill" in offsets else 0
    # Map colors to channels (R/Y/B indices 0..2)
    # Colors: yellow=0, red=1, blue=2; we keep R/Y/B order in channels
    color_map = {1: 0, 0: 1, 2: 2}  # red->0, yellow->1, blue->2 for falling channels
    # Place falling halves
    if 0 <= fc < W:
        base_row = (H - 1) - fr if 0 <= fr < H else None
        offsets = {
            0: ((0, 0), (0, 1)),   # horizontal, 1st color on left
            1: ((0, 0), (-1, 0)),  # vertical, 1st color on bottom
            2: ((0, 1), (0, 0)),   # horizontal, 1st color on right
            3: ((-1, 0), (0, 0)),  # vertical, 1st color on top
        }.get(rotation, ((0, 0), (0, 1)))
        if base_row is not None:
            for color, (dr, dc) in zip((lc, rc), offsets):
                channel_offset = color_map.get(color)
                if channel_offset is None:
                    continue
                row = base_row + dr
                col = fc + dc
                if 0 <= row < H and 0 <= col < W:
                    state[6 + channel_offset, row, col] = 1.0

    # Orientation plane
    state[9, :, :] = float(orient)

    # Scalars → broadcast planes
    grav = (
        _read_byte(ram, offsets["gravity_lock"]["gravity_counter_addr"]) / 255.0
        if "gravity_lock" in offsets
        else 0.0
    )
    lock = (
        _read_byte(ram, offsets["gravity_lock"]["lock_counter_addr"]) / 255.0
        if "gravity_lock" in offsets
        else 0.0
    )
    lvl = (_read_byte(ram, offsets["level"]["addr"]) / 20.0) if "level" in offsets else 0.0
    state[10, :, :] = grav
    state[11, :, :] = lock
    state[12, :, :] = lvl
    state[13, :, :] = 0.0  # spare/settle flag placeholder

    return state
