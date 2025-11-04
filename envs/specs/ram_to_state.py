"""RAM → state-tensor mapper for Dr. Mario with pluggable representations.

Two representations are supported:

1. ``extended`` (default): 16-channel tensor with explicit planes for viruses,
   fixed/falling pills, orientation, scalar broadcasts, and preview metadata.
2. ``bitplane``: 12-channel tensor of mostly binary masks (color planes, entity
   masks, preview positions) plus scalar broadcasts.

Use :func:`set_state_representation` to switch modes at runtime. The change
applies process-wide (matching the previous environment-variable behaviour).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

STATE_HEIGHT = 16
STATE_WIDTH = 8

# Tile/type codes (matches disassembly constants).
T_VIRUS = 0xD0
PILL_TYPES = (0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0)
CLEARED_TILE = 0xB0
FIELD_JUST_EMPTIED = 0xF0
FIELD_EMPTY = 0xFF

COLOR_VALUE_TO_INDEX = {1: 0, 0: 1, 2: 2}  # red -> 0, yellow -> 1, blue -> 2

_PREVIEW_BASE = (0, 3)  # (row, col) centre for HUD preview
_PREVIEW_OFFSETS = {
    0: ((0, 0), (0, 1)),   # horizontal, first on left
    1: ((0, 0), (1, 0)),   # vertical, first on bottom
    2: ((0, 1), (0, 0)),   # horizontal, first on right
    3: ((1, 0), (0, 0)),   # vertical, first on top
}


def _normalize_mode(mode: Optional[str]) -> str:
    if not mode:
        return "extended"
    value = str(mode).strip().lower()
    if value not in {"extended", "bitplane", "policy_v1"}:
        raise ValueError(f"Unknown state representation '{mode}'")
    return value


def _build_state_index_extended() -> SimpleNamespace:
    return SimpleNamespace(
        color_channels=None,
        virus_color_channels=(0, 1, 2),
        static_color_channels=(3, 4, 5),
        falling_color_channels=(6, 7, 8),
        orientation=9,
        gravity=10,
        lock=11,
        level=12,
        preview_first=13,
        preview_second=14,
        preview_rotation=15,
        virus_mask=None,
        locked_mask=None,
        falling_mask=None,
        preview_mask=None,
        clearing_mask=None,
        empty_mask=None,
    )


def _build_state_index_bitplane() -> SimpleNamespace:
    return SimpleNamespace(
        color_channels=(0, 1, 2),
        virus_mask=3,
        locked_mask=4,
        falling_mask=5,
        preview_mask=6,
        clearing_mask=7,
        empty_mask=8,
        gravity=9,
        lock=10,
        level=11,
        virus_color_channels=None,
        static_color_channels=None,
        falling_color_channels=None,
        orientation=None,
        preview_first=None,
        preview_second=None,
        preview_rotation=None,
    )


def _build_state_index_policy_v1() -> SimpleNamespace:
    return SimpleNamespace(
        color_channels=None,
        virus_color_channels=(0, 1, 2),
        static_color_channels=(3, 4, 5),
        falling_color_channels=(6, 7, 8),
        level=9,
        preview_first=10,
        preview_second=11,
        clearing_mask=None,
        empty_mask=None,
    )





def _configure_state_representation(mode: str) -> None:
    global STATE_REPR, STATE_USE_BITPLANES, STATE_CHANNELS, STATE_IDX, STATE_FRAME_SHAPE
    mode_norm = _normalize_mode(mode)
    if "STATE_REPR" in globals() and STATE_REPR == mode_norm:
        return
    if mode_norm == "bitplane":
        STATE_REPR = "bitplane"
        STATE_USE_BITPLANES = True
        STATE_CHANNELS = 12
        STATE_IDX = _build_state_index_bitplane()
    elif mode_norm == "policy_v1":
        STATE_REPR = "policy_v1"
        STATE_USE_BITPLANES = False
        STATE_CHANNELS = 13
        STATE_IDX = _build_state_index_policy_v1()

    else:
        STATE_REPR = "extended"
        STATE_USE_BITPLANES = False
        STATE_CHANNELS = 16
        STATE_IDX = _build_state_index_extended()
    STATE_FRAME_SHAPE = (STATE_CHANNELS, STATE_HEIGHT, STATE_WIDTH)


def set_state_representation(mode: str) -> None:
    """Set the process-wide state representation (``extended`` or ``bitplane``)."""
    _configure_state_representation(mode)


def get_state_representation() -> str:
    """Return the current state representation name."""
    return STATE_REPR


# Initialise globals with default representation.
_configure_state_representation("extended")


def _read_byte(ram: bytes, addr_hex: str) -> int:
    return int(ram[int(addr_hex, 16)])


def _read_optional(ram: bytes, addr_hex: Optional[str]) -> Optional[int]:
    if not addr_hex:
        return None
    try:
        idx = int(addr_hex, 16)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(ram):
        return None
    return int(ram[idx])





def _ram_to_state_extended(
    ram: bytes,
    offsets: Dict,
    *,
    H: int = STATE_HEIGHT,
    W: int = STATE_WIDTH,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.uint8)
    base = int(offsets["bottle"]["base_addr"], 16)
    stride = int(offsets["bottle"]["stride"])
    for r in range(H):
        row = ram[base + r * stride : base + r * stride + W]
        grid[r, :] = np.frombuffer(row, dtype=np.uint8)

    state = np.zeros((STATE_CHANNELS, H, W), dtype=np.float32)
    type_hi = (grid & 0xF0).astype(np.uint8)
    color_lo = (grid & 0x03).astype(np.uint8)

    # Viruses (R/Y/B)
    for color_value, plane_idx in zip((1, 0, 2), STATE_IDX.virus_color_channels):
        state[plane_idx] = ((type_hi == T_VIRUS) & (color_lo == color_value)).astype(np.float32)

    # Fixed pill halves (R/Y/B)
    pill_mask = np.isin(type_hi, PILL_TYPES)
    for color_value, plane_idx in zip((1, 0, 2), STATE_IDX.static_color_channels):
        state[plane_idx] = (pill_mask & (color_lo == color_value)).astype(np.float32)

    # Falling pill halves (R/Y/B) injected via RAM registers
    falling_offsets = offsets.get("falling_pill", {}) if offsets is not None else {}
    fr = _read_optional(ram, falling_offsets.get("row_addr")) or 0
    fc = _read_optional(ram, falling_offsets.get("col_addr")) or 0
    rotation = (_read_optional(ram, falling_offsets.get("orient_addr")) or 0) & 0x03
    lc = _read_optional(ram, falling_offsets.get("left_color_addr")) or 0
    rc = _read_optional(ram, falling_offsets.get("right_color_addr")) or 0

    orient = 1 - (rotation & 1)
    if STATE_IDX.orientation is not None:
        state[STATE_IDX.orientation, :, :] = float(orient)

    if 0 <= fc < W:
        base_row = (H - 1) - fr if 0 <= fr < H else None
        offsets_local = {
            0: ((0, 0), (0, 1)),
            1: ((0, 0), (-1, 0)),
            2: ((0, 1), (0, 0)),
            3: ((-1, 0), (0, 0)),
        }.get(rotation, ((0, 0), (0, 1)))
        if base_row is not None:
            for color, (dr, dc) in zip((lc, rc), offsets_local):
                channel_offset = COLOR_VALUE_TO_INDEX.get(color)
                if channel_offset is None:
                    continue
                row = base_row + dr
                col = fc + dc
                if 0 <= row < H and 0 <= col < W:
                    target_idx = STATE_IDX.falling_color_channels[channel_offset]
                    state[target_idx, row, col] = 1.0

    # Scalars (gravity/lock/level)
    gravity = (
        _read_optional(ram, offsets.get("gravity_lock", {}).get("gravity_counter_addr")) or 0
    ) / 255.0
    lock = (
        _read_optional(ram, offsets.get("gravity_lock", {}).get("lock_counter_addr")) or 0
    ) / 255.0
    level = (_read_optional(ram, offsets.get("level", {}).get("addr")) or 0) / 20.0
    state[STATE_IDX.gravity, :, :] = gravity
    state[STATE_IDX.lock, :, :] = lock
    state[STATE_IDX.level, :, :] = level

    # Preview metadata (matches HUD preview only)
    preview_offsets = offsets.get("preview_pill", {}) if offsets is not None else {}
    next_left = _read_optional(ram, preview_offsets.get("left_color_addr"))
    next_right = _read_optional(ram, preview_offsets.get("right_color_addr"))
    next_rotation = _read_optional(ram, preview_offsets.get("rotation_addr"))

    def _normalize_color(color: Optional[int]) -> float:
        if color is None:
            return 0.0
        return float(color & 0x03) / 2.0

    def _normalize_rotation(rot: Optional[int]) -> float:
        if rot is None:
            return 0.0
        return float(rot & 0x03) / 3.0

    state[STATE_IDX.preview_first, :, :] = _normalize_color(next_left)
    state[STATE_IDX.preview_second, :, :] = _normalize_color(next_right)
    state[STATE_IDX.preview_rotation, :, :] = _normalize_rotation(next_rotation)

    return state


def _ram_to_state_bitplanes(
    ram: bytes,
    offsets: Dict,
    *,
    H: int = STATE_HEIGHT,
    W: int = STATE_WIDTH,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.uint8)
    base = int(offsets["bottle"]["base_addr"], 16)
    stride = int(offsets["bottle"]["stride"])
    for r in range(H):
        row = ram[base + r * stride : base + r * stride + W]
        grid[r, :] = np.frombuffer(row, dtype=np.uint8)

    state = np.zeros((STATE_CHANNELS, H, W), dtype=np.float32)
    type_hi = (grid & 0xF0).astype(np.uint8)
    color_lo = (grid & 0x03).astype(np.uint8)

    color_valid = (
        (type_hi != FIELD_EMPTY) & (type_hi != FIELD_JUST_EMPTIED) & (type_hi != 0x00)
    )
    for color_value, plane_idx in zip((1, 0, 2), STATE_IDX.color_channels):
        mask = (color_lo == color_value) & color_valid
        state[plane_idx] = mask.astype(np.float32)

    virus_mask = (type_hi == T_VIRUS)
    state[STATE_IDX.virus_mask] = virus_mask.astype(np.float32)

    locked_mask = np.isin(type_hi, PILL_TYPES)
    state[STATE_IDX.locked_mask] = locked_mask.astype(np.float32)

    clearing_mask = np.isin(type_hi, (CLEARED_TILE, FIELD_JUST_EMPTIED))
    state[STATE_IDX.clearing_mask] = clearing_mask.astype(np.float32)

    empty_mask = np.isin(type_hi, (FIELD_EMPTY, FIELD_JUST_EMPTIED, 0x00))
    state[STATE_IDX.empty_mask] = empty_mask.astype(np.float32)

    gravity = (
        _read_optional(ram, offsets.get("gravity_lock", {}).get("gravity_counter_addr")) or 0
    ) / 255.0
    lock = (
        _read_optional(ram, offsets.get("gravity_lock", {}).get("lock_counter_addr")) or 0
    ) / 255.0
    level = (_read_optional(ram, offsets.get("level", {}).get("addr")) or 0) / 20.0
    state[STATE_IDX.gravity, :, :] = gravity
    state[STATE_IDX.lock, :, :] = lock
    state[STATE_IDX.level, :, :] = level

    falling_offsets = offsets.get("falling_pill", {}) if offsets is not None else {}
    fr = _read_optional(ram, falling_offsets.get("row_addr")) or 0
    fc = _read_optional(ram, falling_offsets.get("col_addr")) or 0
    rotation = (_read_optional(ram, falling_offsets.get("orient_addr")) or 0) & 0x03
    lc = _read_optional(ram, falling_offsets.get("left_color_addr")) or 0
    rc = _read_optional(ram, falling_offsets.get("right_color_addr")) or 0

    if 0 <= fc < W:
        base_row = (H - 1) - fr if 0 <= fr < H else None
        offsets_local = {
            0: ((0, 0), (0, 1)),
            1: ((0, 0), (-1, 0)),
            2: ((0, 1), (0, 0)),
            3: ((-1, 0), (0, 0)),
        }.get(rotation, ((0, 0), (0, 1)))
        if base_row is not None:
            for color, (dr, dc) in zip((lc, rc), offsets_local):
                channel = COLOR_VALUE_TO_INDEX.get(color)
                if channel is None:
                    continue
                row = base_row + dr
                col = fc + dc
                if 0 <= row < H and 0 <= col < W:
                    state[STATE_IDX.falling_mask, row, col] = 1.0
                    state[STATE_IDX.color_channels[channel], row, col] = 1.0
                    state[STATE_IDX.empty_mask, row, col] = 0.0

    # Preview pill projected into spawn area (matches visible HUD preview exactly)
    preview_offsets = offsets.get("preview_pill", {}) if offsets is not None else {}
    next_left = _read_optional(ram, preview_offsets.get("left_color_addr"))
    next_right = _read_optional(ram, preview_offsets.get("right_color_addr"))
    next_rotation = (_read_optional(ram, preview_offsets.get("rotation_addr")) or 0) & 0x03

    preview_positions = _preview_positions(next_rotation)
    for color, (dr, dc) in zip((next_left, next_right), preview_positions):
        if color is None:
            continue
        channel = COLOR_VALUE_TO_INDEX.get(color & 0x03)
        if channel is None:
            continue
        row = _PREVIEW_BASE[0] + dr
        col = _PREVIEW_BASE[1] + dc
        if 0 <= row < H and 0 <= col < W:
            state[STATE_IDX.preview_mask, row, col] = 1.0
            state[STATE_IDX.color_channels[channel], row, col] = 1.0
            state[STATE_IDX.empty_mask, row, col] = 0.0

    return state


def ram_to_state(
    ram: bytes,
    offsets: Dict,
    *,
    H: int = STATE_HEIGHT,
    W: int = STATE_WIDTH,
) -> np.ndarray:
    """Decode NES RAM bytes into the configured state tensor representation."""
    if STATE_USE_BITPLANES:
        return _ram_to_state_bitplanes(ram, offsets, H=H, W=W)
    return _ram_to_state_extended(ram, offsets, H=H, W=W)


# ---------------------------------------------------------------------------
# Helper utilities to keep downstream code representation-agnostic.
# ---------------------------------------------------------------------------

def _ensure_np(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[1:] != (STATE_HEIGHT, STATE_WIDTH):
        raise ValueError(f"Expected (C,{STATE_HEIGHT},{STATE_WIDTH}), got {arr.shape}")
    return arr


def get_color_planes(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        return arr[list(STATE_IDX.color_channels)]
    planes = np.zeros((3, STATE_HEIGHT, STATE_WIDTH), dtype=np.float32)
    for group in ("virus_color_channels", "static_color_channels", "falling_color_channels"):
        indices = getattr(STATE_IDX, group)
        if not indices:
            continue
        planes = np.maximum(planes, arr[list(indices)])
    return planes


def get_static_color_planes(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        colors = arr[list(STATE_IDX.color_channels)]
        locked = arr[STATE_IDX.locked_mask]
        return colors * locked
    return arr[list(STATE_IDX.static_color_channels)]


def get_falling_color_planes(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        colors = arr[list(STATE_IDX.color_channels)]
        falling = arr[STATE_IDX.falling_mask]
        return colors * falling
    return arr[list(STATE_IDX.falling_color_channels)]


def get_virus_color_planes(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        colors = arr[list(STATE_IDX.color_channels)]
        virus = arr[STATE_IDX.virus_mask]
        return colors * virus
    return arr[list(STATE_IDX.virus_color_channels)]


def get_static_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        return arr[STATE_IDX.locked_mask] > 0.5
    return (arr[list(STATE_IDX.static_color_channels)] > 0.1).any(axis=0)


def get_falling_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        return arr[STATE_IDX.falling_mask] > 0.5
    return (arr[list(STATE_IDX.falling_color_channels)] > 0.1).any(axis=0)


def get_virus_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES:
        return arr[STATE_IDX.virus_mask] > 0.5
    return (arr[list(STATE_IDX.virus_color_channels)] > 0.1).any(axis=0)


def get_empty_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES and STATE_IDX.empty_mask is not None:
        return arr[STATE_IDX.empty_mask] > 0.5
    occupancy = get_occupancy_mask(arr)
    return ~occupancy


def _preview_positions(rotation: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return _PREVIEW_OFFSETS.get(rotation & 0x03, _PREVIEW_OFFSETS[0])


def decode_preview_from_state(frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
    arr = _ensure_np(frame)
    if STATE_USE_BITPLANES and STATE_IDX.preview_mask is not None:
        mask = arr[STATE_IDX.preview_mask] > 0.5
        if not mask.any():
            return None
        colors = arr[list(STATE_IDX.color_channels)] > 0.5
        color_hits = [int((colors[i] & mask).any()) for i in range(3)]
        if sum(color_hits) == 0:
            return None
        first = next((idx for idx, hit in enumerate(color_hits) if hit), None)
        second = next((idx for idx, hit in enumerate(color_hits) if hit and idx != first), first)
        return (first or 0, second or 0, 0)
    if STATE_IDX.preview_rotation is None:
        return None
    rotation = int(round(float(arr[STATE_IDX.preview_rotation, 0, 0]) * 3.0))
    first_color = int(round(float(arr[STATE_IDX.preview_first, 0, 0]) * 2.0))
    second_color = int(round(float(arr[STATE_IDX.preview_second, 0, 0]) * 2.0))
    return (first_color, second_color, rotation)


def get_preview_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    mask = np.zeros((STATE_HEIGHT, STATE_WIDTH), dtype=bool)
    info = decode_preview_from_state(arr)
    if info is None:
        return mask
    _, _, rotation = info
    for dr, dc in _preview_positions(rotation):
        row = _PREVIEW_BASE[0] + dr
        col = _PREVIEW_BASE[1] + dc
        if 0 <= row < STATE_HEIGHT and 0 <= col < STATE_WIDTH:
            mask[row, col] = True
    return mask


def get_gravity_value(frame: np.ndarray) -> float:
    arr = _ensure_np(frame)
    return float(arr[STATE_IDX.gravity, 0, 0]) if STATE_IDX.gravity is not None else 0.0


def get_lock_value(frame: np.ndarray) -> float:
    arr = _ensure_np(frame)
    return float(arr[STATE_IDX.lock, 0, 0]) if STATE_IDX.lock is not None else 0.0


def get_level_value(frame: np.ndarray) -> float:
    arr = _ensure_np(frame)
    return float(arr[STATE_IDX.level, 0, 0]) if STATE_IDX.level is not None else 0.0


def get_occupancy_mask(frame: np.ndarray) -> np.ndarray:
    arr = _ensure_np(frame)
    occ = get_static_mask(arr) | get_virus_mask(arr) | get_falling_mask(arr)
    if STATE_USE_BITPLANES and STATE_IDX.preview_mask is not None:
        occ = occ | (arr[STATE_IDX.preview_mask] > 0.5)
    return occ


def extended_to_policy_v2(state: np.ndarray) -> np.ndarray:
    """Converts a 16-channel extended state to an 8-channel policy_v2 state."""
    C, H, W = state.shape
    if C != 16:
        # This is not an extended state, so we can't convert it.
        # For now, we'll just return the state as is.
        return state

    policy_state = np.zeros((8, H, W), dtype=state.dtype)

    # color_channels (3 channels)
    virus_colors = state[0:3]
    static_colors = state[3:6]
    policy_state[0:3] = np.maximum(virus_colors, static_colors)

    # virus_mask (1 channel)
    policy_state[3] = (virus_colors > 0.1).any(axis=0)

    # locked_mask (1 channel)
    policy_state[4] = (static_colors > 0.1).any(axis=0)

    # level (1 channel)
    policy_state[5] = state[12]

    # preview_first (1 channel)
    policy_state[6] = state[13]

    # preview_second (1 channel)
    policy_state[7] = state[14]

    return policy_state


__all__ = [
    "ram_to_state",
    "set_state_representation",
    "get_state_representation",
    "STATE_REPR",
    "STATE_USE_BITPLANES",
    "STATE_CHANNELS",
    "STATE_HEIGHT",
    "STATE_WIDTH",
    "STATE_FRAME_SHAPE",
    "STATE_IDX",
    "get_color_planes",
    "get_static_color_planes",
    "get_falling_color_planes",
    "get_virus_color_planes",
    "get_static_mask",
    "get_falling_mask",
    "get_virus_mask",
    "get_empty_mask",
    "get_preview_mask",
    "get_occupancy_mask",
    "get_gravity_value",
    "get_lock_value",
    "get_level_value",
    "decode_preview_from_state",
    "extended_to_policy_v2",
]
