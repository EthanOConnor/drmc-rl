from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path
from typing import Optional, Tuple


class DrMarioStatePy(ctypes.Structure):
    """ctypes mirror of `DrMarioState` in `game_engine/GameState.h`.

    This module intentionally lives under `envs.backends` so it is importable from
    multiprocessing spawn workers (the `game_engine/` package is not installed by
    default).
    """

    _fields_ = [
        ("buttons", ctypes.c_uint8),
        ("buttons_prev", ctypes.c_uint8),
        ("buttons_pressed", ctypes.c_uint8),
        ("buttons_held", ctypes.c_uint8),
        ("control_flags", ctypes.c_uint8),
        ("next_action", ctypes.c_uint8),
        ("board", ctypes.c_uint8 * 128),
        ("falling_pill_row", ctypes.c_uint8),
        ("falling_pill_col", ctypes.c_uint8),
        ("falling_pill_orient", ctypes.c_uint8),
        ("falling_pill_color_l", ctypes.c_uint8),
        ("falling_pill_color_r", ctypes.c_uint8),
        ("falling_pill_size", ctypes.c_uint8),
        ("preview_pill_color_l", ctypes.c_uint8),
        ("preview_pill_color_r", ctypes.c_uint8),
        ("preview_pill_rotation", ctypes.c_uint8),
        ("preview_pill_size", ctypes.c_uint8),
        ("mode", ctypes.c_uint8),
        ("stage_clear", ctypes.c_uint8),
        ("ending_active", ctypes.c_uint8),
        ("level_fail", ctypes.c_uint8),
        ("pill_counter", ctypes.c_uint8),
        ("pill_counter_total", ctypes.c_uint16),
        ("level", ctypes.c_uint8),
        ("speed_setting", ctypes.c_uint8),
        ("viruses_remaining", ctypes.c_uint8),
        ("speed_ups", ctypes.c_uint8),
        ("wait_frames", ctypes.c_uint8),
        ("lock_counter", ctypes.c_uint8),
        ("speed_counter", ctypes.c_uint8),
        ("hor_velocity", ctypes.c_uint8),
        ("rng_state", ctypes.c_uint8 * 2),
        ("rng_override", ctypes.c_uint8),
        ("frame_count", ctypes.c_uint32),
        ("frame_budget", ctypes.c_uint32),
        ("fail_count", ctypes.c_uint32),
        ("last_fail_frame", ctypes.c_uint32),
        ("last_fail_row", ctypes.c_uint8),
        ("last_fail_col", ctypes.c_uint8),
        ("spawn_delay", ctypes.c_uint8),  # Frames before new pill can be controlled
        ("reset_wait_frames", ctypes.c_uint8),
        ("reset_framecounter_lo_plus1", ctypes.c_uint16),
        # Batched stepping (driver↔engine). See `game_engine/GameState.h`.
        ("run_request_id", ctypes.c_uint32),
        ("run_ack_id", ctypes.c_uint32),
        ("run_mode", ctypes.c_uint32),
        ("run_frames", ctypes.c_uint32),
        ("run_frames_executed", ctypes.c_uint32),
        ("run_tiles_cleared_total", ctypes.c_uint32),
        ("run_tiles_cleared_virus", ctypes.c_uint32),
        ("run_tiles_cleared_nonvirus", ctypes.c_uint32),
        ("run_buttons", ctypes.c_uint8),
        ("run_last_spawn_id", ctypes.c_uint8),
        ("run_reason", ctypes.c_uint8),
        ("run_reserved", ctypes.c_uint8),
    ]


SHM_SIZE = ctypes.sizeof(DrMarioStatePy)


def open_shared_memory_file(
    path: Path, *, size: Optional[int] = None
) -> Tuple[mmap.mmap, DrMarioStatePy]:
    """Open an engine shared-memory mapping backed by a concrete file path."""

    if size is None:
        size = int(SHM_SIZE)
    p = Path(path)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        current_size = os.path.getsize(p)
        if current_size < size:
            os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    state = DrMarioStatePy.from_buffer(mm)
    return mm, state

