import ctypes
import ctypes.util
import mmap
import os
from pathlib import Path
from typing import Tuple

# Shared memory configuration (must match C++)
SHM_NAME = "/drmario_shm"
SHM_SIZE = 180


class DrMarioStatePy(ctypes.Structure):
    _fields_ = [
    ("buttons", ctypes.c_uint8),
    ("buttons_prev", ctypes.c_uint8),
    ("buttons_pressed", ctypes.c_uint8),
    ("buttons_held", ctypes.c_uint8),
    ("control_flags", ctypes.c_uint8),
    ("_pad0", ctypes.c_uint8),
    ("board", ctypes.c_uint8 * 128),
    ("falling_pill_row", ctypes.c_uint8),
        ("falling_pill_col", ctypes.c_uint8),
        ("falling_pill_orient", ctypes.c_uint8),
        ("falling_pill_color_l", ctypes.c_uint8),
        ("falling_pill_color_r", ctypes.c_uint8),
        ("falling_pill_size", ctypes.c_uint8),
        ("preview_pill_color_l", ctypes.c_uint8),
        ("preview_pill_color_r", ctypes.c_uint8),
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
        ("gravity_counter", ctypes.c_uint8),
        ("lock_counter", ctypes.c_uint8),
        ("speed_counter", ctypes.c_uint8),
        ("hor_velocity", ctypes.c_uint8),
    ("rng_state", ctypes.c_uint8 * 2),
    ("frame_count", ctypes.c_uint32),
    ("frame_budget", ctypes.c_uint32),
    ("fail_count", ctypes.c_uint32),
    ("last_fail_frame", ctypes.c_uint32),
    ("last_fail_row", ctypes.c_uint8),
    ("last_fail_col", ctypes.c_uint8),
    ("spawn_delay", ctypes.c_uint8),  # Frames before new pill can be controlled
    ("_pad1", ctypes.c_uint8 * 1),
]


def _load_libc() -> ctypes.CDLL:
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        libc_path = "libc.dylib"
    return ctypes.CDLL(libc_path, use_errno=True)


def open_shared_memory(
    name: str = SHM_NAME, size: int = SHM_SIZE
) -> Tuple[mmap.mmap, DrMarioStatePy]:
    shm_file_env = os.environ.get("DRMARIO_SHM_FILE")
    if shm_file_env:
        fd = os.open(shm_file_env, os.O_RDWR | os.O_CREAT, 0o666)
        current_size = os.path.getsize(shm_file_env)
        if current_size < size:
            os.ftruncate(fd, size)
        mm = mmap.mmap(
            fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)
        state = DrMarioStatePy.from_buffer(mm)
        return mm, state

    libc = _load_libc()
    shm_open = libc.shm_open
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    shm_open.restype = ctypes.c_int

    fd = shm_open(name.encode(), os.O_RDWR, 0o666)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"shm_open({name}) failed")

    mm = mmap.mmap(
        fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
    )
    os.close(fd)
    state = DrMarioStatePy.from_buffer(mm)
    return mm, state


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
