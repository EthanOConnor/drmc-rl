from __future__ import annotations

"""Native reachability accelerator for the NES-accurate placement planner.

The reference implementation in :mod:`envs.retro.fast_reach` is correct but can
be too slow for training-time use because it performs a full BFS over the
falling-pill per-frame state for every pill spawn.

This module wraps a small C helper (``reach_native/drm_reach_full.c``) that
implements the same BFS in native code and returns, for every locked base pose
``(x, y, rot)``:
  - the minimal frame cost (tau),
  - a compact controller script (per-frame action indices) achieving that pose.

The planner treats this backend as an optional accelerator: if the shared
library is missing, callers should fall back to the Python reference.
"""

import ctypes as C
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from envs.retro.fast_reach import FrameState, HoldDir

GRID_W = 8
GRID_H = 16
POSES = 4 * GRID_H * GRID_W  # (rot, y, x)


class NativeReachError(RuntimeError):
    """Raised when the native reachability helper fails."""

@dataclass(frozen=True)
class NativeReachStats:
    """Optional instrumentation for the last native BFS call.

    Enable collection by setting `DRMARIO_REACH_STATS=1` in the environment.
    """

    visited_states: int
    expanded_states: int
    transitions: int
    locks_found: int
    queue_nodes_enqueued: int
    queue_nodes_expanded: int
    max_depth: int
    depth_processed: int
    wanted_count: int
    found_wanted: int


def _default_library_name() -> str:
    if sys.platform == "darwin":
        return "libdrm_reach_full.dylib"
    if sys.platform.startswith("linux"):
        return "libdrm_reach_full.so"
    if sys.platform == "win32":
        return "drm_reach_full.dll"
    raise RuntimeError(f"Unsupported platform for native reachability: {sys.platform!r}")


def default_library_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "reach_native" / "build" / _default_library_name()


def resolve_library_path(path: Optional[str] = None) -> Path:
    """Return the shared library path to load.

    Priority:
      1) explicit ``path`` argument
      2) env var ``DRMARIO_REACH_LIB``
      3) repo-local default under ``reach_native/build/``
    """

    if path:
        return Path(path).expanduser()
    env = os.environ.get("DRMARIO_REACH_LIB")
    if env:
        return Path(env).expanduser()
    return default_library_path()


def is_library_present(path: Optional[str] = None) -> bool:
    try:
        return resolve_library_path(path).is_file()
    except Exception:
        return False


@dataclass(frozen=True)
class NativeReachability:
    """View of native reachability outputs for one spawn.

    Note: instances may reference buffers that are re-used by the runner that
    produced them. Treat them as valid until the next native BFS call from that
    runner.
    """

    costs_u16: np.ndarray  # (512,) uint16; 0xFFFF == unreachable
    offsets_u16: np.ndarray  # (512,) uint16
    lengths_u16: np.ndarray  # (512,) uint16; 0 == unreachable
    script_buf: np.ndarray  # (used,) uint8

    @staticmethod
    def pose_index(x: int, y: int, rot: int) -> int:
        return int((int(rot) & 3) * (GRID_H * GRID_W) + int(y) * GRID_W + int(x))

    def script_for_pose(self, x: int, y: int, rot: int) -> Optional[np.ndarray]:
        idx = self.pose_index(x, y, rot)
        if idx < 0 or idx >= POSES:
            return None
        length = int(self.lengths_u16[idx])
        if length <= 0:
            return None
        offset = int(self.offsets_u16[idx])
        end = offset + length
        if offset < 0 or end > int(self.script_buf.shape[0]):
            return None
        return self.script_buf[offset:end]

    def cost_for_pose(self, x: int, y: int, rot: int) -> Optional[int]:
        idx = self.pose_index(x, y, rot)
        if idx < 0 or idx >= POSES:
            return None
        cost = int(self.costs_u16[idx])
        if cost == 0xFFFF:
            return None
        return cost


_CDLL_CACHE: dict[str, C.CDLL] = {}


def _load_cdll(path: Path) -> C.CDLL:
    key = str(path)
    cached = _CDLL_CACHE.get(key)
    if cached is not None:
        return cached
    lib = C.CDLL(str(path))
    _CDLL_CACHE[key] = lib
    return lib


class NativeReachabilityRunner:
    """Runner that owns buffers and executes the native BFS."""

    def __init__(self, *, max_frames: int = 2048, lib_path: Optional[str] = None) -> None:
        self._max_frames = int(max(1, max_frames))
        path = resolve_library_path(lib_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Native reachability library not found at {path}. "
                "Build it with: python -m tools.build_reach_native"
            )
        self._lib = _load_cdll(path)

        fn = getattr(self._lib, "drm_reach_bfs_full", None)
        if fn is None:
            raise NativeReachError(f"{path} does not export drm_reach_bfs_full")

        # C signature:
        # int drm_reach_bfs_full(
        #   const uint16_t cols[8],
        #   int sx,int sy,int srot,int sc,int hv,int hold_dir,int parity,int rot_hold,
        #   int speed_threshold,int max_frames,
        #   uint16_t out_costs[512], uint16_t out_offsets[512], uint16_t out_lengths[512],
        #   uint8_t* out_script_buf, int script_buf_cap, int* out_script_used)
        fn.argtypes = [
            C.POINTER(C.c_uint16),
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_uint16),
            C.POINTER(C.c_uint16),
            C.POINTER(C.c_uint16),
            C.POINTER(C.c_uint8),
            C.c_int,
            C.POINTER(C.c_int),
        ]
        fn.restype = C.c_int
        self._fn = fn

        # Optional stats hook (may be missing in older builds).
        self._stats_fn = None
        stats_fn = getattr(self._lib, "drm_reach_get_last_stats", None)
        if stats_fn is not None:

            class _DrmReachStats(C.Structure):
                _fields_ = [
                    ("visited_states", C.c_uint32),
                    ("expanded_states", C.c_uint32),
                    ("transitions", C.c_uint32),
                    ("locks_found", C.c_uint32),
                    ("queue_nodes_enqueued", C.c_uint32),
                    ("queue_nodes_expanded", C.c_uint32),
                    ("max_depth", C.c_uint16),
                    ("depth_processed", C.c_uint16),
                    ("wanted_count", C.c_uint16),
                    ("found_wanted", C.c_uint16),
                ]

            stats_fn.argtypes = [C.POINTER(_DrmReachStats), C.c_int]
            stats_fn.restype = C.c_int
            self._stats_fn = stats_fn
            self._stats_struct = _DrmReachStats()  # type: ignore[attr-defined]

        self._out_costs = np.empty(POSES, dtype=np.uint16)
        self._out_offsets = np.empty(POSES, dtype=np.uint16)
        self._out_lengths = np.empty(POSES, dtype=np.uint16)

        # Worst-case upper bound: every pose uses max_frames actions.
        self._script_cap = int(POSES * self._max_frames)
        self._script_buf = np.empty(self._script_cap, dtype=np.uint8)
        self._used = C.c_int(0)

    @property
    def max_frames(self) -> int:
        return self._max_frames

    def bfs_full(
        self,
        cols_u16: np.ndarray,
        spawn: FrameState,
        *,
        speed_threshold: int,
    ) -> NativeReachability:
        cols = np.asarray(cols_u16, dtype=np.uint16).reshape(-1)
        if cols.shape[0] != GRID_W:
            raise ValueError(f"Expected cols_u16 shape (8,), got {cols.shape!r}")
        cols = np.ascontiguousarray(cols)

        hold_dir = int(getattr(spawn.hold_dir, "value", spawn.hold_dir))  # enum or int
        parity = int(spawn.frame_parity) & 1
        rot_hold = int(getattr(spawn.rot_hold, "value", spawn.rot_hold))  # enum or int

        rc = int(
            self._fn(
                cols.ctypes.data_as(C.POINTER(C.c_uint16)),
                int(spawn.x),
                int(spawn.y),
                int(spawn.rot) & 3,
                int(spawn.speed_counter),
                int(spawn.hor_velocity),
                int(hold_dir),
                int(parity),
                int(rot_hold),
                int(speed_threshold),
                int(self._max_frames),
                self._out_costs.ctypes.data_as(C.POINTER(C.c_uint16)),
                self._out_offsets.ctypes.data_as(C.POINTER(C.c_uint16)),
                self._out_lengths.ctypes.data_as(C.POINTER(C.c_uint16)),
                self._script_buf.ctypes.data_as(C.POINTER(C.c_uint8)),
                int(self._script_cap),
                C.byref(self._used),
            )
        )
        if rc != 0:
            raise NativeReachError(f"Native reachability failed with error code {rc}")

        used = int(self._used.value)
        used = max(0, min(used, int(self._script_cap)))
        return NativeReachability(
            costs_u16=self._out_costs,
            offsets_u16=self._out_offsets,
            lengths_u16=self._out_lengths,
            script_buf=self._script_buf[:used],
        )

    def get_last_stats(self) -> Optional[NativeReachStats]:
        if self._stats_fn is None:
            return None
        st = getattr(self, "_stats_struct", None)
        if st is None:
            return None
        rc = int(self._stats_fn(C.byref(st), C.sizeof(st)))
        if rc != 0:
            return None
        return NativeReachStats(
            visited_states=int(st.visited_states),
            expanded_states=int(st.expanded_states),
            transitions=int(st.transitions),
            locks_found=int(st.locks_found),
            queue_nodes_enqueued=int(st.queue_nodes_enqueued),
            queue_nodes_expanded=int(st.queue_nodes_expanded),
            max_depth=int(st.max_depth),
            depth_processed=int(st.depth_processed),
            wanted_count=int(st.wanted_count),
            found_wanted=int(st.found_wanted),
        )


__all__ = [
    "NativeReachError",
    "NativeReachability",
    "NativeReachabilityRunner",
    "NativeReachStats",
    "default_library_path",
    "resolve_library_path",
    "is_library_present",
]
