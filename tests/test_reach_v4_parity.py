"""Fuzz parity: drm_reach_bfs_v4 must match drm_reach_bfs_full exactly.

v4 (greedy-witness upper bounds + admissible lower-bound pruning) is the
production planner for the cpp-pool warp path. v1 (drm_reach_bfs_full) is the
oracle. Costs for all *in-bounds* poses must be identical; offscreen poses may
legitimately differ at the early-exit depth and are never consumed by the
macro env.
"""

from __future__ import annotations

import ctypes as C

import numpy as np
import pytest

from envs.backends.drmario_pool import default_library_path

GRID_W, GRID_H = 8, 16
SCRIPT_CAP = 512 * 2048


def _load():
    path = default_library_path()
    if not path.is_file():
        pytest.skip("pool library not built (python -m tools.build_drmario_pool)")
    lib = C.CDLL(str(path))
    f1 = lib.drm_reach_bfs_full
    f1.restype = C.c_int
    f1.argtypes = [C.POINTER(C.c_uint16)] + [C.c_int] * 8 + [
        C.c_int, C.c_int,
        C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), C.POINTER(C.c_uint16),
        C.POINTER(C.c_uint8), C.c_int, C.POINTER(C.c_int),
    ]
    f4 = lib.drm_reach_bfs_v4
    f4.restype = C.c_int
    f4.argtypes = [C.POINTER(C.c_uint16)] + [C.c_int] * 8 + [
        C.c_int, C.c_int, C.POINTER(C.c_uint16),
    ]
    return f1, f4


def _in_bounds_mask() -> np.ndarray:
    ib = np.zeros(512, dtype=bool)
    for rot in range(4):
        for y in range(GRID_H):
            for x in range(GRID_W):
                idx = rot * 128 + y * 8 + x
                ib[idx] = (x < 7) if (rot % 2 == 0) else (y >= 1)
    return ib


def _rand_board(rng: np.random.Generator, kind: int) -> np.ndarray:
    cols = np.zeros(8, dtype=np.uint16)
    if kind == 0:
        pass  # empty
    elif kind == 2:
        for x in range(8):
            h = int(rng.integers(0, 13))
            for y in range(16 - h, 16):
                cols[x] |= np.uint16(1 << y)
        for _ in range(int(rng.integers(0, 12))):
            x = int(rng.integers(0, 8))
            y = int(rng.integers(4, 16))
            cols[x] &= np.uint16(~(1 << y) & 0xFFFF)
    else:
        density = rng.uniform(0.05, 0.25) if kind == 1 else rng.uniform(0.3, 0.6)
        for x in range(8):
            for y in range(16):
                if rng.random() < density:
                    cols[x] |= np.uint16(1 << y)
    # keep the spawn cells open
    cols[3] &= np.uint16(~1 & 0xFFFF)
    cols[4] &= np.uint16(~1 & 0xFFFF)
    return cols


def test_v4_matches_oracle_on_fuzzed_boards():
    f1, f4 = _load()
    ib = _in_bounds_mask()

    offs = np.zeros(512, dtype=np.uint16)
    lens = np.zeros(512, dtype=np.uint16)
    buf = np.zeros(SCRIPT_CAP, dtype=np.uint8)
    used = C.c_int(0)

    def u16p(a):
        return a.ctypes.data_as(C.POINTER(C.c_uint16))

    rng = np.random.default_rng(20260609)
    for trial in range(120):
        cols = _rand_board(rng, trial % 4)
        thr = int(rng.choice([1, 5, 13, 21, 37, 69, 127]))
        sc = int(rng.integers(0, thr + 1))
        hv = int(rng.integers(0, 16))
        hd = int(rng.choice([0, 0, 0, 1, 2]))
        p = int(rng.integers(0, 2))
        rh = int(rng.choice([0, 0, 0, 1, 2]))

        c1 = np.full(512, 0xFFFF, dtype=np.uint16)
        rc = f1(u16p(cols), 3, 0, 0, sc, hv, hd, p, rh, thr, 2048,
                u16p(c1), u16p(offs), u16p(lens),
                buf.ctypes.data_as(C.POINTER(C.c_uint8)), SCRIPT_CAP, C.byref(used))
        assert rc == 0

        c4 = np.full(512, 0xFFFF, dtype=np.uint16)
        rc = f4(u16p(cols), 3, 0, 0, sc, hv, hd, p, rh, thr, 2048, u16p(c4))
        assert rc == 0

        if not np.array_equal(c1[ib], c4[ib]):
            bad = np.flatnonzero(ib & (c1 != c4))
            pytest.fail(
                f"trial={trial} thr={thr} sc={sc} hv={hv} hd={hd} p={p} rh={rh}: "
                f"{bad.size} in-bounds pose costs differ, e.g. pose {bad[0]} "
                f"oracle={c1[bad[0]]} v4={c4[bad[0]]}"
            )
