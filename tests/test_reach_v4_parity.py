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

from drmc_rl.envs.backends.drmario_pool import default_library_path
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, simulate_frame

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


@pytest.mark.parametrize("x,rot,held", [(0, 0, 1), (0, 1, 1), (6, 0, 2), (7, 1, 2)])
@pytest.mark.parametrize("velocity", [0, 10, 15])
def test_all_native_variants_and_witnesses_at_bottle_boundaries(x, rot, held, velocity):
    oracle, v4 = _load()
    lib = C.CDLL(str(default_library_path()))
    v2, v3 = lib.drm_reach_bfs_v2, lib.drm_reach_bfs_v3
    v2.restype, v2.argtypes = oracle.restype, oracle.argtypes
    v3.restype, v3.argtypes = v4.restype, v4.argtypes
    cols = np.zeros(8, dtype=np.uint16)
    # An uneven floor exercises rotations and collision-charged repeats after
    # leaving the wall, while the root is clear in both orientations.
    cols[[1, 3, 5]] = (1 << 13) | (1 << 14) | (1 << 15)
    root = (x, 5, rot, 2, velocity, held, 0, 0, 5, 256)
    ptr = lambda a: a.ctypes.data_as(C.POINTER(C.c_uint16))
    costs = np.full(512, 0xFFFF, dtype=np.uint16)
    offsets = np.zeros(512, dtype=np.uint16)
    lengths = np.zeros(512, dtype=np.uint16)
    scripts = np.zeros(SCRIPT_CAP, dtype=np.uint8)
    used = C.c_int()
    args = (ptr(cols), *root, ptr(costs), ptr(offsets), ptr(lengths),
            scripts.ctypes.data_as(C.POINTER(C.c_uint8)), SCRIPT_CAP, C.byref(used))
    assert oracle(*args) == 0
    legal = _in_bounds_mask()
    for variant in (v2, v3, v4):
        actual = np.full(512, 0xFFFF, dtype=np.uint16)
        tail = (None, None, None, 0, None) if variant is v2 else ()
        assert variant(ptr(cols), *root, ptr(actual), *tail) == 0
        np.testing.assert_array_equal(actual[legal], costs[legal])
    for pose in np.flatnonzero(legal & (costs != 0xFFFF)):
        state = FrameState(x=x, y=5, rot=rot, speed_counter=2,
            hor_velocity=velocity, hold_dir=HoldDir(held), frame_parity=0,
            rot_hold=Rotation.NONE)
        script = scripts[int(offsets[pose]):int(offsets[pose]) + int(lengths[pose])]
        for elapsed, action in enumerate(script, 1):
            assert not state.locked
            state = simulate_frame(cols, state, int(action), speed_threshold=5)
        assert state.locked and elapsed == costs[pose]
        assert state.rot * 128 + state.y * 8 + state.x == pose


def test_geometric_bounds_keep_the_combined_slide_and_rotation_edge():
    from tools.test_reach_cuda_parity import load_cpu_debug, regression_cases

    case = regression_cases()[0]
    debug = load_cpu_debug()
    wanted = np.zeros(512, dtype=np.uint8)
    upper = np.zeros(512, dtype=np.uint16)
    distance = np.zeros((512, 512), dtype=np.uint8)
    ptr16 = lambda a: a.ctypes.data_as(C.POINTER(C.c_uint16))
    ptr8 = lambda a: a.ctypes.data_as(C.POINTER(C.c_uint8))
    assert debug(ptr16(case["cols"]), 3, 0, 3, 52, 4, 1, 1, 2, 17,
                 ptr8(wanted), ptr16(upper), ptr8(distance), 512) > 0
    target = list(np.flatnonzero(wanted)).index(272)  # (x=0, y=2, rot=2)
    assert distance[target, 402] == 1  # (x=2, y=2, rot=3): left + rotation
    _, v4 = _load()
    costs = np.full(512, 0xFFFF, dtype=np.uint16)
    assert v4(ptr16(case["cols"]), 3, 0, 3, 52, 4, 1, 1, 2, 17, 256, ptr16(costs)) == 0
    assert costs[272] == 4
