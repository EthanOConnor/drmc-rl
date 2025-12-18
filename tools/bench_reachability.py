from __future__ import annotations

"""Benchmark placement reachability planning (native vs Python).

This script captures a real spawn snapshot from the libretro backend and times
`PlacementPlanner.build_spawn_reachability` for both backends.

Example:
  python -m tools.bench_reachability --core-path cores/quicknes_libretro.dylib \\
      --rom-path \"legal_ROMs/Dr. Mario (Japan, USA) (rev0).nes\"
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from envs.retro.drmario_env import Action, DrMarioRetroEnv
from envs.retro.placement_env import NEXT_ACTION_PILL_FALLING, ZP_CURRENT_P_NEXT_ACTION
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner


def _read_u8(buf: bytes, addr: int) -> int:
    if addr < 0 or addr >= len(buf):
        return 0
    return int(buf[addr]) & 0xFF


def _at_decision_point(env: DrMarioRetroEnv) -> bool:
    state = getattr(env, "_state_cache", None)
    if state is None:
        return False
    ram_bytes = state.ram.bytes
    if _read_u8(ram_bytes, ZP_CURRENT_P_NEXT_ACTION) != NEXT_ACTION_PILL_FALLING:
        return False
    try:
        return bool(state.calc.falling_mask.any())
    except Exception:
        return False


def _capture_spawn(env: DrMarioRetroEnv, *, max_wait_frames: int = 6000) -> Tuple[BoardState, PillSnapshot]:
    for _ in range(int(max_wait_frames)):
        if _at_decision_point(env):
            state = env._state_cache
            assert state is not None
            snap = PillSnapshot.from_state(state, getattr(env, "_ram_offsets", {}))
            board = BoardState.from_planes(state.calc.planes)
            return board, snap
        env.step(int(Action.NOOP))
    raise RuntimeError("Timed out waiting for a pill-falling decision point.")


def _time_build(planner: PlacementPlanner, board: BoardState, snap: PillSnapshot, repeats: int) -> float:
    # Warm up
    planner.build_spawn_reachability(board, snap)
    best = float("inf")
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        planner.build_spawn_reachability(board, snap)
        t1 = time.perf_counter()
        best = min(best, float(t1 - t0))
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=str, default="libretro")
    parser.add_argument("--core-path", type=str, default="cores/quicknes_libretro.dylib")
    parser.add_argument("--rom-path", type=str, default='legal_ROMs/Dr. Mario (Japan, USA) (rev0).nes')
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-python", action="store_true", help="Also benchmark Python reference (slow)")
    args = parser.parse_args(argv)

    core_path = str(Path(args.core_path).expanduser())
    rom_path = str(Path(args.rom_path).expanduser())

    env = DrMarioRetroEnv(
        obs_mode="state",
        backend=str(args.backend),
        core_path=core_path,
        rom_path=rom_path,
        level=int(args.level),
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        board, snap = _capture_spawn(env)
    finally:
        env.close()

    print(
        "spawn:",
        f"id={snap.spawn_id}",
        f"pos=({snap.base_col},{snap.base_row}) rot={snap.rot}",
        f"colors={snap.colors}",
        f"speed_threshold={snap.speed_threshold}",
    )

    native = PlacementPlanner(max_frames=int(args.max_frames), reach_backend="native")
    dt_native = _time_build(native, board, snap, int(args.repeats))
    print(f"native: {dt_native*1000:.2f} ms (best of {int(args.repeats)})")

    if args.include_python:
        py = PlacementPlanner(max_frames=int(args.max_frames), reach_backend="python")
        dt_py = _time_build(py, board, snap, max(1, int(args.repeats)))
        print(f"python:  {dt_py*1000:.2f} ms (best of {int(args.repeats)})")


if __name__ == "__main__":
    main()

