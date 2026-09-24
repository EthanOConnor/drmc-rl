"""Fuzz parity: CUDA ``drm_reach_full_kernel`` versus CPU ``drm_reach_bfs_full``.

Every output array (costs, offsets, lengths, script bytes) must be
byte-identical for all 512 poses. Boards cover empty, sparse, dense and
stacked/cavity layouts; spawn microstates cover every falling-pill field
(position, rotation, speed counter, DAS velocity, held direction, parity,
held rotation) and speed thresholds 0..39, including roots that cannot fit.
Instances the kernel flags must be rare and are reported; they are never
counted as matches.

usage: python -m tools.test_reach_full_cuda_parity [--cases 4000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation
from drmc_rl.planning.native_reach import NativeReachabilityRunner


def random_board(rng: np.random.Generator, kind: int) -> np.ndarray:
    cols = np.zeros(8, dtype=np.uint16)
    if kind == 1 or kind == 3:
        density = rng.uniform(0.05, 0.25) if kind == 1 else rng.uniform(0.3, 0.6)
        for x in range(8):
            for y in range(16):
                if rng.random() < density:
                    cols[x] |= np.uint16(1 << y)
    elif kind == 2:
        for x in range(8):
            h = int(rng.integers(0, 14))
            for y in range(16 - h, 16):
                cols[x] |= np.uint16(1 << y)
        for _ in range(int(rng.integers(0, 12))):
            cols[int(rng.integers(0, 8))] &= np.uint16(~(1 << int(rng.integers(3, 16))) & 0xFFFF)
    if rng.random() < 0.9:
        cols[3] &= np.uint16(0xFFFE)
        cols[4] &= np.uint16(0xFFFE)
    return cols


def random_spawn(rng: np.random.Generator, threshold: int) -> FrameState:
    standard = rng.random() < 0.5
    return FrameState(
        x=3 if standard else int(rng.integers(0, 8)),
        y=0 if standard else int(rng.integers(0, 6)),
        rot=0 if standard else int(rng.integers(0, 4)),
        speed_counter=int(rng.integers(0, threshold + 2)),
        hor_velocity=int(rng.integers(0, 16)),
        hold_dir=HoldDir(int(rng.integers(0, 3))),
        frame_parity=int(rng.integers(0, 2)),
        rot_hold=Rotation(int(rng.integers(0, 3))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-threshold", type=int, default=39)
    args = parser.parse_args()
    from drmc_rl.planning.cuda.full import CudaReachFull

    rng = np.random.default_rng(args.seed)
    boards, spawns, thresholds = [], [], []
    for i in range(args.cases):
        threshold = int(rng.integers(0, args.max_threshold + 1)) if i % 2 else int(rng.integers(0, 14))
        boards.append(random_board(rng, i % 4))
        spawns.append(random_spawn(rng, threshold))
        thresholds.append(threshold)
    cpu = NativeReachabilityRunner()
    tick = time.perf_counter()
    reference = [cpu.bfs_full(b, s, speed_threshold=t).copy() for b, s, t in zip(boards, spawns, thresholds)]
    cpu_seconds = time.perf_counter() - tick
    gpu = CudaReachFull()
    try:
        instances = gpu.pack(np.stack(boards), spawns, thresholds)
        tick = time.perf_counter()
        batch = gpu.solve(instances)
        gpu_seconds = time.perf_counter() - tick
    finally:
        gpu.close()
    flagged, mismatched = [], []
    for i, expected in enumerate(reference):
        if batch.status[i]:
            flagged.append(dict(index=i, status=int(batch.status[i])))
            continue
        actual = batch.reach(i)
        for field in ("costs_u16", "offsets_u16", "lengths_u16", "script_buf"):
            a, b = getattr(expected, field), getattr(actual, field)
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                mismatched.append(dict(index=i, field=field, threshold=thresholds[i]))
                break
    report = dict(cases=args.cases, seed=args.seed, identical=args.cases - len(flagged) - len(mismatched),
                  flagged=len(flagged), mismatched=len(mismatched), first_mismatches=mismatched[:10],
                  first_flagged=flagged[:10], cpu_ms=round(1000 * cpu_seconds / args.cases, 3),
                  cuda_ms_batched=round(1000 * gpu_seconds / args.cases, 3),
                  max_frontier_entries=int(batch.nodes.max()), max_script_bytes=int(batch.used.max()))
    print(json.dumps(report))
    if mismatched or len(flagged) > args.cases // 100:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
