"""Throughput benchmark: CUDA planner vs CPU drm_reach_bfs_v4.

Solves real corpus boards (annotation spawn shape). Reports GPU solves/s at
several batch sizes, CPU single-core solves/s, and the projected wall time
for a full-corpus annotation pass (~34M moves).

Usage: .venv/bin/python -m tools.bench_reach_cuda [--quarks N] [--seed S]
"""

from __future__ import annotations

import argparse
import ctypes as C
import time

import numpy as np

from tools.test_reach_cuda_parity import corpus_boards, load_cpu_v4

CORPUS_MOVES = 34_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu-sample", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    boards = corpus_boards(args.quarks, rng)
    print(f"boards: {len(boards)} (from {args.quarks} quarks)")
    cols = np.stack([b[0] for b in boards])
    parity = np.array([b[1] for b in boards], dtype=np.uint8)
    thr = np.array([b[2] for b in boards], dtype=np.uint8)

    # --- CPU baseline (single core) ---
    cpu = load_cpu_v4()
    out = np.empty(512, dtype=np.uint16)
    m = min(args.cpu_sample, len(boards))
    for i in range(min(8, m)):  # warm
        cpu(cols[i].ctypes.data_as(C.POINTER(C.c_uint16)), 3, 0, 0, 0, 0, 0,
            int(parity[i]), 0, int(thr[i]), 2048,
            out.ctypes.data_as(C.POINTER(C.c_uint16)))
    t0 = time.perf_counter()
    for i in range(m):
        cpu(cols[i].ctypes.data_as(C.POINTER(C.c_uint16)), 3, 0, 0, 0, 0, 0,
            int(parity[i]), 0, int(thr[i]), 2048,
            out.ctypes.data_as(C.POINTER(C.c_uint16)))
    cpu_rate = m / (time.perf_counter() - t0)
    print(f"CPU v4 1-core : {cpu_rate:8.0f} solves/s   "
          f"({1000/cpu_rate:.2f} ms/solve; 4-core ~{4*cpu_rate:.0f}/s)")

    # --- GPU ---
    from reach_cuda import CudaReach
    ctx = CudaReach(max_batch=65536)
    _ = ctx.solve_costs(cols[:256], parity[:256], thr[:256])  # warm/JIT

    for bs in (256, 1024, 4096, 16384, len(boards)):
        if bs > len(boards):
            break
        reps = max(1, 8192 // bs)
        t0 = time.perf_counter()
        for _ in range(reps):
            ctx.solve_costs(cols[:bs], parity[:bs], thr[:bs])
        dt = (time.perf_counter() - t0) / reps
        rate = bs / dt
        print(f"GPU batch {bs:6d}: {rate:8.0f} solves/s   "
              f"({dt*1000:8.2f} ms/batch, {rate/cpu_rate:6.1f}x 1-core, "
              f"{rate/(4*cpu_rate):5.1f}x 4-core)")

    t0 = time.perf_counter()
    ctx.solve_costs(cols, parity, thr)
    best = len(boards) / (time.perf_counter() - t0)
    print(f"\nprojected full corpus ({CORPUS_MOVES/1e6:.0f}M moves): "
          f"{CORPUS_MOVES/best/60:.1f} min on GPU vs "
          f"{CORPUS_MOVES/(4*cpu_rate)/3600:.1f} h on 4 CPU cores")
    ctx.close()


if __name__ == "__main__":
    main()
