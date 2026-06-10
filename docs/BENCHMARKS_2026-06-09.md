# Benchmark Notes — 2026-06-09/10 (planner v4 + warp + sync-free training)

Hardware: Apple M3 Max (16 cores), macOS, local `.venv` (torch 2.9, MPS).
All numbers from this checkout at the commits landing planner v4
(`drm_reach_bfs_v4`), pool warp execution, and the training-loop sync fixes.

## Reachability planner (per spawn, single thread)

| Case | v1 (oracle) | v4 (production) |
|---|---:|---:|
| Empty board, thr=69 | 12.2 ms | 0.21 ms |
| Mid-game board (level 10, real rollout) | ~13 ms | ~0.58 ms |
| Sparse floating-virus boards (worst) | 7–12 ms | 4–9 ms |
| Fuzz average (mixed kinds/thresholds) | 7.1 ms | 1.8–2.1 ms |

Exactness: v4 ≡ v1 on all in-bounds pose costs across 1k+ fuzzed cases
(`tests/test_reach_v4_parity.py`); offscreen poses may differ at the
early-exit depth and are never consumed.

Profile context: before this work, `drm_reach_bfs_full` was **99.7%** of pool
wall time (~750k states / 7–9M transitions per spawn).

## Pool throughput (`tools.bench_multienv`, sync, random feasible actions)

| Envs | FPS | Decisions/s | step ms |
|---:|---:|---:|---:|
| 1 | 19,415 | 332 | 3.0 |
| 16 | 130,145 | 2,227 | 7.1 |
| 32 | 157,073 | 2,697 | 11.7 |
| 64 | 181,799 | 3,077 | 20.4 |

Baseline before warp+v4 (same harness): ~40k FPS / ~680 dec/s at peak.
Note: with warp execution, fall frames are never simulated; FPS here counts
SMDP τ-frames. Decisions/s is the honest planner-bound metric.

## Policy forward (`tools.bench_policy`, candidate_cnn 363k params, synthetic)

| Device | Batch | fwd ms | dec/s |
|---|---:|---:|---:|
| cpu | 16 | 3.9 | 4,109 |
| cpu | 128 | 16.9 | 7,575 |
| mps | 16 | 3.8 | 4,221 |
| mps | 128 | 6.7 | 19,015 |

## End-to-end training (`training.run`, headless, cpp-pool, MPS)

| Configuration | dec/s | frames/s |
|---|---:|---:|
| Before (16 envs, mb 128, per-minibatch syncs) | 168 | 11.0k |
| + inference_mode, CPU sampling, batched aux | 225 | 14.5k |
| + 64 envs, decisions_per_update 2048, mb 512, 3 epochs | 708 | 45.7k |
| + on-device update metrics (one sync/update), dist rewrite | **954** | **63.5k** |

Sanity: a fresh 5M-frame run (~80 s wall) reaches curriculum level −12 and
clears vanilla level 0 in 12.5% of fixed-seed eval episodes
(`tools.eval_policy`), vs 0% for random and greedy-cost baselines.

## Known remaining costs

- PPO update: ~1.1 s per 2048 decisions (≈1.8k dec/s ceiling). Next levers:
  torch.compile, async rollout/update overlap.
- Planner worst case: sparse low-virus boards with deep tuck poses
  (witness-UB gap keeps the exact search broad). Options recorded in
  notes/BACKLOG.md (NEON, factored expansion, explicit horizon knob).
