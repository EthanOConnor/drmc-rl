# Networks report: architectures, scales, and where the capacity should go

2026-07-02, measured on tf3090 (RTX 3090, 4-core host) with the VS training
run live (CPU/GPU numbers taken under that load are flagged). Param counts
are exact; the champion checkpoint was loaded and dissected.

## 1. Inventory

| Net | Role | Arch (active parts) | Params | File size |
|---|---|---|---|---|
| **champion / vs6 learner** (`smdp_ppo_step535164979`) | 1P champion, VS warm-start + league anchor, search prior/value | candidate policy: 8ch board → CoordConv CNN d192 × 4 resblocks; pill/preview OrderedPair embeds (128) + aux v1 (57) FiLM conditioning; per-candidate MLP (pos+cost+patch9²+trunk-gather → 384 → 192); logit = dot(cand, context)/√d; value MLP 192→192→1 | **3.32 M** | 48 MB ckpt (weights + EMA + Adam state ≈ 4× weights) |
| **BC bands v1** (`bc_{lt1600,1600to2000,gt2000}`) | human-style league seeds, gate opponents, dial anchors | same family, d96 × 2 blocks, tx2 (inactive — cnn encoder), no aux | **0.57 M** | 2.1 MB each |
| BC bands v2 (building now, R10) | same, full-corpus | d128 × 3 blocks | ~1.2 M (est.) | — |
| heatmap net (`placement_heads.py`) | legacy alternative | dense 512 head | n/a | unused |
| MLX twins (`mlx_networks.py`) | Apple-only | — | — | dead on this box |

Champion parameter budget by module: board trunk 2.67M (80%), candidate MLP
0.50M (15%), value head 37k, conditioning (pill/aux fusion) ~0.12M.

## 2. Facts worth knowing before scaling anything

- **The "transformer" config knobs are inactive.** `candidate_transformer_*`
  only matter for `candidate_board_encoder: col_transformer`; every production
  config uses `cnn`. And `head_type: dense` belongs to the (unused) heatmap
  head. The live net is a CNN trunk + *pointwise* candidate scorer.
- **Candidates never see each other.** Each candidate is scored independently
  against one global context vector. The net cannot directly express "these
  two placements are near-duplicates" or allocate probability by comparing
  options — the softmax normalizes, but features are per-candidate. A small
  cross-candidate attention block (1-2 layers over ≤128 candidate tokens,
  d192: ~0.6M params) is the single most *structurally* interesting upgrade,
  distinct from just widening the CNN.
- **The GPU is idle.** Training holds the 3090 at 2-3% utilization; the
  learner forward averages 0.25 ms at B=32 (training's own perf counters).
  Inference cost is not a constraint at any plausible scale.
- **Opponent forwards run on CPU.** `vs_opponents` loads frozen nets CPU-side
  and forwards them inside `env.step`. On this 4-core box, under load, that
  is real serial-loop time (my under-load CPU benches were 30-90 ms/forward —
  contention-poisoned as absolute numbers, but the direction is right). Moving
  opponent inference to the GPU is a cheap throughput win and belongs in the
  R2 "Python decision path" work.

## 3. Latency scaling (RTX 3090, measured under training load — treat as upper bounds)

| Net | B=32 | B=128 | B=512 |
|---|---|---|---|
| champion d192/enc4 (3.3M) | 4.2 ms | 12.6 ms | 22.4 ms |
| capacity probe d320/enc6 (12.4M) | 10.5 ms | 25.2 ms | 100 ms |

Even the 12.4M-param probe at rollout batch sizes costs ~10 ms per decision
wave on a busy GPU — versus a ~70 ms wave cadence in training. A 4× capacity
bump is essentially free at rollout time; PPO update time scales linearly
with params (update is currently ~7% of wall).

## 4. Recommendations

1. **Hold the champion architecture until the metagame gate passes** (review
   R7 stands): no recorded failure was capacity-shaped; bundling a net change
   into the clear-win metagame fix would make both unreadable.
2. **Then bump capacity once, deliberately**: d256-d320 × 6 blocks
   (~8-12M params), warm-started via `tools/expand_checkpoint.py`, as its own
   SPRT-gated ladder step. Cost: ~2.5× inference (still negligible), ~3×
   update FLOPs.
3. **Add cross-candidate attention in the same step or the next**: 1-2
   attention layers over candidate tokens before the dot-product scorer.
   This changes the hypothesis class, not just its width — worth its own
   ablation arm.
4. **BC bands: v2 (d128/enc3, full corpus) is being trained now.** If val
   top-1 saturates vs d96, stop there — league seeds need style diversity
   and calibrated strength, not maximum accuracy. Keep them small: they
   forward on the rollout path.
5. **Move opponent-pool forwards to GPU** (batched with the learner forward
   if possible) as part of the Python-decision-path work — likely worth more
   fps than any planner change now that planning is off the CPU.
6. **Search/live nets**: the same champion net serves as search prior+leaf
   value. The 81-combo leaf marginalization batch (up to beam² × 81 rows) is
   the one place where net latency touches the live deadline — at d320 it
   would grow the p95 search decision by roughly 2.5×; re-measure
   `bench_search` before shipping a bigger net into the live agent, or give
   the live agent the smaller net + deeper search (beam is a better
   latency-for-strength trade than width there).
7. **No recurrence yet** (roadmap GRU): the placement SMDP with full preview
   is near-Markov; recurrence buys little until opponent-modeling/garbage
   forecasting matter (post-R6).
