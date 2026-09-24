# Pace-constrained CUDA planner: scope and design

Status: design only. The unconstrained (Frame Perfect) planner is ported and
parity-proven (`drmc_rl/planning/cuda/drm_reach_full.cu`). This note scopes the
same port for `drm_reach_bfs_paced` (`reach_native/drm_reach_full.c`), which
answers every other pace. Measured facts are marked **[measured]**; everything
else is an estimate.

## What the CPU paced planner actually is

`drm_reach_bfs_paced` is not a constrained BFS with a unique answer. Its
witnesses are *realized durations*, not minima, and the output depends on the
order in which a cascade of heuristics is tried. Exact parity therefore means
reproducing each stage's control flow, not just solving the constrained
reachability problem correctly.

1. **Reaction prefix.** Step action 0 for `max(0, reaction - delay)` frames; a
   lock during the prefix is the only answer.
2. **Simple routes.** For every "wanted" pose (timer-free flood fill), try
   `paced_simple_route` for tap ∈ {0,1} × order ∈ {rotate-first, move-first,
   both}; keep the first strictly shortest. Pack scripts in pose order. Return
   if every wanted pose has a route.
3. **Unrestricted filter.** Drop wanted poses the exact v4 planner cannot
   reach at all; return if the rest are covered (a covered pose that v4 calls
   unreachable is error -4).
4. **Tuck waypoints, then a bounded probe.** Per unresolved pose, try staged
   routes through waypoints (row ry from the pose upward × 4 rotations × 15
   column offsets × tap × order, first success wins), else a best-first probe
   (effort heap, ≤4,096 expansions, 32,768 nodes, 64k-slot hash, hash-order
   tie-breaks).
5. **Exhaustive layered search.** Nodes keyed by (x, y, rot, sc, hv while
   holding, previous 18-way action, parity, motion wait); a button change
   contracts the forced dwell (`edge_interval` frames) into one edge, so a
   layer list receives nodes from several earlier layers. Parents are chosen
   by (depth, effort) with first-found ties; a node found later at a shorter
   depth supersedes the old one; a reverse-closure `needed` mask prunes states
   that cannot reach an unresolved pose and is refreshed at each layer.

## Where the CPU time goes **[measured]**

tf3090, 3,000 captured level-14 arena roots per pace (Frame Perfect states
re-planned at each pace, plus captured Super Human requests), CPU
`drm_reach_bfs_paced` per call, and the share of calls that return after
stages 1–3 (the only stages a GPU prototype implemented):

| Pace | CPU ms/call | Returned by stage 2 | + stage 3 | Needs stages 4–5 |
| --- | ---: | ---: | ---: | ---: |
| Frame Perfect (`bfs_full`) | 8.9 | — | — | — |
| Super Human | 0.75 | 1,633 | 0 | 1,367 |
| Top Humans | 0.79 | 1,633 | 0 | 1,367 |
| Fast | 1.14 | 1,587 | 2 | 1,411 |
| Normal | 2.61 | 1,063 | 34 | 1,903 |
| Relaxed | 9.26 | 22 | 0 | 2,978 |
| Sloth | 10.39 | 0 | 4 | 2,996 |

The stage-1/2 prototype (commit `f0481a9`, kernel
`drm_reach_paced_stage1_kernel` plus the existing v4 cost kernel for stage 3)
matched the CPU byte for byte on every call it resolved: 0 mismatches over
1,633/1,633/1,589/1,097/22/4 resolved roots. It was removed from the
production module because it only covers the cheap half of already-cheap
paces.

Conclusions:

- Super Human, Top Humans and Fast cost under 1.2 ms of CPU per call — about
  8–12× less than a Frame Perfect call. Offloading them frees little CPU and
  does not justify a hybrid route. **Super Human is not worth porting for
  throughput.**
- The paces where the GPU would matter (Normal 2.6 ms, Relaxed 9.3 ms, Sloth
  10.4 ms) almost never finish before stage 4, so any useful port must include
  stages 4 and 5 in full.

## Proposed GPU design (stages 1–5)

- **Stages 1–3**: as prototyped — one block per instance, one thread per
  wanted pose, 2 KB route scratch per pose; stage 3 reuses `CudaReach` v4
  costs. Exact by construction (same scalar stepper, same loop order).
- **Stage 4 tuck routes**: per unresolved pose, a warp scans the waypoint loop
  in parallel and takes the *lowest loop index* that succeeds (the CPU's
  first-success order), so parallelism does not change the answer.
- **Stage 4 probe**: inherently serial (heap and open-addressing order decide
  ties). Run one thread per unresolved pose with a global-memory node pool
  (32,768 × 24 B + 256 KB table ≈ 1 MB per pose). Risky for memory when many
  poses need it; fall back to the CPU for an instance whose probe count
  exceeds a budget.
- **Stage 5 layered search**: reuse the ordered-event technique of the Frame
  Perfect kernel. Each insertion carries an order (source layer, source list
  position, action); a layer list position is the minimum order of any
  insertion *at that depth*; the parent is the lexicographic minimum of
  (effort, order) at the minimal depth. Superseding by a shorter depth becomes
  a per-key minimum over depth. The `needed` refresh happens at the layer
  barrier, exactly where the CPU refreshes it. The key space is too large for
  dense arrays (x·y·rot·sc·hv·18 actions·parity·motion-wait ≈ 10⁸·(thr+1)), so
  it needs a device hash table with deterministic slots (insert, then resolve
  winners by order), plus per-layer lists that accept insertions from up to
  `edge_interval` (12 for Sloth) earlier layers.

State and memory per instance (estimate): node record ~24 B (parent, effort,
order, depth, packed microstate, action, motion wait); CPU searches hold
10⁴–10⁶ nodes, so 25–50 MB per resident instance with a 2× hash table. At one
instance per SM that is 2–4 GB, comparable to the Frame Perfect arena.

## Correctness risks

- Effort ties and supersession are order-dependent; a single mis-ordered
  insertion changes a script byte but rarely a cost, so cost-only checks would
  miss it. Every check must compare script bytes.
- Hash-capacity growth, the probe budget, and the 65,535-byte script limit are
  CPU error/fallback paths that must map to explicit CUDA statuses.
- `needed` is refreshed mid-search; refreshing at a different point prunes a
  different set and changes witnesses.

## How parity would be proven

The same bar as Frame Perfect, per pace:
1. `tools/trainer_planner_parity.py compare` over captured arena requests —
   byte-identical native arrays, candidate sets and `execution_for_action`
   output, ≥10,000 roots per pace, with every CPU-routed request counted.
2. Fuzzed roots over all spawn microstates, speed thresholds 0–39 and every
   profile value (`tools/test_reach_full_cuda_parity.py` extended with
   profiles).
3. A mirror arena per pace (CUDA side versus CPU side, same seeds, ≥128 games)
   requiring byte-identical side-swapped move journals.

## Estimate

Stages 1–3: done as a prototype (≈1 day including parity). Stage 4:
2–3 days. Stage 5: 5–8 days including the ordering proof and parity
debugging. Total ≈ 2 weeks of focused work, worthwhile only if Normal,
Relaxed and Sloth arenas become the throughput bottleneck. Their 9–10 ms calls
cost about as much as Frame Perfect ones, so the payoff there would match the
Frame Perfect result.
