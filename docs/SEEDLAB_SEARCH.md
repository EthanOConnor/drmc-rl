# Seedlab Search: Per-Seed Solution Optimization

Deep-dive on the algorithmic structure of "fastest clear for a fixed seed"
and the design of the jagged-progress explorer built on it. Companion to
`docs/SEED_CATALOG.md`.

## Problem structure (what makes this easy-er than it looks)

1. **Deterministic, perfect information.** A (level, speed, seed) game has no
   chance nodes: `seedlab/rng.py` derives the full 128-pill reserve and virus
   board analytically before play. Per-seed optimization is therefore
   single-agent shortest-path planning — state `(board, pill_idx)`, edge cost
   in frames — not expectimax. (The wrap at pill 128 reuses the reserve, so
   even 100+ pill games stay fully predictable.)

2. **Exact additive edge costs, mostly known up-front.** The v4 planner
   reports the exact movement frame cost to every reachable lock pose
   *before* committing (`cost_to_lock`); the residual (clear animations,
   settles, spawn delay) is revealed by stepping the warp engine once
   (~30 µs). g-costs are exact, so f = g + h machinery applies cleanly.

3. **O(1) node restore.** The pool reset spec carries a full extraction
   checkpoint (board, reserve index, pill counters, speed_ups). Restoring an
   arbitrary search node = one reset + one step ≈ 2 engine ops, vs O(depth)
   for replay-to-node. This is the difference between 23 min/seed and
   ~45 s/seed for deep-level tree search.

4. **Transposition structure.** Placement order often commutes; different
   prefixes reach identical `(board, pill_idx)`. Hash-dedup per layer
   multiplies effective beam width for free.

5. **Learned guidance exists.** The trained policy gives child priors
   (effective branching ~50 → top-M ≈ 8) and its aux features are
   constructible from search state, so net-guided ordering works outside the
   training loop.

6. **The core tension is searchable, not learnable-only.** tau-greedy locks
   high (short drops) but starves progress; clearing needs low placements.
   Multi-virus clears amortize the ~per-clear animation stall. These tradeoffs
   are exactly what beam ranking `f = g + λ·viruses_remaining` explores;
   λ ≈ catalog q10(frames)/virus_count per level (optimistic frames-per-virus).

## Ranked algorithmic wins

| # | Win | Payoff |
|---|-----|--------|
| 1 | Treat per-seed as deterministic planning (no sampling noise) | search instead of luck |
| 2 | Checkpoint-restore node expansion | ~30× over replay-to-node at depth 100+ |
| 3 | Policy-prior child pruning (top-M by logits) | branching 50 → 8 |
| 4 | Board-hash transposition dedup per layer | free width |
| 5 | Exact g + admissible bounds → B&B with catalog incumbent | certified optima at low levels |
| 6 | Replay-verify final traces from true reset | checkpoint-restore approximations can never pollute the catalog |
| 7 | Jagged effort allocation across seeds (headroom-weighted + uniform mix) | spend depth where it pays |

Admissible bounds for T3 (and headroom signals), all engine-measured or
combinatorial (`seedlab/bounds.py`, 2026-06-11):

- **Pills**: remaining ≥ max(ceil(v/6), ceil(Σ_c ceil(v_c/3) / 2)) — a pill
  has 2 halves, one half clears ≤3 viruses, and halves are color-specific.
- **Frames per step**: measured minima from extremal boards on the
  rules-exact engine, min over frame parity, per (speed_setting, speed_ups):
  *continuing* steps (grounded checkerboard support one row under the spawn
  → earliest possible drop-failure, no clears, bare next-spawn post-lock;
  37f at MED/HI, 58f at LOW) and *terminal* steps (min of topout-lock and
  immediate-virus-clear stage end; 8f). speed_counter is 0 at every real
  spawn, so checkpoint measurement matches play; admissibility is
  fuzz-asserted against sampled rollouts (`test_step_bounds_admissible`).
- **Schedule-exact future sum**: pill index = step index, so future
  speed-ups are known exactly; the B&B sums per-step minima along the true
  ups schedule, with the final step allowed the terminal minimum.

## Solver tiers

- **T0 — greedy rollout** (existing worker, pass 0): policy argmax, 1
  episode/seed. Baseline coverage.
- **T1 — best-of-K sampled** (existing worker, K dial): temperature samples;
  feeds "typical achievable" distributions and cheap record luck.
- **T2 — policy-guided beam** (`seedlab/search.py:beam_search`): width W ∈
  {8, 32, 128}, top-M policy-prior children, board-hash dedup, rank by
  f = g + λ·viruses_remaining. Anytime: every cleared leaf is a candidate;
  best trace replay-verified then recorded.
- **T3 — exact DFS branch-and-bound** (`seedlab/search.py:exact_search`):
  low levels only (≈ depth ≤ 8–10). Incumbent from catalog best; admissible
  frame bound prunes; transposition table. Returns optimality certificate
  when the search closes → `solutions.verified=2` ("proven optimal") via
  explorer.

## Jagged explorer (`python -m seedlab explore`)

Continuous process, no work-unit queue (that's the systematic-pass axis).
Each iteration:

1. Sample level **width-first with a priority floor** (default 4,
   `--priority-level`): bulk on the frontier — the lowest level ≥ floor whose
   32,767-seed space lacks a best — with geometrically decaying "tentative
   exploration" mass above it (×0.35 per level) and a small uniform residue
   everywhere, which is what randomly backfills levels below the floor
   (0–3 by default). When the frontier completes, mass overflows to the next
   level automatically; if everything ≥ floor is finished the scan falls back
   to the full list.
2. Sample seed: on the frontier, prefer seeds with no best yet (bounded
   rejection sampling); on finished levels, 50% uniform / 50% weighted
   toward the slowest decile of per-seed bests.
3. Pick tier by **share-weighted resources**: each tier (greedy x1 — the
   pass-0 single-argmax-rollout method and cheapest coverage filler —
   rollout x4, beam w8/w32/w128, exact, polish) accumulates wall time in a
   ledger seeded from `search_log`; the tier with the lowest
   spent/share ratio runs next, so cumulative time converges to
   share_t/Σshares. Default shares are 1.0 except **polish = 3.0** (~1/3 of
   total compute). Cheap tiers spend their share via many iterations, deep
   tiers via few. `exact` only applies at level ≤ `--exact-max-level` and
   retargets record seeds (only near-optimal incumbents can close).
   **polish** retargets its iteration at one of the top-16 fastest seeds of
   a random level-with-records (the level-record front): beam width
   escalates with that seed's prior polish count (32→64→128→256, from
   `search_log`), and visits after the first add Gumbel noise to the
   policy-prior child ordering so each repeat explores a different subtree
   instead of re-running the same deterministic beam. Falls back to greedy
   coverage while the catalog has no records yet.
4. Run, replay-verify any improvement, fold into the catalog
   (`record_attempts`; budget-exhausted episodes count as failed attempts),
   and log `(seed, tier, nodes, wall_ms, improved)` to `search_log`.

Note on greedy x1: at level ≥5 the argmax policy can dither on the last
viruses and hit the 400-decision cap without clearing — that is the policy,
not a search bug; the sampled and beam tiers break those cycles, and the cap
bounds the waste.

Note on exact (re-benchmarked 2026-06-11 after the real bounds landed): the
admissible machinery is now genuinely derived — engine-measured per-step
frame minima (continuing 37f MED/HI / 58f LOW, terminal 8f; fuzz-asserted),
schedule-exact future sums, board-aware per-color line-component pills
bound, and virus-progress DFS ordering — yet 300k-node budgets still close
nothing, even at level 0. Two measured reasons: the 37f floor sits well
under the ~50–70f average true step cost, and beam incumbents are not
optimal, so the g-threshold admits deep subtrees. Certificates therefore
remain opportunistic (`closed AND replay-exact` only); the real unlock is a
per-node planner-aware bound (use the node's own cost_to_lock distribution
for future steps), not a tighter global constant. `--exact-max-level`
stays 0.

Every seed keeps getting occasional deep probes (uniform half) while fat
headroom attracts effort (weighted half) — progress is deliberately jagged,
prime95-style, and the frontier is never starved.

Thread discipline: explorer sets `DRMARIO_POOL_WORKERS=1` and
`torch.set_num_threads(2)` by default (`--threads`), so it coexists with
training runs and interactive use.

## Verification invariant

Search runs on checkpoint restores (speed_ups/parity reconstructed
analytically). Before anything touches the catalog, the winning action trace
is replayed from a true level reset; the **replay** frame count is what gets
recorded, and a mismatch discards the candidate (logged). The catalog can
therefore never contain a time that the engine cannot reproduce from reset.

## Engine fast path (2026-06-11, ~3–11×)

Profiling showed every beam child paid two redundant planner runs (~6 ms each
on level-0 sparse boards): the restore re-planned the parent decision we
already cache in the Node, and the step fully planned children of which only
W of W×M survive pruning. Two additive, struct_size-guarded pool features
remove both, with zero trust impact:

- **Plan injection** (`DrmResetSpec.inject_plan/_feasible/_costs`): checkpoint
  restores seed the env's reachability cache with the planner's own outputs
  round-tripped from the parent node; `ensure_planner` cache-hits. Any
  mismatch with the restored state falls back to a normal plan.
- **Lazy decision outputs** (`DrmPoolConfig.lazy_decision_outputs`): step()
  emits exact state-only context (tau, board, viruses, pill colors, terminal
  flags) and skips planning; masks zero / costs saturate so stale data can
  never be emitted. Beam runs phase 1 (cheap expansion) on a lazy pool, then
  plans only frontier survivors via checkpoint restore (phase 2). Training
  and eval pools leave the flag off; replay/rollouts use a normal pool.

The recording trust chain is unchanged: only from-reset replay frames enter
the catalog; certificates void on any replay mismatch.

## Measured baselines (M3 Max, ≤3 threads, ckpt step590M, 2026-06-11)

`python -m seedlab explore --bench` on seed 0x8988, after the fast path
(pool workers 2; in parentheses: before):

| level | width | frames | nodes | wall s | nodes/s |
|------:|------:|-------:|------:|-------:|--------:|
| 0 | 8  | 1304 | 1096  | 0.8 (9.0)  | 1400 (122) |
| 0 | 32 | 1304 | 4168  | 4.5 (35.2) | 925 (118) |
| 7 | 8  | 5647 | 4744  | 3.2 (27.2) | 1502 (174) |
| 7 | 32 | 3940 | 12792 | 13.5 (97.1)| 947 (132) |
| 14 | 8  | — (no clear) | 10184 | 4.5 (14.1) | 2265 (723) |
| 14 | 32 | 8258 | 26952 | 23.9 (63.6)| 1128 (423) |

Observations:
- Search answers are unchanged (±1 frame of search-internal accounting from
  the injected-cost source; replay frames identical).
- Beam beats the pass-0 greedy baseline immediately (level 0 seed 0x8988:
  1651 → 1313 replay frames, −20%).
- Width pays at mid/high levels (level 7: W32 −30% vs W8); W8 can lose the
  clearing line entirely at level 14. Tier weights should shift wider with
  level.
- Remaining cost at width 32 (~1 ms/node) splits between survivor re-plans,
  the CPU policy forward, and python batch assembly. MPS only pays at widths
  ≥64 (small-batch dispatch overhead dominates below that).
