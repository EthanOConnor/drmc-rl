# Inference-Time Search (Policy-Guided Truncated Expectimax)

Status: implemented (`models/policy/search_policy.py`). This doc is the design
spec plus the calibration/measurement notes that fixed its free parameters.

## Summary

A placement decision is refined by a depth-2, beam-K (default 8) truncated
expectimax over the native 1P pool engine:

- **ply 1**: placements of the *current* pill (top-K by policy prior).
- **ply 2**: placements of the *preview* pill (top-K by policy prior, per
  ply-1 branch). The preview is fully known, so there are **no chance nodes
  through depth 2**.
- **leaf metric**: the policy network's value head evaluated on the decision
  context after the ply-2 placement resolves (win-prob-flavored value for VS
  checkpoints, discounted return for 1P checkpoints — same interface).
- **backup** (reward-augmented, training-consistent):
  `Q(a1) = r̂1 + γ^τ1 · max_a2 [ r̂2 + γ^τ2 · V(leaf) ]`, argmax over ply-1.
  γ and the reward replication come from the checkpoint's training config
  (auto-detected 1P vs VS, see below). Backing up the *value head alone* is
  wrong and was measurably harmful: V estimates the *future* return, so a
  branch that clears viruses now has a **lower** leaf value than one that
  hoards them (the reward moved into the past) — a pure-V search actively
  avoids clearing. The first probe build had exactly this bug (degraded clear
  rate and slower play vs the plain policy at 1P level 20; 1W-3L in a small
  VS smoke) before the reward terms were added; with them the VS smoke
  flipped to 8W-2L.

The search is *anytime*: the plain policy argmax is the instant fallback, the
ply-1 value estimates (from the ply-2 prior forward) are a depth-1 refinement,
and the leaf backup is the depth-2 result. A deadline (default 60 ms) is
enforced between stages; whatever depth completed when it expires is what
commits. Beam width is the strength-dial notch.

## Simulation primitive

`envs/backends/drmario_pool.py::build_reset_spec(checkpoint_enabled=True, ...)`
resets a pool env to an arbitrary observed state: `checkpoint_board[128]` (NES
tile bytes, row 0 = top), `checkpoint_falling_colors` / `checkpoint_preview_colors`
(**raw** NES colors 0=Y,1=R,2=B), `checkpoint_speed_ups`, etc. After the reset
the falling pill is the checkpointed one and the planner runs from spawn.

`SearchPolicy` owns one `DrMarioPoolRunner(num_envs=num_sim_envs)` (default 64
= K²) arranged as K blocks of K envs:

1. **reset** all envs to (board, falling=CURRENT, preview=PREVIEW) with the
   caller's planner outputs **injected** (`inject_plan`): no native BFS at
   reset, and the sim accepts exactly the caller-feasible actions (the live
   bridge's mask comes from a mid-fall replan, the sim's would come from
   spawn — injection removes that mismatch).
2. **step ply-1** on one representative env per branch (envs 0..K-1). The
   same call returns the post-resolution board, the feasible mask + costs for
   the preview pill (now falling), `viruses_rem`, event counters for r̂1, and
   `terminated`/`terminal_reason` — the full ply-2 decision context. One BFS
   per surviving branch.
3. **ply-2 priors**: one batched policy forward on the representatives; its
   value output gives the depth-1 estimate `r̂1 + γ^τ1·V(s after ply-1)`.
4. **phase-B re-checkpoint**: fan each surviving branch out to its block of K
   envs by checkpoint reset to the branch's post-ply-1 board (falling = the
   known preview) with the branch's just-computed plan injected — again no
   BFS. **step ply-2**: env *j* of block *i* steps the block's rank-*j*
   candidate; one BFS per *distinct* leaf board (unavoidable: the leaf obs
   feasibility planes need it).
5. **leaf values**: batched marginalized value evaluation (below); terminal
   sims (clear → win, topout → loss) get terminal Qs.

Measured costs (M4-class, 64 envs): checkpoint reset with injection ≈ 0.3 ms,
steps ≈ 1–3 ms on mid-game boards. Caveat: the native per-decision BFS cost
grows sharply on near-empty boards (~7 ms/env vs ~0.01 ms on a 60-virus
board), which is why the reset/step structure above is careful never to plan
the same board twice. Network forwards, not the engine, dominate the budget
on normal boards.

## The unknown pill after the preview (chance at the leaf)

The leaf decision context contains the pill *after* the preview (and its own
preview), which in a checkpoint-reset sim is seed-dependent garbage. Measured
on a real level-14 mid-game state (vs2_02 net): leaf value std across random
seeds with the board held fixed is **0.14**, while the leaf value spread
across different ply-2 placements is only ~0.6 (std 0.17). A single sample
would materially corrupt the argmax, and averaging S engine seeds only shrinks
noise by √S.

Neutralization chosen: **exact analytic marginalization at the value forward,
single engine sim**. This works because the leaf obs is almost
color-independent:

- board planes (0–7) and the feasibility planes (8–11) of
  `bitplane_bottle_conn_mask` depend only on board geometry + speed, not on
  the leaf pill's colors;
- the only color-dependent obs artifact is the training-time symmetry
  reduction, which zeroes obs channels 6–7 when the falling pill is
  same-colored;
- the colors enter the value head only through the pill/preview embeddings in
  the conditioning vector, plus one aux scalar (`placements/options`, halved
  by symmetry reduction for same-color pills).

So the leaf value is computed as the mean over all 81 (pill, preview) ordered
canonical color pairs (assumed uniform; the NES reserve is near-uniform over
the 9 combinations): the expensive board trunk runs twice per leaf (normal +
ch-6/7-zeroed variant), the cheap conditioning/value MLPs run for all 81
combos, and each combo selects the trunk/aux variant matching its same-color
bit. This removes *all* seed dependence from the leaf value, so one sim per
leaf is sufficient and exact (up to the uniform-combo assumption).

The ply-2 *prior* forward still sees a garbage preview (it conditions
candidate selection only, not values) — accepted noise.

## Reward replication and terminal calibration

The backup needs the rewards collected *during* the two plies. The reward
mode and γ are auto-detected from the checkpoint's cfg (`env.backend`:
`cpp-vs-pool` → "vs", else "1p"; `smdp_ppo.gamma`), overridable via
constructor args.

**1P mode** (γ=0.998-style discounted SMDP training): r̂ replicates
`DrMarioPoolVecEnv`'s reward exactly from the pool's event counters
(virus-clear / non-virus / adjacency bonuses from `envs/specs/
reward_config.json` via `_RewardCfg.load()`, terminal clear bonus + time
penalty, topout penalty). Terminal sims have no future value, so their Q is
just the replicated terminal-inclusive reward — env semantics are the
objective, no dominance constants needed.

**VS mode** (γ=1.0, reward = terminal ±1 + 0.05·garbage-sent shaping):

- r̂ is the shaping term with the volley size *estimated* from cleared tiles
  (`lines = tiles_cleared/4; volley = min(4, lines) if lines >= 2`); the 1P
  sim has no garbage plumbing, and the NES attack rule adds the full combo
  counter when ≥ 2 matches land from one pill. The estimate is within ±1 line
  in practice and removes the systematic anti-combo bias that omitting it
  introduces (V drops by the shaping that just moved into the past).
- terminal sims get dominating constants instead of the train-consistent ±1:
  measured on real states, the vs2_02 net's values live in ≈ [0.5, 4.5]
  (gamma-1 shaping accumulates into V — it is *not* a pure win-prob), so a
  train-consistent +1 win-now would lose to any continuation. Defaults
  `win_value=+8.0` / `loss_value=-8.0` sit outside the observed range;
  ply-2 terminals are scaled by `(1 - depth_penalty)` (default 0.01) so an
  immediate win beats a win one pill later. `truncated` sims (wait-cap;
  effectively unreachable in 2 plies) are treated as losses, conservatively.

If a future checkpoint's value scale exceeds ±8, pass larger constants; the
search only compares Qs, so any dominating constant works.

## Known approximations (accepted)

- **1P sim for VS play**: incoming garbage and opponent actions during the
  2-ply horizon are not modeled (opponent model is future work, below).
- **Mid-fall vs spawn state**: the live bridge plans from the pill's mid-fall
  micro-state. Root plan injection makes the sim accept exactly the
  caller-feasible actions with the caller's (mid-fall-exact) lock costs; the
  warp execution then resolves the same lock pose the live script will
  reach, so ply-1 board outcomes are exact. Should a sim still reject a
  ply-1 action (defensive path), that branch is dropped from the beam.
- **Root parity**: `decide()` applies the training-time symmetry reduction
  (same-color pill → orientations 2–3 dropped, obs ch 6–7 zeroed) before the
  root forward. The plain live-bridge checkpoint path does not (pre-existing
  off-distribution quirk there, unchanged).
- **Speed ramp**: the sim seeds `checkpoint_speed_ups` from the caller and
  lets the engine ramp naturally; the visible pill counter is not replicated
  exactly (at most one +1 speed-up of drift over the 2-ply horizon).

## Latency budget (live bridge)

Total plan margin is 6 frames ≈ 100 ms. Search takes `--search-deadline-ms`
(default 60), leaving ≥40 ms for the v1 BFS-full planner + script generation.
Measured with `tools/bench_search.py` on real mid-game states at beam 8 (MPS
leaf device, CPU for the small forwards): p50 ≈ 50 ms, p95 ≈ 79 ms — the
deadline checks sit between stages, so a decide can overshoot by at most one
leaf chunk. Spawn→plan-written (`tools/live_agent_server --bench`, includes
the BFS-full planner + script) p95 ≈ 94 ms, inside the margin. On pure CPU
the batch-64 trunk forward alone is ~100 ms, so the leaf stage chunks
block-by-block (best-first) and commits whatever depth the deadline allows;
use `--device mps`/`auto` on Apple Silicon for full-depth searches inside
the budget.

## Strength dial

`beam` is the notch: 1 ≈ plain policy (plus win/loss detection at ply-1),
8 = default full strength. Latency scales roughly linearly in beam via the
leaf batch (beam² leaves).

## Phase 2 (future work, deliberately not built)

- **Ponder during pill fall**: the ~40–90 frames while the committed script
  executes are idle; pre-expand the next decision's tree from the predicted
  post-lock state and refine when the real spawn arrives.
- **1-ply opponent model**: for VS, simulate the opponent's most likely
  placement (their policy prior argmax) between our plies in a VS-pool sim,
  capturing garbage exchange in the horizon.
- **Search-as-training-targets**: distill the search-improved action
  distribution / backed-up values into the policy (expert iteration), turning
  inference-time gains into weight-space gains.
