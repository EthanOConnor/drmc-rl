# Design

## Fixed contract

The policy chooses a final pill pose, not controller buttons. Actions form a
`4 × 16 × 8 = 512` grid. The planner returns feasibility and exact frames to
lock; PPO discounts over the resulting duration `tau`.

This contract is shared by training, search, seedlab, and live play. It removes
movement legality from the learning problem while retaining frame-perfect
maneuvers and timing as explicit costs.

## Package boundaries

- `drmc_rl.game` owns backend-independent actions, state decoding, mechanics,
  rendering, and specifications.
- `drmc_rl.planning` owns the 512-way placement space and Python, native, and
  CUDA reachability implementations. It may depend on `game`, never libretro.
- `drmc_rl.envs.backends` owns native-pool and emulator runtime bindings.
- `drmc_rl.envs.libretro` owns Gymnasium wrappers, registration, and emulator
  verification utilities. It depends on `game` and `planning`.
- `drmc_rl.training`, `drmc_rl.models`, `drmc_rl.eval`, and `drmc_rl.seedlab`
  are application layers
  over those contracts.

The `vendor/drmario_native` submodule is a pinned build dependency. Engine work
is committed in the standalone `drmario-native` repository first; `drmc-rl`
then updates the submodule revision. Python bindings remain in `drmc_rl.native`
instead of making the vendored source tree part of the application namespace.

## Runtime paths

### Single player

```text
drmario-native pool
  -> board, pill, preview, feasible placements, costs
  -> packed candidate policy
  -> SMDP PPO and curriculum
```

The pool runs in process through ctypes and warps a chosen pill to its lock
pose. `drm_reach_bfs_v4` is the costs-only training planner;
`drm_reach_bfs_full` is its verification oracle.

### VS

The VS pool simulates both boards, garbage, attacks, and terminal outcomes.
Training either exposes both sides for self-play or one learner side against a
frozen opponent pool. Optional CUDA planning batches parked decision states
before reinjecting exact costs into the native pool.

Pure reward-shaped self-play learned ceiling attrition rather than decisive
clearing and attack. The current path begins with exact human-corpus
afterstates, improves them with search, then fine-tunes on actual match outcomes
against human anchors and frozen lineage. Start banks remain curriculum tools,
not substitutes for full-game evaluation.

### Search and live play

Search uses native simulation plus the policy/value network for depth-2 policy
improvement. Pondering spends pill-fall time preparing the next decision. The
live bridge reads RAM at spawn boundaries, plans from the observed microstate,
and sends a verified frame-indexed input script. Training uses warp execution;
live play never does.

### Seedlab

Seedlab distributes fixed-seed attempts, records best traces and distributions,
and runs beam or bounded exact search. Catalog times are native planner times
until independently audited through script replay or an emulator.

## Model

The V3 policy encodes the root and opponent bottles once. Each legal action is
represented by the exact sparse tile changes produced after lock, clear, and
cascade, plus pose, movement cost, current pill, and preview. Cross-candidate
attention compares alternatives directly. Rating-independent heads predict
competitive quality, outcome, clear, top-out, virus progress, and attack;
rating/history condition only the human-style head. A separate empirical model
controls human strength from conditional regret quantiles. It preserves rare
mistake tails across rating and decision-opportunity strata rather than assuming
that average or typical placements differ by skill.

## Independent verification boundary

The native engine and reachability planner are optimized models, not the final
oracle. Libretro with a legal ROM, recorded NES traces, the full native planner,
and controller-script replay remain independent checks. Optimizations must not
replace their own oracle.
