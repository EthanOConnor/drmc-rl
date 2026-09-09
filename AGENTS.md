# Repository guidance for coding and training agents

This repository has one authoritative program. Before changing architecture or
launching work, read in this order:

1. `drmc_rl/program/program.yaml` — machine-readable stage, recipe, product,
   artifact, and gate authority.
2. `docs/DESIGN.md` — architecture and information-flow contracts.
3. `docs/ROADMAP.md` — why the stages are ordered as they are.
4. `docs/OPERATIONS.md` — supported launch, arena, artifact, and recovery
   procedures.
5. `docs/COUNTERFACTUAL_QUALITY_HANDOFF.md` — exact current instructions for
   the staged counterfactual-quality evidence program.

Run `python -m tools.program status` at the start of a session. Long-running
work must be launched through `python -m tools.program launch ...`; do not invoke
an old VS YAML directly. Staged recipes require `--allow-staged`. Blocked
recipes are not implementation suggestions: they identify a missing contract or
gate that must be completed first.

## Governing architecture

- Build **one rating-independent competitive core**. The unrestricted player,
  human-rate player, and human trainer are controlled projections of that core,
  not unrelated agents.
- Preserve the placement SMDP and exact planner. A policy selects a final pose;
  the planner supplies timing and an exact script. Whether lock timing becomes
  an explicit sub-action is governed by `timing-action-gate`, not assumption.
- Deployed actors consume `PublicPairState` only. `PrivilegedPairState` is
  restricted to critics, search teachers, counterfactual labels, and parity.
  Hidden RNG or internal attack state must never leak into a public actor.
  The old native SMDP buffers can also contain a future opponent lock after a
  fall warp. New offline public data must carry `causal-settled-pair-v1`;
  zeroing auxiliary fields alone does not establish a causal observation.
- The target competitive model is the asynchronous **full pair game**. New
  search work implements `drmc_rl.search.joint_event.PairSearchModel`; do not
  extend the own-board depth-2 approximation into another permanent stack.
- Match outcome is the authority. Tactical terms are auxiliary predictions,
  curricula, replay priorities, or bounded secondary preferences. Do not create
  a specialist reward that can compensate for losing.
- Human strength is calibrated competitive regret. Apply style only inside the
  selected regret envelope, then cadence and mechanical execution. Do not use
  temperature, beam width, requested-rating cloning, or random actions as the
  strength definition.
- Human-rate means compliance with a published corpus-derived
  `ExecutionProfile`, including reaction, burst, edge, overlap, correction, and
  soft-drop limits—not merely average APM.

## Current work

- G4 Strong League and its bounded +900M-parent rewarm are complete frozen
  lineages; +900M is the evidenced local maximum and +1.0B is retained.
- Full-corpus V3 afterstate training is complete. Epoch 5 is the balanced
  teacher; Epoch 6 is the sharper imitation reference.
- `pair-state-v2` is complete with canonical native full-pair snapshot/restore,
  reveal boundaries, and no-leak contracts.
- The 512-state counterfactual pilot is **mechanics evidence only**. It proved
  restore, complete candidate enumeration, reveal override, and bounded search,
  but its independent `1/9` reveal probabilities are not a valid mature chance
  model.
- Current work is `v3-counterfactual-quality`: grouped draw-aware calibration,
  public reserve-seed belief, member-specific uncertainty, a balanced 1,440-state
  bank, opponent-beam 1/4/8 convergence, and direct observed-action/V3
  comparison. The executable gate must pass before competitive-head or G5
  quality distillation proceeds.
- `timing-action-gate` remains active. `g5-v3-bootstrap`, joint-event search,
  constrained human execution, and trainer release remain staged or blocked by
  the program registry.
- The bounded `trainer-public-outcome` diagnostic improves the frozen public
  V5 bootstrap from natural full-game results at level 20 and checks level-14
  regressions. The current counterfactual teacher failed its held-out V3
  comparison; none of those labels or legacy hidden auxiliary inputs enter
  this run. This does not open a search-quality or product gate.
- The September review sequence in `docs/ROADMAP.md` runs independent objective,
  public-context/critic and paired-terminal diagnostics. The E1 arms preserve
  one frozen initialization and separate loss reductions. New paired quality
  fitting uses causal banks, full candidate coverage and whole-game holdouts;
  it does not reuse the failed V3 teacher or certify a product model. Continue
  through the named program recipes and preserve each running source snapshot.
- The user explicitly requires the September review program to continue after
  the current held-out strength evaluation: finish live public history/motor
  integration, larger teachers and student distillation, adaptive search and
  anticipation, and persistent expressive play. Follow the completion criteria
  in `docs/ROADMAP.md#required-continuation-after-the-strength-evaluation`.
  A neutral or negative evaluation changes the selected baseline; it does not
  cancel these stages. Resolve missing prerequisites and continue implementation,
  training and evaluation. Staged status is not a reason to stop at another plan;
  promotion still requires the applicable evidence gates.

## Counterfactual quality rules

- The native 128-pill reserve is generated once from a two-byte RNG and the
  public initial virus bottle is generated from the same stream. Future reveals
  are correlated with both that bottle and public pill history. Never assign
  nine independent outcomes probability `1/9` in a quality release.
- Use `PillReserveBelief` and persist `reserve_belief` with every source state.
  A posterior reveal node may have fewer than nine supported outcomes.
- The frozen G4 continuation consumes exact pending-attack scalars and is a
  privileged teacher. Releases must declare
  `privileged-pending-attack-continuation-v1`; do not describe them as fair
  deployable public-information search.
- Aggregate mixture calibration is not sufficient for epistemic uncertainty.
  Fit member-specific grouped Davidson links and export each candidate's
  `member_wdl`, utility standard deviation, and Jensen-Shannon disagreement.
- Calibration and bootstrap confidence are grouped by whole game. Long games
  must not receive more total fitting or evaluation weight because more states
  were sampled.
- Run identical releases at opponent beams 1, 4, and 8. Only the beam may vary;
  source ids, legal actions, chance model, checkpoints, calibration, seed,
  depth, and node budget must match.
- Promotion is determined only by `tools.counterfactual_quality_gate`. Do not
  hand-edit or verbally waive a failed evidence check.

## Engineering rules

- Routine iteration uses focused pytest files or node IDs for the current edit,
  not the full suite or every dirty file in the working tree. Pace constants
  require paced-reach coverage and a backend case exercising the changed actual
  execution delay; planner algorithm changes require the relevant oracle/parity
  tests. Broaden only for failures, shared invariants, or release verification.
- Use the source backend during local trainer development. Do not rebuild the
  frozen sidecar, repeat multi-pace cartridge matches, run benchmarks, or launch
  training/evaluation as a default response to a routine tweak. Package only
  when a distributable is requested. Before external handoff/release, run the
  full applicable backend/planner/package suites and final artifact validation;
  certified promotion still requires every program gate. State verification
  scope accurately, and do not rerun passing checks without a relevant change.
- `drm_reach_bfs_full` remains the independent planner oracle. Changes to
  v4/CUDA planning require parity and fuzz tests.
- `vendor/drmario_native/` is a pinned submodule. Commit engine changes in the
  standalone repository first, then update its pin here.
- Libretro/emulator replay is independent verification, never the throughput
  training backend.
- Candidate truncation is a measured failure. Production and evaluation must
  report zero dropped feasible candidates.
- Every promoted artifact has immutable identity: policy checkpoint hash,
  config hash, repository commit, native submodule revision, observation
  schema, execution profile, search settings, corpus release, and gate evidence.
- Keep tests deterministic and focused. Do not commit ROMs, cores, corpora,
  checkpoints, run outputs, or operator secrets.
- Update existing authority documents when a contract changes. Do not add
  session logs, abandoned plans, duplicate roadmaps, or speculative launch
  configs.
