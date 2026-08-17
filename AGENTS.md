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
