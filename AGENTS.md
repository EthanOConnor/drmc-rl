# Repository guidance for coding and training agents

This repository has one authoritative program. Before changing architecture or
launching work, read in this order:

1. `drmc_rl/program/program.yaml` — machine-readable stage, recipe, product,
   artifact, and gate authority.
2. `docs/DESIGN.md` — architecture and information-flow contracts.
3. `docs/ROADMAP.md` — why the stages are ordered as they are.
4. `docs/OPERATIONS.md` — supported launch, arena, artifact, and recovery
   procedures.

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
- `pair-state-v2` is open with canonical native full-pair restore parity.
- Current architecture work is the stratified counterfactual pilot and
  `timing-action-gate`. Promotion-quality counterfactual labels still require
  a frozen continuation mixture and explicit reveal-time chance integration.
- `g5-v3-bootstrap` and joint-event search remain staged by the program
  registry; constrained human execution and trainer release remain gated.

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
