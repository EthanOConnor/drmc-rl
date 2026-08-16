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

- Primary live campaign: `g4-strong-league-rewarm`, a bounded exploration
  rewarm from the permanent Strong League +1.0B checkpoint. The base
  `g4-strong-league` campaign is complete and remains a frozen teacher lineage.
- Parallel architecture work: `timing-action-gate`, `pair-state-v2`, and
  `human-afterstate-bootstrap`.
- First staged competitive successor: `g5-v3-bootstrap`; it remains staged
  until its declared artifacts and bakeoff gate exist.
- Joint-event search, constrained human execution, and final trainer release
  remain gated by the program registry.

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
