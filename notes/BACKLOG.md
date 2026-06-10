# BACKLOG.md - drmc-rl

Forward backlog for the current architecture. Historical plans that are no
longer active belong in `notes/archive/` or git history, not in this file.

## P0: Keep `cpp-pool` Training Reproducible

- Verify fresh-checkout setup after the `game_engine/` submodule split:
  `git submodule update --init --recursive`, build `libdrmario_pool`, and run
  `training.run --dry_run`.
- Keep local editable installs pointed at this checkout; do not report test
  health from an old venv. Current local repair was done on 2026-05-08.
- Keep README, `AGENTS.md`, setup docs, and command examples aligned with
  `backend: cpp-pool`.
- Keep `tools/bench_multienv.py` and `tools/bench_policy.py` healthy; use their
  JSON outputs before changing env count, worker count, or policy shape.

## P1: Improve Placement Policy Learning

- Treat candidate scoring as the default policy path.
- Run comparable heatmap-baseline and candidate-scoring training experiments
  under the same `ln_hop_back` curriculum.
- Compare `candidate_board_encoder=cnn` vs `col_transformer` using decisions/sec,
  clear rate, stage progression, and policy entropy.
- Keep candidate packing invariants covered: feasible-mask shape, cost sentinel
  handling, deterministic tie-breaks, and chosen-action repacking.
- Keep ordered pill-color embeddings as the default unless a targeted ablation
  shows unordered features improve learning.

## P2: Strengthen Evaluation and Certification

- Add fixed-seed certification at curriculum stage boundaries.
- Use an anytime-valid gate or bounded sequential test before stage advancement
  is treated as durable.
- Extend best-times reporting beyond the single best per `(level, seed)` when it
  helps compare policies.
- Add deterministic smoke runs that exercise `ln_hop_back` advancement, hop-back
  sampling, and soft time-budget behavior.

## P3: Maintain Native/Emulator Parity

- Keep libretro or tetanes-style emulator work as oracle/debug support rather
  than the training default.
- Expand native-vs-emulator trace coverage around spawn, lock, settle, top-out,
  and stage clear.
- Preserve `cpp-engine` only where it still adds compatibility or parity value.
- Document each parity failure with the exact backend pair, seed, level, action
  trace, and first divergent RAM/counter field.

## P4: Remove Historical Weight Safely

- Update stale docs and commands when found; archive instead of layering caveats
  onto obsolete instructions.
- Do not delete active placement planner code just because it lives under
  historical `envs/retro/` package names.
- If a larger package rename is warranted, do it as a dedicated migration with
  import shims, targeted tests, and docs updates in the same change.
- Keep future 2P, pixel-to-state, and EnvPool references out of current docs
  unless they become active scope again.
