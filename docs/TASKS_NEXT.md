# Tasks Next

This is the short forward checklist. Use `notes/BACKLOG.md` for the fuller
roadmap.

## P0: Keep the Current Path Reproducible

- Confirm fresh-checkout setup after the `game_engine/` submodule split:
  `git submodule update --init --recursive`, build `libdrmario_pool`, and run a
  `training.run --dry_run` smoke.
- Fix or document any local venv drift before reporting test health.
- Keep README, `AGENTS.md`, and setup docs aligned with `cpp-pool` as the
  training default.
- Keep the local dev environment on the newest compatible NumPy/OpenCV/Torch
  stack; run `pip check` after dependency changes.

## P1: Train Better Placement Policies

- Treat candidate scoring as the default policy path.
- Use `training/configs/smdp_ppo_heatmap.yaml` only for baseline comparisons
  against 512-way heads.
- Track candidate encoder choice (`cnn` vs `col_transformer`) with decisions/sec,
  clear rate, and curriculum progression.
- Keep `placements/feasible_mask` and cost-to-lock packing tests tight; policy
  bugs here silently poison training.

## P2: Make Evaluation Harder to Fool

- Add fixed-seed certification gates at curriculum boundaries.
- Extend best-time reporting beyond single best per `(level, seed)` where useful.
- Add deterministic smoke runs that exercise `ln_hop_back` advancement and budget
  behavior.

## P3: Maintain the Parity Lane

- Keep libretro/tetanes-style emulator work framed as oracle/debug support, not
  the training default.
- Expand native-vs-emulator trace coverage around spawn, lock, settle, top-out,
  and stage clear.
- Preserve `cpp-engine` only where it still provides compatibility/parity value.

## P4: Remove Historical Weight Safely

- When a stale command, package name, or doc habit is found, either update it to
  the current architecture or move it under `notes/archive/`.
- Do not delete active planner code just because it lives under the historical
  `envs/retro/` package name.
- If a larger rename is warranted, do it as a dedicated migration with import
  shims and tests.
