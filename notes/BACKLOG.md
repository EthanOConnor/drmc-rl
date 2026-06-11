# BACKLOG.md - drmc-rl

Forward backlog for the current architecture. Historical plans that are no
longer active belong in `notes/archive/` or git history, not in this file.

## P0: Top-Play Roadmap (see docs/DESIGN_TOP_PLAY.md)

Sequenced plan toward beating strong humans (speedrun + VS):

1. **Speedrun push**: capacity bump (d_model 256, 4–6 blocks) once curriculum
   reaches positive levels; small GRU over decision embeddings; long runs
   through the level ladder with per-checkpoint `tools.eval_policy` sweeps;
   tighten mastery-gated time budgets toward BestTimesDB floors.
2. **VS engine port**: 2P attack/garbage rules from the disassembly into the
   pool (combo→attack tables, garbage drop scheduling, win/loss); paired-env
   pool ABI; parity tests vs emulator 2P mode.
3. **League training**: snapshot pool + PFSP opponent sampling + exploiter
   agents; round-robin Elo harness (extend tools/eval_policy to head-to-head).
4. **Strength dial**: value-gap sampling + reaction-time model + optional
   persona conditioning; calibrate notches against the Elo ladder.
5. **Eval-time shallow search**: depth-2 expectimax over placements × next
   pill using warp rollouts (~30 µs/sim) with the value head at leaves, for
   exhibition play.

## P0.5: Throughput Tail (post-2026-06-09 substrate)

- PPO update is the current ceiling (~1.1 s / 2048 decisions): try
  torch.compile on the candidate net (MPS + CPU), then async rollout/update
  overlap (collect next rollout while updating).
- Planner worst case (sparse low-virus boards, deep tuck poses, 4–9 ms):
  NEON the v2/v4 inner loop, factor the per-action Y-stage recompute, or add
  an explicit `DRMARIO_REACH_HORIZON` knob (documented approximation, off by
  default). v3's bit-sliced design is a measured dead end — see WORKLOG
  2026-06-09 before attempting anything similar.
- `build_reset_spec` is called for all envs every step in the pool vec env
  (~1% of wall); build specs only for envs actually resetting.
- Demo-parity drift: `tests/test_game_engine_demo.py` fails from pre-existing
  engine/source mismatch (SCRUTINY 2026-06-09). Recover the working source
  state from `../drmario-native` history.

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
- Curriculum state is not saved in checkpoints; resume restarts the ladder (fast with a strong policy, but wasteful). Persist curriculum stage/stats in the checkpoint payload.
- Human-corpus integration: see docs/HUMAN_CORPUS_INTEGRATION.md — fightcadeRatings event schema v2, ingest + planner-annotation tools, replay-parity acceptance tests for the 2P port, WHR dial calibration, BC league seeding.
