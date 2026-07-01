# WORKLOG.md — drmc-rl

Chronological log of work done. Format: date, actor, brief summary.

---

## 2026-06-11 (vs-opp-obs) – Coding Agent (Claude) – Opponent-Board Observability for VS Self-Play

- New `state_repr=bitplane_bottle_conn_mask_vs` (20ch): own bottle 0–7
  unchanged, opponent bottle planes 8–15 (post-reduction copy of the pair
  partner's own planes), feasible planes moved to 16–19. Built entirely in
  `training/envs/drmario_vs_vec.py` from state the vspool already exports;
  no engine changes. Default config stays 12ch — gate is the repr name.
- `aux_spec: v1_vs` (72 = 57 + 15): v1 prefix bit-identical, appended
  opponent scalars (opp viruses/84, garbage pending both directions /4,
  opp pill + preview one-hots) from new `vs/*` info keys.
- `tools/expand_checkpoint.py`: checkpoint surgery (zero-init stem-conv +
  aux-encoder slices, coords relocated, both state_dict and ema, optimizer
  dropped, embedded cfg rewritten). Verified bitwise-identical outputs vs
  the original net — including the real 535M champion at d192 scale.
- Frozen-opponent pool handles mixed archs: per-entry `aux_spec`, old
  8-channel nets read their board prefix from 20ch obs unchanged.
- Tests: `tests/test_vs_opponent_obs.py` (layout, scalars vs vspool
  buffers, v1_vs==v1 prefix, surgery equality); full suite 197 passed
  (1 pre-existing `test_game_engine_demo` failure, also fails on clean
  tree). Docs: `docs/VS_OPPONENT_OBS.md`.

## 2026-06-11 (ponder) – Coding Agent (Claude) – Dead-Time Pondering for the Search Policy

- `models/policy/search_policy.py`: `PonderingSearchPolicy(SearchPolicy)` —
  after a placement is committed, a single background worker (own 64-env
  pool runner, newest job wins) searches the *next* decision during the
  fall/lock/clear dead time: resolve the committed action (deterministic),
  full-width ply-1 over every feasible next placement, depth-1 values for
  all 9 candidate preview pairs in one trunk pass
  (`_pill_conditioned_values`, preview marginalized per pill pair), then a
  per-pair ply-2 beam + marginalized leaves + the existing Q backup
  (budget-checked, `ponder_budget_s` default 1.0). Cache keyed by
  (normalized post-commit board bytes, next pill); `decide()` consults it
  first (hit ≈ 0.03 ms, restricted argmax over the caller's mask), miss =
  normal deadline search + stale-job abort. Refactors: `_step_reward` /
  `_marginal_leaf_values` take an optional `buf`, `_combo_values` extracted.
- `tools/live_agent_server.py --ponder` (implies `--search`): kicks
  `start_ponder` after every plan write; a cache hit at the next spawn
  commits with `PONDER_MARGIN=2` frames instead of 6 (planner+script p95
  ~10 ms < 33 ms; measured hit spawn→plan p50 1–8 ms) and logs `n_options`
  at both margins. Desync/late replans call `ponder_invalidate()`.
  Feasibility gain at speed-ups 40 HI: mean 28.8 options at margin 2 vs
  12.2 at margin 6 (+16.6/decision); ~0 at slow early-game gravity.
- `tools/vs_head_to_head.py --a-ponder`: side 0 ponder-search vs side 1
  plain-search (a real `SearchPolicy`); offline dead time simulated by
  running each ponder job to completion before the env steps (fair: job
  wall p95 < 0.8 s << real spawn-to-spawn dead time, and the report includes
  it). `tools/bench_search.py --ponder`: job wall + hit/miss decide latency.
- Measured (M3 Max/MPS, vs2_02 step540020887, level 14 HI, beam 8): job
  wall p50 0.25–0.7 s / p95 < 0.8 s; hit rate ~97 % in 1P-ish flow; all 9
  pairs reach depth 2 in budget. Tests: `tests/test_ponder_policy.py`
  (cache hit/miss/invalidate with stubbed compute, supersede/abort thread
  lifecycle via events — no sleeps, real end-to-end ponder→hit); full suite
  minus test_game_engine_demo green (174 passed).

## 2026-06-11 (search) – Coding Agent (Claude) – Inference-Time Search + Live-Bridge Wiring

- `models/policy/search_policy.py`: `SearchPolicy` — depth-2 beam-K
  policy-guided expectimax over the 1P pool's checkpoint reset
  (docs/SEARCH_DESIGN.md). Anytime with a between-stages deadline
  (fallback → ply1 → depth1 → depth2); `inject_plan` round-trips so no board
  is BFS-planned twice; the unknown pill-after-preview is neutralized by
  exact analytic marginalization over the 81 (pill, preview) color pairs
  (single engine sim, fully deterministic decide()); reward-augmented
  discounted backup `Q = r̂1 + γ^τ1(r̂2 + γ^τ2 V)` with 1P reward replication
  from `_RewardCfg` and a VS garbage-volley proxy (tiles/4). Key debugging
  find: a pure value-head backup is anti-clear (V excludes the reward that
  just moved into the past) — it *lost* to the plain policy until the reward
  terms were added.
- Wiring: `tools/live_agent_server.py --search [BEAM] --search-deadline-ms
  --device`, `tools/eval_policy.py --search`, new `tools/vs_head_to_head.py`
  (probe) and `tools/bench_search.py` (latency).
- Quality probe (vs2_02 step540020887, level 14 HI, 60 matches,
  tools/vs_head_to_head.py): search beat the plain argmax of the same
  checkpoint **49W-11L = 81.7% win rate, Wilson 95% CI [0.70, 0.89]**
  (search ms p50 63 / p95 77; agreement with the policy argmax ~= 0.30).
  1P probe: champion step535164979, level 20, 32 eps, seed 4242 — plain
  84.4% clear / p50 60986 / p90 108637 frames; search 59.4% clear /
  p50 50901 / p90 72325. The search faithfully optimizes the trained
  objective (time penalty makes clears slower than ~90k frames worse than a
  topout), so it trades clear rate for much faster clears — roughly neutral
  on the training return, a calibration caveat for pure-clear-rate use
  (see SCRUTINY).
  decide() p50 ~= 50 ms, p95 ~= 79 ms at beam 8 (MPS leaves + CPU small
  forwards); spawn->plan-written p95 ~= 88-94 ms inside the 6-frame margin.
- Tests: `tests/test_search_policy.py` (pure backup, fast-aux parity vs the
  adapter shim, decide determinism/deadline/empty-mask; checkpoint-gated on
  the pool lib).

## 2026-06-11 – Coding Agent (Claude) – Seedlab Per-Seed Search + Jagged Explorer

- Deep-dive design (`docs/SEEDLAB_SEARCH.md`): per-seed optimization is
  deterministic perfect-information planning (full pill reserve known
  analytically); exact additive frame costs; O(1) node restore via the
  pool's extraction-checkpoint reset.
- `seedlab/search.py`: `SearchEngine` on raw `DrMarioPoolRunner` (batch
  checkpoint restore + step), policy-guided beam (T2: priors top-M,
  board-hash dedup, rank g + λ·v_rem + κ·Σ−logπ), exact DFS B&B (T3,
  anytime, certificate only when closed AND replay-exact). Key debugging
  find: without the κ policy term the beam degenerates into tau-greedy
  stacking and never clears (SCRUTINY M3).
- `seedlab/explore.py` + `python -m seedlab explore`: jagged scheduler —
  50% uniform / 50% slowest-decile seed sampling, heavy-tailed tier mix
  (rollouts → beam w8/w32/w128 → exact), `search_log` effort table, thread
  caps (default 2), replay-verified recording via new `CatalogDB.record_best`
  (search results don't pollute attempt distributions).
- Perf: memmove board fill in `build_reset_spec` (was a 128-iteration python
  loop), cached per-node ctypes specs. Bench (2 threads): level 0 ~120
  nodes/s (planner-bound; sparse boards are the v4 worst case), level 14
  ~400–700 nodes/s. Beam already beats pass-0 greedy: seed 0x8988 level 0
  1651 → 1313 frames (−20%); level 7 W32 −30% vs W8.
- Tests: `tests/test_seedlab_search.py` (restore parity, beam replay
  invariant, exact incumbent respect, explorer smoke) — checkpoint-gated.
- TUI: `seedlab tui` gained a search-activity panel (last-hour/all-time
  iters · records · nodes/s, per-tier economics incl. rec% and avg Δframes,
  recent-improvements feed) and coverage columns `searched` (distinct seeds
  touched by the explorer) and `proven` (certified-optimal solutions).
- Engine fast path (~3–11× search): additive struct_size-guarded pool
  features — `DrmResetSpec.inject_*` (checkpoint restores seed the planner
  cache with the parent node's own outputs; ensure_planner cache-hits, falls
  back to a real plan on hash/spawn mismatch) and
  `DrmPoolConfig.lazy_decision_outputs` (step emits exact state-only context,
  planning deferred to frontier survivors via phase-2 restore). Beam runs on
  a lazy pool; rollouts/exact/replay on a normal one. Bench level 0 W8:
  9.0 s → 0.8 s, identical replay frames. Plus pool workers thread split
  (~2×). Gotcha logged: Makefile header deps are weak — `make clean` after
  capi changes (and that rebuild surfaced the documented demo-parity drift
  as two failures; SCRUTINY updated).
- Explorer scheduling reworked width-first per Ethan: frontier level (lowest
  with uncovered seeds) gets the bulk, geometric decay above, uniform
  residue everywhere. TUI/report show best times as frames + NTSC seconds.
- Second pass per Ethan: priority floor (default level 4 — build up from
  there, 0–3 random backfill via the uniform residue), new `greedy x1` tier
  (pass-0 single-argmax-rollout method), and tier selection switched from
  weighted sampling to an equal-resource wall-time ledger (`pick_tier`:
  least-spent applicable tier next, seeded from search_log → lifetime 1/n
  split; a new tier catches up by design — Ethan confirmed that semantics
  after initially misreading the all-greedy catch-up phase, so the explorer
  now prints a startup ledger banner naming the laggard tier).
  Budget-exhausted rollouts now recorded as failed attempts. Observed greedy-argmax dithering at L5+ (stalls at ~2
  viruses until the decision cap) — policy behavior, documented in
  docs/SEEDLAB_SEARCH.md, mitigated by the other tiers.

## 2026-06-11 (later) – Coding Agent (Claude) – Admissible Bounds + Polish Tier

- `seedlab/bounds.py`: per-step frame minima measured from the rules-exact
  engine on extremal boards (grounded checkerboard support → earliest drop
  failure; min over frame parity; speed_counter=0 matches every real spawn).
  Continuing 37f (MED/HI) / 58f (LOW); terminal 8f via immediate stage-clear
  (faster than topout). Pitfall found: floating junk rows settle post-lock
  and inflate the measurement — supports must be grounded. Admissibility
  fuzz-asserted (`test_step_bounds_admissible`).
- `exact_search`: schedule-exact future bound (pill index = step index →
  speed-ups known per future step), board-aware per-color line-component
  pills bound, virus-progress DFS ordering. Honest result: still no
  certificate closures at 300k nodes even on level 0 — floor 37f vs ~50–70f
  true average + non-optimal incumbents; next unlock is per-node
  planner-aware bounds (docs/SEEDLAB_SEARCH.md, BACKLOG).
- Greedy-vs-training clarity (Ethan's question): eval at step ~610M shows
  argmax clears L5 92% / L10 ~80% but with 8.7k–29k frame medians; the
  explorer's flat 400-decision rollout cap was truncating real clears.
  Cap now level-scaled (200 + 80·level): L4 greedy 7/8 clears.
- New `polish` tier (Ethan): equal-share slice dedicated to the top-5
  fastest seeds per level; beam width escalates with prior polish visits
  (32→64→128→256 from search_log) and repeat visits Gumbel-perturb the
  policy-prior ordering so each visit searches new subtrees. The exact tier
  also retargets record seeds (only near-optimal incumbents can close).
- Certificates parked per Ethan ("don't worry about certificates for now");
  bounds machinery stays (sound, speeds anytime pruning).
- TUI colors (Ethan): recent-records entries turn orange when they are the
  current level record; per-level bests turn green when faster than the
  human IL world record (speedrun.com drmariones, fetched 2026-06-11 into
  `seedlab/report.py:HUMAN_WR_SECONDS`; realtime-vs-frame-metric caveat
  noted inline). Level 0 already shows green: 273f = 4.5s vs 7.438s human WR.
- Polish boosted (Ethan): share-weighted tier ledger (TIER_SHARES; polish
  3.0 → ~1/3 of compute, others 1/9 each) and the polish field deepened to
  the top-16 seeds per level.
- Engine thread lifecycle fixed (Ethan: "no speedup from more threads"):
  `parallel_for_envs` spawned/joined std::threads PER CALL and each exiting
  thread freed the planner's ~2-3 MB thread-local BFS ctx (re-malloc'd next
  call) — fine for training's few big calls/sec, fatal for search's hundreds
  of small calls/sec. Replaced with a persistent worker pool
  (`DrMarioPool::run_parallel`, condvar dispatch; ctxs freed once at pool
  teardown). Scaling restored: L7 beam w32 864→1053→1197 n/s at 1/2/4
  workers, identical results. Also killed 32 leaked
  `drmario_engine --wait-start` procs from cpp-engine test runs (~3 cores;
  SCRUTINY'd).
- Then profiled the residue: torch.conv2d on 1-thread CPU was 92% of beam
  wall (10.5s of 11.3s; engine just 2.1s). CPU torch threads make it WORSE
  at these batch shapes; wider engine batches (E=256) change nothing. Fix:
  dual solver — wide beams (w32+) run the net on MPS (~35% faster wall,
  8.0s vs 10.7s), narrow work stays on CPU; torch pinned to 1 thread; beam
  engine widened to 256 envs so a layer lands in one call. `--device mps`
  now recommended for the explorer. Smoke: polish on MPS took L12's record
  20,062 → 5,309 in two escalating visits.

## 2026-06-10 – Coding Agent (Claude) – Seed Catalog ("seedlab")

- New `seedlab/` package + `docs/SEED_CATALOG.md`: prime95-style exhaustive
  per-seed catalog of best-known and typical clear times, for agent eval
  baselines, human skill grading (fightcadeRatings v2 events carry init RNG),
  and speedrun research.
- Measured seed-space facts: the NES RNG orbit from `0x8988` has period
  32,767; `0x0000` is a lockup fixed point; the other 32,768 states are
  unreachable on console. Census over levels 0–20 (analytic, no rollouts,
  ~4 s/level): zero game-hash collisions — 688,107 distinct games per speed.
- `seedlab/rng.py`: pure-Python mirror of `rng_step`/`generatePillsReserve`/
  `addVirus` from `GameLogic.cpp`; byte-exact board + pill parity vs the
  engine asserted in `tests/test_seedlab_rng.py`.
- `seedlab/db.py`: WAL sqlite (`data/seed_catalog.sqlite3`) with games census,
  per-(level,speed,seed) aggregates + 64-sample reservoir, best solutions as
  replayable uint16 action traces, and a lease-based `work_units` queue.
- `seedlab/worker.py`: claims units, runs K attempts/seed on
  `DrMarioPoolVecEnv` via the new `seed_provider` hook (per-env exact engine
  seeds, surgical addition to the vec env + `request_reset` for decision
  caps); solver modes checkpoint/greedy-cost/random.
- CLI `python -m seedlab init|worker|report|grade|verify|tui`; rich dashboard.
- End-to-end test with the current best checkpoint: level-0 unit searched,
  solutions stored and re-verified frame-exact on replay
  (`tests/test_seedlab_worker.py`). Catalog initialized and pass-0 queue
  (1,344 units, levels 0–20 speed 2) enqueued.

## 2026-06-09 – Coding Agent (Claude)

- Profiled the `cpp-pool` training path end to end. `sample` on a single-env
  bench shows **99.7% of wall time inside `drm_reach_bfs_full`** (the planner
  BFS); engine frame simulation, script replay, and Python overhead are noise.
- Isolated BFS cost with `DRMARIO_REACH_STATS=1`: a real top-of-board spawn
  visits ~750k–900k states and evaluates 7–9M transitions per spawn
  (**11–13 ms**), even though early-exit triggers at depth ~30–34. Baseline
  pool throughput: 8.7 ms/decision single env; ~43k FPS / 956 dec/s at 16 envs.
- Root-cause analysis of the state-space blowup (basis for planner v2):
  - `hor_velocity` is semantically dead whenever `hold_dir == NEUTRAL`
    (any future press edge-resets it to 0 before it is read), but the BFS keys
    on it anyway → up to ×16 redundant states for neutral-hold states.
    Collapsing it is exact, not an approximation.
  - `frame parity` is a pure function of BFS depth (p = (p0 + depth) & 1) and
    never branches within a depth; keying on it wastes ×2 memory.
  - Long same-direction lateral holds only matter through DAS (hv ≥ 16).
    DAS repeat (6 frames) is strictly slower than tap repeat (2 frames), and
    a single DAS auto-tuck at frame t is replicable by an edge press timed at
    t for equal cost. Capping modeled hold-run length (keeping exact hv
    tracking only inside short runs) should preserve exact minimal costs;
    verify empirically vs the existing BFS as oracle.
  - Parent-pointer/script bookkeeping writes ~5 bytes per visited state into
    ~50 MB arrays (random access). Training only needs feasibility + costs;
    scripts can be reconstructed on demand (or skipped entirely once the pool
    warps to the lock pose instead of replaying controller scripts).
- Plan reordered accordingly: planner v2 first (exact-equivalence-tested),
  then pool warp-execution, then training-stack modernization.
- **Planner rewrite landed** (`reach_native/drm_reach_full.c`):
  - `drm_reach_bfs_v2`: exact hv-collapse for neutral hold_dir (hv is dead
    there — every lateral input from neutral is an edge press that resets it),
    optional costs-only mode. Verified exact vs v1 on 1k fuzz cases. ~2×.
  - `drm_reach_bfs_v3`: bit-sliced (x × speed_counter) blocks. Correct but
    SLOW — sc bits shift every frame so every key re-enters the frontier every
    depth, and 128-byte block ops lose to v1's scalar loop. Kept as a
    documented dead end; do not resurrect without fixing re-expansion.
  - `drm_reach_bfs_v4` (production): greedy-witness upper bounds (pattern
    plans + shaft-descent/rotate-at-bottom tucks + gradient-follow on a
    composite-frame geometric distance field) + admissible lower-bound gate
    (vertical descent rate ⊔ per-pose backward geometric BFS) folded into a
    per-depth 16×4×8 allowance bitmask. Exact (same fuzz harness), 4× on the
    worst sparse-board cases, up to ~60× on open/mid-game boards.
- **Pool warp execution** (`game_engine`): `GameLogic::warp_fall` jumps the
  engine to the lock pose (frame counter += cost, status-row countdown advance,
  sc=0, confirmPlacement, PillPlaced) instead of replaying controller scripts;
  `DrMarioPool` plans with v4 costs-only and skips per-env 1 MB script buffers.
  `DRMARIO_POOL_WARP=0` restores the legacy replay+v1 path. A 300-step
  4-env trajectory comparison (levels 0/5/10/18, speeds 0–2, resets included)
  is byte-identical across tau/boards/masks/costs/adjacency/locks.
- **End-to-end**: bench_multienv random-action: 1 env 19.4k FPS (was ~5k);
  64 envs 182k FPS / 3.1k decisions/s (was ~40k / ~680). Worst remaining
  planner cases are sparse low-virus boards with deep tuck poses (~4–9 ms);
  mid-game boards ~0.5 ms.

## 2026-06-10 – Coding Agent (Claude)

- **Training-loop sync purge** (`training/algo/ppo_smdp.py`,
  `models/policy/placement_dist.py`): live-profiling showed the trainer at
  168 decisions/s vs ~3k env-only — all MPS dispatch/sync overhead. Rollout
  selection now runs under `torch.inference_mode` with one device→host sync
  and CPU-side sampling; per-minibatch metric `.item()` syncs replaced by
  on-device stacking (one sync per update); `MaskedPlacementDist` rewritten
  branchless (`torch.where`, no clone/bool-index/host-sync); aux_v1 features
  built batched (equivalence test added). Config moved to 64 envs /
  2048 decisions-per-update / minibatch 512 / 3 epochs; checkpoint interval
  100k→5M frames (was gzipping every ~2 s at the new speed).
  Result: 168 → **954 decisions/s**, 11k → **64k frames/s** during training.
- **Eval harness** (`tools/eval_policy.py`): fixed-seed, curriculum-free
  checkpoint evaluation across levels (clear rate, frames-to-clear p50/p90,
  baselines: random, greedy-cost). Sanity: an 80-second (5M-frame) checkpoint
  clears level 0 at 12.5% vs 0% for both baselines.
- **Docs**: `docs/DESIGN_TOP_PLAY.md` (system design toward top-human play:
  speedrun track, native 2P VS port, PFSP league, strength dial, eval-time
  shallow search over warp rollouts), `docs/BENCHMARKS_2026-06-09.md`
  (measured numbers), BACKLOG P0 roadmap, README/AGENTS pointers.

- Rewrote the forward-facing docs and backlog around the current `cpp-pool`
  placement-SMDP architecture, keeping emulator/libretro/Stable-Retro material
  in a parity/debug lane instead of the default onboarding path.
- Added `docs/PROJECT_DEEP_DIVE_2026-05-07.md`, a current-state audit covering checkout status, actual training/backend architecture, metaproject reference layout, verification results, documentation drift, and future agent entry points.
- Added the sourced Karpathy-inspired behavior principles bridge in `CLAUDE.md` and expanded `AGENTS.md` / `docs/REFERENCES.md` with the local `/Users/ethan/dev/drmario/` umbrella workspace convention.
- Verified the local `cpp-pool` smoke path without pytest: `training.run --dry_run`, pool library presence, default/candidate config env resets, a two-env pool step, and a candidate-policy forward pass.
- During completion audit, refreshed `docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md`
  as a RAM-map reference rather than the live architecture source, and marked
  `docs/CPP_SIM_NOTES.md` as parity/core-rules notes.

## 2026-05-08 – Coding Agent (Codex)

- Repaired the local editable install from this checkout with `.[dev,rl,viz]`;
  pytest is now available, NumPy/OpenCV are on the newer compatible line
  (`numpy==2.4.4`, `opencv-python==4.13.0.92`), and `pip check` is clean.
- Expanded `tools/bench_multienv.py` with repeat statistics, component timings,
  machine-readable JSON/CSV output, action-selection modes, and bounded batch
  runs for tests.
- Added `tools/bench_policy.py` for policy/network benchmarking, including
  parameter counts, forward latency, candidate-packing overhead, and decisions
  per second.
- Made candidate scoring the default policy in `training/configs/smdp_ppo.yaml`,
  added `training/configs/smdp_ppo_heatmap.yaml` as a controlled baseline, and
  recorded benchmark results in `docs/BENCHMARKS_2026-05-08.md`.
- Made optional MLX imports lazy in `training/speedrun_experiment.py` so
  torch-only tests and training paths do not import MLX during collection.

## 2026-03-26 – Coding Agent (Codex CLI)

- Extracted the native C++ Dr. Mario engine from the tracked `game_engine/` directory into a standalone repo at `EthanOConnor/drmario-native`.
- Made the extracted engine repo self-contained for builds by vendoring the required reachability helper into `third_party/reach_native/` and dropping tracked build artifacts/logs from the new repo.
- Re-integrated the engine back into `drmc-rl` as the `game_engine/` git submodule so existing engine paths/imports remain stable.
- Updated root docs to call out the new submodule-based setup.

## 2025-10-17 – Coding Agent (Codex CLI)

- Confirmed ROM revision (Dr. Mario Japan/USA rev0, CRC32 0xB1F7E3E9).
- Extracted and validated RAM map from disassembly.
- Implemented RAM→state mapping in `envs/specs/ram_to_state.py`.
- Added tests and CLI tooling (`tools/ram_planes_dump.py`).
- Documented RNG, virus placement, and state spec in `docs/`.

## 2025-12-19 – Coding Agent (Codex CLI)

- Fixed SMDP-PPO multi-env rollouts: batch actions per decision, per-env τ accounting, and per-env GAE with `env_id` tracking.
- Added `emit_raw_ram` env option to trim AsyncVectorEnv IPC payloads; debug runs keep raw RAM enabled.
- Reworked debug TUI for multi-env: grid/summary view, env selection hotkeys, restart-only env count changes.
- Added `tools/bench_multienv.py` scaling harness (sync vs async, fps/speedup/efficiency metrics).

## 2025-12-20 – Coding Agent (Codex CLI)

- Normalized Gymnasium vector `info` dicts into per-env info lists inside the debug UI wrapper to avoid array→scalar conversion errors when rendering multi-env boards (`training/envs/interactive.py`).
- Fixed nested info unbatching so dict-valued entries (like `preview_pill`) are split per-env rather than broadcast (`training/envs/dr_mario_vec.py`).
- Added a lightweight benchmark smoke test that runs the multienv harness for a short sync run when the C++ engine binary is available (`tests/test_bench_multienv_smoke.py`).
- Logged curriculum graduation events with frames/episodes totals and deltas in SMDP-PPO (`training/envs/curriculum.py`, `training/algo/ppo_smdp.py`).
- Improved debug TUI restart UX (status TTL + alt-screen) and ensured debug sessions always stop/close on exit (`training/ui/runner_debug_tui.py`, `training/run.py`).
- Added compact grid rendering for multi-env boards, auto-downshifted UI refresh with env count, and a unified timing (ms/frame) breakdown with a wider summary column for readability (`training/ui/board_viewer.py`, `training/ui/runner_debug_tui.py`).
- Hid the per-env reward panel while in summary/grid view to keep the UI focused (`training/ui/runner_debug_tui.py`).
- Debounced `[`/`]` env-count restarts, added a numeric env-count entry mode (`e` + digits + Enter), and expanded the footer to show pending restart state (`training/ui/runner_debug_tui.py`).
- Added a curriculum advancement report script (`tools/report_curriculum.py`).
- Added `tests/test_cpp_backend_multienv.py` for multi-instance C++ backend isolation.
- Removed noisy startup warnings by (a) switching environment package capture to `importlib.metadata` (no `pkg_resources` deprecation warning) and (b) lazy-importing W&B so its pydantic warnings don’t fire unless W&B is enabled (`training/utils/reproducibility.py`, `training/diagnostics/logger.py`).
- Defaulted `training.run` to write each invocation under a unique `run_id` subdirectory (unless `--logdir` is provided) and recorded `run_id`/`logdir` in run metadata (`training/run.py`).
- Emitted curriculum snapshots in SMDP-PPO `update_end` events and displayed curriculum level/goal/success-window and env-level distribution in the Rich TUI (`training/algo/ppo_smdp.py`, `training/ui/tui.py`, `training/ui/event_handler.py`).
- Added a new `ln_hop_back` curriculum mode (probe + ln-tightened hop-backs) and set it as the default for `training/configs/smdp_ppo.yaml`; extended synthetic match-count stages to `-15..-4` (1..12 matches) (`training/envs/curriculum.py`, `envs/retro/drmario_env.py`).
- Made task time budgets “soft”: allow play past budget exceedance and replace the terminal clear bonus with a smooth time-goal reward that’s positive under-budget and negative over-budget (`envs/retro/placement_env.py`).
- Relaxed curriculum confidence defaults to 1-sigma (stage pass) and 2-sigma (mastery), and slowed the ln-style pass-rate ramps via `pass_ramp_exponent_multiplier=1/3` (`training/envs/curriculum.py`, `training/configs/smdp_ppo.yaml`).
- Reworked curriculum gate stability: replaced tiny rolling windows with an EMA-based Wilson lower bound (min effective sample size) and added a `min_stage_decisions` floor; stopped SMDP-PPO rollouts immediately on curriculum advancement to keep PPO updates stage-pure (`training/envs/curriculum.py`, `training/algo/ppo_smdp.py`, `training/configs/smdp_ppo.yaml`).

## 2025-12-21 – Coding Agent (Codex CLI)

- Fixed a Gymnasium `AsyncVectorEnv` crash caused by returning `None` for sometimes-numeric info keys; unset optional fields are now omitted (`envs/retro/placement_env.py`).
- Added per-run `drmario_engine` pidfile tracking and best-effort cleanup on shutdown to reduce orphaned engine processes after crashes/forced worker termination (`envs/backends/cpp_engine_backend.py`, `training/run.py`).
- Normalized virus-clear rewards so the total per-episode virus-clear reward is constant across levels; updated reward config + docs.
- Updated `tools/plot_success_by_level.py` to default to confidence-lower-bound plots and skip the first 10 episodes; added a metric selector for alternate plots.
- Freed thread-local reachability BFS buffers for cpp-pool worker threads to prevent per-step memory leaks during parallel planning (`reach_native/drm_reach_full.c`, `game_engine/DrMarioPool.cpp`).
- Added SMDP-PPO checkpoint warm-start support via `train.init_checkpoint` (with optional optimizer/step restore) for extending runs with new curricula (`training/algo/ppo_smdp.py`).
- Fixed SMDP-PPO resume throughput stats to use steps since resume, and hardened checkpoint IO (atomic saves + clearer load errors) to avoid corrupted resumes (`training/algo/ppo_smdp.py`).
- Implemented an in-process batched C++ pool backend (`game_engine/libdrmario_pool`) that owns N engine instances, integrates the native reachability planner, and emits decision-time masks/obs plus step-time events/counters.
- Added a `cpp-pool` training backend (ctypes wrapper + lightweight vector env) and wired it into the real env factory (`envs/backends/drmario_pool.py`, `training/envs/drmario_pool_vec.py`, `training/envs/dr_mario_vec.py`).
- Added `python -m tools.build_drmario_pool` and a pytest smoke test for the pool backend (`tools/build_drmario_pool.py`, `tests/test_cpp_pool_smoke.py`).
- Updated the SMDP-PPO config to default to `backend: cpp-pool` (`training/configs/smdp_ppo.yaml`).
- Parallelized cpp-pool planner/step work across envs, made the native reachability helper thread-local, and added `DRMARIO_POOL_WORKERS`/`-pthread` support for tuning worker count (`game_engine/DrMarioPool.cpp`, `reach_native/drm_reach_full.c`, `game_engine/Makefile`).
- Added a candidate-scoring placement policy (packed feasible actions + explicit cost-to-lock feature) and wired it into SMDP-PPO (`models/policy/candidate_policy.py`, `models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`).
- Added a candidate-policy config + tests (`training/configs/smdp_ppo_candidate.yaml`, `tests/test_candidate_policy.py`).
- Added a cpp-engine integration smoke test for the candidate policy (skips if `game_engine/drmario_engine` is unavailable) (`tests/test_candidate_policy_cpp_engine_smoke.py`).
- Reconciled candidate-policy implementation with the existing docs/config/tests (restored missing modules + ensured interfaces match SMDP-PPO candidate mode); `pytest -q` passes.
- Suppressed Ctrl+C shutdown tracebacks in debug runs by catching `KeyboardInterrupt` during session/env teardown (`training/run.py`, `training/envs/dr_mario_vec.py`).
- Strengthened candidate-policy correctness + signal: deterministic packing tie-breaks, PPO update asserts repacked candidates contain the chosen macro action, and candidate-local patches now include color+virus planes (not just occupancy) (`models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`, `models/policy/candidate_policy.py`).
- Improved candidate-policy throughput: precomputed patch offsets to avoid per-forward allocations, prepacked candidates once per PPO update (no per-minibatch repacking), sanitized NaN costs, lowered default candidate `Kmax` to 128, and added targeted tests (`models/policy/candidate_policy.py`, `models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`, `training/configs/smdp_ppo_candidate.yaml`, `tests/test_candidate_policy.py`).
- Added a plotting utility to select a run log and plot `curriculum/rate_current` vs steps per curriculum level, breaking lines across gaps when a level is not active (`tools/plot_success_by_level.py`).
- Strengthened candidate scoring with spatial trunk context: gather CNN feature-map features (or column-token features) at the candidate’s two landing cells/columns and feed them to the per-candidate MLP (`models/policy/candidate_policy.py`, `tests/test_candidate_policy.py`).
- Pruned `runs/` to reclaim disk space (kept newest 3 runs under `runs/smdp_ppo_candidate/` and removed older run artifacts under `runs/smdp_ppo/` + `runs/ppo_example/`).
- Tweaked `ln_hop_back` curriculum defaults: skip immediate full hop-backs when a new probe stage is already above the k=1 pass target, hop back to the 3rd-highest mastered hop-back level (not always `start_level`), and add a configurable bailout (fraction of run `total_steps`) for stuck probe stages (`training/envs/curriculum.py`, `training/envs/dr_mario_vec.py`, `tests/test_curriculum_scheduler.py`).

## 2025-12-22 – Coding Agent (Codex CLI)

- Added line-by-line inline documentation to `training/configs/smdp_ppo_candidate.yaml`, cross-referencing the current SMDP-PPO + cpp-pool code paths and flagging unused/ignored knobs in the default experimental setup.
- Logged a new scrutiny item about unordered pill embeddings vs directed macro-action semantics (`notes/SCRUTINY.md`).

## 2025-11-22 – Coding Agent

- Implemented C++ game engine core logic (`game_engine/GameLogic.cpp`).
- Completed RNG, level generation, basic gravity/matching/clearing.
- Set up shared memory IPC between engine and Python.
- Created monitor tool for visual debugging (`game_engine/monitor.cpp`).

## 2025-01-04 – Coding Agent

- Completed SMDP-PPO placement policy implementation.
- Implemented 3 policy heads: dense, shift_score, factorized.
- Created `MaskedPlacementDist` for action masking.
- Added `DecisionRolloutBuffer` with SMDP discounting (Γ=γ^τ).
- Created a one-off training launcher (later removed; canonical entrypoint is `python -m training.run`).
- All 12 unit tests passing.
- Documented in `notes/archive/root_docs/PLACEMENT_POLICY_IMPLEMENTATION.md` and `notes/archive/root_docs/IMPLEMENTATION_COMPLETE.md`.

## 2025-12-16 – Coding Agent (Antigravity)

- Performed comprehensive codebase review.
- Created developer handoff report with prioritized findings.
- Identified critical gaps: no notes system, C++ engine missing DAS/wall kicks.
- Found 64 tests all passing.
- Implemented inter-session notes system (`notes/` directory).

## 2025-12-16 – Decruftification & Runner Refactor

- Deleted stub directories: io-bridge/, streaming/, sim-envpool/, retro/
- Deleted orphan files: patches, package.json, Screenshot
- Dropped the vendored drmarioai Java bot snapshot (kept external link + summarized notes only)
- Updated .gitignore: added cores/, checkpoints/, .venv-*/
- Updated docs/REFERENCES.md with drmarioai, Rich, Textual, WandB
- Created training/ui/tui.py: Rich-based TUI with sparklines (replaces Tkinter)
- Created training/utils/devices.py: unified MLX/PyTorch device resolution
- Created training/utils/wandb_logger.py: WandB integration with graceful fallback
- Enhanced training/run.py: added --ui tui|headless, --wandb, --wandb-project
- Updated pyproject.toml: added rich>=13.0, wandb, bumped version to 0.1.0
- Created training/ui/board_viewer.py: Rich-based board visualization with colored tiles
- Created training/ui/debug_viewer.py: interactive debug viewer with step controls

## 2025-12-16 – Critical Priority Tasks

- Created training/ui/event_handler.py: TUIEventHandler bridges EventBus → TUI
- Modified training/run.py: TUI integration with --ui tui flag
- Verified DAS physics already implemented in GameLogic.cpp (16-frame initial, 6-frame repeat)
- Verified wall kicks already implemented in GameLogic.cpp (kick-left on blocked rotation)
- Updated game_engine/AGENTS.md: corrected implementation status (was outdated from Nov 22)
- Updated notes/BACKLOG.md: marked critical priorities complete

## 2025-12-16 – C++ Engine Parity Testing

- Created tools/game_transcript.py: frame-by-frame recording format
  - JSON + MessagePack serialization
  - Delta encoding for board changes
  - Comparison utilities for divergence detection
  - Demo input parser (RLE → per-frame)
- Created tools/record_demo.py: C++ engine demo recorder
  - Uses --manual-step for synchronized stepping
  - Captures pill positions, board changes
- Ran parity test: initial board matches demo_field exactly
- Found divergence: C++ engine tops out at frame 292 (8 pills) vs NES ~5461 frames
- Root cause TBD: likely timing difference in input processing

## 2025-12-17 – NES Demo Parity Deep Dive

### Recording Layer Verification ✓
- Verified `demo_pills` array in C++ matches NES ROM exactly (45 bytes)
- Confirmed NES recorder captures board state correctly from RAM at `0x0400`
- Validated pills 1-3: positions AND colors match NES exactly (Y-Y, B-R, Y-B)
- Board state byte-for-byte identical after pill 3

### Fixes Applied
- **spawn_delay**: Added 35-frame delay matching NES throw animation
  - Files: `GameState.h` (new field), `GameLogic.cpp` (init/decrement), `engine_shm.py`
- **INPUT_OFFSET**: Changed from 158 to 124 (accounts for spawn_delay in input indexing)
  - File: `tools/record_demo.py`
- **pill_counter off-by-one**: Added second `generateNextPill()` in `init()`
  - Matches NES `level_init.asm` lines 125-126
  - File: `game_engine/GameLogic.cpp` lines 163-165

### Current State
- Pills 1-3: ✓ Full parity (positions, colors, board state)
- Pill 4: First divergence point
  - C++ lands at (4,6) as single tile; NES at (3,6-7) full pill
  - Board identical at spawn → behavioral difference in C++ engine
- Root cause: Cumulative timing drift (C++ 28 frames slower by pill 4)

### Handoff Notes
- Recording layer is rock-solid and trustworthy
- Divergence stems from C++ engine behavior, not recording
- See walkthrough.md for detailed analysis and next steps
- Probable causes: gravity counter timing, DAS timing, spawn_delay interaction

## 2025-12-18 – Coding Agent (Codex CLI)

- Reimplemented `game_engine/GameLogic.cpp` as a parity-first port of the NES frame loop (explicit NMI tick + `nextAction` / `pillPlacedStep` state machines).
- Corrected core ROM tables and counters for rules-exact timing: full NTSC `speedCounterTable`, full 128-byte `demo_pills`, 512-byte demo input stream semantics, and BCD counter behavior (viruses/pill counters).
- Updated demo tooling for ground-truth verification:
  - `tools/record_demo.py` now relies on engine-internal demo replay (no external input feeding) and matches NES recorder stop semantics.
  - `tools/game_transcript.py` comparison fixes + normalization.
- Added regression coverage: `tests/test_game_engine_demo.py::test_demo_trace_matches_nes_ground_truth` asserts full demo trace matches `data/nes_demo.json`.

## 2025-12-18 – Coding Agent (Codex CLI) – Engine Demo TUI

- Added an interactive Rich-based demo player for the C++ engine (`tools/engine_demo_tui.py`) with pause/step/speed/restart controls and live shared-memory state inspection.
- Fixed `training/ui/board_viewer.parse_board_bytes` to correctly accept raw `bytes`/`bytearray` board buffers.
- Improved demo TUI ergonomics + diagnostics: upcoming pill list, smoother FPS estimate, and an integrated benchmark suite (engine freerun vs manual-step vs TUI render costs).
- Fixed a pill-render “seam” artifact by filling pill tiles using background color in `training/ui/board_viewer.py`.
- Switched playback speed control from “seconds per frame” to an `x` multiplier target (e.g. `2.4x` NTSC), with region/base-FPS selection in `tools/engine_demo_tui.py`.
- Refreshed the tracked prebuilt engine artifacts (`game_engine/drmario_engine`, `game_engine/*.o`) to match the parity-correct C++ sources (fixes demo playback stalling/timeouts when running the shipped binary).

## 2025-12-18 – Coding Agent (Codex CLI) – Macro Placement Planner Rewrite

- Re-implemented the placement macro-action stack as a NES-accurate, spawn-latched SMDP wrapper:
  - Canonical 512-way `(o,row,col)` action space in `envs/retro/placement_space.py`.
  - Frame-accurate reachability in `envs/retro/fast_reach.py` (gravity + DAS + rotation quirks).
  - Spawn snapshot decoding + feasibility/cost masks + minimal-time controller reconstruction in `envs/retro/placement_planner.py`.
  - Gym wrapper `envs/retro/placement_env.py` returning `placements/*` masks and `placements/tau`.
- Kept `envs/retro/placement_wrapper.py` as a small compatibility shim for older scripts.
- Restored training/docs ergonomics:
  - `envs/retro/register_env.py` registers both `DrMarioPlacementEnv-v0` and legacy `DrMario-Placement-v0`.
  - Updated `docs/PLACEMENT_PLANNER.md`, `docs/PLACEMENT_POLICY.md`, and `notes/archive/root_docs/QUICK_START_PLACEMENT_POLICY.md` to match the new wrapper.
- Added QuickNES update utility (`tools/update_quicknes_core.py`) and documented it in `docs/RETRO_CORE_NOTES.md`.
- Fixed regressions uncovered by unit tests:
  - Corrected τ=1 bootstrap semantics in `tests/test_placement_policy.py`.
  - Hardened `training/speedrun_experiment.py` episode finalization for missing/legacy `dones` tracking.

## 2025-12-18 – Coding Agent (Codex CLI) – New Runner: Real Env + Debug TUI

- Upgraded the unified runner (`training/run.py`) to support:
  - `--algo ppo_smdp` (uses `training/algo/ppo_smdp.py`).
  - `--ui debug` (Rich board visualization + pause/step + speed controls).
  - Convenience flags for real retro training: `--env-id`, `--core`, `--core-path`, `--rom-path`, `--backend`, `--level`, `--vectorization`.
- Implemented a real vector env factory in `training/envs/dr_mario_vec.py`:
  - Returns a Gymnasium VectorEnv for `DrMario*` env ids (with a wrapper that converts vector `infos` to a list-of-dicts).
  - Keeps `DummyVecEnv` for tests and non-retro configs.
- Added interactive playback control wrappers:
  - `training/envs/interactive.py` provides `PlaybackControl` + `RateLimitedVecEnv` (pause/single-step and target FPS using `placements/tau` when available).
  - `training/ui/runner_debug_tui.py` provides a terminal-based debug UI with board rendering from `raw_ram`.
- Added a unit test for the real env factory running on the mock backend: `tests/test_runner_real_env_factory.py`.
- Runner/config polish:
  - `training/run.py` now defaults `--algo/--engine` from the config file when not provided on the CLI (with `smdp_ppo` → `ppo_smdp` aliasing).
  - `training/configs/smdp_ppo.yaml` updated to `algo: ppo_smdp` and `env.id` (matches the unified runner + env factory).
  - Placement docs (`docs/PLACEMENT_POLICY.md`, `notes/archive/root_docs/QUICK_START_PLACEMENT_POLICY.md`, `notes/archive/root_docs/IMPLEMENTATION_COMPLETE.md`, `notes/archive/root_docs/PLACEMENT_POLICY_IMPLEMENTATION.md`) now recommend `python -m training.run` over bespoke launch scripts.
- Fixed a reset-time state mismatch in `envs/retro/drmario_env.py`:
  - Rebuild `_state_cache` after the auto-start sequence so `reset()` returns observations consistent with the post-start `raw_ram` snapshot.
  - `viruses_remaining` now prefers the raw RAM counter during startup (avoids stale `_state_cache` during reset/start sequences).
- Fixed a `SyntaxError` in `training/ui/runner_debug_tui.py` (`f-string` quoting in the speed display).

## 2025-12-18 – Coding Agent (Codex CLI) – Runner Reset Fix + Native Reachability

- Fixed a training-time double-reset bug in `training/envs/dr_mario_vec.py` that prevented `DrMarioRetroEnv` from running the full 3-press auto-start sequence (led to empty board/viruses=0 on the placement env).
- Added a native reachability accelerator for the macro placement planner:
  - C BFS implementation: `reach_native/drm_reach_full.c`
  - Python wrapper + buffer management: `envs/retro/reach_native.py`
  - Build helper: `python -m tools.build_reach_native`
  - Bench harness: `python -m tools.bench_reachability`
- Integrated native backend into `envs/retro/placement_planner.py` (`reach_backend=auto|native|python`) while keeping `envs/retro/fast_reach.py` as the oracle.
- Surfaced the active planner backend in env `info` as `placements/reach_backend` and displayed it in the debug TUI stats panel.
- Updated docs: `docs/PLACEMENT_PLANNER.md`, `docs/PLACEMENT_POLICY.md`, `notes/archive/root_docs/QUICK_START_PLACEMENT_POLICY.md`.

## 2025-12-18 – Coding Agent (Codex CLI) – SMDP-PPO Minibatch KL Bugfix

- Fixed a crash in `training/algo/ppo_smdp.py` where KL divergence was computed against the full-batch `log_probs_old` instead of the mini-batch slice (`mb_log_probs_old`), causing a 512-vs-128 tensor shape mismatch when `minibatch_size < decisions_per_update`.

## 2025-12-18 – Codex CLI – Auto-start Fix + RNG Randomization Toggle

- Fixed libretro auto-start after terminal episodes by using the correct default `start_presses=3` for backend resets (previous 1/2-press logic left the game in menus with viruses=0 after topout/clear).
- Made level alignment robust by reading the current level from RAM and tapping LEFT/RIGHT to reach the configured level (avoids wrap-around to level 20).
- Added per-env RNG randomization toggle:
  - New env attribute `rng_randomize` (used as the default for `reset(options.randomize_rng)` so Gymnasium vector autoresets still honor it).
  - `training.run` CLI: `--randomize-rng/--no-randomize-rng`
  - Debug TUI hotkey: `r` (shows `rng: on/off` in stats).

## 2025-12-18 – Codex CLI – Debug UI Responsiveness + Batched PPO Update

- Vectorized `SMDP-PPO` policy update in `training/algo/ppo_smdp.py` by computing masked log-probs/entropy for the full minibatch at once (removes per-sample Python loops that could stall the debug UI and reduce throughput).
- Fixed `MaskedPlacementDist` edge-case handling for batched masks (`models/policy/placement_dist.py`).
- Enhanced debug UI performance telemetry:
  - `emu_fps(step)` = frames/sec inside `env.step` only (planner + emu)
  - `emu_fps(total)` = frames/sec including training compute between env steps

## 2025-12-18 – Codex CLI – Placement Env: Skip No-Feasible Spawns

- Fixed `envs/retro/placement_env.py` to treat “decision points” with **zero feasible macro placements** (e.g., spawn-blocked top-out) as non-decision frames and keep stepping NOOP until the env transitions (lock/top-out/reset) instead of returning an empty mask that can cause an infinite invalid-action loop.
- Added regression coverage: `tests/test_placement_env_no_feasible_actions.py`.

## 2025-12-19 – Codex CLI – Reduced Bitplane Observations + Mask Injection

- Added two new state representations in `envs/specs/ram_to_state.py`:
  - `bitplane_reduced` (6ch): type-blind color planes + `virus_mask` + `pill_to_place` + `preview_pill`.
  - `bitplane_reduced_mask` (10ch): reduced + 4 feasibility-mask channels (`feasible_o0..feasible_o3`).
- Implemented feasibility-mask injection at true decision points in `envs/retro/placement_env.py` (fills reserved obs channels from `placements/feasible_mask`).
- Made the debug runner UI less noisy by hiding the channel index→name list by default (toggle with `p`) in `training/ui/runner_debug_tui.py`.
- Updated docs: `docs/STATE_OBS_AND_RAM_MAPPING.md`, `docs/RAM_TO_STATE.md`, `docs/PLACEMENT_POLICY.md`.
- Added tests for the new representations and mask injection:
  - `tests/test_bitplane_reduced_helpers.py`
  - `tests/test_feasible_mask_obs_injection.py`

## 2025-12-19 – Codex CLI – Bottle-Only Obs + Preview-Pill Vectors

- Added bottle-only state representations:
  - `bitplane_bottle` (4ch): bottle color planes + `virus_mask` (no falling/preview projection).
  - `bitplane_bottle_mask` (8ch): bottle-only + feasibility planes injected by placement env.
- Decoded falling/preview pill metadata directly from RAM in `envs/state_core.py` (observation-repr independent), and updated intent wrapper to decode falling coords from RAM.
- Updated the placement policy to condition on both **current** and **preview** pill colors as vectors (no longer requires `pill_to_place`/`preview_pill` planes), and extended the rollout buffer accordingly.

## 2025-12-19 – Codex CLI – Env Step Profiling Breakdown

- Added per-frame env timing keys (`perf/env_*_sec`) in `envs/retro/drmario_env.py` and aggregated them per macro decision in `envs/retro/placement_env.py`.
- Extended `training/envs/interactive.py` perf snapshot to report env breakdown ms/frame and `macro_other_ms/frame`.
- Updated `training/ui/runner_debug_tui.py` Perf panel to display the breakdown.
- Optimized state-mode stepping by avoiding redundant RAM refreshes, reusing the RAM snapshot for `info["raw_ram"]`, and fetching RGB frames lazily in `render()` unless `obs_mode=pixel`.
- Fixed a debug-TUI crash when `info["raw_ram"]` was present but `None` (`training/ui/board_viewer.py`), and added `ms/frame(total|accounted|unaccounted)` rows to reconcile `sps` vs per-component timings.
- Reduced per-frame reward overhead by gating adjacency/height computations on static-tile changes and vectorizing bottle-buffer scans (`envs/retro/drmario_env.py`).
- Reduced C++ engine backend step overhead by using a short spin-then-sleep polling loop (avoid guaranteed oversleep each frame) (`envs/backends/cpp_engine_backend.py`).
- Added optional encoder scaling via `encoder_blocks` (extra 64-channel residual blocks) and updated debug UI to show preview pill colors.

## 2025-12-18 – Codex CLI – Debug TUI: Perf Diagnostics (Inference/Planner)

- Added lightweight perf counters + timing breakdowns for interactive runs:
  - `DrMarioPlacementEnv` now emits planner timings (`perf/planner_build_sec`, `perf/planner_plan_sec`) and legacy planner-step keys (`placements/plan_calls`, `placements/plan_latency_ms_*`).
  - `RateLimitedVecEnv` accumulates inference/planner/update timing and exposes derived `ms/frame` and `ms/call` stats via `perf_snapshot()`.
  - `RunnerDebugTUI` displays inference/planner/update diagnostics alongside FPS.
- Added unit coverage: `tests/test_interactive_perf_counters.py`.

## 2025-12-18 – Codex CLI – Placement Env: Spawn-Latched Decisions (Fix Excess Replanning)

- Fixed `envs/retro/placement_env.py` to expose **exactly one macro decision per pill spawn** by gating decision-point detection on the ROM spawn counter (`pill_counter`, RAM `$0310`) in addition to `currentP_nextAction == nextAction_pillFalling`.
- Marked a spawn as “consumed” once we commit to a macro plan (or when `placements/options == 0`) so we don’t surface new decisions mid-fall (prevents “options ticking down” + hundreds of extra planner/inference calls per spawn).
- Adjusted planner-build timing emission so `perf/planner_build_sec` counts actual reachability builds (invalid-action retries reuse cached ctx).
- Added regression coverage: `tests/test_placement_env_spawn_latch.py`.

## 2025-12-18 – Codex CLI – Episode Stats + Live Return Metrics (Runner Debug UI)

- Added a vector-env wrapper episode-stat injector (`training/envs/dr_mario_vec.py`) that attaches:
  - `info["episode"] = {"r": return, "l": length_frames, "decisions": decisions}`
  - `info["drm"]` with lightweight end-of-episode summaries (e.g., `viruses_cleared`, `top_out`, `cleared`)
  This fixes `ret(last)`, `ret(mean100)`, and `len(last)` reporting for real envs.
- Extended the debug TUI stats panel (`training/ui/runner_debug_tui.py`) to show:
  - live per-episode return (`ret(curr)`), median of last 16 (`ret(med16)`), and current episode progress (`len(curr)`).
- Added unit coverage for episode stats injection: `tests/test_vec_env_episode_stats.py`.
- Documented the current base reward terms in `docs/REWARD_SHAPING.md`.

## 2025-12-18 – Codex CLI – Scripted Curriculum + WandB Wiring

- Wired `training.run --wandb/--wandb-project` into `DiagLogger` by ensuring `"wandb"` is added to `cfg.viz` when enabled.
- Implemented a scripted curriculum based on synthetic negative levels:
  - `DrMarioRetroEnv` interprets `level < 0` as a curriculum stage and patches the bottle RAM at reset time to reduce virus count (`-4..-1`), with `-4` using “any 4-match” (first clear event) as the success condition.
  - Added `training/envs/curriculum.py` (`CurriculumVecEnv`) to schedule levels based on rolling clear rate and optional rehearsal of lower levels.
  - Enabled the curriculum in `training/configs/smdp_ppo.yaml` and surfaced curriculum stats in `RunnerDebugTUI`.
- Updated docs to reflect implemented curriculum (`docs/PLACEMENT_POLICY.md`, `notes/archive/root_docs/QUICK_START_PLACEMENT_POLICY.md`, `notes/archive/root_docs/PLACEMENT_POLICY_IMPLEMENTATION.md`).

## 2025-12-18 – Codex CLI – Curriculum Clear Detection + Reward Breakdown Debugging

- Fixed curriculum `-4` clear detection by using the ROM’s clearing-tile markers (`CLEARED_TILE`/`FIELD_JUST_EMPTIED`) from bottle RAM (`envs/retro/drmario_env.py`) instead of relying on occupancy deltas (which can miss clears).
- Made curriculum stats explicitly “recent window” in the debug UI by surfacing `window_n/window_size` and ensuring terminal-step info reports the episode’s level (plus `next_env_level`) (`training/envs/curriculum.py`, `training/ui/runner_debug_tui.py`).
- Added reward breakdown aggregation for macro steps (`reward/*` totals and counts) and a new Reward column in the debug TUI to audit scoring live (`envs/retro/placement_env.py`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Fixed the default reward config to apply a negative top-out penalty (`envs/specs/reward_config.json`).

## 2025-12-18 – Codex CLI – Fix `-4` False Clears + Debug UI Columns

- Fixed a false-positive in the curriculum level `-4` “any_clear” success detection: empty tiles are `0xFF` (high nibble `0xF0`) and must not be counted as `FIELD_JUST_EMPTIED` (`0xF0..0xF2`). Added a gameplay-mode guard and a regression test (`envs/retro/drmario_env.py`, `tests/test_clearing_tile_counter.py`).
- Corrected bitplane `clearing_mask` / `empty_mask` construction to distinguish `0xFF` empty from `0xF*` just-emptied tiles (`envs/specs/ram_to_state.py`).
- Restructured `RunnerDebugTUI` into 4 columns: board + perf + learning + reward, with wider side panels (`training/ui/runner_debug_tui.py`).

## 2025-12-18 – Codex CLI – Preview + Placement Verification + Curriculum Stages

- Made `preview_pill` consistently structured as a dict (`first_color`, `second_color`, `rotation`) and updated the Rich board renderer to show the next pill above the bottle with correct orientation (`envs/retro/drmario_env.py`, `training/ui/board_viewer.py`).
- Added placement verification metadata to macro steps by capturing the observed falling-pill pose at lock time and comparing it to the planner’s target pose (`placements/pose_ok`, `placements/target_pose`, `placements/lock_pose`) (`envs/retro/placement_env.py`).
- Expanded synthetic curriculum stages to include match-count levels `-10..-4` (1..7 matches) before the 1/2/3-virus stages (`-3..-1`) and updated configs to start at `-10` (`envs/retro/drmario_env.py`, `training/envs/curriculum.py`, `training/configs/smdp_ppo.yaml`).
- Made RNG randomization default-on in the standard training configs, and added spawn-level perf ratios (`infer/spawn`, `planner/spawn`) plus last terminal reason tracking to the debug UI (`training/configs/base.yaml`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Surfaced placement verification status in the debug UI (`pose_ok`, `pose_err`) so mismatches are visible without digging through raw infos (`training/ui/runner_debug_tui.py`).
- Added coverage for the new negative-level mapping (`tests/test_synthetic_level_mapping.py`).

## 2025-12-19 – Codex CLI – Planner Parity + Canonical Clear Counting

- Fixed placement pose verification to capture the locked-pill pose from RAM state transitions (leaving `nextAction_pillFalling` / spawn counter advancing) instead of relying on `pill_bonus_adjusted` (which can be disabled by reward config). Added `placements/lock_reason` and surfaced `target_pose`/`lock_pose` in the debug UI.
- Corrected “down-only” soft drop parity: soft drop triggers when `frameCounter & 1 == 0` (not `== 1`). Updated:
  - Python oracle stepper (`envs/retro/fast_reach._step_state`)
  - Python packed BFS (`build_reachability` fast path)
  - Native BFS helper (`reach_native/drm_reach_full.c`, rebuild via `python -m tools.build_reach_native`)
  This restores frame-accurate agreement between native reachability scripts and the emulator (pose mismatches disappear).
- Replaced non-virus clear reward counting with a canonical bottle-buffer diff (`envs/specs/ram_to_state.count_tile_removals`) and cached bottle snapshots in `DrMarioRetroEnv` to avoid false positives from falling-pill overlays. Updated docs (`docs/REWARD_SHAPING.md`) and added unit coverage.

## 2025-12-19 – Codex CLI – Pose Mismatch Logging

- Added persistent pose mismatch counters (`placements/pose_mismatch_*`) and JSONL logging for rare planner/executor divergences, dumping snapshot + board + feasibility + plan + observed lock pose into `data/pose_mismatches.jsonl(.gz)` (override/disable via `DRMARIO_POSE_MISMATCH_LOG`). Optional per-frame trace capture is gated by `DRMARIO_POSE_MISMATCH_TRACE` (`envs/retro/placement_env.py`, `training/ui/runner_debug_tui.py`).

## 2025-12-19 – Codex CLI – Rotation Edge Semantics + Reward Config Safety

- Fixed reachability/planner rotation semantics to match the ROM: rotate uses `currentP_btnsPressed` (edge), so holding A/B across consecutive frames must not rotate repeatedly. Implemented by tracking a new per-frame `rot_hold` state across the Python stepper, Python packed BFS, and native BFS helper; rebuilt native dylib (`envs/retro/fast_reach.py`, `envs/retro/placement_planner.py`, `reach_native/drm_reach_full.c`, `envs/retro/reach_native.py`).
- Made reward-config failures non-silent and aligned `RewardConfig` dataclass defaults with `envs/specs/reward_config.json` so reward scale can’t unexpectedly jump to legacy “hundreds” defaults (`envs/retro/drmario_env.py`).

## 2025-12-19 – Codex CLI – Native Reachability Planner Performance

- Optimized the native reachability helper by (a) early-stopping once all in-bounds terminal poses are found, (b) pruning the early-stop target set via a timer-free geometric flood fill (to avoid max-depth blowups from sealed cavities), and (c) switching to a frontier-aggregated BFS that batches x positions per counter-state key (8-bit x masks). Added optional stats via `DRMARIO_REACH_STATS=1` and a replay test that validates native plans against the Python per-frame stepper (`reach_native/drm_reach_full.c`, `envs/retro/reach_native.py`, `tests/test_reach_native_smoke.py`).

## 2025-12-19 – Codex CLI – Virus Adjacency Shaping + Bitplane Policy Obs

- Added virus-specific adjacency shaping terms (`virus_adjacency_pair_bonus`, `virus_adjacency_triplet_bonus`) and surfaced the aggregate in the debug UI reward breakdown (`envs/retro/drmario_env.py`, `envs/specs/reward_config.json`, `envs/retro/placement_env.py`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Set the placement SMDP-PPO config to use the `bitplane` state representation by default (type-blind color planes + virus mask) and improved preview decoding for bitplane states to infer rotation (`training/configs/smdp_ppo.yaml`, `envs/specs/ram_to_state.py`).
- Added unit coverage for virus adjacency shaping (`tests/test_virus_adjacency_reward.py`) and updated reward docs (`docs/REWARD_SHAPING.md`).

## 2025-12-19 – Codex CLI – Debug TUI: State Representation + Input Plane Names

- Added `ram_to_state.get_plane_names()` and a small test to keep plane-name lists consistent with channel counts (`envs/specs/ram_to_state.py`, `tests/test_state_plane_names.py`).
- Extended the debug TUI Perf panel to show `state_repr`, per-env observation shape, plane index→name map, next-pill colors, and mask/orientation conventions (`training/ui/runner_debug_tui.py`).

## 2025-12-19 – Codex CLI – C++ Engine ↔ Libretro Ghost Parity

- Added `cpp-engine` backend (shared memory + subprocess) with a synthetic 2 KB NES RAM view so the existing RAM→state pipeline works unchanged (`envs/backends/cpp_engine_backend.py`, `envs/backends/__init__.py`, `game_engine/engine_shm.py`, `game_engine/GameState.h`).
- Made libretro RNG seeding parity-robust by applying `rng_seed_bytes` at the `initData_level` boundary (mode==0x03), removed the engine’s menu-time RNG warmup hack, and added per-reset `frameCounter` low-byte seeding for exact soft-drop timing parity (`envs/retro/drmario_env.py`, `game_engine/GameLogic.cpp`).
- Added a ghosting parity harness that runs libretro and the C++ engine side-by-side and stops on first divergence (`tools/ghost_parity.py`).
- Hardened libretro auto-start: level selection clamps (no wrap-around) and added `start_sync_wait_frames` (`waitFrames` sync) so resets land in a stable post-virus-placement checkpoint across levels (`envs/retro/drmario_env.py`, `tools/ghost_parity.py`).
- Fixed demo recorder startup gating + transcript frame numbering so demo parity remains deterministic under the new frameCounter seeding (`tools/record_demo.py`, `tests/test_rng_randomization.py`), and updated docs (`docs/RETRO_CORE_NOTES.md`, `docs/CPP_SIM_NOTES.md`, `docs/DYNAMICS_SPEC.md`).

## 2025-12-19 – Codex CLI – Multi-Env C++ Backend: Design + P0 Backlog

- Wrote a multi-env scaling design doc (vectorization policy, scaling metrics, debug UI hotkeys, restart semantics) in `notes/DESIGN_MULTIENV.md`.
- Added hierarchical “Up Next / P0” backlog items for multi-env correctness/perf/UI (including a scaling benchmark harness) in `notes/BACKLOG.md`.

## 2025-12-20 – Codex CLI – C++ Batched Stepping for Multi-Env Throughput

- Fixed Gymnasium `AsyncVectorEnv` worker env creation by registering Dr. Mario env ids inside each env factory (subprocess-safe) and making registration idempotent to avoid “Overriding environment …” warnings (`training/envs/dr_mario_vec.py`, `envs/retro/register_env.py`).
- Added a shared-memory batched run protocol to the C++ engine (`run_request_id/run_ack_id`, run modes for fixed-frames and “until next decision”, plus cleared-tile counters) and surfaced it in the Python backend (`game_engine/GameState.h`, `game_engine/main.cpp`, `game_engine/GameLogic.cpp`, `game_engine/engine_shm.py`, `envs/backends/cpp_engine_backend.py`).
- Added `DrMarioRetroEnv.sync_after_backend_run()` and a cpp-engine fast path in `DrMarioPlacementEnv` that executes the planner controller script and wait-to-next-spawn via batched runs (gated by reward config + `DRMARIO_CPP_FAST`), materially increasing decisions/sec in the scaling benchmark while preserving core training reward terms (`envs/retro/drmario_env.py`, `envs/retro/placement_env.py`).
- Fixed an SMDP termination bug in the cpp-engine fast path: match-mode curriculum stages (0 viruses) must not end just because `viruses_remaining==0`; now termination respects `task_mode` and only ends after the configured match count or top-out (`envs/retro/placement_env.py`).

## 2025-12-20 – Codex CLI – Async Scaling Stability + Fast Reset Path

- Removed a multiprocessing spawn footgun: moved cpp-engine shared-memory ctypes bindings into an installable module (`envs/backends/cpp_engine_shm.py`) and updated the backend to import it, so `AsyncVectorEnv` workers no longer depend on `game_engine` being importable.
- Hardened cpp-engine reset behavior under high env counts: `DrMarioRetroEnv.reset()` now retries by restarting the backend (instead of silently falling back to mock dynamics) and records `_backend_last_error` for downstream wrappers.
- Eliminated per-frame stepping during autoresets: `DrMarioPlacementEnv.reset()` now uses `cpp-engine.run_until_decision()` + `sync_after_backend_run()` to reach the first decision point without calling `env.step()` in a loop, reducing timeouts and improving scaling.
- Reduced polling overhead in `CppEngineBackend` (both single-step and batched runs) with gentler backoff to avoid CPU thrash at high env counts.
- Added an `AsyncVectorEnv` regression test to exercise autoresets and ensure the placement env doesn’t lose `_state_cache` (`tests/test_async_vec_env_stability.py`).
- Made shutdown robust for long async runs: `_InfoListWrapper.close()` force-terminates `AsyncVectorEnv` if it has a pending call to avoid `close()` hanging on `step_wait` (`training/envs/dr_mario_vec.py`, `training/run.py`).

## 2025-12-20 – Codex CLI – Time/Spawn Budget Scaffolding (Curriculum)

- Fixed the TUI “Goal” label for synthetic match-count levels (now matches the env mapping `max(1, 16 + level)` for `-15..-4`) so negative levels don’t display as “clear -3 matches” (`training/ui/tui.py`).
- Added optional time-based task budgets (`task_max_frames`, `task_max_spawns`) plumbed via `CurriculumVecEnv.set_attr` and enforced inside `DrMarioPlacementEnv` with per-episode counters and `info` keys under `task/*` (`training/envs/curriculum.py`, `envs/retro/placement_env.py`).
- Added unit coverage for frame/spawn budget truncation and “clear over budget strips terminal bonus” semantics (`tests/test_task_budgets.py`).

## 2025-12-20 – Codex CLI – Confidence-Based Curriculum Windows + Mastery Time Budgets

- Switched curriculum advancement from fixed `window_episodes/min_episodes` gating to a sigma-based one-sided Wilson lower bound check (`p > target`), with window sizes derived from a “near-target” assumption and configurable `confidence_sigmas` (default 2σ). Added a separate perfect-streak window size helper for mastery gating (`training/envs/curriculum.py`).
- Added “time budget after mastery” plumbing: once a level hits a perfect-streak window long enough to certify mastery at `time_budget_mastery_sigmas/time_budget_mastery_target`, the curriculum begins setting a per-level `task_max_frames` that starts at mean clear time and tightens gradually with a MAD-capped drop. Exposed mean/MAD/budget via curriculum snapshots (`training/envs/curriculum.py`).
- Surfaced curriculum Wilson LB + time budget/mean/MAD in both the Rich TUI and debug TUI, and added `tools/report_curriculum.py --confidence-table` to print expected window sizes / requirements (`training/ui/tui.py`, `training/ui/runner_debug_tui.py`, `training/algo/ppo_smdp.py`, `tools/report_curriculum.py`).
- Hardened `DrMarioPlacementEnv` decision-context building to treat missing `_state_cache` as a backend error and return `truncated=True` (instead of raising), improving `AsyncVectorEnv` robustness under high env counts (`envs/retro/placement_env.py`).

## 2025-12-20 – Codex CLI – Stage-Local Hop-Back Stats + Persistent Best Times + Time Goal Demotion

- Made `ln_hop_back` stage tracking stage-local: when revisiting a level with a tighter threshold, success-window stats are now fresh (tracked per stage token/index), avoiding contamination from earlier easier passes (`training/envs/curriculum.py`).
- Added a persistent sqlite best-times DB (`data/best_times.sqlite3`, git-ignored) to track per-(level, rng_seed) best frames/spawns across runs, plus a small reporting script (`tools/report_best_times.py`).
- Extended time-goal logic: once base mastery is achieved, time budgets begin tightening against an increasing `1-exp(-k)` success target; if base-objective mastery drops, time goals are cleared and must be re-earned. Exposed `time_k/time_target` and spawn stats in curriculum info and UIs (`training/envs/curriculum.py`, `training/ui/tui.py`, `training/ui/runner_debug_tui.py`, `training/algo/ppo_smdp.py`).
- Removed a long-standing skipped test by replacing the optional `mlx.core` dependency with a deterministic fake-MLX module in tests, covering both `_mlx_set_row` code paths (`tests/test_discounting.py`).
- Made `--ui tui` shutdown on Ctrl+C cleanly (no traceback spam) by catching `KeyboardInterrupt` around training (`training/run.py`).
- Ignored best-times sqlite WAL/SHM sidecar files and local scratch directories (`notes/human_notes/`, `tests/user_testing/`) in `.gitignore`.

## 2025-12-20 – Codex CLI – Docs/Notes Pass

- Updated README + placement-policy docs to reflect fast `cpp-engine` multi-env training (`--vectorization async`) and new reporting tools (`tools/bench_multienv.py`, `tools/report_curriculum.py`, `tools/report_best_times.py`).
- Refreshed `notes/BACKLOG.md` to mark completed multi-env items and add next steps for best-times/time-goal iteration.

## 2025-12-21 – Codex CLI – Engine/Digital-Twin Interface Design Notes

- Added a design document specifying the engine/digital-twin/planner boundary and the decision-vs-telemetry channel split, grounded in the existing `libdrmario_pool` ABI (`notes/DESIGN_ENGINE_TWIN_PROTOCOL.md`).
- Updated `notes/MEMORY.md` with the architectural decision to keep board internals behind the decision-boundary ABI and co-locate planning with the timing model.
- Added a new scrutiny item for vision shadow-mode desync/drift risks and the need for explicit twin quality signals (`notes/SCRUTINY.md`).

## 2025-12-21 – Codex CLI – Fix VectorEnv Info Merge Crash (Task Budgets)

- Fixed `DrMarioPlacementEnv.reset()` to omit unset `task/max_frames` and `task/max_spawns` keys (instead of returning `None`), matching `step()` behavior and preventing Gymnasium VectorEnv dtype crashes. Added a SyncVectorEnv regression test (`envs/retro/placement_env.py`, `tests/test_task_budgets.py`).

## 2025-12-21 – Codex CLI – SMDP-PPO Aux Inputs (v1) + Speed Setting

- Added `smdp_ppo.aux_spec` (`none|v1`) and plumbed a 57-dim aux vector into the placement policy net + rollout buffer (speed/viruses/level/time/heights/clear-progress + a few cheap extras) (`training/algo/ppo_smdp.py`, `models/policy/placement_heads.py`, `training/rollout/decision_buffer.py`).
- Surfaced `speed_setting` (0/1/2) as a real env option across retro backends and `cpp-pool`, and emitted it in decision-time infos (`envs/retro/drmario_env.py`, `envs/retro/placement_env.py`, `training/envs/dr_mario_vec.py`, `training/envs/drmario_pool_vec.py`).
- Added `drm/viruses_initial` as a backend-agnostic info key to support a scalar “clearance progress” aux feature (`envs/retro/placement_env.py`, `training/envs/drmario_pool_vec.py`).
- Updated the default SMDP-PPO config to enable aux v1 and default to high game speed (`training/configs/smdp_ppo.yaml`).

## 2025-12-21 – Codex CLI – cpp-engine Async Timeout Recovery

- Prevented rare cpp-engine batched-run timeouts from crashing `AsyncVectorEnv`: placement fast-path now truncates the episode, records `placements/backend_error*`, and forces a backend restart on run-request failures (`envs/retro/placement_env.py`).
- Made `CppEngineBackend._run_request` use a progress-based watchdog + a more forgiving total timeout (with better diagnostics) to reduce false timeouts under heavy multi-env load (`envs/backends/cpp_engine_backend.py`).
- Added regression tests ensuring cpp-engine fast-path timeouts truncate instead of raising (`tests/test_cpp_engine_timeout_recovery.py`).

## 2025-12-22 – Coding Agent (Codex CLI) – Ordered Pill Pair Embedding

- Added `pill_embed_type` (`unordered` vs `ordered_onehot`/`ordered_pair`) and implemented an ordered 9-way pair embedding to preserve half identity for mixed-color pills in directed macro action spaces (`models/policy/placement_heads.py`, `models/policy/candidate_policy.py`, `training/algo/ppo_smdp.py`).
- Updated the candidate-policy training config to enable the ordered embedding (`training/configs/smdp_ppo_candidate.yaml`).
- Added unit tests for order sensitivity + selection wiring (`tests/test_placement_policy.py`, `tests/test_candidate_policy.py`).

## 2025-12-22 – Coding Agent (Codex CLI) – Compressed Artifacts + Checkpoint Scanner

- Defaulted run artifacts to gzip-compressed, streamable files: `metrics.jsonl.gz`, `env.txt.gz`, and `*.pt.gz` checkpoints; updated readers/tools to accept `.gz` (e.g., `tools/plot_success_by_level.py`, `tools/report_curriculum.py`, `training/utils/reproducibility.py`, `training/run.py`, `tests/test_adapters.py`).
- Added gzip-aware logging for pose mismatch and ghost-parity JSONL outputs and updated docs to match the new defaults (`envs/retro/placement_env.py`, `tools/ghost_parity.py`, `notes/archive/root_docs/QUICK_START_PLACEMENT_POLICY.md`).
- Added a checkpoint validation tool to scan for corrupt checkpoint files and optionally delete them (`tools/check_checkpoints.py`).
- Added a live-updating curriculum plotter with an in-window chart selector that reuses the existing JSONL parser and avoids background threads (`tools/plot_success_live.py`).
- Fixed a startup ordering bug in the live plotter (status label initialized before trace callback).
- Tweaked live + static plot pickers to show `*.jsonl.gz` files in the file filter (`tools/plot_success_live.py`, `tools/plot_success_by_level.py`).
- Reworked the live picker flow to offer explicit file/dir buttons and centered the chooser to avoid off-screen dialogs (`tools/plot_success_live.py`).
- Made the live picker use the main Tk window (no hidden root) to avoid invisible/off-screen dialogs on macOS, and relaxed file filters so `.gz` isn’t hidden (`tools/plot_success_live.py`).
- Made `.jsonl.gz` readers tolerant of in-progress gzip streams so live plots work while training is still writing logs (`tools/plot_success_by_level.py`, `tools/report_curriculum.py`, `tools/plot_success_live.py`).
- Resized the live plot window after selection so it doesn’t inherit the small picker geometry (`tools/plot_success_live.py`).

## 2025-12-23 – Coding Agent (Codex CLI) – Remove External Trainer Integration

- Removed all external trainer integration points (runner, configs, adapters/callbacks) and the third-party dependency; simplified `training.run` to only support in-repo algorithms (`simple_pg`, `ppo_smdp`) and updated docs/notes accordingly.
- Deleted leftover trainer-integration directories and `__pycache__` artifacts so no named vestiges remain in the tree.

## 2025-12-23 – Repo Hygiene: Archive Root Docs + Remove Root Artifacts

- Moved root-level historical Markdown docs into `notes/archive/root_docs/` and added `notes/archive/README.md` explaining that archived content is non-authoritative.
- Updated docs to use the `notes/` system (instead of root discussion/worklog files) and documented pose mismatch diagnostics in `docs/PLACEMENT_POLICY.md`.
- Removed tracked root artifacts (`archive/`, `writeup/`, `commandlines.txt`) and added root-only `.gitignore` rules to prevent reintroduction.
- Standardized on a single local venv (`.venv`) by removing `.venv-py313*` variants.
- Removed redundant legacy training entrypoints (`training/train_placement_ppo.py`, `training/launches/*`) and their unused helper modules.
- Removed the `re/` reverse-engineering workspace and RE automation scripts (`tools/automation/`), relying on `dr-mario-disassembly/` as the canonical disassembly source; updated docs/tools accordingly.
- Fixed a race in the engine demo reset parity test by waiting for a post-reset sentinel instead of using `frame_count` as a readiness signal (`tests/test_game_engine_demo.py`).

## 2026-05-08 – Coding Agent – Capsule Connection-Edge Observations

- Added `bitplane_bottle_conn` and `bitplane_bottle_conn_mask` state representations with explicit `connected_{up,down,left,right}` channels for ordinary locked pill halves; viruses, singles, and legacy middle-half tile codes emit no connection edges.
- Extended the native `cpp-pool` observation ABI to protocol v2 with connection-edge observation specs, and made `bitplane_bottle_conn_mask` the forward-facing default for placement-SMDP training, benchmarking, and docs.
- Updated candidate PPO wiring so the board trunk consumes non-feasibility bottle channels (`candidate_board_channels: 8`) while local raw patches stay on the first four color/virus planes; added validation to reject feasible-mask planes in the board trunk.
- Added focused unit/smoke coverage for RAM decoder edge planes, plane-name/channel ordering, feasibility injection, candidate forward passes with 12-channel observations, and cpp-pool reset/step behavior for both old and new bottle-mask representations.

## 2026-06-10 – Coding Agent – Fightcade Live-Play Bridge

- Added the live-play bridge so the agent can play real Fightcade matches:
  `tools/fc_live_agent.lua` (runs inside fcadefbneo: exports per-frame state to
  `state.jsonl`, applies a frame-indexed NES button script from `plan.json` via
  `joypad.set`, rollback-safe by construction) and `tools/live_agent_server.py`
  (tails state lines, plans per pill spawn with `drm_reach_bfs_full` seeded
  from mid-fall micro-state rolled forward over a latency margin, picks the
  pose with the candidate policy checkpoint, writes the plan atomically;
  `--dry-run`/`--bench`/`--once` modes; replans on missed windows and on
  trajectory-verification desyncs).
- Local test without Fightcade: `tests/test_live_bridge.py` simulates the Lua
  side (pool-backend snapshot -> state line -> server -> plan), re-simulates
  the script via `fast_reach` to assert it locks at the decided pose, and
  unit-tests the 18-action->button mapping against `DrMarioPool.cpp`.
- Bench (M-series laptop, two training runs live): spawn->plan-written p50
  17.4 ms / p95 33.2 ms with the checkpoint policy (greedy: p50 11.9 / p95
  32.6) — under the 50 ms p95 target.
- `tools/fc_local_match.sh` documents the real-match launch order; blocked on
  the fightcadeRatings agent's answers re live-Lua availability (see the ask at
  the bottom of `../fightcadeRatings/COORDINATION.md`).

## 2026-06-11 – Coding Agent – VS Opponent Snapshot Pool (PFSP)

- Added `training/envs/vs_opponents.py`: `OpponentPool` of frozen policy
  checkpoints with PFSP sampling (`p=(w+1)/(g+2)`, weight `(p(1-p))^2 + 0.05`,
  unplayed = max weight), persisted as `<logdir>/opponent_pool/manifest.json`
  plus checkpoint copies; eviction beyond `max_pool` (12) keeps the protected
  seed champion. Seeds: newest `runs/vs2_*` checkpoint + the 1P placement
  champion (`runs/best_agents/smdp_ppo_step535164979.pt.gz`).
- `DrMarioVsPoolVecEnv` grew an `opponent_pool_cfg` mode: exposes only the N
  learner sides (`num_envs == num_pairs`); P2 sides are driven internally by
  the frozen nets (CPU, batched per opponent, same candidate-packing +
  MaskedPlacementDist path). Per match: result recorded to the pool, PFSP
  resample, `vs/opponent_id` in terminal info. `maybe_snapshot()` freezes the
  EMA weights into the pool every `snapshot_every_matches` (400) matches
  (called from the trainer metrics hook). `get_vs_metrics` adds real
  `vs/opponent_pool` size and `vs/pool_winrate_min/max`; `vs/win_rate` is now
  the learner-vs-pool rate. `enabled: false` keeps the legacy 2N self-play
  path unchanged.
- Tests: `tests/test_vs_opponent_pool.py` (PFSP math, manifest roundtrip,
  snapshot eviction, 2-pair env smoke with a tiny random candidate net).
- 2.5-min CPU smoke (2 pairs, DRMARIO_POOL_WORKERS=2) from the champion
  checkpoint vs the seeded pool: matches completed and per-opponent
  wins/games recorded in the manifest.

## 2026-06-11 (tournament) – Coding Agent (Claude) – Round-Robin Tournaments + SPRT Change Gates

- `tools/tournament.py` (`python -m tools.tournament run|report|sprt`):
  resumable round-robin VS tournaments over a roster yaml (entries:
  name/checkpoint/mode plain|search|ponder/params) on the native VS pool,
  one sqlite row per game (`runs/tournaments/tournaments.sqlite`;
  tournaments + games tables), deterministic per-game NES rng seeds and
  alternating side assignments derived from the tournament seed via the
  env's `seed_provider` hook. `report`: pairwise W-L-D matrix + Wilson 95%
  CIs (decisive games) + Bradley-Terry/logistic Elo MLE (draws=0.5,
  mean-zero anchor, ±95 from the Fisher-information Laplacian pinv).
  `sprt`: sequential A-vs-B fishtest-style BayesElo trinomial LLR gate
  (zero cells regularized at 0.5 so D=0 doesn't stall; bounds
  log(β/(1−α)) / log((1−β)/α)), records into the same store, resumes by
  name. Match running reuses `tools/vs_head_to_head.PlainPolicy` and the
  per-env `SearchPolicy.decide` convention (ponder: simulated dead time,
  recommend `--pairs 1`).
- `tests/test_tournament.py`: 16 tests — Wilson/Elo/SPRT against
  independently computed references and closed forms (400·log10(W/L),
  Fisher SE, trinomial LLR constants), store resumability/crash-resume/
  roster+seed mismatch with a stub runner. `docs/TOURNAMENTS.md`: usage +
  statistical conventions + game-count guidance (~±10 Elo ≙ ~1000
  games/pair; SPRT 0-vs-5 Elo resolves in a few hundred games).
- Smoke: 6 real games vs2-tip (step540020887) vs best-535m
  (step535164979), level 14 HI, 2 pairs — 5-1-0 for vs2-tip,
  Elo +139.8 ±186.5, db rows + report + resume-noop verified.

## 2026-06-12 — Search-distillation targets in SMDP-PPO (phase 1, config-gated)

- `training/algo/search_distill.py` + `ppo_smdp.py` wiring: Gumbel-AZ-lite
  distillation alongside PPO (`smdp_ppo.search_distill`, default OFF; the
  flag-off update path is bit-identical — regression-guarded). A Bernoulli
  `decision_fraction` of rollout decisions is re-analyzed by
  `SearchPolicy.decide` (sim budget = `sims` pool envs, no wall-clock
  deadline) guided by the current net (refreshed per update via the new
  `SearchPolicy.refresh_weights`); targets = completed-Q improved policy
  (`prior + sigma_scale*norm(Q)`, baseline = root V for unevaluated) and
  `v_search = Σ π_target·Q`. Loss adds `beta·KL(π_target‖π_net)` on searched
  rows; `value_mix` blends v_search into value targets. Executed actions stay
  behavior-policy samples (phase 2 = act from search; not implemented).
- Works in 1P and VS rollouts (vs env now emits `info["board"]`; VS search =
  own-board 1P sim approximation, terminal ±1, parked sides skipped).
  Restrictions: candidate policy, 12-ch obs, aux v1 (no v1_vs yet).
- Measured (16 envs, sims=12 beam=8 frac=0.25): rollout dec/s 1736→428
  (tiny net) / 263→83 (d192 prod net, MPS leaves; p50 ≈ 39 ms/searched
  decision); update time unchanged. docs/SEARCH_DISTILL.md has the details.
- `tests/test_search_distill.py`: 14 tests (target math, KL masking, blending,
  flag-off bit-identity, 1P+VS end-to-end smokes). Full suite: 212 pass,
  1 pre-existing fail (game_engine demo NES trace).

## 2026-06-12 — League roles for the VS opponent pool (exploiter/mixed, config-gated)

- `training/envs/vs_opponents.py`: `LeagueConfig` + `parse_league_config`
  (`env.opponent_pool.league`: mode pfsp|exploiter|mixed, main_agents,
  exploiter_fraction; default pfsp = unchanged behavior). Entries gain a
  persisted `league_target` flag (protected from eviction); `sample()` picks
  the eligible subset per mode — exploiter: targets only, PFSP-weighted by
  per-target win rate; mixed: exploiter_fraction coin flip vs targets, else
  the self-history pool. `seed_league_targets` is restart-idempotent by
  checkpoint filename (per-target win/game counts persist in manifest.json).
- `training/envs/drmario_vs_vec.py`: league wiring in the pool setup;
  exploiter mode forces `_snapshot_every=0` (never snapshots itself) and
  skips self-history seeding. `get_vs_metrics` adds `vs/league_targets`,
  `vs/league_win_rate[_min|_max]`, `vs/league_wr_<id>`; dashboard gets a
  conditional "league wr (targets)" row. docs/LEAGUE.md has the contract.
- `tests/test_vs_league.py`: 9 tests incl. exploiter smoke vs the real
  best-535m checkpoint on the native vspool. `pytest -k "opponent or vs or
  league"`: 25 pass.

## 2026-06-12 — BC human-style opponents + Go-Exploit start bank (corpus assets)

- Engine (game_engine @ 6a4215f): `DrmVsResetSpec.checkpoint_*` per-side
  mid-game checkpoint reset via `GameLogic::loadCheckpoint`; wrapper struct
  mirrored in `envs/backends/drmario_vs_pool.py` (struct_size checked —
  rebuild the dylib when pulling: `make -C game_engine libdrmario_pool`).
- `tools/train_bc_opponent.py`: WHR-banded BC dataset (50k moves/band from
  98 quarks; corpus now ~1.9M decisions/894 quarks) + per-band small
  candidate nets (d96, enc2/tx2, aux none) -> runs/bc_opponents/bc_<band>.pt.gz;
  cfg embedded so OpponentPool loads them unchanged.
- `tools/build_start_bank.py` + `training/envs/start_bank.py`: 29,744
  positions (early/mid/late/crisis strata) -> runs/start_bank/start_bank_v1.npz;
  validated 64/64 byte-exact board round-trip + play-to-terminal.
  Env gate: `env.start_bank {enabled,path,fraction}` in DrMarioVsPoolVecEnv
  (disabled = bit-identical resets; no extra RNG draws).
- Tests: tests/test_bc_opponent.py, tests/test_start_bank.py (+ fixture
  tests/fixtures/fc_v2_events_small.jsonl); vs/pool suite green (37 passed).
- drmc-rl commit c3b934a; docs/HUMAN_CORPUS_INTEGRATION.md has rebuild/enable
  runbooks; dated note appended to ../fightcadeRatings/COORDINATION.md.

## 2026-06-13 — Static internal competition dashboard

- Added `tools/export_competition_web.py` plus `web/pool/` static HTML/CSS/JS.
  The exporter reads `runs/tournaments/tournaments.sqlite` and opponent-pool
  manifests, fits static Bradley-Terry ratings for frozen agents, emits
  `web/pool/data.js`, and renders ratings, pairwise records, tournaments,
  pool manifests, and a method note.
- The rating model is intentionally static: frozen checkpoints have one latent
  skill; estimate movement after re-export comes from added games/opponents or
  connected-component changes, not modeled calendar drift.
- Validation: `pytest -q tests/test_competition_web_export.py`,
  `python3 -m tools.export_competition_web`, `node --check web/pool/app.js`,
  and browser smoke checks at desktop + 390px mobile width.
