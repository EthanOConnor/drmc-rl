# BACKLOG.md — drmc-rl

Technical backlog / roadmap. More detailed items than top-level docs.

---

## Up Next / P0: Multi-Env C++ Backend

> See `notes/DESIGN_MULTIENV.md` for full design document.

### 1. Correctness

- [x] **Verify multi-instance isolation**
  - [x] Test: spawn multiple `CppEngineBackend` instances in same process
  - [x] Assert unique `_shm_file` paths, independent stepping
  - [x] Added `tests/test_cpp_backend_multienv.py`

- [x] **Test vector env modes**
  - [ ] `SyncVectorEnv` with `num_envs=4` (same process) + leak checks (nice-to-have)
  - [x] `AsyncVectorEnv` smoke test with `num_envs>=8` (worker processes)
  - [x] Prefer robust shutdown paths for long async runs (force-terminate on close)

### 2. Performance

- [ ] **Direct struct→observation path** (skip synthetic RAM)
  - [ ] Add `BitplaneBuilder.from_engine_state(state)` for C++ backend
  - [ ] Bypass `_refresh_ram_from_state()` + `build_state()`
  - [ ] Benchmark: expect 0.15ms → 0.03ms per step

- [x] **Batched stepping for macro actions**
  - [x] Add `step_frames`/`run_until_decision` style batching in `cpp-engine` backend
  - [x] Use for placement env (decision → lock → settle → next spawn)

- [x] **Measure async scaling**
  - [x] `tools/bench_multienv.py` reports fps/dps + speedup/efficiency across `num_envs`
  - [ ] Track benchmark results over time (CI artifact / doc table)

### 3. Debug TUI Multi-Env Support

- [x] **Hotkeys for env navigation**
  - [x] `Tab` — cycle per-env views (env 0, 1, ..., summary)
  - [x] `1`-`9` — jump to env N
  - [x] `0` — summary view (all boards)

- [x] **Summary view** (new layout)
  - [x] Grid of mini-boards (scales with env count)
  - [x] Aggregate stats panel

- [x] **Speed scaling metrics**
  - [x] `fps(total)`, `fps(per-env)`
  - [x] `speedup_vs_single` — ratio vs num_envs=1 baseline
  - [x] `efficiency` — speedup / num_envs × 100%

- [x] **Env count hotkeys** (restart-only)
  - [x] `[` / `]` — adjust num_envs (restart training)

---

## Near-term (This Sprint)

### Critical Priority

- **Up Next (P0): Multi-env training throughput + debug UI**
  - **P0.1 Fix SMDP-PPO multi-env correctness (true vector actions)**
    - Current SMDP-PPO rollout collection must select actions for *all* envs and call `env.step(actions)` once per decision batch (no “one env at a time” stepping).
    - Handle per-env `placements/tau` correctly:
      - Track `global_step_frames_total += sum(tau_i)` as the canonical “frames processed” counter.
      - Log/emit both `perf/sps_frames_total` and `perf/dps_decisions_total` so scaling comparisons are unambiguous.
    - Ensure episode boundaries are per-env:
      - Emit `episode_end` events for each env independently (return, frames, decisions, terminal_reason).
      - Reset per-env accumulators without affecting other envs.
    - Validate with deterministic seeds:
      - `num_envs=1` behavior must match current baselines.
      - `num_envs>1` must produce identical results to running each env in isolation with the same seeds (modulo different action sampling).

  - **P0.2 Make `cpp-engine` scale with `num_envs` (initial implementation: Gymnasium `AsyncVectorEnv`)**
    - Default vectorization policy:
      - `--ui debug`: use `SyncVectorEnv` (single process) for simpler inspection.
      - Headless training: use `AsyncVectorEnv` for parallel env stepping and planner execution across CPU cores.
    - Add a dedicated benchmark harness to quantify scaling efficiency:
      - Compare `num_envs ∈ {1,2,4,8,16,32,...}` for `sync` vs `async` and report:
        - `fps_total`, `fps_per_env`, `speedup`, `efficiency = fps_total/(num_envs*fps_1env)`.
      - Keep an “env-only” bench mode that bypasses policy updates so we can isolate simulator+planner scaling.
    - Reduce cross-process payload when training at scale:
      - Add an env option to disable `info["raw_ram"]` (and other heavy debug-only fields) during training; keep it enabled for debug UI and parity tools.

  - **P0.3 Debug TUI: multi-env boards + selection + env-count hotkeys**
    - Add a **grid view** showing N boards at once + an aggregate stats panel (overall FPS, total frames/sec, mean/median return, curriculum distribution).
    - Add a **focused view** for a selected env index (current behavior), with per-env perf/reward breakdown.
    - Hotkeys:
      - Increase/decrease active env count (UI-driven): `[`/`]` (exact keys TBD).
      - Switch selected env: `tab`/`shift+tab`, `0-9` quick jump, and `g` to toggle grid/focus.
    - Implementation approach (pragmatic): treat env-count changes as “restart training with new num_envs” (stop thread, rebuild vec env, restart) rather than attempting to mutate Gymnasium VectorEnv sizes in-place.
    - Add a lightweight “scaling HUD” in debug mode:
      - Show speedup vs 1 env and efficiency vs linear scaling, computed from the benchmark harness.

  - **P0.4 Longer-term (optional, high ceiling): batch/in-process cpp-engine backend**
    - Goal: eliminate per-env subprocess polling overhead and process explosion (AsyncVectorEnv + per-env engine process).
    - Two candidate designs (choose one after P0.1–P0.3 are working + benchmarked):
      - **Multi-instance engine process**: one engine binary simulates N games; shared memory exposes arrays of per-instance state; driver issues batched step requests.
      - **In-process engine library**: build a Python extension (pybind11/ctypes) with N engine instances in memory; batched step is a single C++ call.
    - Success criterion: demonstrate superlinear scaling vs the current “N engines in N processes” baseline and validate parity invariants per instance.

- **P0.5 Curriculum certification gate (freeze + anytime-valid test)**
  - Freeze policy at curriculum stage boundaries and run eval episodes (fixed seeds/registry when available).
  - Advance using an anytime-valid sequential test (SPRT / confidence sequence) with feasibility checks (`n_max` vs target).
  - Keep PPO batches stage-pure by running certification only between PPO updates (no cross-stage rollouts).

- **Wire TUI into training loop** ✅
  - Connect `TrainingTUI` to `SimplePGAdapter` update callbacks.
  - Pass metrics from adapter → TUI via update() method.
  - Test with: `python -m training.run --algo simple_pg --ui tui`

### High Priority

- **Placement planner: native acceleration + parity oracle** ✅ (native accel shipped)
  - Keep `envs/retro/fast_reach.py` as the reference implementation.
  - Native BFS accelerator: `reach_native/drm_reach_full.c` + `envs/retro/reach_native.py` (ctypes).
  - Build: `python -m tools.build_reach_native`
  - Benchmark: `python -m tools.bench_reachability` (add `--include-python` for slow oracle timing)
  - Remaining: expand parity-oracle traces beyond the “immediate lock” smoke test.

- **Placement env: decision-point regression traces**
  - Record short emulator traces covering: spawn → lock → settle → next spawn, stage clear, top-out, ending.
  - Assert `DrMarioPlacementEnv` never times out waiting for a decision point.

- **Libretro backend: RAM-only fast path**
  - Add a mode that avoids per-frame 240×256 frame-buffer copies when running `obs_mode=state` without rendering/video.
  - Keep `render_mode=rgb_array` working (either lazy capture or “capture-on-demand”).
  - Benchmark before/after in terms of FPS and reset latency.

- **C++ engine: Parity beyond demo**
  - Add golden traces for non-demo gameplay (seeded levels, edge-case rotations/DAS).
  - Validate 2P / attack logic once implemented.

- **Curriculum: time goals + best-times DB**
  - Extend `BestTimesDB` to record top-K clear times per (level, seed) (not just the best).
  - Add an optional “pill sequence signature” key so we can track best times per (level, seed, sequence).
  - Surface best-time quantiles and current time budget floors in the Rich TUI.
  - Consider quantile-based floors (e.g. p10) instead of clamping to the absolute minimum.

- **Extract diagnostics tracker**
  - Move `NetworkDiagnosticsTracker` from `speedrun_experiment.py` to `training/diagnostics/tracker.py`.
  - Integrate with TUI for gradient/param stats display.

- **Complete evaluator training**
  - Finish `models/evaluator/train_qr.py` skeleton for QR-DQN distributional head.
  - Train on RAM-labeled corpus.

- **Populate seed registry**
  - Capture savestates for 120 seeds per level in `envs/retro/seeds/`.
  - Store first 128 pills and virus grid hash per seed.
  - Add a deterministic training option that samples only from the registry (fixed layouts + pill sequences) to reduce variance for curriculum gating and policy updates.

### Recently Completed ✅

- Unified runner debug mode (`training/run.py --ui debug`) with board viz + playback controls
- Real retro vector env factory in `training/envs/dr_mario_vec.py` (Gymnasium VectorEnv + info normalization)
- Placement macro-action rewrite (NES-accurate reachability + `DrMarioPlacementEnv`)
- QuickNES updater script (`tools/update_quicknes_core.py`) + docs in `docs/RETRO_CORE_NOTES.md`
- C++ engine demo parity suite (frame-perfect vs `data/nes_demo.json`)
  - Tooling: `tools/record_demo.py`, `tools/game_transcript.py`
  - Regression test: `tests/test_game_engine_demo.py::test_demo_trace_matches_nes_ground_truth`
- Interactive C++ engine demo TUI (`tools/engine_demo_tui.py`)
- Rich TUI with sparklines (`training/ui/tui.py`)
- Board state viewer (`training/ui/board_viewer.py`)
- Debug viewer (`training/ui/debug_viewer.py`)
- Device utils (`training/utils/devices.py`)
- WandB logger stub (`training/utils/wandb_logger.py`)
- Enhanced run.py with --ui, --wandb flags
- Cleanup: deleted stubs, archived drmarioai, updated gitignore


---

## Medium-term (This Quarter)

- **Stable-Retro integration**
  - Wire `retro.get_ram()` → `ram_to_state` in `DrMarioRetroEnv`.
  - Add debug mode to print per-channel counts live.

- **Evaluator-based reward shaping**
  - Add `use_potential_shaping` and `kappa` flags.
  - Compute r_shape = γΦ(s') − Φ(s).
  - Log r_env, r_shape, r_total.

- **Complete pixel2state translator**
  - Implement UNet/ViT-tiny in `models/pixel2state/`.
  - Train on emulator frames + RAM labels.
  - Later: fine-tune on HDMI capture frames.

- **PettingZoo 2-player wrapper**
  - Complete skeleton in `envs/pettingzoo/`.
  - Two agents, simultaneous moves, shared RNG.

- **Seed sweeps evaluation**
  - Extend `eval/harness/seed_sweep.py`.
  - Output Parquet/CSV with E[T], Var[T], CVaR metrics.
  - Add risk-conditioned runs at τ∈{0.25, 0.5, 0.75}.

---

## Longer-term (Future)

- **EnvPool C++ simulator**
  - Re-implement core rules in C++ for 10–100× FPS.
  - Golden parity suite against emulator traces.
  - Host in EnvPool with Gym API.

- **Real-console I/O bridge**
  - Video in: HDMI capture from Analogue/FPGA clone.
  - Controller out: RP2040/Arduino emulating NES shift-register protocol.
  - Sync to VBlank at 60 Hz.

- **Curriculum learning**
  - Virus count progression (4 → 8 → 16 → 84).
  - Level progression (0 → 5 → 10 → 20).
  - Adaptive sampling based on success rate.

- **Advanced policy heads**
  - Set-to-Set cross-attention (Perceiver-style).
  - Transformer decoder over placement tokens.
  - Multi-head attention over board.

- **Distributed training**
  - Mixed-precision (FP16).
  - Multi-GPU gradient accumulation.
  - Ray/RLlib integration for cluster scale.
