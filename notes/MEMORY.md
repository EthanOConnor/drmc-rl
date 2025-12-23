# MEMORY.md — drmc-rl

Long-lived architectural memory, design decisions, and "why" behind choices.
Format: short entries, timestamped, with rationale and trade-offs (ADR-lite).

---

## 2025-10-17 – Initial Architecture Decisions

### Environment Design

- **Observation modes (pixel vs state-tensor)**:
  - Pixel: 128×128 RGB, frame-stack 4, normalized [0,1]. For pixel-based RL and real-console camera input.
  - State-tensor: 16×8 grid, 14 channels (3 virus colors, 3 fixed pill colors, 3 falling pill colors, orientation, gravity scalar, level scalar, time_since_spawn, settle_flag). Stack 4 frames.
  - Why: Train fast on state-tensor, then bridge to pixels for real-console via pixel→state translator.

- **Action space**:
  - 10 discrete actions at 60 Hz: noop, left, right, down, rotate_A, rotate_B, left_hold, right_hold, down_hold, both_rot.
  - Hold actions latch until opposite tap, new hold, or lock. No DAS emulation initially.
  - Why: Matches NES input clock while keeping action space tractable.

- **Reward function**:
  - r_t = −1 + 8·ΔV + 0.5·chain_bonus − 0.1·settle_penalty; terminal clear +500.
  - Why: Dense shaping accelerates learning; terminal bonus incentivizes completion.

### Placement Policy

- **SMDP discounting**:
  - Each decision spans τ frames. Use Γ_t = γ^τ for proper credit assignment.
  - Why: Pill placement is a semi-Markov decision—standard per-frame discounting would bias toward fewer-frame placements.

- **Three policy head variants** (dense, shift_score, factorized):
  - All produce [B, 4, 16, 8] logit maps (512 actions: 4 orientations × 16 rows × 8 cols).
  - dense (FiLM-conditioned CNN) is recommended baseline.
  - Why: Ablation-ready design; same interface enables easy comparison.

- **Color-order invariance**:
  - Deep-Sets embedding (sum + product pooling) ensures (c1, c2) == (c2, c1).
  - Why: No manual augmentation needed; reduces effective action space complexity.

- **Single inference per spawn**:
  - Policy runs once when pill spawns; logits cached by spawn_id; replanning reuses logits with updated mask.
  - Why: Efficiency—avoids repeated forward passes during execution.

### C++ Engine

- **Direct port first, optimize later**:
  - Translate assembly logic 1:1 to C++ for accuracy, then refactor for performance.
  - Why: Correctness must be verified against NES before optimizations obscure parity.

- **Shared memory IPC**:
  - 180-byte DrMarioState struct shared between engine and Python agent.
  - Agent writes buttons, engine writes state. Non-blocking.
  - Why: Lowest-latency communication for high-throughput training.

### RNG and Determinism

- **Seed registry via savestates**:
  - Savestate at Level Select with fixed power-on frame counter pins layout + pill sequence.
  - Registry maps seed_id → (state_file, frame_offset).
  - Why: TAS community confirmed layout/sequence are determined by frame count at selection.

---

## 2025-11-22 – C++ Engine Initial Implementation

- **Completed**:
  - RNG: accurate LFSR (Bit 1 EOR + ROR) verified against assembly.
  - Level gen: virus placement using original algorithms/tables.
  - Basic loop: gravity, pill generation (reserve logic), piece locking.
  - Matching: horizontal/vertical detection and clearing.
  - Post-clear gravity: floating blocks fall.

- **Deferred**:
  - DAS (Delayed Auto Shift) physics.
  - Wall kicks for rotation.
  - These are documented in game_engine/AGENTS.md as next steps.

---

## 2025-12-16 – Codebase Review Findings

- **Test health**: 64 tests, all passing. Good coverage of placement policy, discounting, RAM decoding.
- **Gap identified**: No inter-session notes system → now implemented.
- **speedrun_experiment.py complexity**: 5436 lines, 170 functions. Being replaced by modular approach.
- **C++ engine gaps**: Missing DAS and wall kicks prevents accurate high-speed play testing.

---

## 2025-12-16 – Unified Runner Architecture

### Decision: Rich-based TUI (Not Tkinter)

**Context**: Original `speedrun_experiment.py` used Tkinter (784-line `_monitor_worker`).

**Decision**: Replace with Rich library.

**Rationale**:
- Works in any terminal (no X11 needed on servers/containers)
- Modern Python package with active maintenance
- Better for headless logging integration
- Sparklines and color support built-in

### Decision: Modular UI Components

Created `training/ui/` with clear separation:
- `tui.py` (312 lines): Training metrics TUI with sparklines
- `board_viewer.py` (290 lines): Dr. Mario board visualization
- `debug_viewer.py` (210 lines): Interactive step-by-step debugger
- Lazy imports in `__init__.py` to avoid runpy warnings

### Decision: Graceful WandB Integration

- `training/utils/wandb_logger.py`: Stubs WandB API
- Falls back silently when WandB not installed
- Free tier available for personal use

### Decision: Unified Device Resolution

- `training/utils/devices.py`: Single interface for PyTorch + MLX
- `resolve_device("auto")` picks best available (CUDA → MPS → Metal → CPU)

### Current Runner Status

**Completed (Phase 1)**:
- ✅ Rich TUI with sparklines (`training/ui/tui.py`)
- ✅ Board state viewer (`training/ui/board_viewer.py`)
- ✅ Debug viewer (`training/ui/debug_viewer.py`)
- ✅ Device utils (`training/utils/devices.py`)
- ✅ WandB stub (`training/utils/wandb_logger.py`)
- ✅ Enhanced `run.py` with `--ui tui|headless`, `--wandb`
- ✅ Cleanup: deleted stubs, archived drmarioai, updated gitignore

**In Progress (Phase 2)**:
- [ ] Wire TUI into training loop (currently standalone)
- [ ] Extract diagnostics tracker from speedrun_experiment.py
- [ ] Test run.py with --ui tui end-to-end

**Future (Phase 3)**:
- [ ] Mark speedrun_experiment.py as deprecated
- [ ] Remove legacy Tkinter code
- [ ] Full integration testing

---

## 2025-12-17 – NES Demo Parity Testing

### Recording Layer Architecture ✓

**Verified rock-solid** - next agent can trust recordings:
- `tools/record_nes_demo.py`: Reads NES RAM via libretro at 0x0400
- `tools/record_demo.py`: C++ engine recorder with --manual-step
- `tools/game_transcript.py`: JSON serialization, delta encoding, comparison utils
- Pill positions/colors/board state all correctly captured

**Update (2025-12-18):** The below constants were part of an *interim* alignment
hack and are superseded by the rules-exact parity port described in the
2025-12-18 entry (NMI-tick emulation + NES `nextAction` / `pillPlacedStep` state
machines, full ROM tables, BCD counters, etc.). Keep this section as historical
context only.

### Key Constants (interim, superseded)
- `spawn_delay = 35` frames (throw animation approximation)
- `INPUT_OFFSET = 124` (external demo input indexing hack)
- `demo_pills` array: 45 bytes (partial table; later corrected to full 128 bytes)

### Parity Status
- **Pills 1-3**: ✓ Full parity (positions, colors, board state)
- **Pill 4+**: First divergence (cumulative timing drift)
- **Root cause**: C++ engine behavioral difference, not recording

### Files Modified for Parity
| File | Change |
|------|--------|
| `GameState.h` | Added `spawn_delay` field |
| `GameLogic.cpp` | spawn_delay init/decrement, second generateNextPill |
| `engine_shm.py` | Python struct updated for spawn_delay |
| `record_demo.py` | INPUT_OFFSET = 124 |

### Next Investigation Areas
1. Gravity counter comparison (`speedCounterTable` indexing)
2. DAS timing differences (`hor_velocity` counter)
3. spawn_delay interaction with input frame indexing

---

## 2025-12-18 – NES Demo Parity: Rules-Exact Frame Loop Port

### Decision: Emulate the NES’s 2-phase per-frame model

**Context:** Dr. Mario’s gameplay logic is split across:
1) **NMI-time work** (input sampling / demo replay, one-row status rendering, etc.)
2) **Main-thread work** (the `nextAction` dispatcher and its sub-state machines)

The earlier `spawn_delay` / input-offset approach matched a few early pills but
drifted because it did not reproduce the actual NES control flow and NMI-coupled
timing.

**Decision:** Model each frame as:
- `nmi_tick()` at the *start* of `step()` (status-row countdown + input update)
- One main-thread `nextAction` routine per frame (`PillFalling`, `PillPlaced`, `SendPill`, …)

**Why:** This matches how the ROM interleaves rendering/input and gameplay state,
and it eliminates cumulative drift.

---

## 2025-12-20 – Runner Logdir Run IDs

### Decision: Unique per-run log subdirectory (default)

**Context:** Training runs (notably `ppo_smdp`) defaulted to a constant `logdir`
like `runs/smdp_ppo`, causing checkpoints and `metrics.jsonl(.gz)` to collide across
invocations and making it easy to overwrite/append runs unintentionally.

**Decision:** `python -m training.run` now creates a unique `run_id` and writes
outputs to `cfg.logdir/<run_id>` by default. Supplying `--logdir ...` opts out.
The `run_id` and resolved `logdir` are recorded in run metadata.

**Rationale:** Avoids collisions, supports concurrent runs, and keeps artifacts
grouped and attributable.

**Trade-offs:** Existing scripts that assume a fixed path (e.g.
`runs/smdp_ppo/metrics.jsonl(.gz)`) must select a run directory; mitigated by the
`--logdir` override and tooling that can pick the newest run automatically.

### Key parity facts captured in code

- **Status-row gating:** `checkDrop` is gated on `status_row == 0xFF`, which is
  decremented by NMI-time bottle row rendering (modeled explicitly).
- **`nextAction` / `pillPlacedStep`:** Implemented as explicit state machines;
  the demo drift root cause was missing these transitions and micro-steps.
- **BCD counters:** Virus count (`viruses_remaining`) and pill counters use BCD
  semantics. The transcript’s “viruses_cleared” deltas reflect BCD boundaries.
- **ROM tables:** Use full retail ROM data:
  - `demo_pills`: 128 bytes, indexed by `& 0x7F` (includes the “UNUSED” tail).
  - `demo_instruction_set`: 512 bytes (256 pairs); each (btn,dur) lasts `dur+1` frames.
  - `speedCounterTable`: full NTSC table from `drmario_data_game.asm`.
- **Demo capture alignment:** `DEMO_INPUT_PREROLL_FRAMES = 7` matches the start
  state of the recorded ground truth in `data/nes_demo.json`.

### Tooling alignment decision

- **Recorder semantics:** `tools/record_demo.py` stops recording when demo ends
  *before* appending that final frame, matching the NES recorder (frames end at
  5700 while `total_frames = 5701` in the ground truth).

---

## 2025-12-18 – Engine Demo Playback TUI

### Decision: Use manual-step shared-memory driving (no new debug struct)

**Context:** We wanted an interactive “demo player” for the C++ engine that can
pause, single-step, and vary playback speed while showing the board and
important internal counters.

**Decision:** Implement the viewer as a Python Rich TUI that:
- Starts `drmario_engine` in `--demo --manual-step` mode.
- Steps by toggling `control_flags & 0x04` and reads state from shared memory.
- Visualizes “internal state” using the existing shared-memory fields (inputs,
  counters, flags) rather than extending the shared struct with debug-only data.

**Why:** Keeping the shared-memory ABI stable avoids ripple changes in the
training stack. If deeper engine-private variables (`nextAction`, status-row
gate, etc.) are needed later, add them via an explicit, versioned debug channel
rather than silently changing the core IPC struct.

### Decision: Playback rate as an `x` multiplier

**Context:** Human debugging often wants “2.4× realtime NTSC” style controls
instead of thinking in seconds-per-frame delays.

**Decision:** The demo TUI expresses playback rate as `speed_x` against a base
framerate (NTSC/PAL), and schedules steps using a small accumulator to hit the
target rate while keeping the UI responsive. `MAX` mode removes the cap.

---

## 2025-12-18 – Macro Placement Planner: NES-Accurate Spawn-Latched SMDP

### Decision: Canonical 512-way macro action space

**Decision:** Standardize macro actions as a dense `(o,row,col)` grid with shape
`(4, 16, 8)` and a flat index in `[0, 512)`, implemented in
`envs/retro/placement_space.py`.

**Why:** A dense 4×16×8 logit map is the natural interface for CNN policy heads.
We keep the full grid (including boundary actions) and handle legality via masks.

### Decision: Use the NES base-cell coordinate model (not “first-half anchor”)

**Decision:** Treat the ROM’s falling pill position as the **bottom-left cell of
the pill’s 2×2 bounding box** (NES `fallingPillX`/`fallingPillY` convention), and
convert to top-origin `(row,col)` only for planner/policy convenience.

**Why:** This is the actual coordinate system used by the ROM routines. Using a
“first-half anchor” model produces systematic off-by-one geometry errors in
rotation and wall/stack interactions, which makes feasibility masking unreliable.

### Decision: Frame-accurate reachability as the reference planner

**Decision:** Implement reachability by mirroring the ROM’s per-frame update
order (Y move → X move/DAS → rotate), including the relevant counters
(`speedCounter`, `horVelocity`, frame parity, held direction), in
`envs/retro/fast_reach.py`.

**Trade-off:** This is slower than earlier “512-state geometry BFS” approaches,
but it is correctness-first and forms a reliable reference for later native
optimizations (C++/SIMD/bitboard acceleration) without changing behaviour.

### Decision: Native reachability accelerator via a small C shared library

**Decision:** Keep `envs/retro/fast_reach.py` as the oracle, but run the
spawn-time BFS in native code for training-speed performance:

- C implementation: `reach_native/drm_reach_full.c`
- Python wrapper: `envs/retro/reach_native.py` (ctypes)
- Build command: `python -m tools.build_reach_native`
- Runtime selection: `PlacementPlanner(reach_backend=auto|native|python)` and
  `info["placements/reach_backend"]` for debugging.

**Why:** The per-spawn BFS is the dominant cost in the placement macro env when
implemented in Python. A small, dependency-free C helper preserves exact ROM
semantics while making training runs practical (and keeps the Python reference
available for parity checks).

### Decision: Spawn-latched macro environment wrapper

**Decision:** Provide `DrMarioPlacementEnv` (`envs/retro/placement_env.py`) as the
macro (SMDP) environment that:
- Waits for a ROM decision point (`currentP_nextAction == pillFalling`)
- Emits `placements/*` masks + `placements/tau` in `info`
- Executes a minimal-time controller script for the selected macro action

**Why:** “One decision per spawn” keeps learning stable and makes SMDP discounting
(Γ=γ^τ) well-defined. It also cleanly supports “one inference per spawn” caching
via `placements/spawn_id`.

---

## 2025-12-18 – Unified Runner: Debug TUI + Real Retro Env

### Decision: Keep `training.run` as the canonical entrypoint

**Decision:** Prefer `python -m training.run` for day-to-day training/debugging,
and treat `training/speedrun_experiment.py` as legacy.

**Why:** The unified runner centralizes config, diagnostics, and UI plumbing so we
can iterate on training setups without duplicating bespoke harness logic.

### Decision: Real env factory with a dummy fallback

**Decision:** `training.envs.make_vec_env` builds a real Gymnasium VectorEnv when
`env.id` is a registered Dr. Mario env (`DrMarioRetroEnv-v0`, `DrMarioIntentEnv-v0`,
`DrMarioPlacementEnv-v0`), otherwise it returns the lightweight `DummyVecEnv`.

**Why:** This keeps unit tests fast (no emulator dependency) while enabling “real
training” runs from the same codepath.

### Decision: Normalize Gymnasium vector `infos` to list-of-dicts

**Decision:** Wrap Gymnasium VectorEnvs so `reset/step` return `infos` as a
`List[Dict]` (one dict per env), matching the historical `DummyVecEnv` interface.

**Why:** Most of the existing training code and tooling assumes list-of-dicts.
Normalizing at the env boundary avoids pervasive adapter changes and keeps
debugging code straightforward.

### Decision: Playback control as an env wrapper (algorithm-agnostic)

**Decision:** Implement pause/single-step and target FPS throttling as a wrapper
around the vector env (`training/envs/interactive.RateLimitedVecEnv`) controlled
by a thread-safe `PlaybackControl`.

**Why:** This keeps training algorithms (SimplePG/PPO/SMDP-PPO) focused on RL and
makes interactive debugging available uniformly across algorithms.

### Decision: Target FPS is based on emulator frames via `placements/tau`

**Decision:** When available, use `info["placements/tau"]` as the number of
emulated frames consumed by a macro step, and schedule wall-clock playback using
`target_hz = base_fps * speed_x`.

**Why:** This provides a consistent “x realtime” control across both per-frame
controller envs and macro placement envs.

### Decision: RNG randomization is an env attribute (works with vector autoresets)

**Decision:** Store the “randomize ROM RNG on reset” toggle as an env attribute
(`DrMarioRetroEnv.rng_randomize`) and treat it as the default for
`reset(options={"randomize_rng": ...})` when options don’t specify.

**Why:** Gymnasium vector environments perform autoresets by calling `reset()`
without passing options. Keeping the toggle on the env ensures episode-to-episode
RNG randomization can be enabled/disabled reliably in both headless training and
interactive debug runs (including runtime hotkey toggles).

### Decision: `DrMarioRetroEnv.reset()` rebuilds state after auto-start

**Decision:** After running the auto-start/start-press sequence in
`envs/retro/drmario_env.py`, rebuild `_state_cache` from the latest RAM before
returning `obs`/`info`.

**Why:** Auto-start advances emulator frames via `_backend_step_buttons`, which
updates the RAM cache but (by design) does not rebuild `_state_cache` every
frame. Rebuilding once at the end ensures `reset()` returns a consistent snapshot
(state tensor, derived masks, and `raw_ram` all agree), avoids out-of-range
observations, and prevents placement decision-point logic from seeing stale RAM.

### Decision: Prefer raw RAM virus counter during startup

**Decision:** Prefer the raw `p1_virusLeft` RAM byte (via the offsets table) for
`viruses_remaining` when available, falling back to derived counts from the state
tensor only when RAM is unavailable.

**Why:** During reset/startup sequences, `_state_cache` may be temporarily stale.
The raw counter is cheap to read and provides a reliable “game has loaded” signal
for auto-start wait loops.

### Decision: Treat empty-feasible “decision points” as non-decisions

**Decision:** If `DrMarioPlacementEnv` reaches a `pillFalling` decision point but
the placement planner reports **zero feasible in-bounds macro actions**
(`placements/options == 0`), do **not** surface a macro decision to the agent.
Instead, continue stepping NOOP until the underlying env transitions (lock /
top-out / reset) or a later controllable spawn appears.

**Why:** In spawn-blocked/top-out scenarios the capsule can lock offscreen before
any meaningful input window exists. Surfacing an empty mask would otherwise
cause policies to loop on invalid actions without advancing the emulator,
freezing interactive runs and corrupting training data.

### Decision: Interactive perf diagnostics are accumulated in `RateLimitedVecEnv`

**Decision:** For `training.run --ui debug`, accumulate performance counters
(inference/planner/update timings) inside `training.envs.interactive.RateLimitedVecEnv`,
using:
- env-emitted per-step timings (`perf/planner_build_sec`, `perf/planner_plan_sec`)
- optional training-thread hooks (`record_inference`, `record_update`)

**Why:** The debug UI runs in a separate thread/process context and should read a
single thread-safe snapshot (`perf_snapshot()`) without coupling to a specific
training algorithm or backend. This keeps overhead low and makes it easy to
compare “emu step” vs “training compute” bottlenecks.

### Decision: `DrMarioPlacementEnv` decisions are spawn-latched via `pill_counter` (RAM `$0310`)

**Decision:** `DrMarioPlacementEnv` must expose *one macro decision per pill
spawn* (not one per falling frame). Decision-point detection is therefore gated
on the ROM spawn counter (`pill_counter`, RAM `$0310`) in addition to
`currentP_nextAction == nextAction_pillFalling` and a present falling-pill mask.

Concretely:
- The wrapper tracks a `_consumed_spawn_id`.
- A macro decision is only produced when `pill_counter != _consumed_spawn_id`.
- A spawn becomes “consumed” when we commit to a macro plan for it (or when the
  planner reports `placements/options == 0` and we auto-advance).

**Why:** `nextAction_pillFalling` stays active for many consecutive frames while
the capsule falls. Treating every `pillFalling` frame as a decision causes the
policy to re-run inference and rebuild reachability masks *mid-fall* as the
capsule descends (“options ticking down”), inflating planner/inference counts
and breaking the intended SMDP semantics.

### Decision: Attach `episode` stats in the vector-env wrapper (length in emulated frames)

**Decision:** Normalize real Gymnasium vector env outputs to the project’s
historical list-of-dicts `infos` format *and* attach standard episode summaries
(`info["episode"]`) inside `training.envs.dr_mario_vec._InfoListWrapper`.

Concretely, on terminal steps the wrapper injects:
- `info["episode"]["r"]`: cumulative return since last termination
- `info["episode"]["l"]`: episode length in **emulated frames**
  - uses `placements/tau` when present (macro env)
  - falls back to 1 frame per env step (controller env)
- `info["episode"]["decisions"]`: number of wrapper `step()` calls in the episode

**Why:** Multiple training adapters (`simple_pg`, `sf2_adapter`, `ppo_smdp`) and
UIs expect `info["episode"]` to exist (matching `DummyVecEnv`). Using frames for
`l` keeps episode “length” consistent with SMDP training where time-to-clear is
measured in emulator frames.

---

## 2025-12-18 – Scripted Curriculum via Synthetic Negative Levels

### Decision: Encode early curriculum stages as negative `env.level`

**Decision:** Represent early curriculum stages as negative `env.level` values
and implement them by patching the bottle RAM at reset time, while still
selecting ROM level 0 (the only valid menu level for these stages).

Mapping:
- `level=-10`: 0 viruses; **success** = 1 match (“any match”)
- `level=-9`: 0 viruses; **success** = 2 matches
- …
- `level=-4`: 0 viruses; **success** = 7 matches
- `level=-3`: 1 virus
- `level=-2`: 2 viruses
- `level=-1`: 3 viruses
- `level=0`: vanilla level 0 (4 viruses)

**Why:** This keeps the ROM’s real physics, timing, and clearing rules (the same
ones used by the reachability planner) while enabling a simple, high-signal
curriculum without introducing bespoke env ids or synthetic dynamics.

### Decision: Curriculum scheduling lives in the vector-env wrapper

**Decision:** Implement curriculum progression as a vector-env wrapper
(`training.envs.curriculum.CurriculumVecEnv`) that:
- tracks rolling clear rates,
- updates per-env `level` via `set_attr` *after* terminal steps (works with
  Gymnasium’s default `autoreset_mode=NextStep`), and
- optionally rehearses lower levels with a small probability.

**Why:** Training algorithms assume vector autoresets; keeping curriculum logic
outside the env avoids coupling and makes it easy to disable/adjust via config.

### Decision: Detect match events via bottle clearing tile codes (not occupancy deltas)

**Context:** The synthetic curriculum match stages (levels `-15..-4`, “0
viruses, terminate after N match events”) initially used an occupancy-delta
heuristic (`tiles_cleared_total`) to detect clears. This can miss clears when
other occupancy changes happen in the same macro step or when representation
details change.

**Decision:** Detect clear events by scanning the bottle RAM for the ROM’s
explicit clear-animation tile type codes (`CLEARED_TILE` / `FIELD_JUST_EMPTIED`)
and terminate when at least 4 tiles are marked as clearing.

**Implementation note:** `FIELD_EMPTY` is encoded as `0xFF`, which shares the
same high nibble (`0xF0`) as `FIELD_JUST_EMPTIED` (`0xF0..0xF2`). When masking
tile *types* via the high nibble, we must explicitly exclude `0xFF` (full-byte
check) or we will falsely treat an empty bottle as “clearing”. We also gate the
scan on `gameplay_active` to avoid stale bottle bytes during menu/reset frames.

**Why:** This is a direct, rules-exact signal from the game engine and is far
less brittle than occupancy-based heuristics.

### Decision: Aggregate reward components for macro steps and surface in the debug UI

**Decision:** `DrMarioPlacementEnv` now aggregates per-frame reward components
across each macro step and emits them as `reward/*` totals + counts. The debug
UI (`RunnerDebugTUI`) displays these in a dedicated Reward column, with current
episode totals (or last episode totals when between episodes).

**Why:** This makes reward shaping auditable and helps catch subtle mistakes in
curriculum success detection, reward config signs, and per-term scaling without
digging through logs post-hoc.

---

## 2025-12-19 – Planner/Input Parity + Canonical Clear Counting

### Decision: Down-only soft drop parity is `frameCounter & 1 == 0`

**Decision:** In the falling-pill update, “down-only” soft drop is applied on
frames where `frameCounter & 1 == 0` (with DOWN held and no LEFT/RIGHT).

**Why:** Emulator tracing shows the pill advances downward exactly on even-parity
frames under down-only input. This parity gate must match across:
- the Python oracle (`envs/retro/fast_reach.py`),
- the Python packed BFS (reachability fast path), and
- the native reachability helper (`reach_native/drm_reach_full.c`),
or else the planner will generate controller scripts that don’t execute correctly.

### Decision: Pose verification uses RAM state transitions (not reward signals)

**Decision:** `DrMarioPlacementEnv` captures `placements/lock_pose` by tracking
the last valid falling-pill pose while `currentP_nextAction == nextAction_pillFalling`,
then freezing it when we leave that state (or when the spawn counter advances).

**Why:** Reward terms like `pill_bonus_adjusted` are configurable and may be zero
for shaping-focused runs; using them for correctness checks is brittle and can
silently disable verification.

### Decision: Clear counts for reward are derived from bottle-buffer diffs

**Decision:** `tiles_cleared_total` / `tiles_cleared_non_virus` are computed
canonically from the bottle buffer (occupied→non-occupied transitions) using
`envs/specs/ram_to_state.count_tile_removals`, with a cached previous bottle
snapshot in `DrMarioRetroEnv`.

**Why:** This avoids false positives from observation overlays (falling pill,
preview) and keeps reward/curriculum/debug tooling consistent with the ROM’s
own tile encodings.

### Decision: Pose mismatch diagnostics are logged as JSONL events

**Decision:** When `DrMarioPlacementEnv` detects a planner/executor pose mismatch
(`placements/pose_ok == False`), it increments a persistent mismatch counter
(`placements/pose_mismatch_count`) and appends a single JSONL record containing
the decision snapshot, board/bottle bytes, feasibility masks/costs, the chosen
macro action + controller script, and the observed `lock_pose`/`lock_reason`.

**Defaults & controls:**
- Default log path: `data/pose_mismatches.jsonl.gz` (git-ignored)
- Disable/override via `DRMARIO_POSE_MISMATCH_LOG` (`0/off/false/none` disables)
- Optional per-frame trace capture via `DRMARIO_POSE_MISMATCH_TRACE=1`

**Why:** Pose mismatches are rare and hard to reproduce; logging a rich,
single-event snapshot makes them diagnosable post-hoc without slowing down
normal training/inference.

### Decision: Rotation planning models btnsPressed (edge), not btnsHeld

**Decision:** The reachability planner must treat A/B rotation as edge-triggered
(`currentP_btnsPressed`), not “rotate every frame while held”. Concretely:
holding A across consecutive frames rotates at most once, and a second rotation
requires at least one intervening frame where A is released.

**Implementation:** Extend the falling-pill per-frame state with a `rot_hold`
field (which rotate button was held on the previous frame). Apply rotation only
when `rot_hold` changes from NONE→(A/B). This is implemented in:
- Python oracle stepper + packed BFS: `envs/retro/fast_reach.py`
- Native BFS helper: `reach_native/drm_reach_full.c`
- Snapshot decoding: `envs/retro/placement_planner.PillSnapshot.rot_hold`

**Why:** Emulator mismatch logs showed almost all pose mismatches occurred when
the planner emitted consecutive `ROTATE_A` frames; ROM disassembly confirms
rotation checks `currentP_btnsPressed` (edge), so the second frame doesn’t
rotate.

---

## 2025-12-19 – Native Reachability Performance: Frontier x-mask BFS

### Decision: Native reachability batches x positions per counter-state key

**Decision:** The native reachability helper (`reach_native/drm_reach_full.c`)
performs a level-order BFS over the falling-pill per-frame state, but *batches*
the x dimension using 8-bit masks. Concretely, each frontier node represents:

- a counter-state “key” `(y, rot, sc, hv, hd, p, rh)`, and
- an `xmask` (which x positions are present for that key at the current depth).

The BFS aggregates `xmask` per key per depth (frontier merging), so we avoid the
degenerate case where the queue contains one entry per x even when many x
positions share the same counters.

**Why:** This yields a large constant-factor speedup (typically several ×) while
preserving exact per-frame semantics and minimal-time costs. It also reduces
memory pressure vs a “one queue node per full state” BFS.

### Decision: Early termination targets only in-bounds, geometrically reachable terminals

**Decision:** The native helper can terminate early once it has discovered
minimal-cost scripts for all *in-bounds* terminal poses that are reachable in a
timer-free (over-approximate) geometric flood fill.

**Why:** Many boards contain macro-legal lock poses that are *geometrically*
unreachable (sealed cavities). Without pruning, the exact BFS must explore out
to the per-call `max_lock_frames` just to prove those are unreachable, which is
wasted work for macro-action planning.

**Caveat:** This early-stop set intentionally excludes offscreen/top-out
terminal poses (which aren’t valid macro actions). The macro env ignores these
poses anyway.

### Decision: Optional native BFS stats are exposed for profiling

**Decision:** The native helper exposes lightweight per-call stats via
`drm_reach_get_last_stats()` and the Python wrapper `NativeReachabilityRunner.get_last_stats()`,
enabled by setting `DRMARIO_REACH_STATS=1`.

**Why:** Planner throughput is a critical bottleneck during training; having a
first-party “how many states/edges did we explore” counter makes regressions
and board-dependent slowdowns diagnosable without external profilers.

---

## 2025-12-19 – Reward Shaping + Policy Input: Virus Adjacency + Type-Blind Colors

### Decision: Add virus-specific adjacency shaping (separate from generic adjacency)

**Decision:** Introduce two reward terms:
- `virus_adjacency_pair_bonus`
- `virus_adjacency_triplet_bonus`

These award when a newly placed pill creates/extends a same-color run that
includes at least one virus tile (stronger weight than the generic
`adjacency_*` shaping which only considers pill-to-pill adjacency).

**Why:** The early curriculum stages can teach “make clears” without teaching
“prioritize clears that remove viruses”. Virus adjacency shaping provides a
low-latency on-ramp toward the true objective (virus elimination) without
requiring full clears to be frequent early in training.

### Decision: Prefer type-blind color planes + virus mask for placement policy training

**Decision:** For placement-policy training runs (`training/configs/smdp_ppo.yaml`),
use the `bitplane` state representation by default.

**Why:** The default `extended` state splits colors across separate planes
(virus vs locked pill vs falling pill). While a CNN can learn to merge them, a
type-blind “color is color” representation is a better inductive bias for
match-based dynamics (Dr. Mario clears depend on color, not tile type). The
bitplane representation provides (R/Y/B) color planes plus a separate `virus_mask`.

**Caveat:** Bitplane also encodes the HUD preview/falling pill into the color
planes; this is legal information (the next pill is visible to humans), but it
should be monitored for pathological overlap. We improved preview decoding so
the rotation can be recovered from the preview mask.

---

## 2025-12-19 – Reduced Bitplane Observations (and Optional Feasible-Mask Planes)

### Decision: Add minimal reduced bitplane variants for spawn-latched placement

- **Decision:** Add two additional state representations:
  - `bitplane_reduced` (6ch): `color_{r,y,b}`, `virus_mask`, `pill_to_place`, `preview_pill`.
  - `bitplane_reduced_mask` (10ch): reduced + `feasible_o0..feasible_o3`.
- **Rationale:**
  - For spawn-latched macro placement, most of `bitplane` is unnecessary: the agent primarily needs the color field, which tiles are viruses, and the current/next pill.
  - Smaller C improves throughput and reduces the chance the policy overfits to unused channels.
- **Trade-offs:**
  - Omitting scalar broadcasts (level/gravity/lock) assumes the policy can generalize without explicit speed metadata; this is acceptable for the placement SMDP but can be revisited if needed.

### Decision: Inject feasibility planes at decision time (still mask logits)

- **Decision:** In `bitplane_reduced_mask`, reserve feasibility-mask channels that are emitted as zeros by the RAM mapper and filled by the placement wrapper at true decision points from `placements/feasible_mask`.
- **Rationale:** Keeps the policy input self-contained (no runner-side concatenation) while preserving correctness: invalid actions are still masked in the action distribution.
- **Trade-offs:** The policy can learn correlations specific to the planner/mask generator; keep masking as the hard constraint and validate periodically against pose/plan verifiers.

---

## 2025-12-19 – Placement Policy Inputs: Bottle-Only Planes + Pill Vectors

### Decision: Keep the spatial tensor “bottle-only” for placement policies

- **Decision:** Introduce `bitplane_bottle` (4ch) / `bitplane_bottle_mask` (8ch) representations:
  - `color_{r,y,b}` + `virus_mask` are derived solely from the bottle buffer.
  - No falling-pill or preview-pill projection into the 16×8 tensor.
- **Rationale:** The macro-placement decision is “choose where this pill will lock”, so the falling-pill’s *current* intermediate pose is not decision-relevant, and projecting the preview pill into the grid can introduce aliasing near the top rows.

### Decision: Pass pill colors as explicit vectors (current + preview)

- **Decision:** Condition `PlacementPolicyNet` on two unordered color-pair vectors:
  - `next_pill_colors` (current pill to place)
  - `preview_pill_colors` (HUD preview pill)
- **Rationale:** This matches human-visible information and avoids spending spatial channels to represent two cells whose identity is already captured by the color vectors.

### Convention: Separate “raw NES color values” vs “canonical color indices”

- **Raw NES encoding (from RAM / board bytes):** yellow=0, red=1, blue=2.
- **Canonical policy/plane indices:** red=0, yellow=1, blue=2 (matches plane naming order).
- **Rule:** State planes and policy vectors use canonical indices; UI/board-byte tools may use raw NES values and must convert explicitly when mixing.

---

## 2025-12-19 – Emulator/Engine Parity: RNG + Reset Sync Semantics

### Decision: Define `rng_seed_bytes` at the `initData_level` boundary (mode==0x03)

**Decision:** Treat `rng_seed_bytes` as “the RNG state at `initData_level` entry” (Dr. Mario rev0, `$0046 == 0x03`), not as “RNG state at some menu-time frame”.

**Implementation:**
- **Libretro backend:** during auto-start, apply RNG seed bytes exactly when mode transitions to `0x03` (observed at a frame boundary), i.e., *after* the START press begins the game but *before* `initData_level` runs on the next frame.
- **C++ engine backend:** pre-seed `rng_state` before calling `GameLogic::reset()`, and do **no** menu-time RNG warmup in the engine.

**Why:** Menu-time RNG consumption is not stable across seeds/levels. Seeding at the level-init boundary makes virus layout + pill reserve generation deterministic and allows a single seed to reproduce both libretro and the engine exactly.

### Decision: Parity sync point uses `waitFrames` and `frameCounter` low byte

**Decision:** For “ghost parity” resets, align the engine to the emulator’s in-level checkpoint by mirroring:
- `waitFrames` (`$0051`) via `intro_wait_frames` (level-intro delay)
- `frameCounter` low byte (`$0043`) via `intro_frame_counter_lo` (soft-drop timing parity)

**Why:** Soft drop and some animation timing read the NMI `frameCounter` LSB; syncing it prevents cumulative one-frame drift in long runs.

---

## 2025-12-19 – State-Mode Performance: Lazy RGB Frames + Env Timing Keys

### Decision: Treat RGB frames as on-demand in state-mode

- **Decision:** In `DrMarioRetroEnv`, do not fetch/copy RGB frames on every `step()` when `obs_mode="state"`. Mark frames dirty on step and refresh lazily in `render()`; keep eager frame updates only for `obs_mode="pixel"`.
- **Why:** Many backends incur non-trivial per-frame costs to materialize RGB output. State-mode training and debug UIs usually only need RAM-derived board state; forcing per-step RGB work can dominate throughput and mask the benefits of faster emulation backends.
- **Trade-offs:** `render()` becomes the synchronization point for frame freshness. This is acceptable because rendering is UI-driven (low Hz) while stepping can run uncapped.

### Decision: Emit `perf/env_*_sec` timings from the base env and aggregate in the macro env

- **Decision:** The per-frame env (`DrMarioRetroEnv`) emits a small set of timing keys (`perf/env_*_sec`) per step. The spawn-latched macro env (`DrMarioPlacementEnv`) aggregates these across its internal per-frame stepping and surfaces totals on the macro-step `info` dict.
- **Why:** The interactive wrapper (`RateLimitedVecEnv`) only sees one `info` dict per macro decision, but a macro decision can advance many frames. Aggregating inside the macro env keeps UI perf accounting accurate without runner-side hooks.
- **Trade-offs:** These timings are best-effort wall-time samples. They are intended for bottleneck identification and regression detection, not for strict accounting.

---

## 2025-12-19 – Multi-Env C++ Backend: Vectorization + Debug UI Semantics

### Decision: `--ui debug` uses `SyncVectorEnv`, headless uses `AsyncVectorEnv`

- **Decision:** Prefer Gymnasium `SyncVectorEnv` for `--ui debug` (single process, easier inspection), and `AsyncVectorEnv` for headless throughput (parallel stepping/planning across CPU cores).
- **Why:** Debugging wants low-latency inspection and simple teardown; throughput wants parallelism even if per-step IPC is higher.
- **Trade-offs:** `AsyncVectorEnv` introduces cross-process payload costs and more complicated failure modes; these are mitigated via explicit benchmarking and minimal debug fields in `info` for training.

### Decision: Env-count changes are “restart-only”

- **Decision:** Debug TUI hotkeys that change `num_envs` trigger a controlled restart (stop training thread, rebuild vec env, restart) rather than attempting dynamic resizing.
- **Why:** PPO/SMDP buffers and optimizers assume a fixed batch shape; Gymnasium VectorEnv sizes aren’t designed to mutate; restart semantics are robust and easy to reason about.
- **Trade-offs:** Restarts discard in-flight rollouts and briefly pause training, but this is acceptable in debug mode and avoids subtle correctness bugs.

### Decision: Scaling metrics track both “frames/sec” and “decisions/sec”

- **Decision:** Treat `sum(placements/tau_i)` across envs as the canonical “frames processed” counter for throughput, and log both:
  - `perf/sps_frames_total` (frames/sec), and
  - `perf/dps_decisions_total` (macro decisions/sec).
- **Why:** Macro actions span variable τ frames; using only “steps/sec” can mislead scaling comparisons.

### Decision: `raw_ram` in `info` is optional (disabled for headless throughput)

- **Decision:** Add an `emit_raw_ram` env option to control whether `info["raw_ram"]` is included. Default to **on** for debug/parity tools and **off** for headless training (especially AsyncVectorEnv).
- **Why:** `raw_ram` adds cross-process payload and can dominate IPC overhead at high env counts; most training loops only need masks/metrics.
- **Trade-offs:** Tools that render boards or decode preview pills from RAM must explicitly enable the field.

---

## 2025-12-20 – Curriculum Graduation Logging

### Decision: Emit curriculum advancement events via env `info` and log frame/episode totals in SMDP-PPO

- **Decision:** When the scripted curriculum advances, the vec env injects `curriculum/advanced_*` keys into `info`, and SMDP-PPO logs frames/episodes totals + deltas at that moment.
- **Why:** This provides a lightweight, JSONL/W&B-friendly way to report "how many frames/episodes to graduate each stage" without changing the training loop structure.
- **Trade-offs:** Logging is event-driven (only on advancement), so offline summaries must aggregate from sparse events; other algos will need similar hooks if they require curriculum reporting.

---

## 2025-12-20 – C++ Batched Run Requests for Training Throughput

### Decision: Add a batched run protocol to the engine shared memory

- **Decision:** Extend `DrMarioState` with a simple request/ack protocol (`run_request_id`/`run_ack_id`) plus parameters (`run_mode`, `run_frames`, `run_buttons`, `run_last_spawn_id`) and outputs (`run_frames_executed`, cleared-tile counters). The engine processes pending run requests with priority in its main loop.
- **Why:** Per-frame Python handshakes (polling + state rebuilds) were the dominant throughput bottleneck and made multi-env scaling inefficient. Batched runs let the C++ engine advance many frames per call and return just the state/metrics we need at decision boundaries.
- **Trade-offs:** The SHM layout is now an implicit “protocol version”. Python and the engine binary must match; we keep a C++ `static_assert(sizeof(DrMarioState))` and Python `ctypes.sizeof`/parity tests to catch mismatches early.

### Decision: Use batched runs in the placement SMDP env when safe

- **Decision:** When running `backend=cpp-engine`, `DrMarioPlacementEnv` executes the planner controller script and the post-lock wait-to-next-spawn via batched runs, then aggregates the macro reward from engine clear counters + adjacency-at-lock shaping. Disable via `DRMARIO_CPP_FAST=0`.
- **Guardrails:** Fast mode auto-disables when non-linear per-frame reward knobs are enabled (pill placement bonus, column-height penalty, action penalties) or potential shaping is on, and falls back to the per-frame stepping path.
- **Trade-offs:** The fast path does not reproduce every per-frame info signal (e.g., match-event edge bookkeeping), so parity tooling / alternate curricula should keep using the slow path unless we explicitly extend fast-mode support.

---

## 2025-12-20 – Async Spawn Robustness + Reset-Time Fast-Forward

### Decision: Put cpp-engine SHM ctypes in an installed package (`envs.backends`)

- **Decision:** Keep the authoritative `DrMarioState` `ctypes.Structure` + file-backed mmap helper under `envs/backends/cpp_engine_shm.py`, and have `CppEngineBackend` import it.
- **Why:** `AsyncVectorEnv` uses multiprocessing spawn on macOS; workers may not have the repo root on `sys.path`, and `game_engine/` is not part of the installed Python package set. Importing SHM bindings from `envs.*` avoids “No module named game_engine” failures in workers.
- **Trade-offs:** There are now two potential SHM binding locations (`envs.backends.*` and `game_engine/engine_shm.py`); treat `envs/backends/cpp_engine_shm.py` as canonical for the Python driver and keep the layout in sync with `game_engine/GameState.h`.

### Decision: Use batched `run_until_decision` during `DrMarioPlacementEnv.reset()`

- **Decision:** When the backend is `cpp-engine` and supports `run_until_decision`, `DrMarioPlacementEnv.reset()` fast-forwards to the first feasible decision point via batched runs + `DrMarioRetroEnv.sync_after_backend_run()`, rather than stepping per frame in Python.
- **Why:** Reset-time per-frame stepping can time out or dominate CPU under high env counts, especially with `AsyncVectorEnv` autoresets. Fast-forwarding makes resets cheap and improves multi-env stability.
- **Trade-offs:** This path skips per-frame reward accumulation (reset has no reward anyway) and relies on the shared-memory protocol; if the backend is missing or incompatible, we fall back to the slow reset path.

### Decision: Force-terminate `AsyncVectorEnv` on close if a call is pending

- **Decision:** When closing a long-running `AsyncVectorEnv`, prefer a force-terminate shutdown path if the env indicates a pending async call (e.g., stuck in `WAITING_STEP`) rather than blocking on `step_wait`.
- **Why:** In rare cases (worker crash / partial IPC message), Gymnasium can hang in `pipe.recv()` during shutdown. For training, a robust teardown is more important than a graceful close.
- **Trade-offs:** Force-terminate can discard in-flight episode results for the final step and may leave less diagnostic context; treat it as a shutdown-only safeguard, not a normal control flow tool.

---

## 2025-12-20 – Hop-Back Curriculum (ln)

### Decision: Add `ln_hop_back` curriculum mode for placement PPO

**Context:** For `ppo_smdp` placement training we want a curriculum that (a) introduces harder tasks gradually, but (b) repeatedly hops back to earlier tasks with increasingly strict success thresholds to prevent forgetting.

**Decision:** Add `curriculum.mode = ln_hop_back`, which generates a stage schedule that alternates:
- a low “probe” success requirement on a newly introduced level (default 16%), then
- hop-backs on earlier levels with thresholds `1 - exp(-m·k)` where `k` increases the further back we hop and `m = pass_ramp_exponent_multiplier` (default `1/3`).

**Why:** This yields a simple, deterministic schedule with an intuitive “tighten earlier levels over time” shape without adding per-episode level-mixing complexity.

**Trade-offs:** The `1 - exp(-m·k)` thresholds can still become strict for large k, but `m<1` slows the ramp. If this stalls progress, cap k (`ln_hop_back_max_k`), reduce `confidence_sigmas`, or further reduce `pass_ramp_exponent_multiplier`.

### Decision: Extend synthetic match-count levels to `-15..-4`

**Context:** The original synthetic match stages only spanned `-10..-4` (1..7 matches), which was too short to ramp before switching into virus-count stages.

**Decision:** Extend to `-15..-4` and map `level -> match_target` as `max(1, 16 + level)` so the match curriculum reaches ~12 matches before the 1/2/3-virus stages.

---

## 2025-12-21 – `ln_hop_back` Fast-Skip + Shallow Hop-Back + Bailout

### Decision: Skip full hop-backs when the probe is already above the pass target

- **Decision:** When a probe stage (new frontier) passes and the same stage’s stats already meet the k=1 “pass target” (`1 - exp(-m)`), skip the rest of that frontier’s hop-back chain and jump directly to the next frontier probe.
- **Why:** Avoid immediately re-running the early ladder when the current policy is already comfortably above the minimum pass bar on the new level (common when a single PPO update boosts multiple stages).
- **Trade-offs:** Skipping reduces rehearsal coverage for intermediate levels in that frontier’s hop-back block; if this causes regressions/forgetting, reduce skipping by tightening targets (`pass_ramp_exponent_multiplier`), raising `probe_threshold`, or disabling via a code tweak.

### Decision: Start hop-backs at the 3rd-highest mastered hop-back level

- **Decision:** When entering a hop-back chain, jump to the hop-back stage for the 3rd-highest **mastered** level (where “mastered” means having passed a hop-back stage, not just a probe), rather than always returning to `start_level`.
- **Why:** Cuts redundant time spent replaying very early stages once they are already stable, while still keeping some backward rehearsal.
- **Trade-offs:** The “3rd-highest” heuristic is blunt; if it’s too shallow/deep, tune `ln_hop_back_hop_back_mastered_rank`.

### Decision: Bail out of stuck probe stages using a fraction of the run budget

- **Decision:** If a probe stage hasn’t passed after `ln_hop_back_bailout_fraction` of the run’s total step/frame budget, jump into the hop-back chain at the 3rd-highest mastered level.
- **Why:** Prevents spending a large chunk of the run stalled on a new frontier when a remediation loop (practice easier mastered tasks, then work forward again) is more productive.
- **Implementation note:** The env factory passes `cfg.train.total_steps` into the curriculum as `run_total_steps` so the curriculum can compute the bailout threshold.

## 2025-12-20 – Time-Based Task Budgets (Scaffolding)

### Decision: Enforce frame/spawn budgets in the placement wrapper

- **Decision:** Implement optional per-episode task budgets (`task_max_frames`, `task_max_spawns`) in `DrMarioPlacementEnv`, tracking `task/frames_used` via SMDP τ and `task/spawns_used` via spawn consumption (including skipped spawns with no feasible actions).
- **Why:** Keeps budget accounting backend-agnostic (libretro vs `cpp-engine`) and avoids relying on ROM-specific pill-counter encodings (BCD vs binary).
- **Semantics:** Budgets are *soft constraints*:
  - If the underlying env clears, the wrapper replaces the terminal clear bonus with a smooth time-goal reward that is **positive under-budget** and **negative over-budget**, bounded by the clear bonus (upper) and the topout penalty (lower).
  - If the clear is over-budget, the wrapper still sets `goal_achieved=False` so curricula treat it as a failure under the current time goal, but the episode terminates normally (no budget truncation).
  - If the budget is exceeded mid-episode, the wrapper does not truncate; `task/success` remains false unless the agent finishes under-budget.
- **Trade-offs:** Budget counting is wrapper-relative (decision-boundary τ) and currently applies only to the placement SMDP env; extending to per-frame controller envs would require separate counters and reset boundaries.

---

## 2025-12-20 – Confidence-Based Curriculum Windows + Mastery Time Budgets

### Decision: Advance curriculum via a one-sided Wilson lower bound (σ confidence)

- **Decision:** Treat each episode as an i.i.d. Bernoulli trial (success/fail) with latent success probability `p`. For each curriculum stage target `t`, advance when the one-sided Wilson lower bound for `p` over a rolling window exceeds `t`: `LB_wilson(k,n; z) > t`, where `z` is configured in “sigmas”.
- **Window sizing:** Use a “near-target” assumption `p_assumed = (1+t)/2` and choose `n` via the normal-approximation sample size:
  - `n ~= z^2 * p(1-p) / (p - t)^2`, which simplifies to `n ~= z^2 * (1+t)/(1-t)` for `p=(1+t)/2`.
- **Why:** Replaces arbitrary `window_episodes/min_episodes` with a principled, target-aware notion of confidence that scales naturally across probe thresholds (~16%) and harder hop-back thresholds.
- **Trade-offs:** For targets near 1, `n` can become large and stall stages if thresholds are too strict. Mitigate by capping hop-back k (`ln_hop_back_max_k`) or adjusting targets/sigmas; monitor `curriculum/confidence_lower_bound` and `curriculum/window_*` in the TUI.

### Decision: Start time budgets only after “mastery” (perfect streak at 2σ)

- **Decision:** Once a stage achieves a *perfect* success streak long enough to certify `p > mastery_target` at `mastery_sigmas` (defaults: 0.99 at 2σ), begin applying time budgets (`task_max_frames`) for that level.
- **Budget schedule:** Initialize the budget to the mean clear time over a recent window of successful clears. Tighten gradually over time, limiting each decrease to a fraction of the observed MAD (median absolute deviation) to avoid overreacting to noise/outliers.
- **Why:** Separates “learn to succeed” from “learn to be fast” and provides a stable, outlier-robust way to reduce allowed time once success is essentially guaranteed.
- **Trade-offs:** A high `mastery_target` at high σ can require a long no-failure streak before budgets activate; tune `time_budget_mastery_target/time_budget_mastery_sigmas` if budgets start too late.

---

## 2025-12-20 – Stage-Local Curriculum Stats + Persistent Best Times

### Decision: Track `ln_hop_back` stage outcomes per stage index (fresh stats on revisits)

- **Decision:** Keep hop-back advancement stats in stage-local rolling windows keyed by `stage_idx` (not just by `level`), while retaining per-level base histories for time-budget mastery checks.
- **Why:** The same level can reappear with a *different* threshold (probe vs hop-back @ `1-exp(-k)`); stage-local windows prevent earlier easier passes from contaminating later stricter ones.
- **Trade-offs:** Requires propagating a stable stage identifier from the vec wrapper; missing/incorrect tokens can reintroduce contamination (see SCRUTINY R45).

### Decision: Pass a per-env `stage_token` through the vec wrapper to avoid multi-env races

- **Decision:** `CurriculumVecEnv` captures a `stage_token` at the moment it assigns an env’s level (stored as `_env_stage_tokens`) and passes it back into `curriculum.note_episode(...)` when that env terminates.
- **Why:** In vectorized runs, multiple envs can terminate in the same vec-step; the curriculum may advance partway through processing those terminals. The `stage_token` ensures each episode is attributed to the correct stage for stats and prevents “late” episodes from advancing a newer stage.
- **Trade-offs:** Adds small bookkeeping to the wrapper and depends on consistent semantics across `sync`/`async` vectorization.

### Decision: Record best-known clear times across runs in a small sqlite DB

- **Decision:** Add a lightweight sqlite DB (`BestTimesDB`) keyed by `(level, rng_seed)` that records the best (minimum) clear times in frames and spawns across all runs.
- **Why:** Provides “best-known floors” per level to ground time-budget curricula (avoid tightening below anything ever observed) and enables reporting the distribution of best times across seeds for profiling/target selection.
- **Trade-offs:** Sqlite can contend if multiple runs write to the same DB; treat writes as best-effort (ignore failures), use WAL mode, and allow overriding the path via `DRMARIO_BEST_TIMES_DB`.

---

## 2025-12-20 – Curriculum Gate Stability + Stage-Pure PPO Updates

### Decision: Use an EMA-based confidence gate (not tiny rolling windows)

- **Context:** The earlier Wilson-LB gate used a short rolling window sized from the target (e.g. 7–10 episodes for mid-range thresholds). In practice this made `curriculum/confidence_lower_bound` extremely noisy and could stall (or occasionally jump) on hop-back stages, even after many episodes.
- **Decision:** Replace the rolling-window estimate with an **exponentially weighted moving average (EMA)** of stage success, and compute a Wilson-style lower bound from **pseudo-counts** `(S, N)` where both are decayed each episode. Require a minimum effective sample size (`confidence_min_effective_episodes`) before allowing advancement.
- **Why:** EMA keeps the estimate responsive to learning (non-stationary p) while preventing “window smaller than a batch” noise.
- **Trade-offs:** EMA introduces additional hyperparameters (half-life + min effective N) and changes the interpretation of `window_n/window_size` to “effective sample size” rather than literal episode count.

### Decision: Enforce a minimum number of macro-decisions per stage (`min_stage_decisions`)

- **Decision:** For `ln_hop_back`, require a stage to execute at least `min_stage_decisions` macro-actions before it can advance.
- **Why:** Prevents “micro-stages” that advance within a single PPO rollout batch and improves statistical stability under variable pill sequences/layouts.

### Decision: Keep PPO updates stage-pure by stopping rollouts on advancement

- **Decision:** When the curriculum advances, stop rollout collection immediately and run the PPO update, so a single PPO batch never mixes transitions across curriculum stages.
- **Why:** Avoids value/advantage contamination from changing task distribution mid-batch and makes curriculum metrics easier to interpret.

---

## 2025-12-21 – AsyncVectorEnv Robustness + Engine Process Cleanup

### Decision: Omit unset optional info keys (never return `None`)

- **Context:** Gymnasium `AsyncVectorEnv` merges per-env `info` dicts into dict-of-arrays. If a key is numeric in one env and `None` in another, the merge can crash (typed array cannot accept `None`).
- **Decision:** For optional values like `task/max_frames`, `task/max_spawns`, and `match_target`, omit the key entirely when unset instead of returning `None`.
- **Why:** Keeps vectorized info payloads type-stable across envs and across curriculum stage transitions.

### Decision: Track engine pids per run and reap on shutdown (best-effort)

- **Context:** When a training run exits due to a crash or forced worker termination, `drmario_engine` child processes can be orphaned.
- **Decision:** Write per-engine pidfiles under a run-local directory (`$logdir/engine_pids`) and, on shutdown, kill any remaining `drmario_engine` pids recorded there.
- **Why:** Reduces “zombie engine” accumulation in long iteration loops without requiring OS-specific parent-death signals.

---

## 2025-12-21 – In-Process Batched Pool Backend (`cpp-pool`)

- **Decision:** Add an in-process native “engine pool” shared library (`game_engine/libdrmario_pool`) that owns N Dr. Mario engine instances and the integrated reachability planner, and expose a batched C ABI consumed via `ctypes`.
- **Contract:** Python ↔ C++ speaks **SMDP-at-decision-boundaries**: one `step(actions[N])` executes the chosen macro placement and runs until the next controllable spawn (or terminal), returning τ frames plus compact events/stats.
- **ABI stability:** Public structs include `protocol_version` and/or `struct_size` so Python can hard-fail on mismatches rather than silently corrupt buffers.
- **Compatibility:** The pool `step()` supports a per-env `reset_mask` to mimic Gymnasium VectorEnv `autoreset_mode="NEXT_STEP"` (reset on the next `step`, ignore the action), keeping `CurriculumVecEnv` behavior unchanged.
- **Trade-offs:** This is the fastest path (zero IPC, one batched call), but crashes can segfault Python; debug board bytes are optional and off the hot path.

---

## 2025-12-21 – Engine/Digital-Twin ABI: Keep Boards Internal, Co-Locate Planner + Timing Model

- **Decision:** Treat the **decision-boundary engine ABI** as the canonical interface for training, evaluation, and future “play against humans” integrations. The ABI returns feasibility/costs/τ plus compact events/stats; raw boards remain internal (debug-only).
- **Why:** Performance and correctness: the engine/twin has direct access to the canonical state and timing, so it can compute derived signals (clear events, virus clears, heights, progress bars) without Python-side board scans or extra copies.
- **Shadow mode:** A vision/hardware backend should “shadow” the same World Kernel using a digital twin that locks to external clocks/latency and validates observations against rules, but the coordinator still sees the same ABI contract.
- **Planner placement:** Keep reachability/planning inside the kernel (or tightly coupled) because it shares representation + timing semantics with the simulator/twin.
- **Reference:** Full system/components/data-flow/control-flow spec in `notes/DESIGN_ENGINE_TWIN_PROTOCOL.md`.

---

## 2025-12-21 – SMDP-PPO Optional Aux Vector Inputs (`aux_spec=v1`)

- **Decision:** Add an optional auxiliary vector input to the placement policy network and SMDP-PPO rollout buffer, controlled by `smdp_ppo.aux_spec` (`none|v1`).
- **v1 layout:** `float32[57]` packed as:
  - `speed_onehot[3]` (0=low, 1=medium, 2=high)
  - `virus_total/84` (scalar, clipped to [0,1])
  - `virus_by_color/84[3]` (R,Y,B, clipped)
  - `level_onehot[36]` for curriculum/env levels in `[-15..20]` (out-of-range => all zeros)
  - `frame_count_norm` (scalar): `task/frames_used / task/max_frames` if available else `tanh(frames_used/8000)`
  - `max_height/16` (scalar): max occupied column height
  - `col_heights/16[8]` (column heights)
  - `clearance_progress` (scalar): fraction of the active clear target achieved (matches or viruses)
  - Extras (cheap + useful): `feasible_fraction`, `occupancy_fraction`, `virus_max_height/16`
- **Why:** These features are nearly free to compute at decision points, but hard/slow for a CNN over sparse bitplanes to infer reliably (especially progress/targets).
- **Trade-offs:** Aux vectors couple learning to `info` key contracts; keeping the spec versioned (`aux_spec=v1`) makes future changes explicit and reproducible.

---

## 2025-12-21 – Speed Setting as a First-Class Env Parameter

- **Decision:** Treat Dr. Mario’s speed setting as an explicit env config (`speed_setting` in {0,1,2}) and surface it in decision-time infos as `pill/speed_setting` (and `speed_setting`).
- **Backends:** `cpp-engine`/`cpp-pool` take the speed as a reset-time configuration; libretro/menu-based backends set speed during auto-start (with a best-effort RAM-patch fallback for robustness).
- **Validation:** Use `tools/ghost_parity.py` with `--speed-setting` to validate emulator↔engine parity when changing default speeds.

---

## 2025-12-21 – cpp-engine Batched-Run Timeout Policy (AsyncVectorEnv)

- **Context:** In long `AsyncVectorEnv` training runs, a single `CppEngineBackend` instance can occasionally fail to ack a batched run request within the Python timeout, which previously raised `TimeoutError` inside a worker process and crashed the entire job.
- **Decision:** Treat cpp-engine batched-run exceptions (timeouts, unexpected backend errors) inside placement fast-path as an **episode truncation** and force a backend restart (close + recreate on next reset). Do not allow these exceptions to propagate out of `env.step()`.
- **Backend watchdog:** `CppEngineBackend._run_request` uses a **no-progress watchdog** (frame counter not advancing) plus a more forgiving total timeout, and includes request/ack/frame diagnostics in the raised error.
- **Why:** Training runs must be robust to rare backend stalls; truncation + restart preserves the run and keeps the failure visible via `info` keys and `_backend_last_error`.
- **Trade-offs:** Truncated episodes reduce sample efficiency and can mask an underlying engine bug; mitigation is to log/monitor `placements/backend_error_phase` and restart counts, and investigate if the rate becomes non-trivial.

---

## 2025-12-21 – cpp-pool Planner Parallelism

- **Decision:** Parallelize cpp-pool reset/step loops across envs and make the native reachability BFS workspace thread-local to allow concurrent planner calls.
- **Why:** Planner BFS dominates cpp-pool time at decision points; inference is already batched once per spawn, so throughput scales with per-env parallelism.
- **Trade-offs:** Thread-local BFS workspaces increase memory per worker. Provide `DRMARIO_POOL_WORKERS` to cap concurrency and default to hardware threads clamped by env count.

---

## 2025-12-21 – Candidate-Scoring Placement Policy (Packed Feasible Actions)

- **Decision:** Add an alternative placement policy that scores a **packed list of planner-feasible macro actions** instead of producing a dense 4×16×8 (512-way) logit map.
- **Interface:** At each decision point, pack the env’s `placements/feasible_mask` + frames-to-lock (`placements/cost_to_lock` for `cpp-pool`, `placements/costs` for emulator wrapper) into fixed-size arrays:
  - `cand_actions[Kmax]` (flat macro action indices)
  - `cand_mask[Kmax]` (valid slots)
  - `cand_cost[Kmax]` (explicit cost-to-lock feature)
- **Packing order:** Sort candidates by `(cost_to_lock, macro_action_id)` so equal-cost ties are deterministic across runs.
- **Model:** `CandidatePlacementPolicyNet` = global board+pill embedding `g` plus per-candidate embedding `e_i` (pose + local patches + cost), with logits via dot-product `g·e_i`. Includes two board encoders:
  - `cnn` (CoordConv-style conv stem + residual blocks → feature map; global pool for `g` and gather per-cell trunk features per candidate)
  - `col_transformer` (8 column tokens + tiny Transformer; global pool for `g` and gather per-column trunk features per candidate)
- **Local context:** Candidate features include (a) gathered trunk features at the two landing cells/columns and (b) raw local patches from bottle bitplanes (colors + virus) around both landing cells (kernel `k×k`), avoiding feasibility-mask plane leakage.
- **Perf notes:** Patch offsets are precomputed as buffers (avoid per-forward allocations), and PPO updates prepack candidate arrays once per update batch (avoid per-minibatch repacking).
- **Why:** Avoids allocating parameters to a 512-output head that is mostly invalid each step, and makes the reachability cost a first-class feature so the policy can explicitly trade off board outcome vs execution time (speedrunning).
- **Trade-offs:** Requires storing per-step `cost_to_lock` in the rollout buffer for PPO updates; candidate truncation (if `Kmax` < feasible count) can bias the action set, so keep `candidate/*` truncation telemetry on and bump `Kmax` if it shows up. PPO update asserts that the chosen macro action is present in the repacked candidate list (crash-loudly on mismatch).

---

## 2025-12-21 – Virus-Clear Reward Normalization

- **Decision:** Treat `virus_clear_bonus` as the total per-episode reward for clearing all starting viruses; each virus clear earns `virus_clear_bonus / viruses_initial` (0 if `viruses_initial` is 0).
- **Rationale:** Keep reward scale consistent across levels with 1-84 viruses so training dynamics are less sensitive to level-specific virus counts.
- **Trade-offs:** Per-virus reward is larger on low-virus levels; zero-virus curriculum stages receive no virus-clear reward signal.

---

## 2025-12-22 – Pill Conditioning: Ordered vs Unordered Color Embeddings

- **Decision:** Make pill conditioning embedding **configurable** via `smdp_ppo.pill_embed_type` with options:
  - `unordered` (default): swap-invariant (DeepSets-style) for pill colors.
  - `ordered_onehot` / `ordered_pair`: preserves half identity by embedding the 9 ordered color pairs (for 3 colors) for each pill (current + preview).
- **Why:** The macro action space is directed/anchored to a specific half/cell; swap-invariant pill embeddings can erase the distinction between `(a,b)` and `(b,a)` on mixed-color pills, causing ~50% plateaus on tasks where choosing the correct half-to-anchor is pivotal (e.g., some 1-virus clears).
- **Trade-offs:** Ordered conditioning increases the effective input entropy (distinguishes 6 mixed pairs instead of 3 unordered pairs), which can slightly increase sample needs but avoids representational bottlenecks.
- **Notes:** Rotation is intentionally not included in the conditioning yet; if spawn rotation varies by backend/ROM, add a small `rot_onehot[4]` later behind a versioned aux/conditioning spec.

---

## 2025-12-22 – Compressed Run Artifacts (gzip)

- **Decision:** Default run artifacts use gzip-compressed, streamable files: `metrics.jsonl.gz`, `env.txt.gz`, and checkpoints as `*.pt.gz`.
- **Checkpoint format note:** Gzipped checkpoints use PyTorch’s legacy (non-zip) serialization so they remain stream-friendly and avoid zipfile seek requirements.
- **Why:** Significantly reduces disk usage while keeping reads stream-friendly and dependency-free (gzip is in the stdlib).
- **Trade-offs:** Tooling must accept both compressed/uncompressed extensions, and checkpoint/log scans pay a small CPU decompression cost.

---

## 2025-12-23 – Training Stack: No External Trainer Integration

- **Decision:** Remove all external trainer integration and keep the canonical RL/training stack fully in-repo (`python -m training.run` with `ppo_smdp` / `simple_pg`).
- **Why:** Reduce dependency surface area and eliminate “two training stacks” drift; keep resume/checkpoint/logging semantics under our control and aligned with our env contracts.
- **Trade-offs:** We give up a mature distributed actor/learner infrastructure; if we need multi-machine scale later, add it behind a clean adapter boundary (not as a second, semi-maintained training path).

---

## 2025-12-23 – Repo Hygiene: Canonical Docs + Notes Archive

- **Decision:** Keep the repo root minimal (primarily `README.md` + `AGENTS.md`), and treat `docs/` + `notes/` as the canonical places for documentation and decisions. Move historical one-off writeups into `notes/archive/` and avoid keeping tracked scratch artifacts (plots, commandlines, vendored reference code) at the root.
- **Why:** Reduce “stale doc” drift and cognitive load; make it obvious where to look for current instructions vs historical context.
- **Trade-offs:** Some old filenames/paths stop existing; mitigation is to keep archived originals under `notes/archive/` and update current docs/notes to point there only when needed.

---

## 2025-12-23 – Reverse Engineering: Use Disassembly Submodule

- **Decision:** Remove the in-repo `re/` reverse-engineering workspace and rely on the `dr-mario-disassembly/` submodule for annotated disassembly. Keep only the derived, runtime-relevant artifacts in-repo (e.g. `envs/specs/ram_map.py`, `envs/specs/ram_offsets.json`, and tracked parity fixtures like `data/nes_demo.json`).
- **Why:** Avoid duplicating the disassembly/annotation effort and reduce repo bloat; keep the codebase focused on env/training/eval rather than RE pipelines.
- **Trade-offs:** We lose the local RE automation scripts and intermediate outputs; if we need to redo ROM verification on a new revision, recreate minimal tooling under `tools/` (without reintroducing a separate `re/` tree).
