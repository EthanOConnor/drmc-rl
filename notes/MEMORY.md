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
