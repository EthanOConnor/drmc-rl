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
- **Gap identified**: No inter-session notes system → now implementing.
- **speedrun_experiment.py complexity**: 5436 lines, 170 functions. Needs refactoring into smaller modules.
- **C++ engine gaps**: Missing DAS and wall kicks prevents accurate high-speed play testing.
