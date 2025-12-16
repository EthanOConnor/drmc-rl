# BACKLOG.md — drmc-rl

Technical backlog / roadmap. More detailed items than top-level docs.

---

## Near-term (This Sprint)

### Critical Priority

- **C++ engine: Implement DAS physics**
  - Reference: `fallingPill_checkXMove` in `dr-mario-disassembly/prg/drmario_prg_game_logic.asm`.
  - Current: pieces move instantly on input.
  - Target: frame-accurate Delayed Auto Shift timing.

- **C++ engine: Implement wall kicks**
  - Reference: `pillRotateValidation` ($8E70 in asm).
  - Push piece left/right if rotation is blocked by wall/pieces.

- **C++ engine: Python interface**
  - Expand `game_engine/engine_shm.py` to a full driver for input sequences and state assertions.

### High Priority

- **C++ engine: Parity testing**
  - Use demo mode data (`dr-mario-disassembly/data/drmario_data_demo_*.asm`) to verify exact board/moves vs NES.
  - Create test fixtures for CI.

- **Refactor speedrun_experiment.py**
  - Current: 5436 lines, 170 functions in one file.
  - Target: Split into modular components (trainer, viewer, metrics, device utils).

- **Complete evaluator training**
  - Finish `models/evaluator/train_qr.py` skeleton for QR-DQN distributional head.
  - Train on RAM-labeled corpus.

- **Populate seed registry**
  - Capture savestates for 120 seeds per level in `envs/retro/seeds/`.
  - Store first 128 pills and virus grid hash per seed.

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
