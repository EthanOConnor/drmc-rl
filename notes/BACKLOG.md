# BACKLOG.md — drmc-rl

Technical backlog / roadmap. More detailed items than top-level docs.

---

## Near-term (This Sprint)

### Critical Priority

- **Wire TUI into training loop** ✅
  - Connect `TrainingTUI` to `SimplePGAdapter` update callbacks.
  - Pass metrics from adapter → TUI via update() method.
  - Test with: `python -m training.run --algo simple_pg --ui tui`

### High Priority

- **C++ engine: Parity beyond demo**
  - Add golden traces for non-demo gameplay (seeded levels, edge-case rotations/DAS).
  - Validate 2P / attack logic once implemented.

- **Extract diagnostics tracker**
  - Move `NetworkDiagnosticsTracker` from `speedrun_experiment.py` to `training/diagnostics/tracker.py`.
  - Integrate with TUI for gradient/param stats display.

- **Complete evaluator training**
  - Finish `models/evaluator/train_qr.py` skeleton for QR-DQN distributional head.
  - Train on RAM-labeled corpus.

- **Populate seed registry**
  - Capture savestates for 120 seeds per level in `envs/retro/seeds/`.
  - Store first 128 pills and virus grid hash per seed.

- **Complete evaluator training**
  - Finish `models/evaluator/train_qr.py` skeleton for QR-DQN distributional head.
  - Train on RAM-labeled corpus.

- **Populate seed registry**
  - Capture savestates for 120 seeds per level in `envs/retro/seeds/`.
  - Store first 128 pills and virus grid hash per seed.

### Recently Completed ✅

- C++ engine demo parity suite (frame-perfect vs `data/nes_demo.json`)
  - Tooling: `tools/record_demo.py`, `tools/game_transcript.py`
  - Regression test: `tests/test_game_engine_demo.py::test_demo_trace_matches_nes_ground_truth`
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
