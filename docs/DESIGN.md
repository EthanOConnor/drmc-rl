# Design Overview

This document captures the concrete environment spec, training stack, evaluator design, and evaluation methodology for Dr. Mario RL, aligning with the October 2025 plan. It is the working source of truth for implementation details.

## Environment

- Observation modes
  - Pixel: 128×128 RGB from 256×240 NES frames, stack 4, float32 [0,1]
  - State-tensor: 16×8 grid with channels for viruses, fixed pill colors, falling pill halves, orientation (0/1), gravity/lock scalars (broadcast), and level/time (broadcast); stack 4
- Action space (discrete, 60 Hz)
  - 0 noop; 1 left; 2 right; 3 down; 4 rotate_A; 5 rotate_B; 6 left_hold; 7 right_hold; 8 down_hold; 9 both_rot
- Episode & termination
  - One level per episode; terminate on clear or timeout T_max (2–4× median clear time)
  - Info: viruses cleared, total frames, chains, drops
- Rewards (speedrun baseline + shaping)
  - r_t = −1 + α·ΔV + β·chain_bonus − γ·settle_penalty; terminal clear +C
- Seeding/determinism
  - Savestate at Level Select with fixed power‑on frame counter; registry: seed_id → (state_file, frame_offset)
  - Expose RNG controls and seed sweeps for eval

## Training Stack

- Stable‑Retro + Libretro (Mesen/Nestopia) for emulator fidelity and savestate control
- Sample Factory (PPO with IMPALA CNN + optional LSTM) for high‑throughput actor‑learner
- TorchRL utilities optional for distributional losses
- PettingZoo wrapper later for 2‑player vs. mode
- EnvPool C++ re‑implementation later for 10–100× FPS, with golden parity vs emulator

## Evaluator (Distributional Time‑to‑Clear)

- Head: QR (K=51–101) or IQN; inputs: state tensor (optionally after‑action)
- Targets: Monte‑Carlo playout distributions per position; censor at T_max
- Uses: bootstrapped RL with short rollouts; risk‑aware action selection (mean vs quantiles vs CVaR)

## Risk‑Aware Policy

- Add scalar `risk_tau ∈ (0,1]` to observation
- Score actions using evaluator quantiles or CVaR at chosen τ/α; switch behavior at inference without retraining

## Evaluation Harness

- For each seed (layout, pill sequence): run N=100–1000 episodes with fixed seeds
- Metrics: E[T], Var[T], P(T≤t*), CVaRα(T) for α ∈ {5%,25%}; per‑seed distribution logging
- Save parquet + plots; keep savestate replay traces for anomalies

## Data & Curriculum

- RAM‑labeled corpus via Mesen/FCEUX Lua/trace: 8×16 grid, pill halves, orientation, gravity/lock counters, timers, level
- Curriculum: single‑move solves → short stacks → full level 0
- Train evaluator first; use as bootstrap for RL
- Pixel→state translator trained on emulator frames + RAM labels; later fine‑tune on HDMI frames

## Multi‑Agent (later)

- PettingZoo ParallelEnv with `player_0`, `player_1`; same obs/action spaces
- Rewards: zero‑sum win/loss with optional shaped time signals

## Extreme Scale (later)

- EnvPool C++ simulator implementing core rules + RNG; golden parity suite against emulator traces

## Reverse Engineering

- Tools: FCEUX/Mesen trace & CDL; Ghidra (6502) or ca65; NesDev wiki
- Targets: frame counter source, virus placement constraints, pill RNG, RAM map, input read cadence
- Workflow: breakpoints on bottle writes, CDL playout, bank mapping per MMC1, unit tests reproducing layouts

## Stand‑Up (macOS first; Linux/Windows included)

- macOS Apple Silicon
  - `brew install cmake pkg-config lua@5.1`
  - `uv venv && source .venv/bin/activate`
  - `pip install torch torchvision torchaudio`
  - `pip install stable-retro sample-factory gymnasium numpy opencv-python rich`
  - Place NES libretro core (`mesen_libretro.dylib` or `nestopia_libretro.dylib`)
  - `python -m retro.import ~/ROMs/NES`
  - `python -m envs.retro.demo --obs-mode pixel`
- Linux (training)
  - Install CUDA 12.x; use PyTorch CUDA wheels; install Sample Factory
- Windows
  - Supported for dev; prefer Linux for large CUDA training runs

## Open Questions

Record decisions in `DESIGN_DISCUSSIONS.md` (see end of that file for the Q&A section). Key items:
- Final observation channel semantics for state‑tensor (exact planes and scaling)
- Action macro timing (hold durations, repeats, release rules)
- Reward shaping coefficients (α, β, γ) and terminal bonus C by level
- T_max policy and evaluation thresholds per level
- Seed cataloging (frame windows, savestate directories) and naming convention
- Policy input format for risk conditioning across pixel/state modes
- SF config defaults for actor count and rollout size on target hardware
