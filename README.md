# Dr. Mario RL (drmc-rl)

Reinforcement learning environment, training stack, and evaluation harness for NES Dr. Mario. Focus: single‑agent speedrun mode with risk‑aware evaluation, later scalable to 2‑player vs. (PettingZoo) and a high‑FPS rules‑exact simulator (EnvPool).

What you can do here today:
- Run macOS/Linux/Windows stand‑up steps to smoke‑test Stable‑Retro + libretro NES core.
- Use the provided Gymnasium‑style wrappers (skeleton) with pixel or state‑tensor observations.
- Train with Sample Factory configs (skeleton) and evaluate distributions of time‑to‑clear.
- Track deterministic seeds via savestates + frame‐offset registry for reproducible studies.

Legal: You must use your own legally‑obtained ROM. ROMs are not included or distributed.

## Quick Start (macOS Apple Silicon)

- System deps: `brew install cmake pkg-config lua@5.1`
- Python: `uv venv && source .venv/bin/activate` (or `python -m venv venv`)
- Install (once you have internet access):
  - `pip install torch torchvision torchaudio` (MPS-supported wheels)
  - `pip install stable-retro sample-factory gymnasium numpy opencv-python rich`
- Obtain a NES libretro core (Mesen or Nestopia). Place the `*_libretro.dylib` path in your env.
- Import the game: `python -m retro.import ~/ROMs/NES`
- Smoke test: `python envs/retro/demo.py --mode pixel --steps 2000`

If you need me to fetch docs or install packages, I can request elevated network permissions and run the commands for you.

## Legal & ROM Hygiene

- Never commit ROMs. `*.nes` and `legal_ROMs/` are ignored.
- Enable the pre-commit guard:
  - `chmod +x .git-hooks/pre-commit-block-roms.sh`
  - `ln -sf ../../.git-hooks/pre-commit-block-roms.sh .git/hooks/pre-commit`
- See `docs/LEGAL.md` for history purge instructions if needed.

## Repository Layout

- `envs/retro/` Stable‑Retro wrappers, import helpers, seed registry, demo
- `envs/pettingzoo/` PettingZoo ParallelEnv wrapper (skeleton)
- `envs/specs/` RAM offsets, RNG notes, unit tests (placeholders)
- `models/policy/` PPO networks (IMPALA CNN/LSTM) with risk conditioning (skeleton)
- `models/evaluator/` Distributional evaluator heads (QR/IQN) + training (skeleton)
- `models/pixel2state/` Pixel→state translator (skeleton)
- `training/sf_configs/` Sample Factory configurations (skeleton)
- `training/launches/` Launch helpers (skeleton)
- `data/states/` Savestates seeds (not tracked)
- `data/datasets/` Parquet playouts, labeled frames (not tracked)
- `eval/harness/` Evaluation harness and plotting (skeleton)
- `sim-envpool/` Rules‑exact C++ env (later)
- `re/` Reverse‑engineering projects (Ghidra/ca65), CDL files
- `io-bridge/` Microcontroller scripts for controller out (later)
- `docs/` DESIGN, RNG, CONTRIBUTING and discussions

## Status

- Code: Skeletons created for env wrapper, evaluator head, policy net, eval harness.
- Docs: DESIGN, RNG plan, and CONTRIBUTING added. Open questions live in `DESIGN_DISCUSSIONS.md`.
- Next: Connect to Stable‑Retro, implement state extraction, finalize action mapping, and add tests.

## Tools

- Dump state tensors: `python tools/dump_ram_and_state.py --frames 120 --mode state`
- Visualize planes: `from tools.ram_visualizer import grid_show` (see docs/RAM_TO_STATE.md)

## References

See `docs/DESIGN.md`, `docs/RNG.md`, `DESIGN_DISCUSSIONS.md`, `docs/REWARD_SHAPING.md`, `docs/TASKS_NEXT.md`, `docs/ENV_STANDUP_MAC_LINUX.md`, `docs/RAM_TO_STATE.md`, and `docs/REFERENCES.md` for details and links.
