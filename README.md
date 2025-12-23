# Dr. Mario RL (drmc-rl)

Reinforcement learning environment, training stack, and evaluation harness for NES Dr. Mario. Focus: single‑agent speedrun mode with risk‑aware evaluation, later scalable to 2‑player vs. (PettingZoo) and a high‑FPS rules‑exact simulator (EnvPool).

What you can do here today:
- Run macOS/Linux/Windows stand‑up steps to smoke‑test Stable‑Retro + libretro NES core.
- Train the placement policy with `ppo_smdp` using the headless `cpp-engine` backend (fast) or libretro (parity/debug).
- Scale training with `AsyncVectorEnv` (`--vectorization async`) for near-linear multi-core throughput.
- Use the unified runner with a Rich TUI (`--ui tui`) or interactive debug UI (`--ui debug`).
- Track reproducible RNG seeds and best-known clear times per (level, seed) across runs.

Legal: You must use your own legally‑obtained ROM. ROMs are not included or distributed.

## Quick Start (macOS Apple Silicon)

- System deps: `brew install cmake pkg-config lua@5.1`
- Python: `python -m venv .venv && source .venv/bin/activate`
- Install package and extras:
  - `pip install -e .`  (base deps: gymnasium, numpy)
  - `pip install -e ".[retro,dev]"`  (stable-retro, opencv; pytest/ruff/black/etc.)
- Note: Stable-Retro wheels cover Python up to 3.14; if you are on 3.14+ build from source or drop to 3.13/3.12 via `brew install python@3.12` and recreate your venv.
  - For training: `pip install -e ".[rl]"` (or install PyTorch first on MPS: `pip install torch torchvision torchaudio`)
- Obtain a NES libretro core (QuickNES, Mesen, or Nestopia). Set `DRMARIO_CORE_PATH=/path/to/quicknes_libretro.dylib`.
- Point the env at your ROM: `export DRMARIO_ROM_PATH=/path/to/DrMario.nes`
- (Optional) Import the game for Stable-Retro fallback: `python -m retro.import ~/ROMs/NES`
- Smoke test: `python envs/retro/demo.py --mode pixel --steps 2000 --backend libretro`
- Auto-start tuning: `python envs/retro/demo.py --backend libretro --start-presses 3 --start-level-taps 12 --start-settle-frames 180 --start-wait-frames 600`
- Capture frames: `python envs/retro/demo.py --backend libretro --save-frames out_frames`
- Live window (requires system Tk + Pillow): `python envs/retro/demo.py --backend libretro --show-window --display-scale 2`
  - If you see `No module named '_tkinter'`, install Tk and re-create your venv.
    - macOS (Homebrew): match your Python version, e.g. `brew install python-tk@3.13` (or `@3.12`).
      Then create your venv with that interpreter (e.g., `/opt/homebrew/bin/python3.13 -m venv .venv`).
    - Linux: install your distro's Tk package (e.g., `sudo apt-get install python3-tk`).
    - Note: Tk is a system dependency; `uv`/pip cannot install it from PyPI.
  - Optional viewer pacing: `--viz-refresh-hz 60` (caps redraws). `--viz-sync` forces the runner to wait for the viewer per frame (debugging).

### Placement Action Space Debug Logging

- The speedrun harness can enable verbose placement diagnostics:
  - `--placement-debug-log` prints a compact 16×8 board occupancy and pill snapshot when the planner returns zero feasible options, plus planner search stats.
  - Off by default. Intended for development; emits to stdout only on meaningful changes to avoid spam.
  - QuickNES ships with a muted palette; for faithful colors switch to the Mesen core (`mesen_libretro.dylib`).

### Backend selection

- cpp-engine (recommended for training throughput): `DRMARIO_BACKEND=cpp-engine`. Requires building `game_engine/drmario_engine` (see `make -C game_engine`). No ROM required.
- libretro: `DRMARIO_BACKEND=libretro`. Requires a libretro core (`DRMARIO_CORE_PATH`) and ROM (`DRMARIO_ROM_PATH`). QuickNES and Mesen cores are known good options.
- Top-outs are detected (virus count jumps to menu) and incur a penalty; the env automatically returns to level 0 using configurable START/LEFT sequences (`--start-presses`, `--start-level-taps`, etc.).
- stable-retro: `DRMARIO_BACKEND=stable-retro`. Requires Stable-Retro install and imported game assets.
- mock: `DRMARIO_BACKEND=mock` for deterministic mock dynamics (CI / docs).

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
- `training/configs/` Training configs (in-repo algorithms)
- `data/states/` Savestates seeds (not tracked)
- `data/datasets/` Parquet playouts, labeled frames (not tracked)
- `eval/harness/` Evaluation harness and plotting (skeleton)
- `dr-mario-disassembly/` Disassembly + annotations submodule (no ROMs)
- `docs/` DESIGN, RNG, CONTRIBUTING and discussions

## Status

- Code: multi-env `ppo_smdp` placement training, Rich TUI/debug UI, and a headless C++ engine backend for throughput.
- Docs: design notes + RAM/state specs + placement-policy/training docs.
- Next: tighten engine parity, add more eval harness coverage, and extend to vs/2P play.

## Tools

- Dump state tensors: `python tools/dump_ram_and_state.py --frames 120 --mode state`
- Visualize planes: `from tools.ram_visualizer import grid_show` (see docs/RAM_TO_STATE.md)
- Unified runner (board + playback controls in terminal):
  - `python -m training.run --cfg training/configs/smdp_ppo.yaml --ui debug --env-id DrMarioPlacementEnv-v0 --backend libretro --core quicknes --rom-path /path/to/DrMario.nes --num_envs 1`
  - Controls: Space pause/run, `n` step, `f` +60 steps, `+/-` speed, `0` max, `q` quit
- Multi-env scaling benchmark: `python tools/bench_multienv.py --backend cpp-engine --vectorization both --num-envs 1,2,4,8,16`
- Curriculum confidence table + reporting: `python tools/report_curriculum.py --confidence-table`
- Best-known clear times (sqlite): `python tools/report_best_times.py --db data/best_times.sqlite3` (override via `DRMARIO_BEST_TIMES_DB`)
- C++ engine demo TUI (frame-accurate demo playback + step controls):
  - Build: `make -C game_engine`
  - Run: `python tools/engine_demo_tui.py` (Space pause/run, `n` step, `+/-` speed, `0` max, `b` benchmark)
  - Start at 2.4× NTSC: `python tools/engine_demo_tui.py --speed-x 2.4` (use `--region pal` for PAL)
  - Benchmark suite: `python tools/engine_demo_tui.py --benchmark`

## References

See `docs/DESIGN.md`, `docs/RNG.md`, `docs/REWARD_SHAPING.md`, `docs/TASKS_NEXT.md`,
`docs/ENV_STANDUP_MAC_LINUX.md`, `docs/RAM_TO_STATE.md`, and `docs/REFERENCES.md` for details and links.
Project decisions and work history live under `notes/`.
Older non-authoritative artifacts are archived under `notes/archive/` (read only if needed).
