# Dr. Mario RL (drmc-rl)

Reinforcement-learning environment, training stack, and evaluation harness for NES Dr. Mario. The repo currently focuses on the
single-player speedrun ruleset with risk-aware evaluation, a libretro backend, a placement planner that translates high-level
actions into precise controller inputs, and Sample Factory integration for PPO training.

ROMs are **not** included—use your own legally obtained copy. `*.nes` files remain git-ignored and the optional pre-commit hook
blocks accidental commits (see [Legal & ROM Hygiene](#legal--rom-hygiene)).

## Feature Highlights

* **Gymnasium environment (`DrMarioRetroEnv-v0`)** supporting pixel or RAM-derived state observations, configurable reward
  shaping, risk conditioning, and deterministic savestate resets.
* **Placement translator (`envs/retro/placement_wrapper.py`)** that turns “place capsule at column/orientation” intents into
  button-level control, with spawn detection latching and adaptive timing retries.
* **Intent + planner tooling** for experimenting with action abstractions (see `envs/retro/intent_wrapper.py` and
  `envs/retro/placement_planner.py`).
* **Live state visualizer** (`envs/retro/demo.py --show-window`) that annotates state tensors, planner masks, and emulator timing.
* **Sample Factory launcher** (`training/run_sf.py`) with baseline configs for pixel/state observation modes plus
  evaluation-specific variants.
* **Docs + reverse-engineering artifacts** covering RAM maps, RNG notes, and hardware stand-up steps.

## Quick Start (macOS Apple Silicon)

```bash
brew install cmake pkg-config lua@5.1
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[retro,dev]"   # stable-retro + tooling
pip install -e ".[rl]"          # Sample Factory / PyTorch extras (install torch first on MPS)

# Point to your emulator assets
export DRMARIO_CORE_PATH=/path/to/mesen_libretro.dylib
export DRMARIO_ROM_PATH=/path/to/DrMario.nes

# Optional Stable-Retro fallback assets
python -m retro.import ~/ROMs/NES

# Smoke test (press Ctrl+C to exit)
python -m envs.retro.demo --mode pixel --steps 2000 --backend libretro
```

Stable-Retro wheels currently target CPython ≤3.14. If you use 3.14+, build Stable-Retro from source or recreate the virtual
environment with Python 3.13/3.12.

## Quick Start (Linux, CUDA training)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[retro,dev]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[rl]"

export DRMARIO_CORE_PATH=/path/to/quicknes_libretro.so
export DRMARIO_ROM_PATH=/path/to/DrMario.nes
python -m envs.retro.demo --mode state --backend libretro --steps 1200 --show-window
```

For headless validation, run `python -m envs.retro.demo --backend mock --steps 600` to exercise deterministic mock dynamics.

### Demo CLI Cheatsheet

The demo script exposes most emulator toggles for inspection or debugging:

* `--mode {pixel,state}` – select observation type. State mode supports `--state-repr {extended,bitplane}`.
* `--show-window` (requires Pillow with ImageTk) – spawn an interactive Tk window. Use `space` to pause, `n` to single-step, and
  `+`/`-` to adjust emulator/viz ratios when sync is enabled.
* `--viz-refresh-hz` and `--emu-target-hz` – throttle viewer refresh and emulator stepping.
* `--randomize-rng` – rewrite in-game RNG bytes each reset in addition to savestate seeds.
* `--start-presses`, `--start-level-taps`, etc. – override auto start sequences for level restarts.
* `--save-frames path/` – dump PNG frames (or `.npy` if Pillow is unavailable).

Backends are selected via `--backend` or `DRMARIO_BACKEND`:

| Backend        | Requirements                                      | Notes |
|----------------|----------------------------------------------------|-------|
| `libretro`     | `DRMARIO_CORE_PATH`, `DRMARIO_ROM_PATH`            | Default path; tested with QuickNES and Mesen. |
| `stable-retro` | Stable-Retro install and imported assets          | Matches OpenAI Gym Retro integration. |
| `mock`         | None                                               | Deterministic mock used for CI and documentation. |

Top-outs are detected automatically (virus count returning to menu). The env applies configurable penalties and uses the
auto-start macro to return to level 0 before the next episode.

## Training with Sample Factory

Baseline configs live in `training/sf_configs/`:

* `state_baseline.yaml` / `state_baseline_sf2.yaml` – RAM/state observations with placement wrapper enabled.
* `pixel_baseline.yaml` / `pixel_baseline_sf2.yaml` – pixel observations with the same reward settings.
* `state_eval_bootstrap.yaml` – evaluation-focused run for distributional policy heads.

Launch via:

```bash
python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml
```

Override timeouts or state visualizer cadence with `--timeout` and `--state-viz-interval`. The launcher registers the env id and
shells out to `python -m sample_factory.launcher.run --run <cfg>`.

## Repository Layout

* `envs/retro/` – libretro backend glue, Gymnasium env, placement/intent translators, demo viewer, and seed registry.
* `envs/specs/` – RAM offsets, deterministic timeout tables, and state representation helpers.
* `envs/pettingzoo/` – 2-player ParallelEnv prototype (inactive but kept for future expansion).
* `models/` – policy/evaluator scaffolding for risk-conditioned PPO.
* `training/` – Sample Factory configs and launcher.
* `eval/` – evaluation harness and plotting utilities.
* `tools/` – RAM dumpers, visualization helpers, and debug scripts.
* `re/` – reverse-engineering references (no ROMs committed).
* `docs/` – design notes, RNG research, contributor guides, and stand-up instructions.

## Legal & ROM Hygiene

* Never commit ROMs. Paths are ignored via `.gitignore`.
* Optional pre-commit guard:

  ```bash
  chmod +x .git-hooks/pre-commit-block-roms.sh
  ln -sf ../../.git-hooks/pre-commit-block-roms.sh .git/hooks/pre-commit
  ```

* See `docs/LEGAL.md` for history purge steps if a ROM is accidentally added.

## Current Status & Next Steps

* **Environment** – libretro backend parity against emulator RAM verified; placement planner handles spawn synchronization,
  stickiness, and input retries. State observations support extended and bitplane layouts.
* **Training** – Sample Factory configs produce risk-aware PPO rollouts; evaluator head integration is in progress in
  `models/evaluator/`.
* **Roadmap** – expand PettingZoo wrappers, finish evaluator bootstrapping, wire up the C++ sim in `sim-envpool/` for
  high-throughput regression testing, and prototype the row/column/orientation intent translator plus asynchronous decision
  cadence described in the 19 Oct 2025 `docs/EOCLabbook.md` entry to cut inference frequency.

For deeper design and reverse-engineering notes see `docs/DESIGN.md`, `docs/RNG.md`, `docs/REWARD_SHAPING.md`,
`docs/STATE_OBS_AND_RAM_MAPPING.md`, and `docs/REFERENCES.md`.
