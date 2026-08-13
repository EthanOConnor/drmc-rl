# Environment Stand-Up

The default setup is for native `cpp-pool` placement training. Emulator setup is
optional and only needed for parity/debug work. Python 3.14 is the supported
runtime.

## macOS

```bash
brew install cmake pkg-config
git submodule update --init --recursive
uv sync --all-extras
uv run python -m tools.build_drmario_pool
uv run python -m drmc_rl.training.run --cfg drmc_rl/training/configs/smdp_ppo.yaml --dry_run true
```

The lock selects PyTorch's native macOS wheel; Apple Silicon training uses MPS.

Small live run:

```bash
uv run python -m drmc_rl.training.run --cfg drmc_rl/training/configs/smdp_ppo.yaml \
  --ui tui --backend cpp-pool --num_envs 16
```

## Linux

```bash
git submodule update --init --recursive
uv sync --all-extras
uv run python -m tools.build_drmario_pool
uv run python -m drmc_rl.training.run --cfg drmc_rl/training/configs/smdp_ppo.yaml --dry_run true
```

The Linux lock selects PyTorch 2.13's CUDA 13 build. Long CUDA training runs
should use Linux with a current NVIDIA driver. Keep the same runner and config
path unless a specific experiment says otherwise.

## Emulator Parity Setup

Only do this for libretro/Stable-Retro parity, frame capture, or ROM debugging.

```bash
export DRMARIO_CORE_PATH=/path/to/quicknes_libretro.dylib
export DRMARIO_ROM_PATH=/path/to/DrMario.nes
uv run python -m drmc_rl.training.run --cfg drmc_rl/training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path "$DRMARIO_ROM_PATH" \
  --num_envs 1
```

## Notes

- `cpp-pool` needs the native submodule and build artifact, not a ROM.
- `drmc_rl/envs/libretro/` is restricted to emulator integration; reusable
  mechanics and reachability live under `drmc_rl/game/` and `drmc_rl/planning/`.
- If Tk/Pillow or libretro UI dependencies are missing, that should not block
  native `cpp-pool` training checks.
