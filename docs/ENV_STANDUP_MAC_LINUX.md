# Environment Stand-Up

The default setup is for native `cpp-pool` placement training. Emulator setup is
optional and only needed for parity/debug work.

## macOS

```bash
brew install cmake pkg-config
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl,viz]"
python -m tools.build_drmario_pool
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true
```

For Apple Silicon, install the PyTorch packages appropriate for MPS first if the
generic install does not select the wheel you want.

Small live run:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml \
  --ui tui --backend cpp-pool --num_envs 16
```

## Linux

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,rl,viz]"
python -m tools.build_drmario_pool
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true
```

Long CUDA training runs should use Linux when available. Keep the same runner
and config path unless a specific experiment says otherwise.

## Emulator Parity Setup

Only do this for libretro/Stable-Retro parity, frame capture, or ROM debugging.

```bash
export DRMARIO_CORE_PATH=/path/to/quicknes_libretro.dylib
export DRMARIO_ROM_PATH=/path/to/DrMario.nes
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path "$DRMARIO_ROM_PATH" \
  --num_envs 1
```

Stable-Retro is a legacy compatibility path. If it is needed, install it
explicitly in the local environment and keep those commands out of the default
training setup.

## Notes

- `cpp-pool` needs the native submodule and build artifact, not a ROM.
- `envs/retro/` contains active placement planner code despite the historical
  package name.
- If Tk/Pillow or libretro UI dependencies are missing, that should not block
  native `cpp-pool` training checks.
