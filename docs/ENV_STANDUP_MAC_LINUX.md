# Environment Stand-up (macOS & Linux)

## macOS (Apple Silicon)
1. Install prerequisites:
   ```bash
   brew install cmake pkg-config lua@5.1
   ```
2. Python env + core libs (CPython ≤3.14 recommended for Stable-Retro wheels):
   ```bash
   python3.13 -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -e ".[retro,dev]"  # stable-retro, Pillow, ruff, pytest, etc.
   pip install -e ".[rl]"         # Sample Factory / RL extras (install torch first on MPS)
   ```
3. Libretro core (NES): install RetroArch to fetch a NES core (Mesen/Nestopia/QuickNES) or download a nightly core build, then export:
   ```bash
   export DRMARIO_CORE_PATH=/path/to/mesen_libretro.dylib   # or quicknes_libretro.dylib
   export DRMARIO_ROM_PATH=/path/to/DrMario.nes
   ```
4. (Optional Stable-Retro fallback) Import the game you legally own:
   ```bash
   python -m retro.import ~/ROMs/NES
   ```
5. Run your smoke test:
   ```bash
   python -m envs.retro.demo --mode pixel --steps 400 --backend libretro --show-window
   ```
   *`space` toggles pause, `n` steps a single frame, `+`/`-` adjust the emulator/viewer speed ratio.*
6. Override auto-start macros as needed:
   ```bash
   python -m envs.retro.demo --backend libretro --start-presses 3 --start-level-taps 12 --start-settle-frames 180
   ```
7. Frame capture for debugging:
   ```bash
   python -m envs.retro.demo --backend libretro --save-frames out_frames
   ```
   *Frames are PNG if Pillow is installed, `.npy` otherwise.*

## Linux (training)
1. Install CUDA 12.x drivers/toolkit (if using GPUs) and create a Python env.
2. PyTorch CUDA wheels, Sample Factory, TorchRL:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -e ".[retro,dev]"
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -e ".[rl]"
   ```
3. Install a libretro NES core. Set `DRMARIO_CORE_PATH`/`DRMARIO_ROM_PATH`, then import ROMs if you plan to use the Stable-Retro backend.
4. Register env id and launch training:
   ```bash
   python - <<'PY'
   from envs.retro.register_env import register_env_id
   register_env_id()
   print("Env registered")
   PY
   python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml --state-viz-interval 600
   ```

Notes
- macOS development is great for wiring and tests; prefer Linux for long PPO runs.
- If LuaJIT errors occur on macOS, stick to Lua 5.1 (brew formula shown above).
- Switch backends via `DRMARIO_BACKEND` (`libretro` default, `stable-retro`, or `mock`).
- Use `python -m envs.retro.demo --backend mock --steps 600` to validate docs/tests without emulator assets.
