# Environment Stand-up (macOS & Linux)

## macOS (Apple Silicon)
1. Install prerequisites:
   ```bash
   brew install cmake pkg-config lua@5.1
   ```
2. Python env + core libs (Python ≤3.14 recommended for Stable-Retro wheels):
   ```bash
   python3.13 -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -e ".[retro,dev]"  # installs stable-retro wheels (Python ≤3.14)
   # For RL training on macOS: install torch MPS wheels
   pip install torch torchvision torchaudio
   ```
3. Libretro core (NES): install RetroArch to fetch a NES core (Mesen/Nestopia/QuickNES) or download a nightly core build, then export:
   ```bash
   export DRMARIO_CORE_PATH=/path/to/quicknes_libretro.dylib  # or mesen_libretro.dylib
   export DRMARIO_ROM_PATH=/path/to/DrMario.nes
   ```
4. (Optional for Stable-Retro fallback) Import the game you legally own:
   ```bash
   python -m retro.import ~/ROMs/NES
   ```
5. Run your smoke test: `python envs/retro/demo.py --obs-mode pixel --steps 200 --backend libretro --start-presses 3`
6. Optional live preview (requires Pillow/Tk): `python envs/retro/demo.py --backend libretro --show-window --display-scale 2`
   - Install Tk if missing (`brew install python-tk` on macOS, then recreate your venv).

## Linux (training)
1. Install CUDA 12.x drivers/toolkit and create a Python env.
2. PyTorch CUDA wheels (+ optional TorchRL helpers):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -e ".[rl]"
   ```
3. Install Stable-Retro and a libretro NES core. Set `DRMARIO_CORE_PATH` and `DRMARIO_ROM_PATH`, then import ROMs if you plan to use the Stable-Retro backend.
4. Register env id and launch:
   ```bash
   python -c "from envs.retro.register_env import register_env_id; register_env_id(); print('OK')"
   python -m training.run --cfg training/configs/smdp_ppo.yaml --ui tui --env-id DrMarioPlacementEnv-v0 --backend cpp-pool --num_envs 16
   ```

Notes
- macOS development is great for wiring and tests; prefer Linux for long PPO runs.
- If LuaJIT errors occur on macOS, stick to Lua 5.1 (brew formula shown above).
- Switch backends via `DRMARIO_BACKEND` (`libretro` default, `stable-retro`, or `mock`).
