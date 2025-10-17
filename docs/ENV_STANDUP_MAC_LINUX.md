# Environment Stand-up (macOS & Linux)

## macOS (Apple Silicon)
1. Install prerequisites:
   ```bash
   brew install cmake pkg-config lua@5.1
   ```
2. Python env + core libs:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio  # MPS wheels are fine for dev
   pip install stable-retro gymnasium numpy opencv-python
   ```
3. Libretro core (NES): install RetroArch to fetch a NES core (Mesen/Nestopia) or build a core and note its path.
4. Import the game you legally own:
   ```bash
   python -m retro.import ~/ROMs/NES
   ```
5. Run your smoke test: `python envs/retro/demo.py --obs-mode pixel --steps 200`

## Linux (training)
1. Install CUDA 12.x drivers/toolkit and create a Python env.
2. PyTorch CUDA wheels, Sample Factory, TorchRL:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install sample-factory torchrl[atari] gymnasium
   ```
3. Install Stable-Retro and a libretro NES core. Import ROMs as above.
4. Register env id and launch:
   ```bash
   python -c "from envs.retro.register_env import register_env_id; register_env_id(); print('OK')"
   python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml
   ```

Notes
- macOS development is great for wiring and tests; prefer Linux for long PPO runs.
- If LuaJIT errors occur on macOS, stick to Lua 5.1 (brew formula shown above).

