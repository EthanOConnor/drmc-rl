# Stable-Retro & Libretro Core Notes

- Set the libretro NES core path via `DRMARIO_CORE_PATH` env var.
- Point the backend at your ROM via `DRMARIO_ROM_PATH`.
- Game name and initial state are configurable via `DRMARIO_GAME` and `DRMARIO_STATE`.
- Example:
  ```bash
  export DRMARIO_CORE_PATH=/usr/local/lib/libretro/mesen_libretro.dylib
  export DRMARIO_ROM_PATH=~/ROMs/NES/DrMario.nes
  export DRMARIO_GAME=DrMario-Nes
  export DRMARIO_STATE=LevelSelect
  ```
- Default backend is libretro (`DRMARIO_BACKEND=libretro`); switch with `DRMARIO_BACKEND=stable-retro` or `DRMARIO_BACKEND=mock`.
- QuickNES (`quicknes_libretro.dylib`) and Mesen (`mesen_libretro.dylib`) cores are both supported. QuickNES is lightweight and ships cleanly on arm64; Mesen offers higher fidelity if available.
- The demo auto-presses START (default 3 taps on first boot, then 1 on subsequent resets). Override with `--start-presses`, `--start-level-taps`, `--start-settle-frames`, and `--start-wait-frames` as needed.
- Use `--randomize-rng` in `envs/retro/demo.py` to reseed the ROM's two-byte RNG on every reset,
  providing varied virus layouts and pill sequences.
- When the agent tops out, the env automatically presses START to exit the game-over screen, spams LEFT to return to level 0, and presses START again to resume.
- See `envs/retro/stable_retro_utils.py` for how Stable-Retro specific env vars are consumed.
