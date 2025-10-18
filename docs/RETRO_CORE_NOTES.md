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
- See `envs/retro/stable_retro_utils.py` for how Stable-Retro specific env vars are consumed.
