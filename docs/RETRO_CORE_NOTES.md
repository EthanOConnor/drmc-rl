# Stable-Retro Core Notes

- Set the libretro NES core path via `DRMARIO_CORE_PATH` env var if needed.
- Game name and initial state are configurable via `DRMARIO_GAME` and `DRMARIO_STATE`.
- Example:
  ```bash
  export DRMARIO_CORE_PATH=/usr/local/lib/libretro/mesen_libretro.dylib
  export DRMARIO_GAME=DrMario-Nes
  export DRMARIO_STATE=LevelSelect
  ```
- See `envs/retro/stable_retro_utils.py` for how these are consumed.
