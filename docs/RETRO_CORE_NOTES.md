# Emulator Parity Notes

This document is for emulator oracle/debug work. It is not the default training
setup; use `cpp-pool` for normal placement-SMDP training.

## Libretro

Set a NES core and a legally obtained ROM:

```bash
export DRMARIO_CORE_PATH=/usr/local/lib/libretro/mesen_libretro.dylib
export DRMARIO_ROM_PATH=~/ROMs/NES/DrMario.nes
```

QuickNES and Mesen are both supported. QuickNES is lightweight; Mesen is useful
when visual fidelity matters.

Fetch/update QuickNES on macOS arm64:

```bash
python tools/update_quicknes_core.py --force
```

The upstream nightly URL is:
`https://buildbot.libretro.com/nightly/apple/osx/arm64/latest/quicknes_libretro.dylib.zip`

Run a parity/debug session:

```bash
python -m drmc_rl.training.run --cfg drmc_rl/training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path "$DRMARIO_ROM_PATH" \
  --num_envs 1
```

## Stable-Retro

Stable-Retro remains a legacy compatibility path. It may require a separate
install and imported game assets:

```bash
python -m retro.import ~/ROMs/NES
```

Do not use Stable-Retro setup as the default onboarding path.

## Useful Controls

- `--start-presses`, `--start-level-taps`, `--start-settle-frames`,
  `--start-wait-viruses`, and `--start-sync-wait-frames` tune menu auto-start.
- `--randomize-rng` reseeds the ROM RNG during reset.
- The env applies RNG seed bytes at the `initData_level` boundary for parity.

When the agent tops out, emulator wrappers can press START, return to level 0,
and resume. Native `cpp-pool` training does not use this menu path.
