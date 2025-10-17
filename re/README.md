# Dr. Mario (NES) Reverse‑Engineering Toolkit

This folder gives the coding agent a tool‑land pipeline to disassemble, trace, and interpret a legally owned Dr. Mario ROM and produce:
- Labeled 6502 disassembly (Ghidra project + emulator labels)
- MMC1 bank map (PRG switching timeline)
- RAM map (addresses, names, semantics)
- Pseudocode for core systems (RNG, virus placement, pill spawn, gravity/lock, clear/settle, input)
- Parity fixtures (savestates + seeds) to validate your later C++ simulator

Legal hygiene: Do not commit ROMs. Point tools at a local ROM path via env vars. `.gitignore` already blocks `*.nes` and `legal_ROMs/`.

## Quick start

1. Parse ROM header
- `python re/ines/parse_ines.py --rom "$DRMARIO_ROM" --out re/out/ines.json`

2. Dynamic marking & traces (option A: Mesen)
- In Mesen: enable Code/Data Logger; play intro → level select → a few spawns/clears.
- Export labels + disassembly + trace into `re/mesen/out/`.

3. Dynamic marking & traces (option B: FCEUX/BizHawk)
- FCEUX: run Code/Data Logger to create `.cdl`, export `.nl` labels.
- BizHawk: run `re/bizhawk/trace_mmc1_and_io.lua` to log MMC1 writes, `$4016` input, and frame counts.

4. Import to Ghidra
- Install Ghidra + GhidraNes; import the ROM as NES ROM; import labels; run analysis.
- Use vectors at `$FFFA–$FFFF` to locate RESET/NMI/IRQ; start labeling and bank marking.

5. Build bank map
- `python re/tools/build_bank_map.py --mmc1-log re/out/mmc1_writes.csv --ines re/out/ines.json --out re/out/bankmap.json`

6. Find targets (guided hunts)
- `re/pipelines/virus_placement_hunt.md`
- `re/pipelines/pill_rng_hunt.md`
- `re/pipelines/frame_counter_seed.md`

7. Emit RAM map + pseudocode
- Fill `re/out/ram_map.json` and `re/pseudocode/*.md` using the template.
- `python re/tools/emit_ram_map_py.py --in re/out/ram_map.json --out envs/specs/ram_map.py`
