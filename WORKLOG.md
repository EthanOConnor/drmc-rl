# Worklog — DrMC RL (Dr. Mario NES)

Date: 2025-10-17
Author: Coding Agent (Codex CLI)

## Summary
We confirmed our ROM revision, extracted and validated the RAM map from the disassembly, implemented a faithful RAM→state mapping, added tests and tooling, and documented RNG, virus placement, and the state spec. This puts us in a strong position to wire Stable‑Retro, proceed with RL training, and later port a max‑FPS C++ simulator.

## What’s confirmed on our ROM
- ROM: Dr. Mario (Japan, USA) (rev0)
  - CRC32 0xB1F7E3E9, SHA1 01de1e04c396298358e86468ba96148066688194
  - Parsed via `re/ines/parse_ines.py` → `re/out/ines.json`
- Bottle and player RAM (P1; P2 mirrors):
  - Bottle base: `$0400`, stride `8`, 16×8 grid (P2 `$0500`).
  - Field encoding: high nibble = type, low nibble = color (Y=0, R=1, B=2).
  - Falling pill: X `$0305`, Y `$0306`, Rotation `$0325` (bit0), Colors `$0301/$0302`.
  - Preview pill: Colors `$031A/$031B`, Rotation `$0322`, Size `$0323`.
  - Scalars: speedCounter `$0312` (gravity timing), pillPlacedStep `$0307` (settle micro‑step proxy), Level `$0316`, Viruses left `$0324`.
  - Inputs: P1 pressed/held `$00F5/$00F7`; P2 `$00F6/$00F8`.
  - RNG: `rng0=$0017`, `rng1=$0018`; update at `$B78B`; seed at init to `$89/$88`.

## Code + assets added/updated
- RAM offsets (runtime): `envs/specs/ram_offsets.json`
- Full spec (expanded): `re/out/ram_map.json` → emitted to `envs/specs/ram_map.py`
- State decoder: `envs/specs/ram_to_state.py`
  - Decodes hi:type/lo:color; channels for viruses/fixed/falling (R/Y/B), orientation, gravity, settle proxy, level, spare.
- Exports: `envs/specs/__init__.py` now exposes `BOTTLE`, `FALLING_PILL`, etc.
- CLI tooling: `tools/ram_planes_dump.py` to print per‑channel counts from a 0x800 RAM snapshot.
- Env docstring: `envs/retro/drmario_env.py` points to mapping docs and offsets override.

## Documentation
- Updated: `docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md` (ROM IDs, confirmed RAM, state spec, RNG/placement)
- Added:
  - `docs/STATE_OBS_AND_RAM_MAPPING.md` — definitive 14‑channel spec and addresses
  - `docs/RNG_AND_PLACEMENT_NOTES.md` — RNG state/update/seed, addVirus/generateNextPill summaries
  - `docs/CPP_SIM_NOTES.md` — C++ sim outline (state layout, per‑frame order, constants, parity)

## Tests
- New: `tests/test_ram_decode_field.py` — sanity for field decoding (virus/fixed halves → channels)
- Existing tests pass: `PYTHONPATH=. pytest -q` → 4 passed

## Rationale
We now rely on authoritative labels in the disassembly rather than external candidates. The mapping is explicit and documented, avoiding guesswork and future rework. The CLI tool and tests give quick verification against emulator RAM dumps.

## Open items (tracked decisions)
- Settle/lock plane uses `pillPlacedStep` as a scalar proxy; we can refine if we identify a distinct lock counter.
- Runtime offsets read from JSON for flexibility; can switch to code constants if desired.
- Stable‑Retro RAM dump integration not yet wired; planned next.

## Next steps (proposed)
1) Stable‑Retro integration (state mode first)
   - Bind core and wire `retro.get_ram()` → `ram_to_state` in `DrMarioRetroEnv`.
   - Add a debug mode in `envs/retro/demo.py` to print per‑channel counts live.
   - Record a short session to produce sample RAM snapshots for parity checks.
2) RL environment completion
   - Finalize action adapter (hold latching, one‑frame taps) and reward shaping per `RewardConfig`.
   - Implement termination/timeouts per level; expose cleared flag and stats.
   - Add seeding interface matching ROM seeding (level + frame offset snapshot).
3) Data & fixtures
   - Capture parity fixtures (fields + previews over 3–5 seeds) to guard future changes and feed C++ sim tests.
4) C++ simulator scaffold
   - Define state struct and step order from `CPP_SIM_NOTES.md`.
   - Implement field encoding/decoding and gravity/lock/match resolution.
   - Create a headless stepping harness and a hash‑based parity test fed by recorded fixtures.
5) Observability
   - Add tiny visualizers (ASCII grid, channel overlays) to ease debugging.

## How to run
- Tests: `PYTHONPATH=. pytest -q`
- Emit constants from spec: `python3 re/tools/emit_ram_map_py.py --in re/out/ram_map.json --out envs/specs/ram_map.py`
- State from RAM dump: `python tools/ram_planes_dump.py --ram path/to/ram.bin`

