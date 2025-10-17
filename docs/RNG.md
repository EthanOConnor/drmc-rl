# RNG & Seeding Plan

Goals
- Deterministic seeds for (layout, pill sequence) via Level Select savestates and frame offsets.
- Reverse engineer RNG sources to implement a byte‑exact simulator and parity tests.

Seed Registry
- A JSON registry maps `seed_id -> { savestate, frame_offset }`.
- Savestate is captured at Level Select with a fixed power‑on frame counter.
- `frame_offset` is the number of frames from state load to the selection input.
- Store per‑seed metadata: level, initial virus count, pill color distribution hash, first N pill IDs.

Reverse Engineering Targets
- Frame counter source used at level selection (confirms TAS claim: fixes layout + sequence)
- Virus placement constraints (no 2nd‑neighbor same color, etc.)
- Pill sequence RNG (expect LFSR/LCG or shift‑xor patterns)
- RAM map for bottle grid, falling pill halves, orientation, gravity/lock, timers

Workflow
- FCEUX/Mesen: enable Code/Data Logger (CDL) during boot→level select→several levels.
- Breakpoints on bottle RAM writes; log call trees.
- Export .cdl and use as code/data mask in disassembly (Ghidra or ca65 project).
- Map MMC1 banks: fixed bank $C000–$FFFF, switchable $8000–$BFFF.
- Unit tests: capture RNG state at selection and regenerate layout to assert byte‑for‑byte equivalence.

Data Products
- `envs/retro/seeds/registry.json` with entries and hashes
- `data/datasets/` parquet tables with per‑state Monte‑Carlo time‑to‑clear samples
- `docs/RNG.md` updates with discovered addresses, constants, and algorithm sketch

Open Items (fill during RE)
- Addresses: frame counter, RNG state, virus placer entry point
- RNG function: constants/taps, period, seeding rules
- Determinism caveats: PPU timing, input polling cadence, debounce logic
