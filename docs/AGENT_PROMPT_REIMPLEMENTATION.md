# PROMPT — Finalize RAM map and implementation facts (agent-owned)

You own: verifying RAM addresses on the legally owned ROM, filling missing fields, and producing final constants & pseudocode.

1) Identify ROM (CRC32/SHA1) and paste into docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md §0.
2) Validate addresses against `dr-mario-disassembly/` and (optionally) a Code/Data Logger trace (Mesen/FCEUX) over reset→title→level start→first spawn/clear.
3) Update the canonical in-repo mapping:
   - `envs/specs/ram_offsets.json`
   - `envs/specs/ram_map.py`
4) Update docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md with the confirmed table and any corrected routine addresses.
5) Regenerate/verify parity fixtures if needed:
   - Engine demo ground truth: `python tools/record_nes_demo.py --output data/nes_demo.json`
   - Optional RAM/planes snapshots: `python tools/capture_parity.py --out data/parity`
6) Run `pytest -q` to ensure no regressions.
