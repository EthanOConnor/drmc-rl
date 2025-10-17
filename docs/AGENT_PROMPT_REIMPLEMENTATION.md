# PROMPT — Finalize RAM map and implementation facts (agent-owned)

You own: verifying RAM addresses on the legally owned ROM, filling missing fields, and producing final constants & pseudocode.

1) Identify ROM (CRC32/SHA1) and paste into docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md §0.
2) Dynamic marking: run Code/Data Logger (Mesen/FCEUX) over reset→title→level start→first spawn/clear; export labels/traces.
3) Bank map: log MMC1 writes; build re/out/bankmap.json.
4) RAM confirm: validate each entry in re/out/ram_map_external_candidates.json; fill bottle/preview/falling/gravity/lock/timers/RNG.
5) Write re/out/ram_map.json and emit envs/specs/ram_map.py via emit_ram_map_py.py.
6) Capture routines and write JSONs: virus_placement.json, pill_rng.json, seed_path.json, game_step.json (include bank+addr).
7) Fill re/pseudocode/*.md with faithful logic & constants.
8) Generate parity fixtures: first 128 previews and initial virus grid for ≥3 seeds; store under re/out/parity/.
9) Update docs/IMPLEMENTATION_FACTS_AND_RAMMAP.md with a CONFIRMED table of final addresses and mark the candidates as resolved.
