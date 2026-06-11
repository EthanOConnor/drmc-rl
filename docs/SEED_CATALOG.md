# Seed Catalog ("seedlab")

Prime95-style long-running search that exhaustively catalogs, per Dr. Mario
seed, the **best known clear time** and the **distribution of good-but-
suboptimal clear times**. Consumers:

- **Agent training/eval**: per-seed frame floors and quantiles as objective
  baselines (curriculum floors, eval normalization, "% of best-known" curves).
- **Human game analysis / skill assessment**: fightcadeRatings v2 replay
  events carry the init RNG state, so a human game maps to an exact catalog
  row; their clear time lands at a percentile of the known distribution.
- **Speedrun research**: per-seed fastest-known solutions with replayable
  placement traces (TAS-adjacent lower bounds on standard categories).

## Seed space (measured facts)

- Engine RNG is the NES 16-bit shift register (`GameLogic.cpp:rng_step`,
  carry = bit1(r0)^bit1(r1), ROR across both bytes).
- The console writes `0x89 0x88` at init and steps once per frame; every
  reachable seed lies on the orbit of `0x8988`, which has **period 32,767**.
  `0x0000` is a lockup fixed point; the remaining 32,768 states are transient
  and never occur on hardware. Catalog universe = the 32,767 orbit states.
- A game is fully determined by `(level, speed_setting, seed)`:
  `generatePillsReserve()` (128 pills) runs first, then virus placement, both
  consuming the same RNG stream from the seeded state with no warmup
  (`GameLogic.cpp` reset path).
- Levels 0–20 canonical ((level+1)*4 viruses, capped tables). Speed HI
  (`speed_setting=2`) is the default catalog dimension; schema carries speed
  so MED/LOW sweeps can be added.
- `seedlab/rng.py` mirrors pill/virus generation in Python (parity-tested
  against the engine) and computes `game_hash` = sha1(virus board ‖ pill ids)
  for dedup/cross-referencing. Census result (2026-06-10): **zero hash
  collisions** — every (level, seed) pair in levels 0–20 is a unique game,
  and virus placement always reaches the full (level+1)*4 count. The catalog
  universe is exactly 21 × 32,767 = 688,107 distinct games per speed.

## Scale estimate

21 levels × 32,767 seeds = 688,107 games per speed. Measured eval throughput
(M3 Max, cpp-pool + candidate policy, 64 envs) supports roughly 40–80
episodes/s depending on level length:

- Greedy pass (1 attempt/seed, all levels): ~3–5 hours.
- +8 sampled attempts/seed for distributions: ~1.5–2 days.
- Long tail: unbounded — further passes keep improving bests (prime95 mode).

Single machine suffices for the base catalog; the work-unit design still
shards cleanly across processes/machines.

## Storage (`data/seed_catalog.sqlite3`, WAL)

- `games(level, speed, seed, game_hash, virus_count, orbit_pos)` — census,
  computed analytically by `seedlab init` (no rollouts).
- `seed_stats(level, speed, seed, n_attempts, n_clears, min/max_frames,
  sum/sumsq_frames, reservoir BLOB, best_frames, best_spawns, best_solver,
  best_at)` — one row per game; reservoir = ≤64 packed uint32 clear times for
  quantiles of the "achievable" distribution.
- `solutions(level, speed, seed, frames, spawns, actions BLOB, solver,
  created_at, verified)` — best-known placement trace (uint16 action ids,
  one per decision); replayable via the pool env for verification.
- `work_units(level, speed, pass_idx, seed_lo, seed_hi, status, leased_by,
  leased_at, done_at)` — shardable work queue with atomic lease claim
  (`status: todo → leased → done`; stale leases reclaimable).
- `meta(key, value)` — schema version, census provenance.

Existing `data/best_times.sqlite3` (trainer's opportunistic per-curriculum-
level bests) stays as-is; the catalog is a separate, canonical-level artifact.

## Search worker (`python -m seedlab worker`)

- Claims a work unit, runs every seed in it through K attempts on a
  `DrMarioPoolVecEnv` (per-env exact seeds via `seed_provider`):
  - pass 0: deterministic greedy (argmax) + (K−1) temperature samples;
  - pass ≥1: sampled attempts (more diversity, distribution mass).
- Records per-episode: cleared, frames (sum of planner tau — exact NES frame
  cost under warp), decisions (=spawns), action trace. Updates aggregates and
  best solutions in one transaction per unit; marks unit done.
- Resumable: kill at any time; leased units have partial results already
  committed per unit, stale leases get reclaimed.
- Solvers are pluggable (`--policy checkpoint|greedy-cost|random`); solver id
  is recorded with each best so provenance survives checkpoint upgrades.

## Reporting / UI

- `python -m seedlab report` — coverage per level, best/q10/q50/q90 frames,
  fastest seeds, recent records.
- `python -m seedlab grade --level L --frames F [--seed S]` — percentile of a
  given clear time vs the catalog (the human-assessment hook).
- `python -m seedlab verify` — replay stored solutions, assert frame counts.
- `python -m seedlab tui` — live dashboard (coverage bars, throughput,
  record feed, active leases), `tools/vs_dashboard.py` style.

## Verification semantics

Frame counts come from the warp path (planner v4 exact costs). Solutions are
re-runnable with `DRMARIO_POOL_WARP=0` (controller replay) or the v1 oracle
for byte-exact audits; `verify` re-executes the trace in warp mode and checks
the frame total. Byte-exact spot audits are a SCRUTINY follow-up, not a v1
blocker.

## Future (recorded in BACKLOG, not built now)

- Adaptive pass scheduling: spend attempts where best-vs-quantile gap or
  variance is highest.
- Per-seed optimistic lower bound (sum of per-spawn min lock costs) to flag
  seeds with remaining headroom.
- Depth-2 expectimax solver pass for record-hunting on the frontier.
- Cross-machine federation (export/import unit results as JSONL shards).
