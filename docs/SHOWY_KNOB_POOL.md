> Superseded in the pool release by the general knob registry (`drmc_rl/style/knobs.py`,
> docs/RATING_POOL.md "Knobs"): entrants use `settings.knobs = [{id: "showy-t2", version: 1,
> lambda, model}]` and capabilities `knob:<id>@<version>`; the `showy_*` settings below are
> no longer accepted.

# Showy knob entrants in the rating pool (proposal)

Branch `style/showy-knob` (based on `trainer/rating-pool` @79c3e1fd). The pool
agent applies it; nothing here is deployed.

## What changes

- `tools/trainer_planning_arena.py` `variant_policy`: when a variant has
  `showy_lambda` != 0, the actor is wrapped in `drmc_rl.style.showy_knob.ShowyPolicy`
  (inline `showy_model`, optional `showy_tier_bar`, optional `showy_terms`).
  Without those keys, or with λ = 0, the actor is unchanged. That was checked
  byte-identical over 928 placements. Unknown `showy_*` keys raise `ValueError`,
  and the worker reports that failure as `incompatible`.
- `drmc_rl/pool/conditions.py`:
  - A build advertises the `knob:showy-v1` capability only when it contains the
    knob module and the `variant_policy` hook.
  - `settings_requirements(settings)` derives that capability from any `showy_*`
    setting.
  - `validate_showy_settings` requires a positive finite λ and an inline
    `drmc-showy-knob-v1` spec (at most 64 KB). Paths are refused, because
    workers never read local files.
- `drmc_rl/pool/store.py`:
  - `entrant_requirements` adds the derived capability, so a knob entrant is only
    scheduled to workers that can apply the knob.
  - `validate_entrant` refuses a knob entrant unless `requires` lists
    `knob:showy-v1` explicitly (see step 1 of the migration for why).
- `drmc_rl/eval/big_clear.py` follows `trainer/big-clear-showcase`: T1 at 27,
  plus `horizontal_lines`. `drmc_rl/pool/style.py` changes with it: the docstring
  and the human T1 reference (0.28 per 100 at the new bar). **This moves the
  pool report's T1 column.** Drop these two files from the merge if T1 must stay
  at 20.
- `tests/test_showy_knob.py`.

## Why an un-upgraded worker cannot play a knob entrant as plain

1. Workers must run the coordinator's source revision, or the coordinator rejects
   them (`PermissionError`). After the upgrade, old workers get no leases at all.
2. When the source check is relaxed (`allow_source_mismatch`), a worker's
   capabilities are the intersection of what it claims and what the coordinator
   has. Old code never claims `knob:showy-v1`, so the scheduler never offers it
   a knob pairing.
3. `requires: ["knob:showy-v1"]` is stored on the record. A coordinator that
   predates the derived rule still honours `requires`, and no old worker claims
   the capability, so the entrant is never scheduled at all. That leaves it
   idle rather than wrong.

## Migration

1. Merge this branch into `trainer/rating-pool`. Run the pool tests and
   `tests/test_showy_knob.py`.
2. Stop the coordinator and all workers. Deploy the new revision on mombox and
   restart the coordinator. Restart the workers on the same revision. The Mac
   uses `start-mac-workers.sh`; any other host restarts its own workers.
3. Register the entrants. Each one's inline model is `drmc_rl/style/models/showy_t2k4_v1.json`:

   ```
   python -m tools.rating_pool bootstrap knob-entrants.json
   # entrants: [{"id": "knob-champ-l1.5", "loader": "plain", "era": "style",
   #             "checkpoint": <champion record>, "requires": ["knob:showy-v1"],
   #             "settings": {"showy_lambda": 1.5, "showy_model": {...inline spec...}},
   #             "lineage": {"parent": "champion-retention-mixed-v2", "recipe": "showy-knob-v1"}}]
   ```

   The worker's policy cache is keyed by settings, so a knob entrant and its
   unbiased parent never share a loaded policy.
4. Check that the first knob games play differently from the parent (style
   counters T2+ ≈ 5× at λ = 1.5). A worker that does not advertise the knob
   shows "nothing to play" for knob pairings.

## Cost

About 13 ms of CPython per decision (60 candidates) on the knob side. In
head-to-head runs, throughput went from 126 to 108–129 decisions/s.
