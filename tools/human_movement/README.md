# Human movement vs rating

Analysis of the Fightcade human corpus (fightcadeRatings docs/HUMAN_CORPUS.md) run on
24 Sep 2026; results are published as "How humans move, by rating".

1. `sample_rows.py ROOT PER_MONTH DEST` (on mombox, fightcadeRatings venv): samples placements
   with complete inputs from every monthly release under ROOT/releases/.
2. `slack.py SAMPLE.parquet OUT.parquet WORKERS` (drmc-rl venv, native planner built): replays
   each placement with the audited FBNeo alignment (held byte before spawn, raw[:-1], parity
   xor 1), keeps exact recorded-lock matches, and adds the fastest route to the same pose plus
   path metrics (sideways travel, reversals, overshoot, rotation changes, late moves, pauses).
3. `timing_report.py OUT.parquet RATINGS.parquet REPORT.json`: rating-band tables per speed,
   regressions with geometry controls, player profiles, and style clusters.

`timing_stats.py` is the earlier timing-only version. `report-2026-09-24.json` is the output
for the 1,167,065-placement sample (25,000 per month, 49 months).
Script paths inside slack.py assume the drmc-rl checkout at /Users/ethan/dev/drmario/drmc-rl.
