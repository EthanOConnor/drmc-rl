# Metrics

## Correctness

- planner oracle mismatches and script replay mismatches;
- candidate legal count, packed width, and dropped count;
- pair-engine/emulator divergence;
- public-state hidden-field audit failures;
- invalid actions, desyncs, deadline misses, and profile violations.

## Competitive

- W/D/L and credible intervals;
- pairwise payoff matrix and active-opponent minima;
- clear/topout/horizon terminal causes;
- PSRO mixture value, best responses, and saddle gap;
- clean-start versus curriculum-start results;
- side and seed split.

## Model

- policy target KL/top-k agreement;
- W/D/L Brier/reliability;
- tactical-head calibration;
- teacher disagreement and student error by opportunity;
- recurrent/effect-token ablations;
- value uncertainty.

## Human trainer

- achieved strength versus requested rating;
- regret quantiles/tails by context;
- style identifiability and strength leakage;
- decision latency and motor metrics separately;
- burst, edge, chord, correction, soft-drop, and complexity distributions;
- temporal form/error autocorrelation;
- matched-human outcomes and pedagogy measures.

## Performance

- simulated frames/s and decisions/s;
- candidates/s and planner states/s;
- update wall time and accelerator utilization;
- inference/search latency p50/p95/p99;
- search nodes, depth, cache hits, and coverage;
- strength gain per wall-clock hour and per inference millisecond.

## Skill grade

`tools/skill_grade.py` regresses fightcadeRatings crown metrics onto WHR-C.
Every input describes what the graded side did: volleys sent per minute
(`cpm`), mean garbage pieces per sent volley (`cur`, 0 with no volleys),
ending speed value (`spd`), SALT seconds inflicted per minute
(`salt_per_min`), pills per minute and garbage pieces sent per minute.
`skill_game_features` in `drmario_vs_vec.py` computes the PPO-logged inputs
with those definitions. `skill_history.jsonl` rows without
`"skill_features": 2` fed garbage received as SALT and an all-clear 0/1 as
CUR, so their grades are invalid.

SALT per volley is 16 × n frames, n = the most rows any piece of the volley
falls: the first occupied row below the spawn row in rows 1–15, else 16.
Seconds are frames / 60.0988, as in the fit data. The measured ROM cost of a
volley is 16 × depth + 4 frames; the established definition leaves out that
fixed 4-frame overhead, and we keep it.
