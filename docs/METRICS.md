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
