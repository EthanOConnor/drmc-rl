# Evaluation and release gates

## Evidence hierarchy

From strongest to weakest:

1. predeclared human matches and independent emulator/script verification;
2. clean-start full native games against a permanent diverse population;
3. paired counterfactual/search evaluation on immutable state suites;
4. held-out corpus calibration and behavior metrics;
5. training return and auxiliary losses.

A lower tier may diagnose or select experiments. It cannot replace a required
higher-tier release gate.

## Common protocol

Every comparative result records:

- immutable agent manifests;
- exact engine/planner revisions;
- side alternation;
- seed and start-state identity;
- clean-start versus curriculum-start flag;
- execution profile;
- search settings and deadlines;
- game horizon and treatment of truncation;
- full W/D/L counts and pairwise matrix;
- uncertainty interval and stopping rule.

Report clear wins, topout wins, draws, horizon truncations, pills, pair-clock
frames, garbage, virus progress, and invalid/candidate-drop counts. Elo is a
summary, never the only result.

## Correctness gates

### Planner

- v4/CUDA feasible masks and costs match `drm_reach_bfs_full` on fuzz and
  recorded states;
- scripts replay to the predicted pose;
- zero unreported candidate truncation;
- same-color canonicalization preserves equivalent actions.

### Pair engine

- native versus emulator boards, counters, volleys, and terminal ordering;
- exact garbage frame/color/columns;
- same-frame clear/topout behavior;
- strict forced-lock advancement;
- throughput approximation differences measured explicitly.

### Public information

- serialized public fixtures contain no forbidden hidden keys;
- every actor field has an observation/inference provenance;
- future RNG is absent from actor, teacher labels, and claimed live search;
- privileged consumers are named and tested.

## Timing-action gate

Report by state stratum:

- probes and valid delays;
- fraction with different next-event state;
- `Q(delay) - Q(earliest)` distribution;
- uncertainty and continuation policy mixture;
- beneficial-delay frequency above a predeclared epsilon.

A transition difference proves timing is dynamically relevant. Adopt a timing
action only when competitive gains justify added policy and planner complexity.

## Competitive-model gates

### Offline

- held-out candidate policy cross-entropy/KL;
- W/D/L Brier score and reliability curves;
- value calibration by game phase and pressure;
- tactical-head calibration;
- teacher disagreement and student error by opportunity;
- identity, player, replay, and time holdouts;
- candidate coverage and effect-token ablation.

### Full games

- W/D/L and credible intervals;
- pairwise payoff matrix;
- probability of being best and rank interval;
- clean-start and tactical-suite results;
- worst active-opponent result;
- side/seed split;
- strength gained per millisecond and decision throughput.

### Promotion

A candidate must:

- beat or improve on the current main under the declared sequential test;
- improve expected value against the PSRO mixture;
- remain within the regression bound against every permanent anchor and active
  exploiter;
- show no objective collapse, candidate loss, side bias, or horizon exploit;
- carry a complete artifact manifest.

## Search gate

Compare identical weights with and without strict joint-event search on paired
states/seeds. Report:

- root action agreement and overrides;
- W/D/L improvement;
- search depth completed;
- nodes, cache hits, opponent/chance coverage;
- deadline p50/p95/p99 and misses;
- public versus privileged input mode;
- approximation fallbacks.

Search remains a teacher until improvement is statistically credible and the
public-information path meets its deadline. If search behavior enters training,
record the exact behavior policy and correction algorithm.

## Population gate

Report:

- connected payoff graph;
- regularized meta-strategy;
- saddle gap/exploitability estimate;
- best-response identities and values;
- role coverage: main, main exploiter, league exploiter, human/execution
  exploiter;
- mixture-value change and worst-matchup change at promotion.

A scalar rating cannot hide a cyclic or catastrophic matchup.

## Unrestricted superhuman release

Predeclare a top-human cohort, match format, side/seed protocol, and stopping
rule. Require:

- lower 95% credible match-win bound above 50%;
- clean starts and broad seeds;
- no hidden-information advantage;
- exact live scripts and negligible deadline/desync failure;
- no catastrophic active-exploiter matchup;
- independent audit of the released artifact identity.

Wins over self-play checkpoints or human imitation alone do not establish this
claim.

## Human-rate superhuman release

All unrestricted gates apply, plus:

- named signed execution profile;
- zero hard-profile violations;
- zero invalid/replay-divergent scripts;
- reaction, edge, burst, chord, correction, and soft-drop distributions reported
  against held-out profile data;
- superhuman result achieved while the profile is active.

Do not describe average-APM matching as a complete human-rate constraint.

## Human trainer release

### Strength

- requested rating versus achieved arena strength is monotone;
- adjacent levels remain ordered across opponents and difficult-state strata;
- matched-rating human outcomes are approximately even;
- regret tails, not only means, match held-out players.

### Style

- style is identifiable after conditioning on strength;
- changing style does not materially change achieved rating;
- attack/clear/safety/setup/tuck/tempo axes have observable behavioral effects;
- named styles generalize to held-out players and states.

### Cadence and motor behavior

- decision latency and motor execution are evaluated separately;
- burst, holds, corrections, soft drop, and lock slack match held-out data;
- temporal error/form autocorrelation is realistic;
- mechanical errors preserve recognizable intent rather than becoming random
  placements.

### Coaching and pedagogy

- human typicality and competitive quality remain separate outputs;
- explanations are derived from exact afterstates/events;
- human players judge styles distinct and plausible;
- a small predeclared crossover study shows targeted practice improves a
  held-out performance measure before broader claims.

## Performance reporting

Always publish:

- simulated frames/s and decisions/s;
- policy candidates/s;
- planner states/s;
- training update time;
- search nodes and decisions/s;
- inference and end-to-end plan latency quantiles;
- memory and accelerator utilization;
- strength gain per wall-clock hour and per millisecond at inference.

## Executable release evidence

`drmc_rl.eval.release_gates` and `tools.release_gate` turn predeclared JSON
counts into reproducible competitive, execution, or trainer pass/fail evidence.
For the strict superhuman claim, draws are non-wins in the posterior win
probability; side coverage and the worst active-opponent payoff are separate
gates. Human-rate evidence additionally requires zero profile violations and
replay divergences. Trainer evidence checks monotonicity, matched-rating score,
and style-strength leakage.

```bash
uv run python -m tools.release_gate competitive \
  --input evidence/top-human-matches.json \
  --output runs/program/gates/unrestricted-superhuman.json
```
