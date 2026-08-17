# Unified player roadmap

## Destination

The program stands up three validated products from one competitive core:

| Product | Strategic intelligence | Mechanical action set | Decoder |
| --- | --- | --- | --- |
| Unrestricted superhuman | strongest public-information policy plus validated joint-event search | every exact reachable placement and useful timing | maximize calibrated match value |
| Human-rate superhuman | same quality/search system | scripts inside a named elite human envelope | maximize match value; no intentional errors |
| Human trainer | same quality oracle | rating-appropriate human-feasible scripts | calibrated regret, then style, cadence, form, and plausible execution |

The machine-readable authority is `drmc_rl/program/program.yaml`. This document
explains stage ordering; it does not override gate status or launch recipes.
The exact current counterfactual instructions are in
`COUNTERFACTUAL_QUALITY_HANDOFF.md`.

## Completed scientific contracts

The following are now foundations, not open design questions:

- one competitive core serves all products;
- placement-level SMDP control remains primary pending the timing gate;
- public actor and privileged teacher state are separate contracts;
- PairState v2 has canonical native full-pair snapshot/restore support;
- native search stops immediately before a private reserve pill becomes
  visible and resumes only after an explicit reveal override;
- candidate completeness is measured and silent truncation is prohibited;
- V3 human/style/timing training is frozen: epoch 5 is the balanced teacher and
  epoch 6 the sharper imitation reference;
- G4 Strong League lineages are frozen continuation teachers and robustness
  anchors;
- search teaches the policy before it is allowed to control rollout behavior;
- product strength is competitive regret or constrained quality, never
  temperature, beam width, or random play;
- every promoted artifact is hash-addressed and gate-backed.

## Current gate — counterfactual quality

The first 512-state production-shaped pilot established important mechanics:

- exact restore of unique full-pair states;
- full legal-candidate enumeration with zero truncation;
- strict reveal-boundary stopping and explicit reveal continuation;
- bounded depth-2 search without node-budget exhaustion;
- calibrated frozen Strong League continuation in place of the diagnostic leaf
  heuristic.

It did **not** establish mature quality. The pilot treated each reveal as nine
independent ordered color pairs at probability `1/9`. The NES instead creates
the whole 128-entry reserve from a two-byte RNG. Reserve entries are
nonuniform and correlated with already visible pills. The pilot is retained as
mechanics evidence; its candidate values must not be used to open the gate.

### Corrected chance model

`PillReserveBelief` enumerates the uniform two-byte seed prior used by randomized
native resets, conditions it on every publicly observed falling/preview entry,
and predicts the next reveal. `BeliefNativePairSearchModel` still overwrites the
hidden native reserve byte before reveal, but assigns probability only from the
public posterior. Some nodes have fewer than nine supported outcomes.

A mature source bank stores `reserve_belief` on every row. A release records
chance model `nes-reserve-seed-belief-v1`. Independent one-ninth branching is
never promotion-eligible.

### Privileged continuation scope

The frozen G4 policies were trained with exact pending-attack scalars in their
`v1_vs` auxiliary vector. They are therefore privileged continuation teachers.
That is useful for label generation but is not a fair deployed actor. Every
release declares `privileged-pending-attack-continuation-v1`; the eventual G5
actor is trained and evaluated on public state only.

### Evidence required to open the gate

1. **Grouped draw-aware calibration.** Aggregate and member-specific Davidson
   links use equal total weight per game, grouped cross-fitting, natural draw
   evidence, and paired game-bootstrap improvement over the identity link.
2. **Balanced bank.** A frozen competitive rollout supplies an oversampled
   source, then deterministic quotas select 1,024–2,048 states across level,
   speed, and tactical stratum. The default 1,440-state plan uses 24 states per
   4 × 3 × 5 cell.
3. **Complete member-wise targets.** Every candidate exports aggregate W/D/L,
   all checkpoint-specific W/D/L values, weighted utility standard deviation,
   and weighted Jensen–Shannon disagreement.
4. **Beam convergence.** Identical releases at opponent beams 1, 4, and 8 are
   aligned by source and action. Beam 4 must converge to beam 8 under
   predeclared top-action, value, and policy thresholds.
5. **Direct V3 comparison.** At the observed human action, counterfactual W/D/L
   must improve over the frozen V3 bootstrap with paired whole-game confidence
   for Brier and log loss.
6. **Mechanical integrity.** Full candidate coverage, zero candidate
   truncation, zero node-budget exhaustion, complete public reserve history,
   immutable member/calibration hashes, and explicit information scope.

`tools.counterfactual_quality_gate` is the only promotion authority. A staged
or failed check is work to do, not a threshold to waive.

## Parallel gate — timing as an action

The strict native ABI supports forced locks at exact pair frames.
`earliest_lock_dominance.py` compares earliest and delayed valid locks and
separates:

```text
clock divergence
structural next-event divergence
value_delta(a, d) = Q(state, a, delay=d) - Q(state, a, earliest)
```

Dynamic divergence alone shows timing matters to the simulator. A hierarchical
placement-plus-timing action is adopted only if delayed options produce
meaningful continuation-value gains at a predeclared rate and magnitude. This
work can proceed in parallel with the counterfactual-quality gate.

## Next stage — mature competitive teacher

After the gate passes:

1. Fine-tune the V3 competitive head on counterfactual/search/outcome targets.
2. Keep human action cross-entropy solely on human/style outputs.
3. Preserve candidate W/D/L distribution, calibrated regret, tactical
   consequences, and uncertainty rather than distilling only top-1 choices.
4. Prioritize teacher/student disagreements and high-opportunity states.
5. Freeze the mature teacher release and its exact source/evidence bundle.

Human data remains responsible for human choices, style, timing, cadence, and
immediate tactical representation. It no longer defines competitive candidate
ranking.

## G5 representation bakeoff

Use a common parent, source release, seeds, opponent mixture, compute budget,
and arena schedule to compare:

1. root-only G5;
2. V3/counterfactual-distilled root-only G5;
3. G5 plus exact effect tokens;
4. G5 plus recurrent public event belief;
5. the combined treatment only after single-family effects are understood.

Required measurements:

- held-out candidate W/D/L and regret calibration;
- teacher/student policy and uncertainty error by tactical stratum;
- paired clean-start arena W/D/L and the full payoff matrix;
- decisions and candidates per second;
- strength per millisecond;
- zero candidate truncation in training and evaluation.

Promotion uses paired full-game evidence, never training return alone.

## Tactical curriculum and outcome population training

Build a versioned full-pair archive containing:

- near-clear and race conversion;
- attack conversion;
- imminent topout defense;
- incoming garbage;
- high-speed difficult reachability;
- style-divergent choices;
- teacher/student disagreement;
- exploiter-discovered failures.

Sampling becomes adaptive to value error, uncertainty, search disagreement,
learning progress, and active exploiters. Start states are curriculum data;
untouched clean-start matches remain the evaluation authority.

Outcome population training then uses a game-theoretic mixture with four roles:

- main;
- main exploiter;
- league exploiter;
- human/execution exploiter.

Dense tactical quantities may be auxiliary predictions, replay priorities, or
early curriculum signals, but the mature competitive objective is W/D/L.

## Joint-event search policy iteration

`JointEventSearch` and the native adapter support:

- one-side decisions;
- simultaneous joint actions;
- deterministic causal advancement;
- private reserve reveal boundaries with public-belief chance branching;
- expectation or minimax opponent backups;
- transposition caching;
- full root policy targets and calibrated W/D/L.

Search remains an offline teacher until same-weight paired evaluation beats the
unsearched policy with confidence. A later behavior-search phase must use an
explicit search-policy or off-policy algorithm rather than being described as
ordinary on-policy PPO.

Public deployment also requires replacing privileged continuation features or
showing that a public recurrent belief/student recovers their useful signal.

## PSRO and exploitability hardening

The PSRO-lite loop is:

1. refresh the empirical payoff matrix;
2. compute a regularized meta-strategy;
3. train best responses to the main and mixture;
4. add successful responses as immutable entrants;
5. promote only when mixture value improves and active-opponent regressions
   remain bounded.

Scalar rating remains useful for display. The pairwise payoff matrix and
worst-active-opponent result remain promotion authorities because the game may
be non-transitive.

## Human-rate and trainer products

### Human-rate superhuman

Fit signed operation profiles from raw replay-verified scripts. Profiles include
reaction, inter-edge, burst, overlap, correction, reversal, soft-drop, and
complexity statistics. Build constrained reachability and fine-tune the same
competitive core under the execution profile used for evaluation. Release
requires zero hard-profile violations, zero replay divergence, and a
superhuman result under the named profile.

### Human trainer

The trainer decodes the common quality oracle through:

- conditional empirical regret tails;
- context-residualized regret;
- rating-residualized style;
- slowly varying form;
- independent decision latency and motor cadence;
- profile-valid, intent-preserving execution;
- block-level Bayesian adaptive sparring.

Release requires monotone achieved strength, approximately even matched-rating
outcomes, style identifiability after controlling strength, held-out human
plausibility, cadence/error-distribution fidelity, and evidence that targeted
practice improves a held-out measure.

## Immediate order of work

1. Recollect natural Strong League outcomes with enough games and natural draws
   for grouped aggregate and member-specific calibration.
2. Generate a large frozen-mixture pair-state source carrying accumulated
   public reserve belief.
3. Fill the balanced 1,440-state quota bank without shortfall.
4. Generate complete member-wise releases at opponent beams 1, 4, and 8 using
   the corrected seed-posterior chance model.
5. Audit coverage, chance support, uncertainty, hashes, and search budgets.
6. Compare beam 4 against beam 8 and retain beam 1 as sensitivity evidence.
7. Export held-out observed-action V3 bootstrap W/D/L and run the paired
   whole-game comparison.
8. Run the executable `v3-counterfactual-quality` gate; do not start mature
   quality distillation until it passes.
9. Complete the timing-action gate in parallel.
10. Fine-tune/distill the mature teacher and run the matched G5 bakeoff.
11. Outcome-train and exploitability-harden the population.
12. Fit constrained execution and trainer systems only after the common quality
    core is frozen.
