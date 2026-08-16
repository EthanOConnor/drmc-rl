# Unified player roadmap

## Destination

The program stands up three validated products from one competitive core:

| Product | Strategic intelligence | Mechanical action set | Decoder |
| --- | --- | --- | --- |
| Unrestricted superhuman | strongest policy plus validated joint-event search | every exact reachable placement and useful timing | maximize calibrated match value |
| Human-rate superhuman | same quality/search system | scripts inside a named elite human envelope | maximize match value; no intentional errors |
| Human trainer | same quality oracle | rating-appropriate human-feasible scripts | calibrated regret, then style, cadence, form, and plausible execution |

The machine-readable authority is `drmc_rl/program/program.yaml`. This document
explains the stage ordering; it does not override gate status or launch recipes.

## Phase 0 — scientific contracts

Freeze before successor campaigns become authoritative:

- public versus privileged pair state;
- whether lock timing is part of the action;
- candidate-completeness policy;
- human execution profiles;
- immutable benchmark seeds and state suites;
- superhuman and trainer release criteria;
- artifact identity and gate evidence.

Implemented foundations:

- `PublicPairState` and `PrivilegedPairState` v2 data contracts;
- recursive public-state hidden-field audit;
- execution-profile metrics and Pareto filtering;
- program registry, gated launcher, evidence, and artifact manifests;
- PR-level pure-Python CI.

Open work:

- bind full public/privileged state extraction to the native pair engine;
- freeze golden serialized fixtures and emulator-derived no-leak evidence.

## Phase 1 — correctness and causal fidelity

Expand parity beyond settled boards:

- exact garbage frame, color, and columns;
- simultaneous clear/topout ordering;
- strict versus throughput advancement;
- earliest versus delayed lock;
- live script replay and desynchronization;
- candidate completeness;
- public-state observability.

### Timing-action gate

The strict native ABI already supports forced locks at exact pair frames.
`vs_forced.py` exposes it and `earliest_lock_dominance.py` runs versioned JSONL
probe banks. The report separates mere clock divergence, structural next-event
divergence, and competitive value improvement:

```text
transition_changed(a, d)
value_delta(a, d) = Q(state, a, delay=d) - Q(state, a, earliest)
```

A nonzero transition difference proves timing is dynamically material. An
explicit timing head is adopted only if delayed options also produce meaningful
continuation-value gains at a predeclared frequency/magnitude.

## Phase 2 — exact human and counterfactual teachers

Continue V3 exact-afterstate learning for human style, immediate tactics, and
cadence. Then replace its mature competitive ranking with full-pair
counterfactual labels for **every** legal candidate:

1. restore a complete pair state;
2. apply the candidate and timing option;
3. advance causally to the next event;
4. evaluate continuation against the population mixture;
5. store W/D/L, expected score, uncertainty, tactical consequences, and
   win-logit regret.

Human action cross-entropy remains on the human/style head only. The
counterfactual release is immutable and keyed by source corpus, engine, planner,
teacher, and state schema.

Implemented foundations:

- backend-independent full-candidate counterfactual teacher;
- W/D/L and win-logit-regret target schema;
- adapter-driven release CLI that refuses to assume a legacy simulator.

Open work:

- native full-pair checkpoint adapter;
- continuation policy-mixture adapter;
- candidate tactical-target export and V3 fine-tuning.

## Phase 3 — G5 bakeoff and distillation

Use matched compute, parents, seeds, and opponents to compare:

1. root-only G5;
2. V3-distilled G5;
3. G5 plus exact effect tokens;
4. recurrent/event-state G5.

Teachers have distinct roles:

- V3: exact afterstates, human structure, timing;
- strongest frozen G4: long-horizon policy/value bootstrap;
- joint-event search: policy improvement.

Implemented foundations:

- V3-to-V5 distillation path already present;
- deterministic exact-effect-token builder and projection block;
- compact recurrent public-event belief encoder for the event-state arm;
- hard candidate-coverage evidence and no-truncation guard;
- machine-readable bakeoff gate and artifact identity.

Required measurements:

- held-out candidate W/D/L calibration;
- paired arena W/D/L and full payoff matrix;
- decisions per second, candidates per second, and strength per millisecond;
- candidate truncation count, required to be zero;
- teacher/student disagreement by decision opportunity.

The current G4 Strong League remains active as a robustness baseline and
teacher while G5 matures.

## Phase 4 — tactical state curriculum

Build a versioned full-pair archive with:

- near-clear conversion;
- attack conversion;
- imminent topout defense;
- incoming garbage;
- race finishes;
- high-speed difficult reachability;
- style-divergent choices;
- exploiter-discovered failures.

Sampling becomes adaptive to value error, search disagreement, learning
progress, and active exploiters. Start states remain curriculum data; untouched
clean-start matches are always the evaluation authority.

## Phase 5 — outcome population training

Train the main against a game-theoretic mixture. Tactical signals may shorten
credit assignment early but anneal toward pure W/D/L. Maintain distinct roles:

- main;
- main exploiter;
- league exploiter;
- human/execution exploiter.

The arena retains every promoted lineage and human/style anchor. It publishes
both scalar ratings and the pairwise payoff structure.

## Phase 6 — joint-event search policy iteration

`JointEventSearch` supports:

- one-side decisions;
- simultaneous joint actions;
- deterministic advancement;
- newly revealed chance events;
- expectation or minimax opponent backups;
- transposition caching;
- root policy targets and calibrated W/D/L.

The remaining heavy dependency is a fast restorable native full-pair adapter.
Search first generates offline targets. It controls live or rollout behavior
only after same-weight paired evaluation opens the gate. If search later
becomes behavior during learning, use an explicitly search-policy or off-policy
algorithm rather than calling it ordinary on-policy PPO.

## Phase 7 — PSRO and exploitability hardening

The PSRO-lite loop is:

1. fit/refresh the empirical payoff matrix;
2. compute a regularized meta-strategy;
3. train approximate best responses to the main and mixture;
4. add successful responses as immutable entrants;
5. promote only when mixture value improves and active-opponent regressions
   stay inside limits.

`meta_strategy.py` implements entropy-regularized multiplicative-weights solving,
a population mixture, saddle-gap/exploitability evidence, and active-regression
reporting. The next integration step reads the arena database directly and
writes opponent-pool mixture weights.

## Phase 8 — execution and trainer products

### Human-rate superhuman

Fit signed profiles from raw input scripts. The profile includes reaction,
inter-edge, burst, overlap, correction, reversal, soft-drop, and complexity
statistics. Build constrained reachability and fine-tune the competitive policy
under the same profile used for evaluation. Release requires zero hard-profile
violations and replay divergence.

Implemented foundations:

- script metric extraction;
- quantile profile fitting and validation CLI;
- Pareto frontier over timing, edges, bursts, reversals, and complexity;
- constrained quality decoder.

### Human trainer

The trainer does not clone checkpoints at requested ratings. It uses:

- counterfactual quality;
- conditional empirical regret tails;
- context-residualized regret adjustment;
- rating-residualized style latent;
- slowly varying form;
- independent cadence;
- profile-valid execution;
- block-level Bayesian adaptive sparring.

Implemented foundations:

- unified product decoder;
- contextual regret multiplier fitting;
- rating-residualized player style space;
- temporal form state;
- adaptive sparring controller.

Open work:

- train the context and style systems from immutable corpus releases;
- model intent-preserving late corrections and motor failures;
- integrate decoder output into the backend protocol;
- run held-out human plausibility and pedagogical crossover studies.

## Immediate order of work

1. Continue and landmark the active G4 Strong League run.
2. Build a stratified strict timing-probe bank and complete the timing gate.
3. Bind native/emulator evidence to PairState v2 and open its gate.
4. Complete exact afterstate annotation/training and freeze V3 teachers.
5. Run the matched G5 bakeoff, including effect tokens and event state.
6. Add native full-pair checkpoint/restore and bind joint-event search.
7. Generate counterfactual labels and remove observed-action supervision from
   mature competitive quality.
8. Distill search/counterfactual teachers into G5 and outcome-train the
   population.
9. Integrate the PSRO mixture with arena scheduling/opponent sampling.
10. Fit elite execution profiles, constrained reachability, style, context,
    cadence, and trainer calibration.
11. Conduct predeclared top-human, human-rate, plausibility, and pedagogy gates.
