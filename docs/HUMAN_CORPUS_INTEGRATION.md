# Human corpus integration

## Ownership

The corpus producer owns replay acquisition, legal/raw storage, frame-exact
re-emulation, segmentation, consent, and continuously estimated human ratings.
`drmc-rl` owns planner annotation, exact afterstates, pair-game semantics,
quality/search labels, human modeling, timing/execution profiles, and coaching.

Consumption is through immutable content-hashed releases mounted read-only.
Training code does not open the producer's live database or source blob store.
Core replay facts remain separate from derived planner/model annotations so new
models do not mutate historical evidence.

## Required decision facts

Each player decision needs:

- replay/game/player identity and source frame;
- decision-time visible bottle, current pill, preview, speed, and phase;
- visible opponent state and its exact age;
- chosen lock pose and lock frame;
- outcome and terminal cause;
- continuously interpolated rating and uncertainty;
- raw controller stream when execution/cadence consent permits.

Full-pair counterfactual work additionally needs a restorable strict pair-state
release. Hidden future RNG is never an actor feature.

## Derived releases

1. **Planner annotation:** every reachable action, minimum cost, chosen slack,
   tuck/complexity, and candidate completeness.
2. **Exact afterstates:** sparse resolved candidate deltas and immediate
   clear/topout/virus/attack consequences.
3. **Counterfactual pair labels:** W/D/L, uncertainty, win-logit regret, and
   search policy targets for every action.
4. **Execution profiles:** named cohort quantiles over reaction, edge, burst,
   chord, correction, hold, soft-drop, and complexity metrics.
5. **Style space:** player behavior residualized against rating before latent
   extraction.
6. **Evaluation suites:** immutable identity/time/player holdouts and tactical
   state strata.

Every release carries source release ID, tool commit, native/planner revisions,
schema, parameters, and hashes.

## Human model roles

V3 learns human choice/style, exact local consequences, and timing. Its mature
competitive head is fine-tuned from counterfactual pair labels rather than the
observed human action. Rating changes the regret distribution and cadence, not
competitive quality.

Consented live evidence is private, revocable, and analyzed separately from
public visibility. Coaching reports human typicality and competitive quality as
distinct axes.
