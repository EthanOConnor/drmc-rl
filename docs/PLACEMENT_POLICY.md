# Placement policy and training

## Fixed SMDP contract

One inference occurs at each pill spawn. The action is an exact reachable final
pose; the transition lasts `tau` frames and returns the resolved next decision
state. PPO uses `gamma ** tau` in deltas, GAE, and returns.

The timing-action gate may add a small hierarchical execution/timing choice. It
will not replace the planner with frame-level policy exploration.

## Candidate policy

The production path scores packed feasible candidates. Candidate order is
stable and deterministic; cost-to-lock is an explicit feature. G5 adds:

- shared own/opponent bottle encoder;
- pill/context-conditioned residual processing;
- cross-bottle column interaction;
- candidate-set attention;
- distributional value.

Exact effect tokens are available for the matched representation bakeoff. They
summarize deterministic resolved candidate changes and tactical consequences.

## Candidate completeness

The 128-slot implementation cap is an optimization, not a semantic limit.
Training and evaluation must log the actual legal count and fail/expand when it
exceeds the configured width. Silently keeping the cheapest candidates is not
acceptable for promoted artifacts.

## Teacher hierarchy

- V3 exact-afterstate teacher: human style, timing, immediate tactics, initial
  quality prior.
- Frozen G4: long-horizon bootstrap and robustness baseline.
- Full-pair counterfactual teacher: W/D/L and win-logit regret for every action.
- Joint-event search: improved policy/value targets after its gate opens.

Human requested rating never enters rating-independent quality or tactical
heads.

## Training sequence

1. offline V3/G4 structural initialization;
2. matched G5 representation bakeoff;
3. outcome PPO against the population mixture;
4. counterfactual/search distillation on disagreement and high-opportunity
   states;
5. exploiter hardening;
6. constrained-policy fine-tuning for named execution profiles.

Search remains an auxiliary target until same-weight evaluation validates it.

## Objective

Final competitive value is W/D/L. Clear, topout, virus, attack, and timing are
auxiliary heads/curriculum signals. Shaping is annealed or potential-based and
cannot compensate for a loss.

## Metrics

Report:

- decisions/s and candidates/s;
- candidate count and dropped count;
- policy KL/entropy/clip fraction;
- W/D/L calibration and Brier score;
- clean-start arena payoff matrix;
- effect-token and event-state ablations;
- teacher disagreement;
- strength gained per millisecond.
