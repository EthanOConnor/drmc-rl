# Roadmap

## Objective

Build one strong Dr. Mario decision system, then derive calibrated human
opponents and coaching from it. Tournament outcomes are real comparative
evidence. Labels such as source rating band are hypotheses until play verifies
them, and field-relative Elo is not Fightcade Elo until identified human games
anchor it.

## Current program

1. **Exact alternatives.** For every sampled Fightcade decision, simulate every
   reachable placement through the native engine. Store exact sparse afterstate
   deltas and immediate clear, top-out, virus, attack, and duration targets.
2. **Afterstate pretraining.** Encode the root and opponent bottles once, then
   score exact candidate deltas. Jointly learn rating-independent competitive
   value, tactical outcomes, human style, game outcome, and execution timing.
3. **Human strength calibration.** Measure each observed human choice's regret
   against its legal alternatives. Fit monotone conditional regret quantiles by
   rating and decision opportunity, preserving rare error tails when typical
   placements are identical. Requested strength changes planning reliability,
   mistake cost, and timing; it does not merely perturb placement predictions.
4. **Policy improvement.** Search from the afterstate value model, retain
   improved targets, and distill them into the fast competitive head. Increase
   depth only when strength per millisecond improves.
5. **Outcome self-play.** Fine-tune on actual clears, attacks, top-outs, and
   wins against human anchors plus frozen historical lineage. Search supplies
   improvement targets; PFSP prevents narrow or cyclic promotion.
6. **Deployment.** Professor Pills remains a thin non-blocking host. The V3
   backend exposes exact-afterstate quality, calibrated regret, timing, health,
   and coach explanations. Consented identified games calibrate the internal
   ladder to real human strength.

## Gates

- Offline: held-out action NLL, outcome Brier score, tactical calibration,
  low-versus-high-rating regret-tail separation, and identity/time holdouts.
- Full games: W/L/D, clear and top-out rates, pills and wall time per game, and
  pairwise matrices—not Elo alone.
- Promotion: beat the current champion and a representative frozen lineage with
  confidence; do not remove historical entrants.
- Strength dial: adjacent requested levels must be ordered across diverse
  opponents and difficult-position strata. Real players provide the absolute
  anchor.
- Performance: report decisions/sec, candidate afterstates/sec, inference
  latency, search nodes, and strength gained per millisecond.
