# Counterfactual quality handoff

This document is the operational continuation brief for the
`v3-counterfactual-quality` gate. It supplements `program.yaml`; it is not a
second roadmap.

## Frozen inputs

The following artifacts and decisions are immutable inputs to the next run:

- Native reveal/snapshot implementation: `drmario-native` commit
  `6cfba6bf793a28eb9e49a5f4f1fcf7c8dbfa0f47`.
- Counterfactual pilot implementation: `drmc-rl` commit
  `6f1ffeb6d4d6b6e5c05745bb094fef38eae7f625`.
- V3 balanced human/style/timing teacher: epoch 5.
- V3 sharper imitation reference: epoch 6. It is a comparison reference, not
  the mature competitive teacher.
- Frozen Strong League continuation mixture manifest SHA-256:
  `37d94a13be471637406953fef6f48b78a48b7f5f26b6452b2fe1f5c6820d74a6`.
- Pilot calibration SHA-256:
  `373fb77a6165c196180d4a4e0716e72f5288f1059a3b5e8cc01c016f5245f58e`.
- Mechanics-pilot aggregate identity:
  `a0868725a9f45629bcf2c95f1eb7307429913b6478d471bc6e662e81db5f0ba7`.

The 512-state pilot remains useful evidence that native restore, exact candidate
coverage, reveal stopping, reveal override, and bounded search execute reliably.
It is **not** a promotable quality release.

## Critical chance-model correction

Do not branch future pills as nine independent outcomes at probability `1/9`.

The NES generates the entire 128-entry reserve once from its two-byte RNG. In
`GameLogic::generatePillsReserve`, each entry depends on the preceding pill id
and the next RNG byte. Therefore:

- the unconditional distribution at a reserve index is not exactly uniform;
- future entries are correlated with already visible falling/preview pills;
- public observation history changes the posterior predictive distribution;
- some posterior reveal nodes have fewer than nine possible outcomes.

The native reveal ABI is still correct: it stops before reveal and overwrites
the selected reserve entry before resuming. Probability assignment belongs in
the search layer. Use:

- `drmc_rl.search.pill_belief.PillReserveBelief`;
- `drmc_rl.search.belief_native_pair.BeliefNativePairSearchModel`;
- chance model id `nes-reserve-seed-belief-v1`.

A source-bank row must carry `reserve_belief`, accumulated from every visible
falling/preview entry observed on its trajectory. A release produced from only
the current two visible pills is useful for sensitivity analysis but does not
pass the quality gate.

## Information scope

The frozen G4 continuation checkpoints consume exact pending-attack scalars in
the `v1_vs` auxiliary vector. That is privileged native state under the public
actor contract. Counterfactual releases using these checkpoints must declare:

```text
privileged-pending-attack-continuation-v1
```

This is acceptable for a teacher target. It is not acceptable as evidence that
a deployed public-information search agent is fair. Later G5/student training
must consume only public actor features, while the privileged teacher provides
targets.

## Required evidence bundle

Promotion is evaluated by:

```bash
uv run python -m tools.counterfactual_quality_gate \
  --audit "$AUDIT" \
  --calibration "$CALIBRATION" \
  --beam-sweep "$BEAM_SWEEP" \
  --bank-manifest "$BANK_MANIFEST" \
  --bootstrap "$BOOTSTRAP_COMPARISON" \
  --output runs/program/gates/v3-counterfactual-quality.json
```

The command exits `0` only when every check passes; a staged result exits `2`.
Do not edit the evidence JSON by hand.

### 1. Grouped W/D/L calibration

Collect substantially more natural games than the pilot, including natural
draw outcomes. Fit and evaluate by whole game:

```bash
uv run python -m tools.calibrate_strong_league_wdl \
  --mixture-manifest "$MIXTURE_MANIFEST" \
  --output "$CALIBRATION" \
  --games-per-stratum 32 \
  --pairs 16 \
  --folds 5 \
  --bootstrap-samples 4000 \
  --device cuda
```

This performs equal-total-weight-per-game fitting, grouped cross-fitting, and a
paired game bootstrap against the identity link. It also fits a separate
Davidson link for every frozen member, so member disagreement is expressed in a
common W/D/L space rather than raw value-head units.

Stop and expand collection if:

- there are no natural draw games;
- any cross-fit fold lacks independent games;
- the 95% paired bootstrap upper bound is not below zero for both Brier and log
  loss;
- member-specific calibration is materially worse or numerically unstable.

Do not manufacture draw labels. A separate draw-focused natural suite may be
added, but it must remain game-group separated and be identified in the
artifact.

### 2. Oversampled competitive state source

Generate at least several times the final bank size with the frozen mixture,
not random actions:

```bash
uv run python -m tools.build_pair_state_pilot \
  --output "$OVERSAMPLED_BANK" \
  --states 8000 \
  --mixture-manifest "$MIXTURE_MANIFEST" \
  --wdl-calibration "$CALIBRATION" \
  --device cuda
```

The tool records exact native checkpoints, legal candidates, tactical strata,
and the accumulated public reserve belief. Its manifest must say
`frozen-strong-league-mixture-argmax`; a random-action manifest is diagnostic
only.

### 3. Balanced 1,024–2,048-state bank

Select fixed quotas across level, speed, and tactical stratum:

```bash
uv run python -m tools.balance_pair_state_bank \
  --input "$OVERSAMPLED_BANK" \
  --output "$BALANCED_BANK" \
  --per-cell 24 \
  --rollout-policy frozen-strong-league-mixture-argmax \
  --rollout-policy-manifest-sha256 \
    37d94a13be471637406953fef6f48b78a48b7f5f26b6452b2fe1f5c6820d74a6
```

The default cross product is 4 levels × 3 speeds × 5 tactical strata × 24 =
1,440 states. The selector is content-stable and fails on quota shortfall.
Oversample missing cells rather than passing `--allow-shortfall` for a mature
release.

### 4. Member-wise beam 1/4/8 releases

Use the same bank, source identities, chance model, member calibrations, seed,
depth, and node budget. Only `opponent_beam` changes. Use full root coverage.
The adapter is:

```text
drmc_rl.search.strong_league_memberwise:frozen_strong_league_memberwise_factory
```

Representative beam-8 command:

```bash
uv run python -m tools.counterfactual_teacher \
  --input "$BALANCED_BANK" \
  --output "$RELEASE_B8" \
  --adapter \
    drmc_rl.search.strong_league_memberwise:frozen_strong_league_memberwise_factory \
  --depth-events 2 \
  --own-beam 512 \
  --opponent-beam 8 \
  --chance-beam 9 \
  --max-nodes 100000 \
  --mixture-manifest "$MIXTURE_MANIFEST" \
  --wdl-calibration "$CALIBRATION" \
  --continuation-mixture strong-league-mixture-v1 \
  --corpus-release "$CORPUS_RELEASE" \
  --native-revision 6cfba6bf793a28eb9e49a5f4f1fcf7c8dbfa0f47 \
  --planner-revision "$PLANNER_REVISION" \
  --device cuda \
  --resume
```

Repeat with beams 1 and 4. Distributed shards are acceptable, but each beam
must have a complete, non-overlapping shard set with identical non-shard
settings.

Audit each release:

```bash
uv run python -m tools.audit_counterfactual_pilot \
  "$RELEASE_B8"/manifest.json \
  --calibration "$CALIBRATION" \
  --source "$BALANCED_BANK" \
  --output "$AUDIT"
```

Then compare aligned releases:

```bash
uv run python -m tools.compare_counterfactual_releases \
  --release "1=$RELEASE_B1/manifest.json" \
  --release "4=$RELEASE_B4/manifest.json" \
  --release "8=$RELEASE_B8/manifest.json" \
  --reference-beam 8 \
  --output "$BEAM_SWEEP"
```

Default convergence requirements for beam 4 versus beam 8 are:

- best-action agreement at least 95%;
- p95 state-wise maximum candidate win-probability delta at most 0.02;
- p95 root-policy Jensen–Shannon divergence at most 0.01.

Beam 1 is diagnostic. If beam 4 does not converge to beam 8, raise the mature
teacher beam; do not relax the threshold based on runtime inconvenience.

### 5. Direct observed-action/V3 comparison

Export held-out rows keyed by counterfactual `source_id`:

```json
{
  "source_id": "...",
  "game_id": "held-out-game-id",
  "outcome": "win",
  "observed_action": 117,
  "baseline_wdl": [0.41, 0.08, 0.51],
  "stratum": ["10", "2", "midgame"]
}
```

`baseline_wdl` must come from the frozen observed-action/V3 bootstrap without
using the new counterfactual labels. Compare it at the same observed action:

```bash
uv run python -m tools.compare_counterfactual_bootstrap \
  "$RELEASE_B8"/manifest.json \
  --bootstrap "$V3_BOOTSTRAP_ROWS" \
  --bootstrap-samples 4000 \
  --output "$BOOTSTRAP_COMPARISON"
```

Promotion requires the counterfactual-minus-V3 paired game-bootstrap 95% upper
bound below zero for both Brier and log loss. Decision-row significance is not
sufficient.

## Member uncertainty

A mature label uses schema `drmc-counterfactual-pair-label-v3`. Every candidate
contains:

- weighted aggregate W/D/L;
- `member_wdl` in manifest member order;
- weighted utility standard deviation in `uncertainty`;
- weighted Jensen–Shannon divergence in `uncertainty_js`.

The audit rejects missing, nonfinite, or incomplete member values. Preserve the
mixture member ids and weights in every row; do not replace the frozen mixture
with an unrecorded checkpoint selection.

## Stop conditions and escalation

Stop a run and fix the cause when any of the following occurs:

- candidate coverage differs from the source bank;
- candidate packing truncates or silently drops an action;
- any search reports budget exhaustion;
- a reveal branch uses the hidden reserve value or independent `1/9` mass;
- source belief observations are contradictory or impossible;
- a member checkpoint hash or calibration hash differs from its manifest;
- shard settings differ beyond shard index;
- beam comparison source/action sets differ;
- draw calibration is unidentified;
- a privileged continuation release is presented as deployable public search.

Do not proceed to G5 quality distillation merely because the mechanics pilot
was large or clean. Proceed only after the executable gate passes and the gate
artifact is stored at the authority path in `program.yaml`.

## After the gate

Once the mature release passes:

1. Train the V3 competitive head only from counterfactual/search/outcome
   targets; retain human cross-entropy on the human/style head.
2. Distill W/D/L distribution, policy targets, tactical consequences, and
   uncertainty into the matched G5 variants.
3. Run the root-only, V3-distilled, exact-effect-token, and recurrent-event G5
   bakeoff under common seeds, parent, compute, and opponent mixture.
4. Keep joint search as an offline teacher until same-weight clean-start paired
   evaluation passes its own gate.
