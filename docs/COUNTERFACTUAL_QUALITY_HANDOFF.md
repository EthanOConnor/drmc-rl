# Counterfactual quality handoff

This is the operational continuation brief for the `v3-counterfactual-quality`
gate. `drmc_rl/program/program.yaml` remains the machine-readable authority.

## Scientific status

The first 512-state pilot established useful mechanics evidence:

- native full-pair snapshot/restore works;
- reveal stopping and explicit reveal injection work;
- every legal root candidate can be labeled without truncation;
- the reported depth-2 pilot completed without node-budget exhaustion;
- a frozen Strong League continuation mixture is materially better than the
  former diagnostic leaf heuristic.

It is **not** a promotable competitive-quality release. The pilot used
independent `1/9` reveal mass, had only nine held-out calibration games and no
natural draws, exported no member-wise uncertainty, used a pressure-heavy
source bank, and searched only one opponent continuation.

The next run must replace those limitations rather than merely scale the same
pilot.

## Frozen inputs

- `drmario-native` reveal/snapshot implementation:
  `6cfba6bf793a28eb9e49a5f4f1fcf7c8dbfa0f47`.
- Initial `drmc-rl` mechanics pilot:
  `6f1ffeb6d4d6b6e5c05745bb094fef38eae7f625`.
- Balanced V3 human/style/timing teacher: epoch 5.
- Sharper V3 imitation reference: epoch 6. It is a comparison reference, not
  the mature competitive ranking.
- Frozen Strong League continuation mixture manifest SHA-256:
  `37d94a13be471637406953fef6f48b78a48b7f5f26b6452b2fe1f5c6820d74a6`.
- Pilot calibration SHA-256:
  `373fb77a6165c196180d4a4e0716e72f5288f1059a3b5e8cc01c016f5245f58e`.
- Mechanics-pilot aggregate identity:
  `a0868725a9f45629bcf2c95f1eb7307429913b6478d471bc6e662e81db5f0ba7`.

The last two pilot identities are historical evidence only. A mature release
must use newly generated, content-addressed artifacts.

## Non-negotiable information contracts

### Reserve chance

The native game generates the complete 128-pill reserve once from its two-byte
RNG, then generates the publicly visible initial virus bottle from the same
stream. Later pills are correlated with that bottle and with publicly visible
falling and preview pills. Do not use independent uniform ordered-pair
branching. The current experiment declares the uniform two-byte reset-seed
prior used by randomized native resets; do not call it the retail boot prior.

Use:

- `drmc_rl.search.pill_belief.PillReserveBelief`;
- `drmc_rl.search.belief_native_pair.BeliefNativePairSearchModel`;
- chance model id `nes-reserve-public-seed-belief-v2`.

Every bank row carries the complete public observation history accumulated on
its trajectory. At a reveal node, branch over every posterior-supported pill;
`chance_beam` must be at least nine so no supported outcome is pruned.

### Continuation information

The frozen G4 checkpoints consume exact pending-attack scalars in `v1_vs`.
Releases using them therefore declare:

```text
privileged-pending-attack-continuation-v1
```

That is valid for an offline teacher. It is not evidence that a deployed public
search agent is fair. Students and deployed actors remain public-state-only.

## Default promotion thresholds

`tools.counterfactual_quality_gate` now fails closed on one coherent evidence
bundle. Defaults are intentionally stronger than the mechanics pilot:

- 1,024–2,048 balanced source states;
- 12 level/speed calibration cells with at least 16 independent games each
  (192 games minimum; the recommended collection is 32 per cell = 384);
- at least eight naturally drawn calibration games;
- at least five whole-game cross-fit folds, with draws represented in every
  training fold;
- full root candidate coverage, zero candidate truncation, zero node-budget
  exhaustion, and `chance_beam >= 9`;
- every candidate has every frozen member W/D/L plus finite utility standard
  deviation and Jensen–Shannon disagreement;
- beam 4 versus beam 8: aggregate top-1 agreement at least 95%, p95 maximum
  candidate win delta at most 0.02, and p95 policy JS at most 0.01;
- every tactical cell: top-1 agreement at least 85%, win-delta p95 at most
  0.04, and policy-JS p95 at most 0.02;
- the beam-8 release beats the frozen observed-action/V3 bootstrap on Brier and
  log loss with paired whole-game 95% confidence intervals entirely below
  zero;
- at least 48 independent bootstrap-comparison games, including a natural
  draw;
- clean committed code and exact cross-artifact hash agreement.

Do not relax thresholds after seeing results. A changed threshold is a new,
versioned experiment with a written rationale.

## Recommended artifact layout

Keep all generated data outside Git:

```text
runs/counterfactual-quality-v2/
  calibration/
    strong-league-wdl-v3.json
  source/
    oversampled.jsonl.gz
    oversampled.jsonl.gz.manifest.json
  bank/
    balanced-1440.jsonl.gz
    balanced-1440.jsonl.gz.manifest.json
  release-b1/
  release-b4/
  release-b8/
  audit-b8.json
  beam-sweep.json
  v3-bootstrap-comparison.json
  gate.json
```

Copy or symlink frozen checkpoints into an operator-controlled artifact area;
do not commit checkpoints, corpora, state banks, or run output.

## Execution sequence

Set variables once:

```bash
ROOT=runs/counterfactual-quality-v2
MIXTURE_MANIFEST=/absolute/path/to/strong-league-mixture-v1/manifest.json
CALIBRATION="$ROOT/calibration/strong-league-wdl-v3.json"
OVERSAMPLED_BANK="$ROOT/source/oversampled.jsonl.gz"
OVERSAMPLED_MANIFEST="$OVERSAMPLED_BANK.manifest.json"
BALANCED_BANK="$ROOT/bank/balanced-1440.jsonl.gz"
BANK_MANIFEST="$BALANCED_BANK.manifest.json"
CORPUS_RELEASE=<immutable-corpus-release-id>
PLANNER_REVISION=<exact-reach-planner-revision>
V3_BOOTSTRAP_ROWS=/absolute/path/to/heldout-v3-bootstrap.jsonl.gz
mkdir -p "$ROOT"/{calibration,source,bank,release-b1,release-b4,release-b8}
```

### 1. Collect and fit draw-aware W/D/L calibration

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

The artifact uses equal total weight per game, deterministic whole-game
cross-fitting, paired game bootstrap, and a separate positive-slope Davidson
link for each frozen member.

Stop and collect more games when:

- fewer than eight natural draw games were observed;
- any level/speed cell has fewer than 16 completed games;
- any training fold lacks natural draw evidence;
- either aggregate Brier or log-loss paired CI reaches zero;
- any member link is non-finite, non-positive-slope, or draw-unidentified.

Do not synthesize draw labels. A draw-focused suite may oversample conditions
that naturally draw, but each game remains independently simulated, clearly
identified, and held out by game.

### 2. Roll complete games into an oversampled source

```bash
uv run python -m tools.build_pair_state_pilot \
  --output "$OVERSAMPLED_BANK" \
  --states 8000 \
  --states-per-game 16 \
  --max-decisions-per-game 320 \
  --max-games 4000 \
  --mixture-manifest "$MIXTURE_MANIFEST" \
  --wdl-calibration "$CALIBRATION" \
  --device cuda
```

The collector now rolls a complete bounded game before selecting rows. It
round-robins across globally underrepresented tactical strata and classifies
stack pressure from player-created pill material, not high initial viruses.
The source manifest must report:

```text
rollout_policy = frozen-strong-league-mixture-argmax
per_game_selection = whole-game-global-tactical-round-robin-v1
diagnostic_only = false
chance_model = nes-reserve-public-seed-belief-v2
reserve_initial_board_conditioned = true
```

Inspect `tactical_counts`. If a quota cell remains rare, increase game count or
add a deterministic targeted natural start suite. Do not duplicate or relabel
states to fill a quota.

### 3. Select the balanced promotion bank

```bash
uv run python -m tools.balance_pair_state_bank \
  --input "$OVERSAMPLED_BANK" \
  --input-manifest "$OVERSAMPLED_MANIFEST" \
  --output "$BALANCED_BANK" \
  --per-cell 24
```

The default cross product is 4 levels × 3 speeds × 5 tactical strata × 24 =
1,440 states. The tool verifies the source artifact hash, chance model, rollout
mixture, diagnostic status, and whole-game sampling method. It then fails on
quota shortfall.

Do not use `--allow-shortfall`, `--allow-missing-reserve-belief`, or
`--allow-unverified-source` for promotion evidence. Those switches create a
manifest marked diagnostic.

### 4. Generate member-wise beam 1/4/8 releases

Use exactly the same bank, seed, adapter, depth, root beam, chance beam, node
budget, mixture manifest, calibration, native revision, and planner revision.
Only `--opponent-beam` and output directory change.

```bash
make_release () {
  beam="$1"
  output="$2"
  uv run python -m tools.counterfactual_teacher \
    --input "$BALANCED_BANK" \
    --output "$output" \
    --adapter \
      drmc_rl.search.strong_league_memberwise:frozen_strong_league_memberwise_factory \
    --depth-events 2 \
    --own-beam 512 \
    --opponent-beam "$beam" \
    --chance-beam 9 \
    --max-nodes 100000 \
    --seed 20260817 \
    --mixture-manifest "$MIXTURE_MANIFEST" \
    --wdl-calibration "$CALIBRATION" \
    --continuation-mixture strong-league-mixture-v1 \
    --corpus-release "$CORPUS_RELEASE" \
    --native-revision 6cfba6bf793a28eb9e49a5f4f1fcf7c8dbfa0f47 \
    --planner-revision "$PLANNER_REVISION" \
    --device cuda \
    --resume
}

make_release 1 "$ROOT/release-b1"
make_release 4 "$ROOT/release-b4"
make_release 8 "$ROOT/release-b8"
```

Distributed shards are acceptable. Each beam must have a complete,
non-overlapping shard set and identical non-shard settings. Never use
`--allow-budget-exhausted` for a release candidate.

### 5. Audit beam 8

```bash
uv run python -m tools.audit_counterfactual_pilot \
  "$ROOT/release-b8"/manifest.json \
  --calibration "$CALIBRATION" \
  --source "$BALANCED_BANK" \
  --output "$ROOT/audit-b8.json"
```

For sharded runs pass every shard manifest. The audit validates source identity,
exact legal-action coverage, W/D/L normalization, ranks and policy mass,
reserve-belief integrity, constant teacher ids/weights, complete member values,
artifact hashes, repository cleanliness, and source/release equality.

### 6. Compare beams

```bash
uv run python -m tools.compare_counterfactual_releases \
  --release "1=$ROOT/release-b1/manifest.json" \
  --release "4=$ROOT/release-b4/manifest.json" \
  --release "8=$ROOT/release-b8/manifest.json" \
  --reference-beam 8 \
  --output "$ROOT/beam-sweep.json"
```

The comparison now rejects any difference besides `opponent_beam`, including
source bytes, mixture/calibration hashes, chance model, information scope,
seed, depth, node budget, and root/chance beams. It also reports every tactical
cell separately. If beam 4 does not converge, increase the production teacher
beam; do not loosen the evidence threshold for runtime convenience.

### 7. Compare directly with frozen V3

Each bootstrap row is held-out and keyed to a release `source_id`:

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

`baseline_wdl` comes from the frozen observed-action/V3 bootstrap without
consuming counterfactual labels.

```bash
uv run python -m tools.compare_counterfactual_bootstrap \
  "$ROOT/release-b8"/manifest.json \
  --bootstrap "$V3_BOOTSTRAP_ROWS" \
  --bootstrap-samples 4000 \
  --output "$ROOT/v3-bootstrap-comparison.json"
```

Use at least 48 independent held-out games and include natural draws. Decision
rows from one game are one bootstrap unit.

### 8. Evaluate the gate

```bash
uv run python -m tools.counterfactual_quality_gate \
  --audit "$ROOT/audit-b8.json" \
  --calibration "$CALIBRATION" \
  --beam-sweep "$ROOT/beam-sweep.json" \
  --bank-manifest "$BANK_MANIFEST" \
  --bootstrap "$ROOT/v3-bootstrap-comparison.json" \
  --output "$ROOT/gate.json"
```

Exit `0` means every default check passed. Exit `2` means the gate remains
staged. Copy the passing evidence to
`runs/program/gates/v3-counterfactual-quality.json` only after independent
review of the hashes and source manifests.

## Stop conditions

Stop and fix the cause when any of the following occurs:

- hidden reserve bytes or future RNG reach a continuation network;
- independent `1/9` reveal mass is used;
- a supported posterior pill is pruned;
- a source belief is contradictory, impossible, or fails its count/hash check;
- candidate coverage differs from the source bank;
- candidate packing truncates or silently drops an action;
- any search reports budget exhaustion;
- a checkpoint, mixture, calibration, bank, or release hash differs;
- release code is dirty or shard settings differ;
- beam releases differ in anything besides opponent beam;
- aggregate convergence hides a failing tactical cell;
- draw calibration is unidentified in any training fold;
- member W/D/L or uncertainty is missing or non-finite;
- privileged teacher evidence is presented as deployable public search.

## After this gate passes

1. Train the V3 competitive head from counterfactual/search/outcome targets;
   keep human cross-entropy on the human/style head only.
2. Distill policy targets, W/D/L distribution, tactical consequences, and
   uncertainty into matched G5 students.
3. Run the root-only, V3-distilled, exact-effect-token, and recurrent-event G5
   bakeoff under common parent, seeds, compute, and opponent mixture.
4. Keep joint search offline until same-weight clean-start paired evaluation
   passes the separate `joint-event-search` gate.
5. Preserve this release as immutable teacher evidence; do not continually
   overwrite it with later checkpoints.
