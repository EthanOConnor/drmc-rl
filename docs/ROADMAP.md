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

The local Professor Pills build is playable while these scientific gates remain
open work. It bundles the full V3 epoch-5 regret/cadence model and a public V5
Maximum opponent improved with public outcome training, opens real two-player setup through recorded inputs,
and validates scheduled controller scripts. This does not establish the full
target skill range or a human operation profile.

Current diagnostic evidence identifies the next product gaps:

- The corrected 960-game level-14 HI round robin separates Maximum strongly
  from the V3 rating settings (306 wins in 320 games). The five rating settings
  are ordered overall, but every adjacent pair still has an overlapping 95%
  interval. A separate fixed-state audit has zero regret reversals across
  47,520 paired comparisons. The displayed corpus target is not an
  achieved human rating; calibration must use whole-game evidence and a common
  competitive quality scale.
- Public outcome PPO reached 10,133,254 frames with finite losses and zero
  candidate truncation. In a fresh 256-game comparison per opponent and level,
  it beats the previous public maximum 155–101 at level 14 HI and 177–79 at
  level 20 HI. Side-swapped-seed bootstrap score intervals are 53.9–67.6% and
  61.7–76.6%. Against G4 it scores 241–15 and 198–58 respectively. This clears
  the predeclared local adoption check; it does not establish a human rating.
  These tests use fastest placements. Real-ROM high-board starts can still
  top out early, so pressure robustness under human cadence remains work.
- A further 40M-frame continuation did not pass adoption. Its selected 20M
  checkpoint scored 126–130 at level 14 and 133–123 at level 20 against the
  installed 10M core on fresh 256-game comparisons. Paired score intervals were
  41.4–57.0% and 44.1–59.8%. The final 50M checkpoint regressed to 10–54 in the
  level-20 screen. Keep the installed core; more of this unchanged training
  recipe is not an evidenced improvement.
- The September 7 v20 checkpoint is tagged `trainer-internal-v20-20260907`
  in Professor Pills and drmc-rl: **quite good — good enough to get testers**.
  Spunky is an internal tester; this is not an external-release certificate or
  a calibrated claim about human strength. The app's baseline manifest records
  the package, model, and authored motor-profile identities.
- A fresh 48-game repertoire probe retained 46 selected public roots and
  explicitly reported two early terminations. At Frame Perfect, horizontal
  clears were available in nine roots and selected in two. The human-trained
  parent and selected 20M continuation had the same count on identical hashed
  roots. The one available cascade was selected by all three. This small
  fixed-root sample does not establish their overall style distributions or
  the competitive value of the rejected horizontal moves.
- Sloth's compulsory reaction window locked the pill in 12 of those 46 roots,
  all at level 20, including opening positions. Relaxed locked in three;
  Normal in none. Among actionable roots, Sloth averaged 12.7 candidates versus
  Frame Perfect's 27.6. Holding the feasible set fixed and replacing paced
  costs with unrestricted costs changed one Sloth choice and one Normal
  choice. None of the sampled costs reached the actor's clipping threshold.
  The main intervention to investigate is preparation for future constrained
  turns; cost rescaling cannot rescue a spawn that locks during reaction.
- The September 8 pace-strategy diagnostic implements that intervention as
  a small conditional adapter on the frozen public core. Its 160-game natural
  outcome pilot completed 9,763 learner decisions with finite updates and
  checked controller execution, but only 997,430 simulated frames. Its 960-game
  held-out round robin is inconclusive. Equal game counts severely undersampled
  slow-pace decisions, so the continuation now targets 100M frames plus at least
  100,000 learning decisions at every pace, with larger slow-pace batches and
  25M/50M milestones. A 2,176-game evaluation follows. No strength improvement
  or promotion is established. Super Human and Frame Perfect retain exact
  parent outputs.
  In 64 fresh opening checks, Sloth at 14 HI always had choices (6.1 mean
  candidates), while every 20 HI opening locked during its reaction window.
  This limits useful Sloth training pressure; it does not justify bypassing
  the declared motor limits.
- The immutable 54,819-decision input sample passed byte/window checks but is
  **not validated profile data**. The producer zero-filled unrecorded input
  spans. Its FBNeo snapshot/input convention also differs from the live host's
  boundary. The corrected private August release distinguishes missing input
  and preserves prior held state. With explicit FBNeo frame alignment and the
  retail gravity/wall-repeat corrections, 4,029 of 4,031 eligible sampled moves
  reproduce their exact lock pose and frame (2,446 games; 65 menu/chord
  exclusions). Two long windows remain unexplained and ineligible for profile
  fitting. This is independent stepper evidence, not a ROM replay certificate. Initial held
  buttons, operation bursts, and reaction distributions must drive the motor
  profile; an average-cadence scalar is insufficient. The extreme rating bands
  are sparsely represented, and the underlying V3 conditioning range is about
  718–2451, so 500 and above-3000 performance still need independent evidence.
- Style controls must be exposed and shown to vary independently of strength.
  The focused trainer currently offers uncalibrated skill 0–10, Max, and seven
  named motor paces. Pace restricts feasible moves before strategic selection, with
  no faster fallback; the presets still require human-distribution calibration.
- Live inference now selects available acceleration. On this Mac, a 48-request
  comparison reduced median planning time from 54 ms on CPU to 19 ms on Metal,
  with identical actions and scripts. Bounded Metal shapes must be warmed
  before readiness; without this, new candidate counts caused 200+ ms pauses.
  Subsequent source-build level-14 ROM checks completed with zero missed
  decisions or execution mismatches in Maximum and 1600 modes. Level-20 starts
  still reveal the separate strategic/cadence weakness.
- Anticipatory execution is the selected Max scheduler after the September 8
  controller-frame study: 768 screening games, a fresh 1,536-game level-14 HI
  round robin, and a separate 384-game level-20 HI field. Each matchup used
  side-swapped seeds; confirmation seeds excluded the screen. At 14 HI,
  prepared/four-frame fallback beat legacy eight-frame execution **367–145**
  (71.7%, paired 95% interval 65.9–76.8%) and four-frame reactive **306–206**
  (59.8%, 53.7–65.6%). Reactive beat legacy **328–184**. The three-agent
  Davidson fit gives preparation +165 Elo [127, 202] and reactive +98 [61, 135]
  relative to legacy, counting each paired seed as one effective observation.
  These are experiment-relative estimates, not human ratings. At 20 HI,
  preparation scored 79–47–2 against legacy and 79–49 against reactive;
  the latter's paired interval still includes 50%. All 1,920 confirmation and
  pressure games finished naturally, with median duration 3.2 minutes and
  maximum 15.0 minutes, below the 60,000-frame cap.
- Preparation reused 110,190 of 126,741 Max decisions in the primary round
  robin (86.9%), reducing mean spawn wait from eight frames to 0.52. It retains
  the previous opponent observation but requires exact own-state and input
  agreement. Requiring unchanged opponent context reused just one of 623
  turns in the initial smoke. Zeroing opponent inputs with frozen weights lost
  32–96 to full context; this does not rule out a separately trained
  opponent-blind actor. Twelve sampled games (six side-swapped seeds) replayed
  exactly; refreshing opponent context changed two of 96 sampled choices,
  the same position on both sides. None of those selected targets became
  unreachable after four/eight idle frames. This small sensitivity sample
  suggests faster tempo contributes more than extra move access; it is not
  a win-probability regret estimate.
- On-device measurement supports a combined design: complete warm decisions
  took 18.0 ms median / 23.4 ms p95 on Metal; one nine-preview batch took
  20.7 / 23.6 ms versus 109.9 / 116.7 ms for nine separate calls. Complete
  next-bottle preparation, including both spawn parities and all nine previews,
  took 33.3 / 47.0 ms, maximum 51.4 ms across 46 public roots. The arena charged
  six frames for this preparation and four for fresh decisions. Three-frame
  fresh deadlines were unreliable with two live Mac sidecars, despite winning
  offline. The app therefore uses an adaptive four-frame floor with measured
  end-to-end headroom; the latest cartridge/Super Human and ROM-free checks
  made 129 decisions with zero missed deadlines or wrong placements, and one
  caught controller-read correction. Load occasionally raised the fallback to
  five frames. The arena's fixed four-frame comparison is not a claim of
  identical timing on every host. All 11,100,884 checked input frames across
  the main study matched their predicted microstates. Focused scheduler,
  backend, anticipation, frame-API, memoization and rating checks passed;
  packaging and full release verification were not part of this study.
- Exact-planner auditing corrected a late-game gravity table, bottle-wall DAS,
  and undersized geometric graph tables. The latter could discard a legal
  four-frame slide/rotation and emit a truncated six-frame GPU witness. Native
  sanitizer checks and a 500-case GPU sweep now pass, including 10,277 emitted
  scripts with no replay errors. Future arena/training runs must record the
  corrected native revision; historical matches keep their original identity.

Run artifacts for these diagnostics are retained under
`runs/trainer-baseline-v1/` and `runs/trainer-anticipation-v1/`; full planning
move archives and remote outputs are on mombox under
`trainer-output/anticipation-20260908/`. They do not override any promotion gate.

The recovered 1,440-state beam sweep has 99.93% beam-4/8 action agreement and
at least 95.83% in every one of its 60 tactical cells. However, the direct
631-game comparison is worse than V3: Brier delta +0.0433 (95% whole-game CI
+0.0213 to +0.0648) and log-loss delta +0.0407 (+0.0102 to +0.0702).
These labels remain ineligible for promotion. The bounded
`trainer-public-outcome` recipe instead trains the existing public V5 student
from natural game outcomes at level 20, with level-14 regression evaluation.
It neither waives the search-quality gate nor certifies the trainer's ratings.

The subsequent boundary correction removes unsupported critic calls but does
not rescue that teacher: Brier is 0.5108 versus V3's 0.4616 on the same 631
games. The current public outcome-trained core improves the candidate-search
diagnostic to 0.4728, with a significant Brier improvement over the original
teacher, but still does not reliably beat V3. Its pre-action decision value
scores 0.4547; this is not a candidate-regret result. Further full-candidate
labeling is deferred while the common outcome-trained core is improved.

The first 512-state production-shaped pilot established important mechanics:

- exact restore of unique full-pair states;
- full legal-candidate enumeration with zero truncation;
- strict reveal-boundary stopping and explicit reveal continuation;
- bounded depth-2 search without node-budget exhaustion;
- calibrated frozen Strong League continuation in place of the diagnostic leaf
  heuristic.

It did **not** establish mature quality. The pilot treated each reveal as nine
independent ordered color pairs at probability `1/9`. The NES instead creates
the whole 128-entry reserve from a two-byte RNG, then generates the public
initial virus bottle from the same stream. Reserve entries are nonuniform and
correlated with both the initial bottle and already visible pills. The pilot is
retained as mechanics evidence; its candidate values must not open the gate.

### Corrected chance model

`PillReserveBelief` enumerates the uniform two-byte reset-seed prior used by the
randomized native experiment, conditions it on the public initial virus bottle
and every publicly observed falling/preview entry, and predicts the next reveal.
This is exact under that declared experimental prior; it is not mislabeled as
the boot-to-game retail console prior. `BeliefNativePairSearchModel` still
overwrites the hidden native reserve byte before reveal, but assigns probability
only from the public posterior. Some nodes have fewer than nine supported
outcomes.

A mature source bank stores `reserve_belief` on every row. A release records
chance model `nes-reserve-public-seed-belief-v2`. Independent one-ninth branching is
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

The first repertoire intervention samples exact opportunities from original
human VS positions. Commentary from the July speed brackets highlights
horizontal setups, linked clears, and named shapes, but also praises passing
up a flashy combination for a better finish. Those race comments guide motif
discovery, not VS action-value labels. The inspected video transcripts include
checkpoint-fit fallbacks and sequence gaps; exact video timing alone does not
make them replay-certified imitation data.

The original replay extraction yields 63,112 positions from 2,046 sessions.
A bounded scan produces 1,626 training roots from 1,102 sessions and 408 held-out
roots from 267 sessions. It excludes one unsettled original root. By contrast,
4,032 of 4,088 examined roots from the old virus-reduced practice derivative
were unsettled; do not mine its apparent pre-gravity lines as human motifs.
The matched 5M-frame pilot changes only a 25% repertoire reset mixture, keeping
the current public core, terminal outcome objective, and a clean-start control.

Both arms completed with finite metrics and zero candidate truncation: 5,020,775
frames for curriculum and 5,210,954 for control (whole-update budget rounding).
On 96 identical held-out roots from separate replay sessions, horizontal clears
were selected in 11/33 available cases by the incumbent and 12/33 by each arm.
The same single root changed in both arms. All three selected 30/32 available
cascades. This is no distinct repertoire gain from the curriculum.

The 384-game screen also rejected curriculum: it lost 25–39 to control at
level 14 and 29–35 at level 20; against incumbent it scored 38–26 and 30–32–2.
Control's promising 44–20 and 36–28 incumbent screen did not replicate. On
512 independent confirmation games it scored 129–127 at level 14 and
123–131–2 at level 20. Whole side-swapped-seed bootstrap 95% score intervals
were 43.4–57.4% and 40.6–56.3%, respectively. **Retain the incumbent.** Neither
arm is an evidenced replacement, and the conditional G4 confirmation is
unnecessary after the incumbent gate fails. The frozen experiment plans,
checkpoint hashes, complete reports, and decision manifest are retained under
`runs/trainer-repertoire-v1/` and tf3090's
`/home/ethan/.cache/drmc-rl/trainer-output/repertoire-20260907/`.

The next repertoire curriculum should begin two to six placements before a
verified human setup completes, so the learner practises construction as well
as conversion. This pilot's negative result is specific to one short run and
opportunity-based reset mixture; it does not reject curriculum learning in
general. Human setup sequences must pass replay/settled-board checks before
they define sampling priorities, and all final policy choices still optimize
match outcome.

The next strength experiments should discriminate mechanisms, not just add
frames: conditional slow-pace adaptation with the fast route frozen; exact
afterstate/effect features and public timing/event context; and opponents that
exploit specific weaknesses in the current core. G5's cross-bottle summaries
pool by column, making additional row/spatial features an architecture
hypothesis to test, not an established explanation for vertical play. Require
matched compute, fresh paired seed evaluations, and retained human/style
anchors. Validate the failed quality-teacher contracts before using search
labels or interpreting policy-logit gaps as a safe style envelope.

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
7. Fit the frozen V3 observed-action Davidson link on a content-addressed
   natural-game calibration split, apply it to disjoint held-out games, and run
   the paired whole-game comparison.
8. Run the executable `v3-counterfactual-quality` gate; do not start mature
   quality distillation until it passes.
9. Complete the timing-action gate in parallel.
10. Fine-tune/distill the mature teacher and run the matched G5 bakeoff.
11. Outcome-train and exploitability-harden the population.
12. Fit constrained execution and trainer systems only after the common quality
    core is frozen.
