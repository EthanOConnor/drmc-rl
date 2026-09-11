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

Historical native-SMDP comparisons below retain their original observation
contract. The September review found that its raw opponent buffer could expose
a future lock. New public native runs require strict causal advancement and
separate visible snapshots; the historical scores cannot certify that corrected
contract. Controller-frame tournaments and human play tests use a different
observation path and remain separate evidence.

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
  slow-pace decisions. The larger continuation completed 101,229,824 console
  frames and the 100,000-learning-decision floor at every pace, with larger
  slow-pace batches and 25M/50M milestones. Its 2,176-game scaled evaluation
  completed; the ongoing larger field supplies further evidence. No promotion
  is established. Super Human and Frame Perfect retain exact
  parent outputs.
  Before the large continuation, exact event batching, shared frozen-core
  inference, cost-only feasibility and asynchronous planning improved a matched
  strict-FP32 64-game rollout from 2,934 to 29,856 frames/s on tf3090.
  Full-game comparisons at all trained paces
  preserve input trajectories and outcomes. The independent frame runner stays
  in evaluation; throughput changes do not constitute strength evidence.
  In 64 fresh opening checks, Sloth at 14 HI always had choices (6.1 mean
  candidates), while every 20 HI opening locked during its reaction window.
  This limits useful Sloth training pressure; it does not justify bypassing
  the declared motor limits.

## September 2026 review: implementation and evidence sequence

The external review was checked against source, the exact gradient
counterexample, native transitions and instrumented experiments. Its proposed
directions are hypotheses, not reproduced strength gains. The implemented
actor objective removes per-game inverse-length credit; the toy's true
gradient is +0.1 while the historical normalized update is negative.
This proves a possible failure, not its size in Dr. Mario. A further native
audit found the future warped-opponent observation leak described in DESIGN;
the first new state collection was stopped and regenerated with causal views.

| Experiment | Implemented foundation | Evidence and next decision |
| --- | --- | --- |
| E0: objectives and public inputs | Exact gradient/finite-difference tests; separately named losses; hidden-commitment input test; causal native observations and complete placement collection across pair events; full frontier and bond preservation | Native PPO updates pass in dictionary and direct-array modes without waiting-event samples. Preserve historical artifacts. Never reinterpret a masked private input as a fully public timeline. |
| E1: episodic actor credit | Same frozen 25M adapter, parent, seeds, pace exposure and full categorical KL budget; 50M additional frames per arm; 16,384 reserved arena games | Training and all 16,384 evaluation games completed with no censoring. Corrected credit wins 55.9% at 14-HI Top Humans and 54.4% at Fast against the historical objective; other trained paces are inconclusive. Details below. Keep the corrected objective for further training; independent training seeds and cross-pace confirmation remain required for adoption. |
| E2: critic and public context | Candidate-attending value query; separate terminal/candidate WDL heads; versioned per-side public conditioning; baseline/critic/context/combined fitting modes | The expanded 64-game study completed all four 100-epoch fits on the same 48 training / 16 validation games. Aggregate outcome errors improved, but held-out gap/ranking losses worsened; details below. No architecture is selected. New-schema live history/motor emission remains required before paced deployment. |
| E3: paired candidate improvement | Full-root, complete-reserve panel with two public continuations; chance variance and continuation sensitivity separated; reference-relative KL-bounded targets; supervised gap/WDL fitting | The 520-continuation pilot has exact serial CPU/batched GPU parity. The expanded study completed 2,090 candidates and 8,360 natural continuations across 64 independent 14-HI roots, with no unknown mass. Only 32 roots distinguish moves under this panel. Learned ranking and real-game improvement need substantially more prospective evidence; the failed V3 gate remains failed. |
| E4: diverse experience | Frozen public opponent pool in pace training; empirical regularized mixture; exact states 4/8/16 own placements before natural losses | Initial bank had 16 states from four natural 14-HI games; the expanded study uses 64 independent source games. Expand source diversity and mix openings/midgame/finishes with predecessors. No curriculum or league change enters the E1 arms. Clean-start population results govern adoption. |
| E5: effects, memory and scale | Existing exact-effect and public-event modules audited; supervised auxiliary phase with policy KL; layer updates and representation-rank diagnostics | Integrate observed event history and exact effects before spatial wave/support/access labels and next-known-pill motor opportunity maps. Compare current 320×8 with 384×12 and 512×12 only after target quality; equal GPU time and equal exposure are separate reports. Distill a successful large teacher into the fast student. |
| E6: search and anticipation | Cooperative frontier batching, complete mixed matrices and bounded adaptive root allocation; exact requested-side inference, complete cache identities and explicit unknown entries | Synthetic and native matrix checks pass. Adaptive allocation reduces work on four development roots while retaining response bounds; this does not establish useful Q ranking or stronger play. Tactical extensions and a late-conditioned anticipation trunk remain separate experiments. Charge measured computation delay before full-game evaluation. |
| E7: expression and calibration | Whole-game paired noninferiority analysis with a declared margin, seed design, comparison family and fixed confirmation set; censoring blocks certification | Persistent 2–6-placement proposals still need replay-aligned human setups, held-out local-regret calibration and blind clip preferences. No permanent horizontal/combo reward, independent random blunders or uncalibrated Elo claim. |

The E1 arena completed at 11:16 UTC on September 9. All 16,384 journal records
form complete side-swapped pairs, using 768 distinct evaluation seeds with no
overlap with either training journal. Each arm trained for 75 updates and
22,080 natural game experiences: historical credit used 50,369,235 additional
frames / 446,868 learning decisions; corrected credit used 50,201,899 / 445,535.
Realized median update KL was comparable across paces, not identical.

| 14-HI pace | Corrected score vs historical (1,024 games each) | Paired 95% interval | Five-comparison family interval | Score-equivalent Elo difference |
| --- | ---: | ---: | ---: | ---: |
| Sloth | 50.00% | 48.54–51.46% | 48.05–51.95% | 0.0 |
| Relaxed | 50.63% | 47.56–53.66% | 46.53–54.59% | +4.4 |
| Normal | 49.02% | 45.70–52.25% | 44.73–53.32% | −6.8 |
| Fast | 54.39% | 51.07–57.71% | 50.00–58.79% | +30.6 |
| Top Humans | 55.86% | 52.44–59.28% | 51.37–60.35% | +40.9 |

Intervals resample complete seed pairs on this fixed evaluation; the family
intervals use Bonferroni adjustment across the five primary comparisons. Sloth
has 862 natural draws, not timeouts. Separate 20-HI checks have 512 games each:
Normal scores 53.13% (paired 95%: 49.80–56.45%); Top Humans scores 57.03%
(52.15–61.91%). Only Top Humans excludes parity after adjusting those two
pressure comparisons together. Elo here only transforms each condition's match
score; it is not a human rating or a claim of transitivity across conditions.

Against the older 100M+ adapter, corrected credit scores 48.24%, 47.27%, 51.95%,
55.27% and 55.27% from Sloth through Top Humans (512 games per pace). The two
fastest trained paces are positive in exploratory paired intervals; this is
not uniform superiority. The corrected objective is the next training default,
while the deployed model stays unchanged. One training seed cannot establish
the reliability of the training procedure. Live history/motor integration now
proceeds as required; further independent-seed confirmation need not defer it.
The complete assessment is `runs/review-20260909/objective-arena-assessment.json`.

The journal audit also found an old viewer-database uniqueness constraint that
collapsed repeated seeds across paces. Complete journals and journal-derived
standings were intact. Match identity now includes its comparison condition;
the viewer was recovered from all complete, uncensored journal pairs without
rerunning games. Snapshot connections are explicitly closed, and sync excludes
SSHFS temporary snapshots and SQLite sidecars.

The review's commentator excerpts identify replay windows and qualities such
as efficient low-drop clears, connected structures, delayed payoff, access,
resourceful cleanup and finishing. They do not establish board geometry or VS
attack timing. Mine and replay-align those windows before labeling motifs;
include earlier construction states, not only the final flashy clear. Test the
actual persistent style continuation, including replanning after garbage, and
report losing tails and the strength/preference tradeoff. Local tolerated
regret does not prove whole-game noninferiority.

Other research choices remain deliberate ablations: spatial/row pair
attention, longer public-event memory, independently trained uncertainty
members, long skips/curvature-aware optimizers, conditional geometry caching,
and hierarchical timing around material event boundaries. None is justified
as a default by a literature result alone. Unsearched actions remain unknown;
do not duplicate deterministic trajectories to simulate sample size. The
placement SMDP and timing-action gate remain authoritative.

The review artifacts live under `trainer-output/review-20260909` on mombox,
with compact local reports in `runs/review-20260909`. The 520-rollout panel took
3,876.5 seconds on mombox CPU and 263.0 seconds on tf3090 with CUDA inference
and two native workers. This compares complete execution paths on different
hosts, not an isolated native-threading speedup. Optional parallel native
stepping also passes deterministic serial parity tests. Neither throughput
nor the four fitting smoke tests establishes a stronger player.

The expanded study completed on tf3090 from committed source `d5c7c54` at
09:22 UTC on September 9. It collected causal predecessor/tactical states,
selected 64 distinct natural source games and completed all 8,360 continuations
in 3,639 seconds. All four same-size architectures completed their predeclared
100 auxiliary epochs with the same 48-game/16-game split, matching minibatch
order and matching shared critic initialization. The 16 validation games are
development evidence, not a fresh confirmation set for the next iteration.

| Architecture at epoch 100 | Candidate Brier | Candidate gap MSE | Ranking loss | Validation policy KL |
| --- | ---: | ---: | ---: | ---: |
| Initial neutral WDL heads | 0.5797 | 0.2126 | 0.3033 | 0 |
| Baseline | 0.2089 | 0.2896 | 0.6414 | 0.1462 |
| Candidate-attending critic | 0.1944 | 0.2881 | 0.6965 | 0.1042 |
| Public context | 0.2186 | 0.2645 | 0.5863 | 0.2358 |
| Critic and context | 0.2017 | 0.2585 | 0.6221 | 0.1258 |

Lower is better in the error columns. Every fit improves average outcome
prediction but has worse held-out action-gap and ranking point estimates than
the initial equal-valued candidates. Training gap MSE is 0.0184–0.0366, versus
0.2585–0.2896 on validation: this small fit overfits rather than establishing a
useful search-quality head. The final training-policy preservation KL is
0.0040–0.0060. A reporting bug left validation `anchor_kl` at an unmeasured zero
in the original fit reports. A separate inference-only audit measured the
validation values above against each model's post-migration initial policy:
0.1042–0.2358, with a maximum individual-state KL of 1.1984. The training-only
anchor does not preserve the policy on unseen states. Future fitting reports
now measure both splits against explicit fixed reference distributions; a
focused end-to-end fit test verifies the held-out KL against the saved model.
Original study artifacts remain unchanged.

Exactly 32 of 64 roots give every candidate the same outcome utility under the
panel (30 all losing, two all winning). Nine of the 16 validation roots are
flat, so only seven contribute move-ranking information. Across all roots,
27 incumbents have a better alternative under the frozen panel. The learned
Q heads also show an unconfirmed top-choice signal: mean validation panel regret
is 0.3438–0.4063, versus 0.6563 for the initial actor. Descriptive paired-bootstrap
intervals for that improvement touch or cross zero; these are development-set
comparisons, not optimal-play or real-game strength claims. The next quality
iteration must substantially expand independent source games and tactical/
temporal coverage, add broad public-policy anchor replay disjoint from
validation, and retain a representative prospective confirmation set. Do not
optimize against validation drift measurements. Demonstrate useful within-state
ranking and policy preservation before search allocation or quality distillation.
Assessment and original reports are in
`runs/review-20260909/quality64-review/assessment.json`; remote originals remain
under `trainer-output/review-20260909/quality64-study`. The separate
`policy-drift.json` retains the retrospective inference audit. The objective arena
has now completed as reported above. No diagnostic checkpoint is installed or promoted.

### Required continuation after the strength evaluation

The user's September 9 instruction makes the remaining review work an explicit
continuation of this program. Once the current 16,384-game strength evaluation
is complete and analyzed, proceed through all four items below. Do not wait for
a positive objective result: retain the best evidenced baseline if the result
is neutral or negative, and continue the program. Additional independent-seed
confirmation needed for promotion can proceed alongside subsequent engineering.
The 64-source-game quality study supplies the next architecture evidence; repair
or replace an unsuccessful experiment rather than leaving its dependent work
indefinitely staged.

- [ ] **Live history and motor integration (E2/E5).** Emit and consume the same
  versioned causal public history, observed effects, phase/age and motor context
  in training, evaluation and the live trainer. Integrate exact effects and
  future constrained-movement opportunities. Verify information boundaries,
  execution parity and device latency, then train and compare the integrated
  policy at every supported pace. A schema and an offline fitting smoke test
  alone do not complete this item.
  The controller-frame and batched-event runners now emit the same exact native
  spawn/lock/clear/released-volley/terminal history, both visible previews and
  poses, ages, and charged execution context into new-schema policies. Legacy
  players keep their original encoding. Full frontier coverage and history/
  motor-sensitive memoization pass; stale legacy anticipation/ablation is
  rejected for new-context actors. Exact frame/event input parity, clear/volley
  fixtures, snapshot tests and the independent NES demo transcript pass.
  The shared desktop/browser scheduler now emits the same bounded public wire
  view from native effects or passive cartridge instruction hooks. Context-aware
  cores consume it in the source backend; frozen players retain their existing
  inputs. Native/WASM parity covers 64,800 frames including exact effects, and
  neural live-backend tests cover Sloth and Top Humans. Exact conditional
  motor-opportunity labeling now resolves every current candidate, carries its
  horizontal controller state into the visible next pill, applies the next
  gravity speed-up and actual reaction/compute delay, and retains both spawn
  parities. Two-placement native controller tests cover Sloth, Top Humans and
  Frame Perfect, including clear effects and terminal success. The CPU bank
  recipe uses exact v2 public controller replay, excludes reserved arena seeds
  and holds out whole reset seeds. Incoming garbage remains an explicit
  condition; these are auxiliary geometry targets, not match-value labels.
  The supervised fitting path now adds effect/access/cost auxiliary heads,
  supplies observed own controller geometry, updates the shared encoder and
  preserves the policy on a separate broad replay anchor. Holdout seeds never
  enter anchor gradients or rollback decisions; ordinary inference skips the
  heads. A real native-data fit test verifies shared gradients and checkpoint
  reload. Fresh confirmation is implemented through
  `trainer-motor-confirmation`: reserved seeds excluded from all training and
  anchors, natural public controller replay, complete conditional labels,
  original-predictor and training-only prevalence comparisons, and whole-seed
  intervals reported separately at each level/pace. Its native end-to-end test
  covers fitting, fresh games, exact labeling and prediction assessment.
  The first 20-epoch fit completed on tf3090 in 414 seconds: 33,580 accepted
  root presentations from 1,679 training roots, 369 validation roots and 4,096
  independent policy-anchor games. Validation reach Brier improved from 0.2517
  to 0.1240, clear Brier from 0.2474 to 0.0690, and cost error from 0.1511 to
  0.0150; validation policy KL was 0.00101. These comparisons with the initial
  predictor are preliminary; the independent prevalence baseline, fresh-seed
  confirmation, device latency and full-game strength remain outstanding.
  Fresh confirmation completed all nine conditions, 1,152 natural games and
  1,728 exact roots at September 10 11:38:29 UTC. Every condition finds reach/clear
  prediction worse than the training-only pace/cell prevalence baseline. The
  random-initialization comparison therefore does not establish useful motor
  prediction. Complete the separate strength study and make a bounded revision
  of initialization and auxiliary fitting using training data before another
  independent confirmation; do not promote this first fit. The fitter now
  supports training-only cell-prior initialization, fixed head-only epochs on
  cached public features, and separate head/core learning rates. Core-state
  identity must survive warmup; cached features are discarded before shared
  updates. The fixed revision completed 40 head-only and 20 shared-core epochs,
  100,740 root presentations, followed by all nine fresh confirmation conditions
  (1,152 natural games and 1,728 roots). Reachability beats training-only
  prevalence in every condition; clear prediction improves in seven, with
  Sloth and 20-HI Top Humans inconclusive. These conclusions survive a shared
  whole-seed family correction over all 18 reach/clear comparisons. Conditional
  no-incoming-garbage prediction is now supported; playing strength remains a
  separate question. The immutable refit is in a fixed 16,384-game Mac arena
  against its 25M parent and the current product opponent at each pace, using
  new reserved seeds excluded from both prediction confirmations.
  No new core has been installed. Full-network controller outcome
  learning is implemented under `trainer-controller-core`; focused tests verify
  board/context gradients, exact collection likelihoods, checkpoint reload and
  complete public replay shards. The full 320×8 core run now targets 10M learner
  decisions and 1B console frames, including at least 500k learner decisions
  at each pace. The 32,768-game initial/25M study completed September 10 at
  00:46 UTC with no censored games. The 25M core recovered over its initial
  migration but scored only 45.2%, 44.3%, 42.2%, 41.3% and 42.0% against corrected
  E1 from Sloth through Top Humans, respectively (1,024 games each). All five
  remain below 50% in the simultaneous primary-family intervals. Frame Perfect
  is inconclusive at 48.9% versus the parent (paired 95% 45.4–52.3%). The core
  beats the parent at 14-HI Sloth and 20-HI Normal, but does not replace E1.
  `core-initial-25m-assessment.json` retains whole-seed intervals, execution
  counts and natural draw rates. The fixed 100M study also completed all 16,384
  games without censoring at 03:20 UTC September 10. Against E1, its 14-HI
  scores are Sloth 55.7%, Relaxed 52.1%, Normal 46.1%, Fast 41.3% and Top Humans
  38.6%. Sloth improves and Fast/Top Humans lose under the simultaneous primary
  family intervals; Relaxed/Normal remain unresolved by that family check.
  Super Human/Frame Perfect remain near 50% against the parent. The separate
  20-HI Normal score is 58.6% versus E1, while Top Humans scores 40.0%.
  `core-100m-assessment.json` retains the complete allocation and whole-seed
  uncertainty. A fixed 4,096-game follow-up compared this core with the
  original final Sloth adapter and 50M Relaxed adapter, using 1,024 additional
  training-excluded reset seeds not used in the earlier core studies. All games
  finished naturally. Sloth scored 52.17% (two-condition simultaneous 95%
  interval 50.30–54.05%), with 1,383 draws among 2,048 games; Relaxed scored
  44.73% (41.60–47.85%). The core is a narrowly stronger Sloth candidate under
  this execution model, but loses to the selected Relaxed specialist.
  `core-slow-specialists-assessment.py/.json` retains whole-seed uncertainty
  and complete execution counts. Verify live integration and actual device
  latency before replacing the Sloth adapter in a tester build.
  Substantial later learning and device evidence remain pending. Sloth's many
  natural simultaneous top-outs and loss of reachable actions require explicit
  attention to earlier preparation and curriculum support; merely accumulating
  forced or unplayable frames does not demonstrate strategic learning.
  A migration audit also exposed immediate policy/value drift from switching
  the opponent's bottle to its own pill conditioner. New migrations introduce
  that side-specific path through learned zero-initialized residual scales.
  The real parent preserved all outputs on 288 unchanged controller decisions;
  the earlier direct migration changed 12 choices. Gradient, replay and old-run
  resume checks pass. The active billion-frame study retains its frozen source;
  evaluate this corrected initialization in the next independent training
  branch rather than reinterpreting or silently replacing that study.
  A fixed 16,384-game Mac initialization ablation now compares the corrected
  initial core with the frozen parent and original initial core, preserving
  every shared weight, the previous controller-native libraries, reserved
  seeds and compute charge. It completed without censoring at 06:06 UTC
  September 10. Against the original initial core, every 14-HI score lies
  between 49.0% and 51.0%; the simultaneous family resolves no improvement.
  Against the parent, Top Humans/Super Human/Frame Perfect score
  43.75%/42.68%/42.58%, all below 50% in that family. Other 14-HI cells are
  inconclusive; the separate 20-HI Normal score is 50.0%, Top Humans 45.7%.
  `core-residual-initial-assessment.py/.json` retains 20,000 shared whole-seed
  bootstrap draws. Its common absolute-score family band handles Sloth's
  zero-variance side-pair scores without asserting population equivalence.
  Equal-input preservation remains useful migration evidence but did not
  recover whole-game strength. Attribute the remaining live input/frontier
  changes on training-only public roots before another initialization branch;
  retain the current frozen training run and selected product baselines.
  The training-only attribution audit uses 768 positions from 192 recorded
  seed/shard groups (14-HI Normal/Top Humans and separately 20-HI Frame Perfect).
  Restoring both bond inputs changes 11 physical placements; expanding the
  same-color frontier alone changes 25; all changes together affect 30.
  These are same-position policy differences, not attribution of whole-game
  losses. An explicit parent-behavior alignment fit is now implemented with
  full student inputs/frontiers, whole-seed validation and fixed-epoch selection.
  It must demonstrate held-out policy preservation and controller strength
  before starting another outcome branch. The source backend also now retains
  a context actor's chosen same-color orientation rather than replacing its
  exact witness with the historical canonical orientation; native tests cover
  both formerly rewritten orientations.
  The fixed eight-epoch alignment fit is now complete: 38,616 supervised
  presentations, 4,827 training positions and 1,207 whole-seed validation
  positions. Validation KL fell from 0.14333 to 0.03085; choice agreement
  increased from 91.14% to 94.31%. The fixed 16,384-game controller tournament
  against the parent and corrected initialization completed without censoring
  at September 10 11:19:46 UTC. It reuses the reserved initialization bank and
  is not independent confirmation. Against the parent, 14-HI scores are
  Sloth 45.21%, Relaxed 47.66%, Normal 50.78%, Fast 48.93%, Top Humans 49.90%,
  Super Human 47.46% and Frame Perfect 49.41%. None of the 14 primary
  comparisons establishes a gain under the common 4.6875-point simultaneous
  band; Sloth versus parent is below 50% (40.53–49.90%). The separate 20-HI
  Normal results are 43.16% versus parent and 42.77% versus corrected
  initialization, with individual whole-seed intervals below 50%; Top Humans
  remains near 50%. `public-input-alignment-arena-assessment.py/.json` retains
  all comparisons, natural draws and 20,000 shared seed resamples. Alignment
  is not an all-pace replacement. A possible broader alignment revision must
  cover every pace with training-only replay; keep the substantial outcome
  run's model and optimizer unchanged, and await the fixed 300M comparison
  before allocating another outcome branch.
  The fixed 300M comparison is now complete: all 16,384 games, no censoring.
  At 14 HI, Super Human scores 59.67% and Frame Perfect 61.52% against the
  parent; both gains survive the full primary-family correction (54.31–65.02%
  and 56.28–66.77%). Sloth also beats E1 at 59.23%; Relaxed/Normal/Fast/Top Humans
  are near 50% against E1, recovering the earlier fast-pace deficits without
  yet establishing superiority there. The 20-HI Normal score versus E1 is
  56.54%, Top Humans 48.05%, reported separately. The next fixed 8,192-game
  candidate confirmation is also complete on 1,024 additional candidate-unseen
  reserved seeds. Against actual product routes, Sloth scores 57.54%, Relaxed
  46.00%, Super Human 61.08% and Frame Perfect 61.72%. Shared whole-seed
  simultaneous intervals over all four comparisons establish gains in the
  first and fastest two conditions, and a loss at Relaxed. Retain the original
  50M Relaxed adapter. The 300M core is the selected Sloth/Super Human/Frame
  Perfect candidate for representative actual-app checks before adoption;
  retain the current outcome optimizer and other pace baselines.
  V2 mixed-core live selection is implemented, including per-pace input
  encoding, warmup, anticipation capabilities and portable packaging. The
  explicit tester-300m manifest retains Relaxed/E1 routes. Six native/ROM
  scheduler checks completed 373 placements with no placement mismatches;
  two ROM guarded replans and missed requests remain in the report. The GUI
  loaded the portfolio, but synthetic sub-frame Start taps did not start a
  game. The explicit bounded autoplay mode now enters through normal controller
  input without changing preferences. Ten valid GUI games completed 412
  placements with no initial-state or placement mismatches; missed requests and
  five guarded ROM replans remain in the report. Four short visible follow-ups
  cover the missing fast display cases and revised Sloth: 140 more completed
  placements, 4,466 actual presentations, no corrections or mismatches. The explicit
  tester manifest selects the three supported 300M routes. The browser package
  now serves that same lineup: full public-core selection before encoding, exact
  ONNX parity, and per-pace preparation negotiation. Browser/source exchanges,
  native and ROM WebGPU play, and CPU fallback passed before deployment.
  The September 10 Apple Silicon/macOS 15 internal package also includes the
  same portfolio. Its frozen backend matches all 42 source/browser reference
  exchanges; three final native/ROM level-14 HI games complete 179 placements
  without state or placement mismatches. Six missed requests and one guarded
  ROM correction remain in the evidence. The 275 app and 115 backend/planner/
  package checks pass. This locally signed package is not a notarized release.
  ROM divergence investigation and the broader research program remain open.
  At the user's request,
  live Sloth/Relaxed reaction floors are now 45/30 frames, with unchanged motor
  limits. Frozen training and tournament evidence still use 60/36, and must not
  be relabeled as revised-preset confirmation. Stage one is still open.
  The motor-refit arena subsequently completed all 16,384 games without
  censoring. No strength gain survives the 18-condition simultaneous family;
  product comparisons regress at 14-HI Sloth/Relaxed/Fast/Top Humans and at
  20-HI Top Humans. Reject that refit as a replacement despite better motor
  prediction. Preserve the main outcome run and finish the distinct original
  auxiliary arena; prediction quality alone does not justify actor promotion.
  `HumanBackend` now supports a verified per-pace manifest with one shared
  competitive parent and small frozen residuals. Live scoring and preparation
  supply the same actual pace/gravity/delay context as the arena; preparation
  uses the predicted next gravity and zero execution delay. Warmup covers each
  selected residual and the untouched parent path. Focused native tests verify
  complete choices and controller scripts at all seven paces, exact fastest
  parent outputs, unchanged lower-skill decoding and portable artifact loading.
  The source CLI discovers a companion manifest, and packaging can copy and
  verify it explicitly. Unconfigured source builds keep the plain parent;
  the current browser and Mac packages select the mixed portfolio above.
  Real-checkpoint device
  checks on 12 real public roots now match all 84 portfolio and 12 Sloth-core
  choices/scripts against CPU arena references. All 648 prepared branches match.
  With CPU reference inference separated from timing, the normal JSONL flow
  takes 15–17 ms median per decision and 32–36 ms for preparation on Metal.
  The initial September 10 provisional tester-build shortlist was the
  original final adapter for Sloth, the 50M adapter for Relaxed, corrected E1
  for Normal/Fast/Top Humans, and the 10M public parent at the two fastest
  paces; the completed 300M evaluation and live checks above supersede its
  Sloth and fastest-pace choices. Selection is a product judgment from completed studies, not a claim that
  every selected checkpoint is a uniquely established winner; several nearby
  rankings remain unresolved. The completed 100M specialist comparison now
  supports a Sloth upgrade, subject to its live context and device checks;
  it does not support changing Relaxed. The existing V3 regret decoder remains the current 0–10 path;
  a common-core calibrated ladder and expressive selection remain unfinished.
  Outcome training stopped before update 379 on a rare FP32 probability-audit
  outlier. Replaying all 11,326 preserved decisions against the same model with
  a rare independent CPU FP64 reference passed the unchanged error bounds,
  retaining actual behavior probabilities and detecting real corruption in
  focused tests. The unchanged update-378 model/optimizer resumed from a new
  frozen source; update 379 subsequently completed. No numerical limit, input
  schema, reward, sampling policy or native execution contract was relaxed.
- [ ] **Larger teachers and fast students (E3–E5).** Establish useful held-out
  candidate targets, then compare 320×8, 384×12 and 512×12 with separate equal-data
  and equal-GPU-time analyses, independent teacher members and diverse public
  opponents/source games. Train at a scale capable of showing learning, measured
  in learner decisions and console frames. Distill useful larger-teacher gains
  into the deployment student and test actual strength and inference latency.
  Implemented configuration switches or a tiny pilot do not complete the study.
  The broader source collector now batches causal public inference across
  independent cold-start games, with distinct fit/anchor/confirmation reset
  seeds, bounded whole-game temporal/tactical sampling and durable per-game
  resume. Terminal labeling can batch complete panels across multiple roots
  and retain each finished root through interruption. Serial/batched native
  source parity and recovery checks cover these paths. Broader fitting now
  requires independent policy-anchor games, groups repeated reset seeds, uses
  bounded device batches and retains held-out complete-frontier predictions.
  Only training and anchor KL govern rollback. Dense bottle growth now implements
  384×12 and 512×12 with a fixed 320-wide token/candidate interface. Partitioned
  normalization and identity residual/projection initialization preserve the
  real parent's policy, value and candidate representations on 288 Mac CPU
  decisions; gradient and reload tests exercise the added capacity. These
  variants have 48.5M and 75.3M parameters versus the comparable 28.7M baseline.
  A common `split_seed` keeps held-out whole games fixed across independent
  training seeds. This is architecture and initialization evidence. Useful
  held-out ranking, substantial independent teacher training, separate measured
  equal-data/equal-GPU allocations and fast-student distillation remain open.
  A dedicated allocation supervisor now monitors GPU ownership, enforces a
  cutoff independently of the fitter, and exports only a complete checked
  checkpoint stored before that cutoff. Local snapshots retain two versions;
  failures preserve recovery files. Focused process tests cover interruption,
  contention, stalled-child termination and honest unused-budget reporting.
  A substantial exclusive-GPU study has not yet run under this controller.
- [ ] **Adaptive search and anticipation (E6).** Complete the public full-pair
  search experiments: candidate-dependent allocation, simultaneous-action mixed
  strategies, tactical extensions, and shared geometry with late preview
  conditioning. Use useful Q ranking and the correct reserve posterior; charge
  measured compute delay and test opponent-context ablations. Integrate the
  successful approach into live planning and run substantial controller
  tournaments. Frontier batching alone does not complete this item.
  Native event bookkeeping now implements an explicitly selected V2 public
  timeline independent of internal search polling, with versioned observer
  restore and hidden-commitment protection. Dense, sparse and automatic reveal
  advancement agree across eight natural games; 964 physics snapshots in 32
  games exactly match the prior engine. Actual public inference and Mac/Linux
  snapshot checks pass. Complete-reserve hypothesis execution and explicit V2
  source collection are now implemented as opt-in paths. Focused native and
  Python checks cover atomic private installation, complete public decision
  parity, natural outcomes, slot reuse and source resume. Twenty Linux checks
  pass. The order-balanced real-checkpoint comparison completed on four fresh
  V2 roots: all 1,206 complete tails, 43,287 public decisions including roots,
  and candidate targets matched exactly across all four runs, with no censoring.
  Prefilling reduced mean total job time from 300.8 to 270.1 seconds (10.2%)
  while sharing the GPU with training and V1 labeling. This supports using the
  opt-in path for subsequent V2 search experiments, not a strength claim;
  the running V1 source/label studies keep their frozen contracts.
  The initial bulk-reserve prototype changed the retained opponent bottle on
  some decisions and was rejected; its reproduction is retained under
  `runs/review-20260909/prefilled-reserve-audit`. The separate native reveal
  ordering fix prevents chance nodes from bypassing an earlier parked input.
  Geometry-only preparation is now implemented and consumed by the source
  backend. It retains complete own frontiers for both parities at the actual
  pace/delay, then scores the real preview/opponent/history on arrival. Fresh
  public context remains mandatory; legacy cached scores stay rejected for
  history actors. Native tests cover all nine previews, both parities,
  same-color rotations, gravity changes, carried controller state, exact
  witnesses and ordinary misses. A real 100M-core Metal JSONL audit matched
  all 61 playable decisions across Sloth/Normal/Top Humans/Frame Perfect;
  three Sloth positions were unreachable in both paths. Of the playable
  decisions, 52 reused geometry and nine replanned after own-board changes.
  Median paired savings were small and variable (about -0.2 to 2.7 ms across
  these paces) on the shared Mac. Keep the existing compute allowance. This
  implements exact geometry reuse with fresh scoring, not shared neural
  features, host integration or a demonstrated full-game strength gain.
  Complete simultaneous mixed-strategy backups are now implemented in both
  recursive and cooperative full-pair search. The opt-in mode retains all
  actions and correlated chance support, exports both players' strategies and
  measures the numerical saddle gap and solver time. Exhaustion or failed
  convergence suppresses targets. Analytic asymmetric games, reversed player
  perspective, dominated actions, nested simultaneous boundaries, correlated
  reveals and target rejection pass; existing expectation/minimax checks also
  pass. The initial 2,048-step mirror-prox matrix solve failed its numerical
  gap check on a real root (about 0.06 utility); common-offset conditioning
  did not fix it. The replacement bounded HiGHS dual-simplex solver converges
  and retains those failed probes. On four additional V2 native 14-HI roots,
  all 4,732 joint actions and 10,358 search nodes were covered in both drivers.
  Maximum joint-payoff disagreement was 6.45e-6; each mixture's best-response
  gap on the other matrix was below 3.67e-6. Three cases also met the original
  strict vector tolerance; the fourth retains that failure, reflecting
  mixture sensitivity to tiny neural rounding changes. Both paths evaluated
  3,302 neural rows; cooperative batches reduced calls from 3,302 to 152 and
  total measured time from 58.5 to 20.8 seconds on the shared Mac. Warm matrix
  solving took 1.3–1.7 ms; its separate cold start took 174 ms. This establishes
  numerical game equivalence, not calibrated candidate quality, exclusive
  throughput or stronger play.
  The existing unconditioned encoder now exposes exact reusable own/opponent
  bottle features, with fresh late preview/history/motor/candidate scoring.
  Stale model/input/autocast checks and inference-only use are enforced;
  conditioned models reject preparation. A registered diagnostic separates
  within-model reuse parity/cost from the untrained behavior change caused by
  disabling trunk conditioning. Distillation and full-game preservation of
  strength remain prerequisites; the current trained models are unchanged.
  The fixed 14-root Metal diagnostic (two recorded 14-HI roots per pace)
  now matches all 126 prepared preview choices, with maximum probability error
  9.54e-7 and value error 2.39e-7. Per-root median costs are 17.90 ms for
  nine fresh queries, 7.19 ms preparation plus 6.93 ms afterward; the median
  charged total is 14.23 ms. One query costs 9.93 ms fresh or 8.38 ms with
  only our bottle cached and the opponent freshly encoded. Preparation moves
  work earlier and is not free. Turning off conditioning changes 12 of 14
  original choices (median policy KL 3.40), so this is a fast-student option
  for the planned distillation study, not a current-model optimization.
  Retain this completed audit; do not spend another tiny fit on these 14 roots
  or change the product's execution charge from an offline measurement.
  Bounded root allocation is now implemented on this same queued traversal.
  Every legal action remains represented; unknown joint values retain [-1, 1]
  intervals and null W/D/L. Two interval games guide allocation along unresolved
  best responses and bound the returned mixture's response gap. Twenty distinct
  focused adaptive/mixed cases pass, covering player reversal, complete chance
  support, independent full-matrix bounds and rejection of unfinished batches.
  A separate four-root V2 native audit at depth one checked both 16- and
  32-action allocation batches against complete matrices. All eight interval
  and response checks passed with the declared 1e-5 numerical evaluation
  tolerance and 0.02 response-gap target. Complete evaluation covered 3,724
  joint actions, 9,150 native nodes and 3,728 neural rows in 10.48 seconds.
  Allocation batches of 16/32 covered 1,152/1,184 joint actions and
  1,160/1,192 neural rows in 4.07/3.57 seconds on the shared Mac. The largest
  actual response gap was 0.016375, within its retained interval bound. This
  demonstrates bounded allocation and a workload-specific computation saving;
  four development roots do not establish search strength or mature critic
  quality. Opt-in nested allocation is now implemented on the same cooperative
  traversal: chance and single-side backups propagate lower/upper bounds,
  deeper simultaneous nodes solve interval security games, and all depths share
  work budgets. Incomplete values retain null W/D/L and broad intervals rather
  than a neutral fallback. Analytic mixed, reversed-player, correlated-reveal,
  single-side and budget-failure checks pass. A native depth-four comparison
  on one V2 development position now passes independent complete-matrix
  interval and response checks for both root-only and nested allocation.
  Complete/root/nested runs take 317.48/289.64/135.64 seconds; nested inference
  batching uses 1,434 calls versus 12,058 for root allocation. Both inspect
  464 of 900 root pairs; nested work records 60 matrix certificates and 57,120
  actual simultaneous transitions across all depths. This demonstrates the
  mechanism and a saving on this position, not general performance or strength.
  Bounded public tactical extensions are now implemented and checked on two
  fixed 14-HI V2 development roots (eight variant runs). Top-four-row occupancy
  or one to four remaining viruses can extend leaf decisions; the original
  global work limits and complete chance/action inventories remain. The quiet
  root has identical values/policy and no extensions. The pressured root records
  1,260 extended decisions and takes 279.97/184.97/164.23 seconds for complete,
  root-adaptive and nested search, versus 7.38 seconds without extensions.
  Its policy total variation is 1.0 and maximum utility change 0.23666. Both
  adaptive modes pass independent complete-matrix bounds with zero violations;
  neither prunes any of this pressured root's 1,260 joint actions. The observed
  timing differences include cache/inference scheduling on a shared Mac.
  `tactical-native-assessment-v1.json` retains the full comparison. This proves
  the bounded mechanism changes a horizon-sensitive choice, not that the choice
  is better. Extensions remain off by default. Useful quality estimates,
  shared neural anticipation and substantial live strength evaluation remain
  open. Unilateral P1/P2 root allocation is now implemented with a complete
  action inventory, a directly checked regret bound and shared descendant
  work budgets. Reversed-player, mixed/chance continuation, unknown alternatives,
  budget failures and independent precise-vector checks pass. It does not
  create an inactive opponent root action. A fixed native comparison now
  checks three 14-HI V2 unilateral roots with and without bounded tactical
  extensions (12 runs). All six independent interval/regret checks pass, with
  zero actual regret against complete search; every root action is evaluated.
  Complete/root/nested times are 3.07/3.26/3.19 seconds for pressured P2,
  3.40/2.17/2.07 for pressured P1, and 0.126/0.143/0.141 for quiet P1.
  A separate depth-four comparison on the same quiet root passes both checks
  in 5.91/5.60/5.51 seconds, again with full root coverage. Neither native
  study reaches simultaneous descendants; that combined path is analytically
  tested but not claimed as native coverage. Retain both completed studies;
  do not keep deepening this root as another mechanics exercise. These timings
  reflect caching/batching on the shared Mac, not root pruning or live latency.
  Broad unknown upper bounds often require complete unilateral inventories;
  useful tighter allocation needs defensible bounds, not policy truncation.
  Partial matrices never
  enter quality training; these offline costs are not a live latency claim.
- [ ] **Persistent expressive play (E7).** Replay-align commentator windows
  and earlier construction states, then implement persistent 2–6-placement
  proposals with event termination and replanning after garbage. Train the
  proposer and evaluate actual persistent play using calibrated local regret,
  whole-game strength noninferiority and blind same-root clip preferences.
  Include horizontal, connected and delayed-payoff play without making a motif
  reward compensate for losses. A clip bank or a noninferiority calculator alone
  does not complete this item.

  The initial persistent proposal network and registered source/fitting recipes
  are now implemented. It retains causal root memory and a geometry goal for
  2–6 placements, scores actual new pills, and terminates on payoff, garbage,
  mismatched own state or its placement limit. The source extractor verifies
  complete post-lock own bottles, with original bonds, against the next replay
  observation and breaks at missing events or garbage. One original replay
  yielded 1,362 verified placements and 823 construction windows, excluding
  298 garbage intervals and seven incorrect/invalid locks. Three focused tests
  cover sequence corruption, persistent fresh inputs and event termination,
  strict checkpoint reload and identical learned weights after modifying only
  held-out sessions. Broader extraction completed on 512 sampled replay
  sessions: 781,583 verified transitions and 48,864 construction windows,
  containing 171,605 placement examples. It excluded 124,113 garbage intervals,
  4,706 mismatched boards, 1,228 invalid locks and 13 unsettled spawns. The
  registered eight-epoch CPU fit on mombox completed 304,400 construction and
  1,069,520 action presentations in 852 seconds. These are auxiliary
  proposals, not candidate values or
  style permission; the competitive core and installed trainer remain unchanged.
  Commentary alignment, learned preference, quality admission and persistent
  full-game evaluation are still open.
  The supplied commentary evidence also has an executable alignment review:
  three of nine passages match candidate video events; eight candidates have
  internally consistent reconstruction prefixes. Two passages lack a video
  identity in the excerpt. Across the relevant video games, 79 complete
  recorded prefixes match, 59 stop at missing timestamps and 89 at board
  mismatches. No candidate is video-verified or eligible for training by this
  check alone; player referents, timing and the actual video still need review.
  A subsequent broadcast keyframe review identifies atalito's praised Fat Log
  at April Gold 01:49:44: two adjacent blue/red horizontal rows clear together
  after a preceding vertical clear and falling half prepare the board.
  `commentary-fat-log-video-v1.json` retains this exact payoff and reconstructed
  earlier states; only its listed keyframes are video-verified. It is a concrete
  construction example, not a broad motif preference or strength label, and
  has not entered training. Continue the other passages and longer setup review.
  The final held-out assessment spans 10,814 windows from 113 sessions.
  Action NLL is 4.2125 versus 5.8331 for a training-only conditioned frequency
  prior. Automatic goal/horizon NLL is slightly worse than its prior,
  2.7461 versus 2.7346. Replacing root memory with current-state features also
  improves NLL slightly, to 4.2003; this is an input ablation, not a trained
  stateless control. The prototype learns placement patterns but does not
  establish useful persistent intent. Keep this negative result. The bounded
  next revision should represent specific spatial constructions on shared
  competitive features and compare a separately trained stateless control,
  with fresh replay confirmation before persistent full-game and preference
  evaluation. Do not promote it or extend the run on validation fluctuations.

  The second, spatial proposal study is complete. It uses exact colored
  clear-location targets and frozen competitive own-bottle mean/max features,
  with a separately optimized stateless control of the same size, initialization
  and training order. Both arms completed eight epochs and 1,069,520 action
  presentations each; feature extraction and fitting took 542 seconds on the
  shared Mac. Payoff geometry/duration are labels only; both move decoders
  consume predicted plans. The earlier development sessions are not a
  new holdout: all 512 original sessions, including the 113 validation sessions,
  were excluded by both session ID and replay-content hash from confirmation.
  The fresh source covers 256 reserved sessions, 394,902 verified placements
  and 24,392 constructions/85,199 action examples; 255 sessions have scored
  windows. Fixed-model confirmation completed without optimizer updates.
  Spatial prediction beats training-only goal/pill/preview frequency priors,
  but persistence remains worse than the trained stateless control: action
  NLL 3.7185 versus 3.6249, paired difference +0.0936 (95% 0.0861–0.1012).
  Earlier setup placements also worsen by 0.0634 (0.0514–0.0756); selected
  spatial-anchor hits are 14.90% versus 19.75%. These are recorded-prefix
  predictions, not autonomous construction completion or strength. Retain
  the learned spatial predictor and this second negative persistence result;
  do not install the persistent head. Code inspection confirms that the spatial
  move decoder already sees both current-state features and fixed root memory.
  Its spatial/duration plan stays rooted at the initial decision. The next
  bounded mechanism is implemented as explicit `recurrent-public-v1`: a GRU
  updates location and remaining-duration predictions from actual decision
  prefixes. Its separately trained control has identical capacity and inputs
  but resets hidden history. Both hold the root-selected intent during the
  recorded-prefix comparison. Runtime commits history once per completed
  placement, so preview revisions and repeated rankings cannot advance memory
  or extend the initial placement budget. This is a new mechanism, not evidence
  that persistence now helps. The fixed eight-epoch comparison is complete:
  1,069,520 action presentations per arm, 272.8 seconds with cached features.
  Fresh confirmation excludes both earlier banks' 768 session/content
  identities and scores 24,212 constructions from 256 new sessions. Persistent
  action NLL is 3.7084 versus 3.7013 for the history-reset control; difference
  +0.0071 (individual paired 95% −0.0014–0.0167), inconclusive. Early setups are
  also inconclusive; payoff NLL worsens by 0.0153 (0.0019–0.0298) and spatial
  anchor hits fall by 0.57 percentage points (−0.98–−0.17). Both arms beat
  training-only spatial, duration and intent priors. These are a new sample and
  changed control, so do not attribute improvement over the earlier experiment
  from cross-sample point estimates. Retain the tested event-update mechanism
  and all fixed models; this comparison has not established useful carried
  history. Do not extend fitting or tune on the new confirmation bank. Further
  confirmation must exclude all 1,024 session/content identities.
  The next useful evidence must measure actual plan completion/abandonment,
  reachable proposal choices and competitive/style effects, rather than repeat
  another recorded-prefix memory comparison. Calibrated local quality admission, actual persistent
  games and blind preferences remain open; this result does not cancel E7.
  The autonomous controller observer is now implemented: both frozen heads
  rank complete reachable frontiers while the competitive actor retains every
  choice. Actual lock/effect/bottle agreement verifies conditional payoffs;
  original targets, revised targets, garbage interruptions and unfinished
  terminal transitions stay distinct. A fixed 256-game allocation covers all
  seven 14-HI paces and separate 20-HI Normal/Top Humans, using the current
  45/30-frame slow reactions. These are compatibility diagnostics under
  unchanged competitive play, not a proposal-controlled strength evaluation.
  The allocation is complete: 256 natural games, 2,465,306 console frames,
  25,026 observed decisions per head and 21,684 verified transitions. There
  were no unexplained nonterminal mismatches; 3,182 garbage-interrupted
  transitions and 117 unresolved terminal transitions remain excluded.
  Carried/reset heads reached 428/13,546 and 503/13,573 original targets.
  Only 172/207 were first reached after multiple placements. Their initial
  budget was two in 13,498/13,522 plans and three in the remainder; neither
  proposed a longer budget or a crossing-clear intent. First preferences were
  reachable in 19,818/19,677 decisions, but agreed with the actual actor in only
  3,090/3,141. Target cells changed in 9,256/9,045 plans. This diagnoses short,
  frequently shifting predictions and low compatibility with unchanged play;
  it does not measure what would happen if proposal actions controlled play.
  Whole-seed descriptive intervals, separate pace/level results and actual
  first-hit examples are retained in `spatial-execution-assessment-v1.json`.
  These reused actor-evaluation seeds are not fresh strength confirmation.
  Do not promote either head or repeat prefix fitting. The next expressive
  work needs target-conditioned actual continuations and local quality
  admission, followed by full-game strength and blind style comparison.
  The next mechanism now explicitly conditions route selection on an immutable
  requested colored cell, geometry goal and six-placement cap. Hindsight cells
  are declared training requests; the live root proposer is frozen and reads
  only the actual board and two visible pills. Causal route memory updates after
  verified placements, while garbage/terminal/mismatch events terminate plans.
  A separately gated non-training controller now executes these proposals with
  complete paced witnesses, alongside common-seed unchanged controls. Focused
  tests verify real native tapes, exact shadow preservation, unchanging goals,
  causal memory, whole-session fitting isolation and strict model reload.
  The fixed eight-epoch fit and 512-game actual-control allocation are complete.
  Training presented 304,400 windows/1,069,520 actions, with zero outcome frames;
  development requested-goal NLL improved from 3.6649 to 3.5780. The final epoch
  was retained without selecting a development minimum. That imitation gain
  did not transfer: the controller reached zero original targets in 802 plans,
  versus 354 in 5,960 unchanged-control plans (254 after multiple placements).
  All targets stayed fixed and actual locks/effects/bottles were verified.
  At 14 HI, controlled-minus-unchanged match scores were −28.1/−34.4/−65.6/
  −59.4/−62.5/−78.1/−65.6 percentage points from Sloth through Frame Perfect.
  Whole-seed simultaneous seven-pace intervals exclude zero for every pace
  except Sloth. Separate 20-HI Normal/Top Humans differences were −31.3/−43.8
  points, inconclusive under their two-condition family. This small, reused
  actor-evaluation bank is sufficient to reject this severe regression; it is
  not independent confirmation or a completed expressive-play stage.
  `target-construction-assessment-v1.json` independently checks all 512 journals,
  identities, side allocations and original payoff geometry. Reject this route
  controller and retain the competitive lineup. The next approach must ground
  goals/routes in actual reachable intermediate states and admit choices using
  demonstrated competitive quality, before substantial noninferiority and blind
  same-root preferences. Do not repeat prefix comparisons, extend imitation
  epochs or let a target payoff compensate for losing.

This task owns the continuation; its four-hour follow-up checks the active studies
and resumes the first unfinished item. Keep the dashboard and this existing
roadmap current with the concrete next action and evidence. Use tf3090 for large
training, the Mac and mombox for suitable evaluation, and mombox for overflow
storage. Level 14 HI remains primary, with level 20 HI reported separately.
Use focused checks during development and full applicable checks before release.
Finish implementation, representative training and evaluation; promote only
supported changes. An unsuccessful variant needs an explicit result and a next
approach, not a silent deletion of the requested capability. None of these boxes
is complete merely because the work is documented or scheduled.

The user authorized this hourly continuation as a finite four-stage program,
including training, evaluation, code changes and commit/push across these three
machines. Judge adoption against stronger absolute Max play, stronger
motor-constrained play, and more expressive play with competitive strength
preserved. Do not hide a pace regression inside aggregate results. Keep
implementation, evaluation and promotion separate: a completed, tested variant
can fail its adoption criteria. Once all four capabilities have been implemented
and substantively evaluated, report the three outcome assessments and pause the
follow-up. Do not silently add further stages or keep optimizing indefinitely
until a favorable result appears.

Primary methodological references: [episodic policy gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html),
[PPG](https://arxiv.org/abs/2009.04416), [population responses](https://arxiv.org/abs/1711.00832),
[targeted archive-state search](https://arxiv.org/abs/2302.12359),
[Gumbel planning](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf),
[gradient stability](https://arxiv.org/abs/2506.15544), and
[joint model/data/compute scaling](https://arxiv.org/abs/2508.14881).
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
