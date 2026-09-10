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
| E6: search and anticipation | Cooperative frontier batching with exact requested-side inference, complete cache identities, fail-closed reserve-history cache, full-root output and unknown exhausted labels | Synthetic and real neural/native parity pass. CPU/Metal timing is workload-dependent, so batching remains opt-in. Adaptive/Gumbel allocation waits for useful Q ranking; mixed simultaneous matrices, tactical extensions and a late-conditioned anticipation trunk are separate experiments. Charge their measured computation delay. |
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
  Fresh Frame Perfect, Sloth, Relaxed and Normal confirmations find reach/clear
  prediction worse than the training-only pace/cell prevalence baseline. The
  random-initialization comparison therefore does not establish useful motor
  prediction. Complete the remaining conditions and strength study, and make a
  bounded revision of initialization and auxiliary fitting using training data
  before another independent confirmation; do not promote this first fit.
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
  counts and natural draw rates. The separate fixed 100M study is running;
  substantial later learning and device evidence remain pending. Sloth's many
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
  Make public observation updates independent of internal search polling
  cadence before collapsing complete-reserve rollouts into strict transitions.
  The initial bulk-reserve prototype changed the retained opponent bottle on
  some decisions and was rejected; its reproduction is retained under
  `runs/review-20260909/prefilled-reserve-audit`. The separate native reveal
  ordering fix prevents chance nodes from bypassing an earlier parked input.
- [ ] **Persistent expressive play (E7).** Replay-align commentator windows
  and earlier construction states, then implement persistent 2–6-placement
  proposals with event termination and replanning after garbage. Train the
  proposer and evaluate actual persistent play using calibrated local regret,
  whole-game strength noninferiority and blind same-root clip preferences.
  Include horizontal, connected and delayed-payoff play without making a motif
  reward compensate for losses. A clip bank or a noninferiority calculator alone
  does not complete this item.

This task owns the continuation; its hourly follow-up checks the active studies
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
