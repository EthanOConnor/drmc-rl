# Unified player system design

## 1. System objective

`drmc-rl` builds one strong game-theoretic decision system and derives three
products from it:

```text
public/belief pair state
    -> rating-independent competitive quality model
    -> optional strict joint-event search
    -> exact execution planner and operation-profile filter
    -> product decoder
         unrestricted: quality argmax
         human-rate: constrained quality argmax
         trainer: calibrated regret -> style -> cadence -> motor execution
```

There may be several teachers, exploiters, and historical checkpoints, but
there is one common notion of competitive move quality. Rating and style never
change what the system believes is good; they change which boundedly suboptimal
choice and execution are selected for the trainer.

## 2. Placement SMDP

The primary action remains a final pill pose from the fixed `4 x 16 x 8 = 512`
space. The exact reachability planner supplies:

- feasibility;
- minimum frames to lock;
- an exact frame-indexed controller script when execution matters;
- eventually, Pareto execution alternatives under named operation profiles.

Training advances the native engine directly to the next decision boundary.
Returns and GAE use `gamma ** tau`, where `tau` is the actual elapsed frame
count. Emulator and live paths replay exact scripts rather than warping.

The action contract is provisionally `(placement)`. The timing-action gate tests
whether any valid delayed lock changes pair transition state or continuation
value. If material, the contract becomes hierarchical:

```text
placement -> execution/timing option
```

It will not become an unstructured placement-by-hundreds-of-delays action grid.

## 3. Information boundary

### PublicPairState

The only state accepted by a deployable policy. It contains visible or
reconstructible information:

- both visible settled bottles;
- visible current and preview pills;
- visible active pill pose/phase when available;
- observable relative timing and snapshot age;
- visible spawn, lock, clear, volley, and terminal events;
- the player's own exact controller/planner microstate.

`audit_public_mapping` rejects known hidden-state keys recursively. A future
native adapter must document how every field is observed or inferred.

### PrivilegedPairState

Training-only state for centralized critics, counterfactual teachers, parity,
and strict search:

- full pair clocks and native phases;
- both decision flags and committed actions;
- internal pending attacks;
- a restorable engine checkpoint;
- terminal state.

Future RNG is never exposed as a structured field or model feature. Exact
native restore bytes are an opaque teacher capability and may contain private
engine continuation state; policies, public-state hashes, and leaf evaluators
must not inspect or condition on that encoding. Reveal-time chance integration
belongs in the search adapter. Deployment code must request a `PublicPairState`,
not downcast a privileged object implicitly.

### Public belief/history

The public state carries bounded semantic event history. Recurrent policy work
should encode one token per spawn, lock, clear, volley, and terminal event; it
must not process hidden native tensors or require frame-by-frame video.

## 4. Competitive model hierarchy

### V3 exact-afterstate human teacher

V3 remains the high-fidelity source for:

- exact one-placement afterstates;
- human choice/style;
- timing and cadence;
- immediate clear, topout, virus, and attack consequences;
- an initial competitive prior.

The mature competitive head must no longer be defined by cross-entropy toward
the human's observed action. Human-choice supervision belongs only to human and
style heads.

### Full-pair counterfactual teacher

For every legal action, the teacher clones a complete pair state, advances
causally to the next event, and estimates continuation W/D/L against the current
population mixture. Labels include:

- calibrated `P(win)`, `P(draw)`, and `P(loss)`;
- expected score and win-minus-loss utility;
- win-logit regret relative to the best action;
- clear/topout, virus, attack, and timing consequences;
- teacher disagreement/uncertainty;
- improved root policy targets.

`drmc_rl.teachers.counterfactual` enforces full legal-action coverage. A
candidate omitted by a beam is an error, not an implicit low-value label.

The frozen comparison teacher uses the hash-verified Strong League ensemble in
`drmc_rl.search.strong_league`. Its labels failed the held-out V3 comparison and
are not eligible for training. The diagnostic adapter in
`drmc_rl.search.public_policy` instead uses the deployed outcome-trained public
core's policy and value, with an independently fitted W/D/L link. This is still
an offline teacher with privileged native transitions. No diagnostic opens a
quality or product gate.

The terminal-rollout pilot tests a separate, outcome-defined teacher under the
installed public core. Every root action continues through the asynchronous
pair game until natural termination. Distinct complete reserves are enumerated
with their exact public-posterior mass, and every newly revealed preview is
overridden before the actor observes it. No critic or horizon-as-draw fallback
supplies missing outcomes. This bounded experiment is motivated by the failed
shallow values and changed historical trajectories; it does not replace the
existing gate or authorize distillation from unvalidated labels.

Decision-trained critics are evaluated only at actionable boundaries. Search
finishes forced deterministic and reserve-reveal events after the nominal
depth expires, then evaluates an acting side and reverses calibrated W/L if
that side is the opponent. It does not choose an additional player action after
depth expiration. The versioned leaf contract is
`acting-decision-after-forced-events-v1`; budget-exhausted searches are rejected.

The native search ABI stops immediately before a private reserve pill becomes
public. `PillReserveBelief` conditions the declared reset-seed prior on the
public initial bottle and observed pill history; search integrates every
posterior-supported reveal before causal advancement. Independent `1/9` mass
is retained only in the historical mechanics pilot. The unrevealed reserve value
is never presented to a continuation network. W/D/L is produced through a
positive-slope Davidson link calibrated on game-group-held-out natural Strong
League continuations; horizon-truncated games are excluded rather than labeled
as draws.

### G5/V5 fast student

G5 is the rollout and deployment policy. It uses shared bottle encoding,
pill-conditioned processing, cross-bottle interaction, candidate-set attention,
and a distributional value. It is initialized from:

1. the exact V3 teacher;
2. the strongest frozen G4 lineage for long-horizon structure;
3. strict joint-event search targets as they become available.

The frozen corpus bootstrap uses a 72-wide zero auxiliary vector. Its explicit
`zero_v1_vs` checkpoint contract preserves that width in outcome PPO, frozen
opponents, and evaluation without reading pending attacks or other legacy
context. Natural full-game outcome learning may improve this public bootstrap
while search-quality evidence remains staged; failed search labels never enter
that run. Such a checkpoint still requires paired arena evidence before use and
does not establish calibrated candidate regret or human ratings.

Semantic bottle planes always retain capsule bonds. The frozen VS actors
(including the public outcome bootstrap) use the historical encoding that
hides horizontal bonds on each side's same-color pill turns. Apply that lossy
encoding only at their network boundary, using both public falling/last-pill
colors. V3 exact afterstates must decode the native board bytes, never those
masked actor tensors. The live bridge sends complete semantic bottles and the
opponent's public pill colors so Maximum receives its training representation.

The current V3 diagnostic trainer selects among non-overlapping, ordered
log-regret bands. Their width is twice the former envelope tolerance, evaluated
at the fixed corpus mean rating. Live play holds the human/style conditioning
at that same population reference; requested rating changes regret. Named pace
independently sets mechanical limits and consequently also affects match strength.
This prevents style and overlapping windows from reversing a strength change
on a fixed position at a common regret quantile. Coaching retains the requested
human condition. These bootstrap score bands do not certify win-odds regret,
absolute rating, independently selectable styles, or a complete skill range.

The local trainer applies named motor limits inside native frame reachability,
before regret or Maximum selects a move. Sloth through Super Human constrain
reaction, controller-change spacing, actual horizontal/rotation spacing
(including DAS), and button overlap. Frame Perfect uses unrestricted exact
reachability. Profile-valid monotone routes accelerate ordinary positions;
exhaustive constrained search covers all unresolved candidates. There is no
unrestricted fallback or candidate truncation. Costs for constrained modes are
realized witness durations, not necessarily globally minimum costs. Independent
Python replay validates the selected witness and its full mechanical envelope.
These authored product presets are not corpus-certified ExecutionProfiles;
Top Humans is a setting name, not a validated percentile. Sustained burst,
correction, and human-distribution calibration gates remain separate.

### Anticipatory execution

Maximum can prepare the next turn while its committed controller script runs.
The public current preview supplies that turn's pill. A one-placement native
simulation predicts the settled own bottle; it is not a search teacher or an
opponent rollout. Exact reachability covers both possible spawn parities, and
one batch scores the nine possible new previews for each parity. These are
conditional answers selected after the actual reveal, not nine independent
events assigned uniform probabilities. No reserve, RNG or queued attack state
enters the actor.

Prepared execution requires an exact observed own bottle, pill, gravity speed
and controller microstate. Incoming garbage, a different lock, or a controller
correction invalidates it. The selected implementation retains the previous
opponent observation. The exact-context comparison additionally requires the
same public opponent input; neither variant relaxes own-state validation.
An opponent-input ablation zeros those
channels with fixed weights; it does not establish the capability of a retrained
opponent-blind policy. Candidate buffers must be owned across native BFS calls.

The host measures queue, serialization, pipe and inference round-trip time.
Adaptive execution uses a four-frame floor plus an eight-millisecond margin
over the recent maximum, without retiming an outstanding decision. Preparation
removes compute wait on a hit; named reaction and motor limits still apply to
the witness. Skip preparation when the human reaction floor already covers
computation: it saves no movement time and would age the opponent observation.
Lower skill settings keep their regret decoder and existing timing.
The legacy eight-frame path remains available for controlled comparison.
The measured timing and full-game selection evidence are in `ROADMAP.md`;
this execution improvement does not open the separate calibration/search gates.

### Repertoire and motor-aware strategy

The current Maximum actor chooses policy logits, not calibrated candidate win
probabilities. A small logit gap does not certify a small competitive loss.
Do not add a horizontal/combo bonus to that decoder and describe it as free
style. Repertoire first enters through outcome-only start-state sampling and
exact tactical auxiliary predictions. Named motifs require verified sequences
of placements, bonds, falls, and clear waves; a horizontal capsule alone does
not identify horizontal play or a Fat Log.

`human/repertoire.py` measures first-wave horizontal/vertical lines, their
intersection, length, and union size on a settled raw-NES bottle. The audit
checks those labels against native clear resolution and measures later clear
events separately. Raw-NES and canonical policy colors are converted only at
their respective boundaries. Source roots that already fall or clear when
resolved are excluded from the curriculum. Old synthetic practice banks that
remove viruses can violate this requirement even when their reset succeeds.

The human curriculum is grouped by complete replay session before splitting.
It selects situations with reachable combinations, not a preferred action or
an additional reward. Its inherited asynchronous two-board snapshots, omitted
in-flight attacks, and randomized future make it synthetic training data,
not an exact historical continuation or a public-posterior quality label.
Matched clean-start training and held-out full-game outcomes are needed to
attribute a gain to this sampling change.

The public outcome bootstrap learned unrestricted movement. Its feasible set
and realized costs change at live pace, but it receives no explicit motor
envelope describing what future pills will be able to do. Pace training must
teach board preparation across successive constrained turns. A root cost
ablation alone cannot test this. Use the same reaction window, gravity,
mechanical limits, and causal opponent/garbage progression during rollouts;
replay representative scripts against the live host before trusting a
cost-injected macro approximation as a motor-training environment.

A conditional adapter on the common frozen core is the first isolation
experiment for slow strategy. Its public context includes the motor envelope
and cartridge gravity, not skill rating or hidden opponent state. Preserve
the existing fast route exactly until paired per-pace results support a
shared-weight replacement. Root locks during the compulsory reaction window
are physical limits: training cannot rescue an already doomed spawn, though
it can learn to avoid creating such boards on earlier turns.

Live inference selects an available accelerator and warms its kernels before
readiness. Metal candidate shapes are padded to bounded buckets; first-use
shape compilation must not consume a gameplay deadline. Padding changes
neither legal candidate coverage nor the policy's information scope.

The representation bakeoff compares:

- root-only G5;
- V3-distilled G5;
- G5 with exact effect tokens;
- recurrent/event-state G5.

`effect_tokens.py` supplies deterministic summaries of resolved candidate
changes, heights, holes, top pressure, terminal type, clear/attack targets, and
uncertainty. Integration into the hot model is gated by measured arena strength
per millisecond.

## 5. Joint-event search

The correct competitive process is an asynchronous pair game. A search node is
a restorable full-pair state at one of five boundaries:

- P1 needs an action;
- P2 needs an action;
- both need actions;
- deterministic/chance advancement;
- terminal.

`drmc_rl.search.joint_event` implements the backend-independent search and
requires a `PairSearchModel` adapter. At simultaneous boundaries it evaluates
joint actions, integrating the opponent policy or applying a minimax stress
mode. Chance branches represent only information newly revealed by the game.
Depth is measured in pair events, not the learner's pill count.

Search begins as an offline teacher. It does not control PPO rollout behavior
until paired same-weight evaluation opens the joint-search gate. The existing
own-board depth-2 search remains a diagnostic/legacy teacher; it is not the
architecture target.

## 6. Execution layer

`ExecutionProfile` defines a named operation envelope over exact scripts:

- reaction latency;
- minimum edge interval;
- edge bursts over 250 ms, 1 s, and 10 s;
- simultaneous buttons and forbidden chords;
- direction reversals and correction bursts;
- rotation and soft-drop behavior;
- total complexity.

`script_metrics` and `pareto_frontier` support profile fitting, validation, and
selection among scripts with different lock time, burst, edge, and complexity
costs. Profiles used for claims must be signed, versioned corpus artifacts; the
built-in elite profile is explicitly provisional.

Scripts start after the observed spawn boundary. Pass the already-held
`initial_buttons` to metrics and validation so carried inputs are not counted
as fresh presses. Inter-edge intervals measure distinct controller-change
frames; simultaneous changes are counted by chord and burst metrics.

The human-rate product maximizes competitive value after filtering scripts by
one profile. It never introduces an intentional strategic mistake.

## 7. Human trainer decoder

The trainer uses the following strict order:

1. score all mechanically feasible candidates with the common quality oracle;
2. convert quality to calibrated win-probability logit regret;
3. sample a target regret from the requested rating and decision opportunity;
4. adjust the target from corpus-fitted context and slowly varying form;
5. retain the closest regret envelope;
6. apply human likelihood and explicit style only inside that envelope;
7. sample decision cadence;
8. select an exact script satisfying the execution profile.

This separates:

- **strength:** surrendered competitive value;
- **style:** choice among similarly valued moves;
- **cadence:** when the move is executed;
- **motor execution:** how the intent is realized.

`StyleSpace` residualizes behavior features against rating before extracting
player-level latent axes. `HumanFormState` supplies temporally correlated error
and hesitation. `AdaptiveSparringController` updates player skill over blocks
and limits target changes to avoid per-game rubber-banding.

## 8. Population training

The permanent population contains four roles:

- main agent;
- main exploiter against the current main;
- league exploiter against the historical/meta mixture;
- human/execution exploiter targeting human styles and constrained players.

Arena Elo is descriptive. Promotion uses the full payoff matrix and a
regularized PSRO-lite meta-strategy from `drmc_rl.arena.meta_strategy`.
Candidates must improve mixture value while avoiding catastrophic regressions
against active opponents and permanent human/style/execution anchors.

## 9. Objective

Final competitive optimization is W/D/L:

```text
win  +1
Draw   0
loss  -1
```

Tactical quantities are used as auxiliary heads, curriculum priorities,
start-state selection, or annealed potential shaping. Style is lexicographic:
first remain inside an allowed competitive-value loss, then optimize the style
preference. No accumulated shaping term may compensate for losing.

## 10. Verification boundary

The native engine and fast planner are optimized models, not their own oracle.
Independent evidence includes:

- `drm_reach_bfs_full` parity and fuzzing;
- recorded NES traces;
- emulator/controller-script replay;
- exact forced-lock pair advancement;
- hidden-information audits;
- candidate-completeness telemetry;
- immutable artifact manifests and gate evidence.

Every claimed player identity includes checkpoint and config hashes, repository
and native revisions, observation schema, execution profile, search settings,
corpus release, parents, and promotion evidence.
