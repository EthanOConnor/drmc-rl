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

It explicitly excludes future RNG. Deployment code must request a
`PublicPairState`, not downcast a privileged object implicitly.

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

### G5/V5 fast student

G5 is the rollout and deployment policy. It uses shared bottle encoding,
pill-conditioned processing, cross-bottle interaction, candidate-set attention,
and a distributional value. It is initialized from:

1. the exact V3 teacher;
2. the strongest frozen G4 lineage for long-horizon structure;
3. strict joint-event search targets as they become available.

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
