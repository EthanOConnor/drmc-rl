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

The native placement simulator's raw output is **not a simultaneous public
snapshot**. `warp_fall` can write the committed opponent lock and advance its
clock before the other side reaches that time. The September 2026 audit
reproduced two hidden opponent commitments with different future bottle cells
and endpoint clocks while the viewer was still parked at the same decision.
Removing pending-attack scalars alone did not prevent this leak.

The frozen public studies use `causal-settled-pair-v1`. Starting at
a fresh joint decision, capture carries the previous public view through exact
branches. An ahead-of-time warped side retains its last observed bottle and
pill context, with explicit snapshot age and unknown active phase. It refreshes
when the causal timeline catches up. Future endpoint clocks remain private.
This is a conservative public observation of settled bottles, not a rendering
of the falling animation. Native physics and restore bytes are unchanged.
For new native PPO runs, `env.public_observations: true` also selects strict
causal advancement. The legacy vector scheduler could request actions from
both sides at different simulated times; filtering its buffers after that
would not be sufficient. The public mode retains separate visible boards,
pills and ages through partial resets and the direct array-input path.
This is a new experiment contract, not a silent continuation of an old run.
Public zero-aux PPO rejects the legacy mode; the historical scheduler remains
available for its original privileged experiments and source reproduction.
The causal PPO collector joins intervening pair events into one placement
transition, including wait rewards and exact public elapsed time. It closes
only at the learner's next feasible choice or natural termination. Fixed
per-learner quotas drain under one unchanged behavior policy; each last
transition keeps its own successor-boundary bootstrap. Waiting and forced
states are not actor samples, and pending samples never cross an optimizer
update. Censored terminations abort instead of becoming draw targets. This
initial collector supports gamma-one outcome training and uses host-side
transition storage; accelerator retention and complete live public-context
emission need separate throughput/information validation.
Historical `legacy-warp-buffer-v1` rows cannot initialize this timeline and
are rejected by `PublicPolicyContinuation`. Preserve them as historical
teacher evidence, never relabel them as fair public observations. Frame/event
controller rollouts use actual per-frame state and do not use this warp path.

V1 retains the last Python-captured public bottle. It is conservative, but
different polling schedules can retain different bottles at the same later
decision. The rejected complete-reserve shortcut reproduced this without a
physics difference. Preserve V1 and its fixed polling schedule for existing
studies; their inputs and labels have not been silently migrated.

`causal-settled-pair-v2` moves settled-observation bookkeeping into native
events. Select it explicitly with `capture_native_state(..., event_public=True)`;
subsequent branches inherit it. Each side retains its visible sample and at
most two pending samples: a one-frame-ahead boundary and an atomic fall
endpoint. Samples become visible only when the causal pair clock catches up.
Reads are const, and the public API never exports a future warp endpoint or
committed pose. A parked P1 can retain its one-frame-ahead own decision state,
consistent with the engine's within-frame ordering. The contract still omits
falling animation; the actual frame controller keeps its separate exact public
history and incurs no settled-observer copies.

Native snapshot V2 appends this observer state to the unchanged physics body.
Restoring V1 preserves its original bytes and leaves the new observer unavailable;
legacy non-strict and forced-spectator stepping also invalidate it. Reset starts
a new valid timeline. A missing observer fails rather than inventing history.
V2 public trajectories require their V2 snapshots and cannot be made by relabeling
V1 rows. On 32 natural audit games, all 964 physics snapshots matched native
`e0162ed` exactly and all observer restores matched. Dense, sparse and automatic
reveal advancement also produced identical V2 inputs across eight natural games.
This enables further bulk-reserve experiments; it does not itself certify a
new continuation policy, Q corpus, throughput gain or search-strength gain.

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

`paired_terminal` generalizes this diagnostic to a named frozen
continuation/opponent panel. Every feasible root action uses exactly the same
distinct complete reserves and panel members. Utilities use W−L, an affine
transform of the arena score W+0.5D, so their scale is explicit. Chance variance,
continuation sensitivity and paired action differences are separate outputs.
Exact finite-panel enumeration has no Monte Carlo reserve standard error;
continuation sensitivity is not a confidence bound on optimal play. Paced
rollouts require a separately verified execution adapter; the initial panel
explicitly uses native SMDP execution.

Complete panels produce a conservative reference-relative policy target:
`pi_ref * exp(shrunk_advantage / eta)`, with eta chosen for a declared KL cap.
The sensitivity penalty is a risk preference, not a calibrated lower bound.
A missing/censored candidate suppresses the policy target. Supervised fitting
uses candidate WDL, state WDL under the reference root policy, paired gaps,
ranking and optional policy improvement. It splits and weights by source game,
rejects mixed reanalysis contracts and never treats teacher choices as PPO
behavior likelihoods. An auxiliary phase preserves the current policy with
measured full-dataset KL and rollback. These are fitting diagnostics; predictive
accuracy, action-ranking improvement and equal-latency game improvement remain
three different evidence requirements.
Successive supervised phases retain the learned heads and effective inference
weights. A mode/schema change is rejected; new architecture ablations migrate
from the same frozen core instead of silently discarding learned tensors.

Compute comparisons distinguish an equal maximum GPU allocation from equal
data exposure. `budgeted_quality_fit` gives one child process a fixed wall-time
allowance on a named physical GPU. Startup, reference inference, updates,
validation and local snapshots consume that allowance. An independent watchdog
terminates the child at cutoff; only a complete checkpoint stored earlier can
be exported. Snapshots follow successful training/anchor policy checks and
finite descriptive validation. Holdout scores never select a snapshot or
change the next learning rate. Partial epochs and unfinished writes cannot
replace the last checked model. Actual allocation, unused time, cutoff overrun,
checkpoint age and selected-model counters are retained; export to overflow
storage happens after the GPU child exits. A process query checks ownership
before launch and approximately every second. Detected contention invalidates
the comparison; these sampled checks cannot exclude a transient external job
that starts and ends between queries. Independent teacher studies must also
schedule the GPU for exclusive use.

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

Opt-in G5 critic context `candidate_attention` adds a value query over all
valid candidate and pair-interaction tokens. Its residual projection starts at
zero, preserving the old value on migration. Padding and no-action boundaries
are covered explicitly. Separate three-class state and candidate WDL heads
are fitted by `quality_supervision`; the legacy 51-atom scalar is not renamed
or assumed calibrated.

`public_pair_context_v3` is a separate checkpoint/input contract. It preserves
all bonds and feasible poses and encodes both visible current/preview pills,
active-pose availability, public phases, observation age, bounded public event
history and the viewer's own motor/gravity/compute envelope. Each bottle gets
its own pill/context conditioning. Unknown fields have masks; hidden native
bytes are not model tensors. The controlled migration copies compatible
weights and zero-initializes new conditioning columns. Ordinary checkpoint
loading stays strict. Critic-only, context-only and combined ablations precede
increasing model size. The controller-frame and batched-event arenas now use
the same native in-tick event emitter and charged motor context. New-context
policies retain all planner poses, including same-color rotations; frozen
actors retain historical deduplication. Policy memoization includes public
history and execution. Legacy speculative answers and opponent ablations
cannot be reused by a context actor. The shared desktop/browser scheduler now
emits `public-controller-history-v1`: a fixed 32-event history collected from
exact native effects or passive verified cartridge instruction hooks, with
round-relative active time, both visible previews/poses and decimal virus counts.
The source backend validates and converts that wire view to the same public
model input used in controller arenas. Gravity names the actual counter period
(ROM threshold plus one), including the one-frame limit. Full-core learning,
future motor-opportunity features and device/strength evaluation remain pending;
missing history or motor features must not be fabricated as known inputs.

The source backend also supports geometry-only next-turn preparation. A
committed placement predicts only the settled own bottle, next known pill,
gravity and carried horizontal controller state. Both spawn parities retain
complete frontiers at the requested pace and execution delay. The bounded
two-entry cache contains no policy scores or predicted public history. A
single-use opaque token accompanies the actual next decision; own geometry,
microstate, observation contract and execution profile must match exactly.
Scoring then receives the actual preview, opponent board/pill and live public
history. Opponent changes alone do not invalidate geometry, while incoming
garbage or other own-state changes cause ordinary fresh planning. Legacy scored
anticipation remains unavailable for context actors. This shares native
feasibility, not a cached neural trunk; host scheduling, larger latency studies
and full-game evaluation of any changed compute allowance remain separate.

The controller outcome trainer can now update the full G5 on this input
contract. It reuses the measured causal controller collector and episodic loss,
but stores exact public model inputs instead of frozen trunk features. A fixed
post-migration initial policy supplies KL regularization; no failed quality
labels enter PPO. Outcome gradients reach the board encoder and public context.
Optional replay shards retain the complete legal inventory and natural outcomes
under their observed continuation, with game identities excluded from actor
inputs. These do not label unchosen alternatives or calibrate the separate WDL
heads. New full-core checkpoints still require held-out controller tournaments.

New public-context migrations set `candidate_context_residual: true`. Each
bottle initially uses the legacy acting-pill conditioner; two learned scales
introduce deviations from its own side-specific public conditioner. Zero scales
and zero new auxiliary columns preserve the parent's policy and value on equal
model inputs/frontiers. The original direct-side migration changed the
opponent's FiLM input immediately despite zero new columns. On 288 recorded
Normal/Top Humans/Frame Perfect decisions with the real 320×8 parent, it changed
12 greedy actions and shifted a value by as much as 0.815; residual migration
preserved every probability and value exactly on the Mac CPU. This is migration
evidence, not full-game or deployment-latency certification. Existing context
checkpoints without the option retain their direct-side graph, including their
original regularization reference on resume. New side scales and context weights
receive outcome gradients; no history input is replaced with a fabricated value.

Equal tensors are not equal live input contracts. A training-replay attribution
audit now separates own bonds, opponent bonds and the same-color frontier.
The expanded frontier is the largest observed source of changed decisions.
`trainer-public-input-alignment` therefore supplies an explicit migration fit:
the frozen parent sees its historical public encoding/frontier, while the
student sees full bonds, public context and every legal orientation. The target
mixes the parent's probability with a declared small uniform component over
the full student frontier; this preserves support for new controller actions.
Unsupported teacher roots are excluded from fitting and counted. These are
behavior targets, never candidate quality or optimality labels. Whole reset
seeds stay together across shards; validation is descriptive and the fixed
final epoch is retained. Held-out full games must establish whether alignment
recovers strength before a new outcome-training branch. Existing runs keep
their original graph, reference and checkpoints.

The live backend executes the exact orientation selected from that frontier.
Same-color canonicalization belongs to the frozen actor's frontier construction;
applying it after a context actor's choice can change the witness, timing and
carried horizontal state despite an equivalent visible placement.

Motor effect/access heads can first learn on the frozen competitive candidate
representation. New output layers start from cell/parity prevalence and cost
means fitted only on training games, avoiding a random 50% prior for sparse
opportunities. A fixed head-only phase uses lossless cached public features;
future labels remain separate targets. The core's parameters and buffers must
remain identical throughout that phase. Joint training then discards the cache,
recomputes actual public features and uses separate core/head learning rates
under the existing training/anchor policy-KL limit. Neither development nor
confirmation metrics select optimizer updates. The auxiliary heads remain
absent from ordinary inference, and their learning cannot supply a motif reward
or certify playing strength.

Larger quality teachers can expand the dense bottle encoder with
`grow_bottle_encoder`. In this comparison, 320×8, 384×12 and 512×12 explicitly
mean bottle channels × residual blocks; token attention and the candidate
interface remain 320 wide. Existing channels retain their original GroupNorm
populations, added channels use separate populations, and old output channels
initially ignore new inputs. A learned projection starts as identity on the old
channels and zero on new channels. Added blocks start as residual identities.
The added readouts receive gradients immediately; their random features become
trainable without first destroying the parent policy. This is an encoder-specific
form of function-preserving growth, motivated by
[Net2Net](https://arxiv.org/abs/1511.05641), not a claim that normalization permits
arbitrary prefix copying when widening the whole transformer.

The real combined-head parent has 28,656,219 parameters; its grown 384×12 and
512×12 variants have 48,532,059 and 75,338,715. All three preserved every policy
probability, value and candidate representation on 288 unchanged public
controller decisions in the Mac CPU FP32 audit. Small deterministic tests also
exercise learned projections, repeated growth, all prediction heads and gradients
into new channels/layers. The RTX 3090 FP32 audit retained every greedy choice;
maximum total variation was 7.83e-6 and maximum log-probability error 6.75e-5,
within the existing 1e-4/1e-3 policy bounds. Its value and representation errors
(5.89e-5 and 6.64e-4) exceeded the stricter initialization probe, so that failed
probe remains recorded. The modified bottle computation matched exactly in
FP64 on 16 bottle inputs, supporting a shape-dependent rounding explanation.
Substantive training, equal-data/equal-compute evaluation and distillation remain
separate requirements.
Keep training/anchor/validation banks identical across sizes: `split_seed` is
independent of the model and optimizer `seed`. Architecture size is not evidence
of a better teacher, and wall time under competing GPU jobs is not an exclusive
GPU allocation comparison.

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

`human/expressive_sequences.py` supplies a separate construction-imitation
diagnostic. Recorded raw-color lock poses must be supported, collision-free,
and reproduce the entire next observed own bottle after exact bond-aware
cascade resolution. Gaps, missing locks, mismatched previews, unsettled roots,
and incoming garbage split sequences. It never repairs a wrong recorded lock
by choosing an arbitrary action with a matching payoff. Current labels are
observed horizontal, crossing, large-wave and cascade clears, not named
community motifs, preference judgments, or candidate values. Each 2–6-move
window precedes the first occurrence of its selected geometric payoff.

`human/expressive_proposer.py` learns auxiliary proposal order. The root's
public own-board/pill/preview embedding remains fixed through the construction;
each step additionally reads the actual current own board, pill and preview,
the persistent geometry goal and remaining placement count. Future queue
entries, the observed future payoff and hidden opponent state are targets or
absent, never actor inputs. A separate intent head proposes a goal/horizon at
the root. Terminal events, observed payoff, exhausted budget, incoming garbage
or an own-state mismatch end the proposal before another ranking. Duplicate
completion notifications do not consume extra turns. Ranking accepts the
complete actual feasible action inventory and returns proposal order only.
It does not authorize overriding the common competitive core, claim a
calibrated quality allowance, or change the installed trainer.

Fitting holds out whole Fightcade sessions, gives each training session equal
total weight and each construction equal weight within its session, and retains
the fixed final epoch. Action imitation and goal/horizon prediction have
separate diagnostic losses. Commentary timing remains an independent evidence
link: video reconstruction gaps and checkpoint guesses cannot certify a
historical construction. Useful replay-aligned preferences, calibrated quality
admission, persistent full-game noninferiority and live integration remain
required before expressive behavior can enter the product.

`human/spatial_proposer.py` implements the bounded second proposal experiment.
Its goal predicts a colored location in a horizontal, crossing, large-wave or
cascade clear, together with a duration. A frozen competitive own-bottle
encoder supplies mean/max features from the actual board and visible pills;
its historical same-color bond encoding is preserved, while the small spatial
branch receives complete bonds. This reuses the competitive encoder without
inventing an opponent or missing public history. Own-only replay cannot feed
a full-context core and that combination is rejected. The frozen public
parent's explicit `zero_v1_vs` contract still has 72 auxiliary columns; those
columns receive their defined constant zeros, not fabricated history.
Payoff cells are
supervised targets only: the action decoder receives predicted spatial and
duration distributions during both fitting and execution. It never receives
the future bottle or true remaining construction length. A selected persistent
plan keeps its root features and spatial target while reading current inputs,
and ends on the matching colored location/geometry, surprise, terminal or
placement budget. A distinct stateless network of the same size and initial
weights is trained on the same sessions, predicting afresh each placement.
The comparison reports early setups separately from payoff placements and
retains both observed-goal and predicted-goal action scores. These replay-prefix
scores still use actual human intermediate states; they are not autonomous
play, motor-feasibility evidence or competitive noninferiority. The existing
validation sessions are development data, so fresh replay confirmation and
quality-admitted persistent games remain necessary.

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

The first executable adapter keeps G5's candidate and global representations
frozen. A 158k-parameter policy/value residual receives those representations
and an eight-scalar public motor/gravity context. The actor residual is bounded
to four logits in either direction. Zero initialization preserves the parent;
an explicit gate preserves its outputs for Super Human and Frame Perfect even
after training. Sloth through Top Humans share one conditional adapter.

Training uses categorical exploration over the complete paced feasible set
and executes the selected script in the controller-frame VS runner. Frozen
features and the actual behavior likelihood are retained only for that update.
PPO uses full natural-terminal W/D/L returns and gamma one. Actor credit sums
decision score terms along each game, with one common collection scale.
Historical per-trajectory `1/T` actor weighting can reverse the objective's
gradient and remains an explicit comparison option. Actor, value, entropy,
parent-KL reductions and advantage normalization are separately named; the
last four retain historical settings in the isolated actor experiment.
Actual categorical KL is checked on all collected actions after optimization,
with parameter and optimizer rollback when the update budget is exceeded.
Changing objective on resume is rejected; use a weights-only initialization
in a new run. A small KL penalty anchors the parent. Unfinished games supply no
targets. Planner failures abort; only explicitly unreachable physics states
become uncontrolled falls. Evaluation uses deterministic argmax on disjoint
side-swapped seeds and reports every pace separately. Adapter checkpoints are
bound to their parent hash and context schema; they are experimental artifacts,
not an automatic replacement for the installed core.

Frozen milestones may enter a concurrent controller-frame tournament as soon
as their atomic checkpoint is available. This evaluation consumes its reserved
seed schedule independently of the continuing optimizer. Historical public
cores and earlier adapters remain in the connected field. Live ratings combine
compatible experiment phases within each level and pace, with complete
side-swapped seeds as the observation unit; they do not collapse the motor
settings into an uncalibrated single strength score.

Budget this experiment in both simulated console frames and actual learner
placement decisions, reported separately for every pace. Earlier PPO `steps`
also count elapsed frames, but equal frame or game counts do not provide equal
learning coverage: slow-pace games can end after very few controllable moves.
Use larger game batches for those paces and require a per-pace decision floor
as well as a global frame target. Reaction-locked falls and unfinished games
cannot satisfy that floor. Chunk collection without updating between chunks
so every batch retains one behavior policy.

For throughput, the event rollout parks each independent VS pair at its next
policy boundary, then batches ready public observations across pairs. The two
players in a pair always share a clock. The native loop executes every console
frame in the original player order, validates both controller microstates
before applying either input, and preserves garbage and reveal timing. This
changes host scheduling without changing simulated timing or adding fall warps.
The one-frame runner remains the independent evaluation/replay reference.
With asynchronous planning, ready decisions from other pairs continue while a
difficult root is pending. Each pending pair stays parked at its own boundary;
its two players and the public snapshot supplied to its planner cannot advance.
Completed requests form short inference batches while CPU workers plan ahead.

Worker threads own native planner buffers and share a bounded cache of complete
exact results, coalescing identical concurrent requests. Paced search uses the
exact cost-only v4 solver for its unrestricted feasibility upper bound; the
full BFS remains the independent oracle. Unrestricted witnesses never execute
as paced input. The frozen core scores learner and parent rows in one forward
pass, with the residual disabled for parent rows. Only learner rows sample
actions or retain PPO features and likelihoods. No feasible candidates are
dropped. Explicit strict FP32 disables TF32 convolution as well as matrix
multiplication: otherwise nearly tied scores can change with inference batch
shape even though tensors have float32 dtype. This is a compute setting, not a
different checkpoint or a decision tie-breaking rule.

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

The opt-in `mixed` opponent mode instead solves a simultaneous zero-sum
matrix. Its row and column strategies are chosen together; the opponent does
not get to observe our sampled action before choosing its own. Every legal
action on both sides and every supported correlated reveal remains in the
matrix/tree, regardless of beam settings. The default solver uses
[HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html)
through the optional SciPy search dependency: maximizing the row player's
guaranteed value yields its strategy and the inequality duals give the column
strategy. It has explicit iteration/time limits and independently checks the
unregularized best-response gap of its returned distributions. A numerical gap is
not uncertainty in the learned values. A failed convergence check or exhausted
tree budget withholds teacher targets. At a simultaneous root, `policy_target`
must be sampled; `best_action` is only its most probable display representative.
Recursive and cooperative inference drivers implement the same backups, and
report solver wall time separately from native nodes and neural work. This
does not certify the current critic or authorize live search.
The solver subtracts the matrix's common payoff offset before optimization,
which preserves strategies. The earlier entropy mirror-prox variant remains
available as an explicit diagnostic; centering prevents a large common value
from reducing its step size, but did not cure its observed convergence failure.
Exported utilities and convergence gaps remain in the original units.
For numerical comparison, equilibrium vectors need not be unique or equally
well-conditioned as values. The optional `mixed-game-certificate-v1` probe
compares every joint payoff at the existing 1e-5 tolerance, checks both
strategies' unregularized gaps, and evaluates each strategy pair on the other
matrix. If the largest payoff difference is epsilon, a fixed strategy pair's
gap can increase by at most twice epsilon. This is a numerical bound, not a
confidence interval for the critic. Strict vector agreement is still recorded;
earlier failed probes are retained and are not retroactively relabeled.

`AdaptiveJointEventSearch` allocates work at a simultaneous root while keeping
both complete action inventories. An unexamined joint action retains utility
interval [-1, 1] and null W/D/L; policy priors only break allocation ties. The
lower matrix supplies our security strategy p, and the upper matrix supplies
the opponent's security strategy q. For any matrix M within those intervals,
its response gap is bounded by `max(U @ q) - min(p @ L)`. Allocation evaluates
unknown entries along the currently dangerous row and column until that bound
meets the declared tolerance or a work budget expires. A failed inner solve or
expired traversal discards the entire unfinished allocation batch. Known cells
retain a separately declared numerical evaluation tolerance; this is not a
learned uncertainty estimate. Interior search, forced advancement and complete
correlated reveals use the existing queued traversal. This root-only mechanism
does not yet implement nested allocation or tactical depth extensions.
Its certificate applies only to the configured finite-depth critic game.
Partial matrices never supply quality-training labels, and uncertified results
cannot be sampled by the action decoder. A registered native audit independently
checks the intervals and response bound against a complete matrix and retains
actual native nodes, neural calls and elapsed time.

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
