# Operations

## Authority and supported entrypoints

Run this before any work:

```bash
uv run python -m tools.program status
uv run python -m tools.program validate --check-paths
```

`drmc_rl/program/program.yaml` is the launch authority. A recipe has one status:

- `active`: supported now;
- `staged`: code/config exists but requires explicit review and open gates;
- `blocked`: a prerequisite contract is missing;
- `complete`: retained evidence, not a launch target;
- `retired`: must not be launched.

Launch through the registry:

```bash
uv run python -m tools.program launch g4-strong-league --dry-run
uv run python -m tools.program launch g4-strong-league
```

Runtime placeholders are explicit:

```bash
uv run python -m tools.program launch timing-action-gate \
  --set timing_probes=/data/drmc/timing/probes-v1.jsonl \
  --set timing_report=/data/drmc/timing/report-v1.json
```

A staged recipe additionally requires `--allow-staged`; closed gates still
block it unless `--ignore-gates` is deliberately used for local debugging. Do
not use `--ignore-gates` for a scientific run.

## Environment standup

Use the pinned Python 3.14.7 runtime and uv >=0.12.10. Training selects `rl`;
portable packaging selects `inference` instead. Do not use `--all-extras`, which
would request incompatible CPU-packaging and CUDA-training variants. See README
for the isolated packaging command and the PyTorch 2.14 continuation caveat.

```bash
git submodule update --init --recursive
uv sync --locked --extra rl --extra viz --extra eval --extra corpus
uv run python -m tools.build_drmario_pool
uv run python -m drmc_rl.training.run --dry_run true
uv run pytest -q
```

The native pool is the training backend. A legal ROM and libretro core are only
needed for independent verification:

```bash
uv run python -m drmc_rl.training.run \
  --cfg drmc_rl/training/configs/smdp_ppo.yaml \
  --backend libretro --core quicknes \
  --rom-path legal_ROMs/DrMario.nes --ui debug --num_envs 1
```

Do not place ROMs, private corpora, checkpoints, or run output in the repository.

## Active compute roles

### Training accelerator

The active G4 and staged G5 recipes are sized for the CUDA training host. Keep:

- checkpoint and run directories on local storage;
- the human corpus mounted read-only;
- native build and source revisions pinned;
- one run ID per process;
- telemetry visible to the arena dashboard.

Never edit an in-progress config in place. Stop, write a new config/recipe or
recorded override, and resume with a new run identity.

### Local trainer diagnostics

`trainer-repertoire-audit` collects one public decision per fresh incumbent
self-play game across levels 10/14/20, MED/HI, and early/later decisions. Supply
`competitive_checkpoint` and `trainer_repertoire_report`; append
`-- --games 48 --device cuda`. `--evaluation-checkpoint PATH` scores another
model on the same collected roots. Compare the saved root hashes before
pairing reports. `--source-bank PATH` instead samples one synthetic root per
source replay from a held-out curriculum bank. Native first-wave checks,
complete feasible candidate details, reaction-window locks, and the fixed-mask
cost ablation are diagnostics, not paced full-game win rates. Use v2 reports;
the initial v1 afterstate event counts used incorrect raw pill colors and are
superseded. Its geometric and mechanical counts were unaffected.

`trainer-repertoire-source` runs the lightweight extractor on mombox with
`repertoire_replay_db`, `repertoire_fcr_root`, and
`trainer_repertoire_source`. The populated archive is under
`/home/ethan/fightcadeRatings/data/`; development DB copies may lack blobs.
The default bounds extraction to 2,048 replay sessions and does not modify
the source. Follow it on tf3090 with `trainer-repertoire-curriculum`, supplying
that NPZ as `trainer_repertoire_source` and a fresh `trainer_repertoire_bank`
directory. The default examines at most 4,096 source rows, keeps at most two
per replay, excludes unsettled roots, and uses native resolution to identify
reachable horizontal, crossing, large first-wave, and cascade opportunities.
`train.npz` and `heldout.npz` have disjoint replay sessions; the manifest records
source hashes, exclusions, opportunity counts, and inherited approximations.

Repertoire training uses `trainer-public-outcome` with frozen recorded config
overrides, a common parent, and a matched control. Change reset sampling only;
keep terminal W/D/L as the reward. Predeclare frame budgets, evaluation seeds,
checkpoint selection, and adoption criteria before starting. The September 7
pilot uses 25% curriculum resets versus none, two 5M-frame arms, and optimizer
resets in both. The untouched tester baseline remains
`trainer-internal-v20-20260907` in both repositories. Results belong under
`runs/trainer-repertoire-v1/` and do not change any product gate.

`trainer-terminal-quality-pilot` forces every legal root move and measures
natural results with both sides subsequently controlled by the installed public
core. Supply `trainer_control_states`, `competitive_checkpoint`,
`trainer_control_checkpoint` (the V3 reference), and
`trainer_terminal_quality_report`. Append `-- --states 24 --batch-size 32
--device cuda`. It selects independent games across level/speed cells and
enumerates complete reserve hypotheses from the public posterior. Missing
terminal results are reported as incomplete, never as draws or critic values.
The report includes candidate coverage and paired public/V3 choice scores.
This pilot is a new teacher experiment, not a promotion-quality label release.

`trainer-rollout-consistency-audit` measures whether an older source bank still
describes the corrected continuation adapter and native engine. Supply
`trainer_control_states`, `counterfactual_mixture_manifest`,
`counterfactual_wdl_calibration`, and `trainer_rollout_consistency_report`.
Append `-- --states 96 --batch-size 16 --device cuda`. The audit selects one
position per game across tactical cells, forces the recorded root move once,
then continues both sides with the frozen ensemble argmax. It reports changed
initial choices, natural-outcome agreement, and incomplete continuations
separately. Exact hidden reserves are replay capabilities here; these results
must never be promoted as public counterfactual probabilities or training labels.

`trainer-search-input-audit` counts supported critic boundaries under frozen
search priors. Supply `trainer_control_states`,
`counterfactual_mixture_manifest`, `counterfactual_wdl_calibration`, and
`trainer_search_input_report`. Counts are unweighted search visits; an inactive
root is valid only when the evaluator uses the acting opponent's perspective.

`trainer-search-prediction-audit` scores exact held-out observed actions before
spending compute on every alternative. In addition to those source/mixture
paths, supply `trainer_search_bootstrap`, `trainer_search_bootstrap_manifest`,
`trainer_search_reference_release`, and `trainer_search_prediction_report`.
Its single-action search path has the same continuation values as full root
enumeration. It is diagnostic and cannot substitute for candidate coverage.
Append `-- --public-checkpoint PATH --calibration-bank PATH` to screen the
public core's pre-action value using an independent calibration bank. Append
`-- --public-checkpoint PATH --public-search-calibration REPORT` to evaluate
candidate continuations with that frozen public value link. Both modes preserve
the whole-game split and keep the failed historical teacher as a reference.

`trainer-control-audit` checks the current regret decoder on a frozen pair-state
bank. Supply `trainer_control_checkpoint`, `trainer_control_states`, and
`trainer_control_report`; the default samples two positions in each bank cell
and evaluates paired quantiles across 800/1200/1600/2000/2400. It saves model
scores alongside the report. Population style stays fixed while strength
changes. Zero fixed-state inversions is a decoder check, not evidence that
neighboring presets are separated in games or match absolute human ratings.

`trainer-baseline` compares frozen checkpoints before more training. Supply
`trainer_roster`, `trainer_benchmark_db`, `trainer_benchmark_name`,
`trainer_benchmark_games`, `trainer_benchmark_pairs`, `trainer_benchmark_level`,
`trainer_benchmark_speed`, `trainer_benchmark_seed`, and
`trainer_benchmark_device` through `tools.program launch --set`. Run from an
activated environment so the recipe's `python` resolves correctly. Use a new
name and database when checkpoint hashes or entrant parameters change.
On CUDA hosts, append `-- --gpu-planner` after the recipe name to use the
parity-tested batched reachability solver. CPU reachability remains the default
for independent comparisons. Report uncertainty over side-swapped seed pairs;
the two games sharing a seed are not independent samples.

The native tournament uses fastest reachable placements. It measures strategic
strength, not live human cadence or absolute WHR-C. The public V5 entrant must
set `params.public_only=true`, or declare `aux_spec=zero_v1_vs` in its checkpoint;
its auxiliary vector remains zero, matching the corpus distillation contract.
G4 references with pending-attack inputs remain
privileged teachers.

V3 entrants receive complete semantic bonds from native bottle bytes. Earlier
arena runs that reconstructed afterstates from the legacy actor observations
used different physics on same-color turns and cannot certify live trainer
calibration. Keep those results as historical diagnostics and use a fresh
database after the correction. The Strong League search adapter also needs
the opponent-side bond mask and complete candidate packing; older teacher
releases must be regenerated and recalibrated before promotion.

`trainer-execution-sample` prepares input traces from an immutable corpus
release. Supply `human_corpus_root`, `human_corpus_release`, and
`execution_scripts`. Sample schema v2 uses the audited FBNeo boundary:
`held_before_spawn` initializes controller history, `raw[:-1]` supplies movement
inputs, and recorded frame parity is inverted. Every accepted row must match
its recorded lock pose and frame. It retains the initial bottle/microstate,
rating, and whole-game/player holdout identity. Both producer coverage and
prior-held contracts are mandatory; the old unverified-coverage override has
been removed. Earlier byte/length checks could accept zero-filled missing input.
Sampling does not certify an operation profile or open its gate. Profile
metrics must honor the initial held buttons. Keep traces and model artifacts
in ignored run directories.

`trainer-execution-replay-audit` checks recorded lock poses and times with the
independent frame stepper. Supply `human_corpus_root`, `human_corpus_release`,
and `execution_replay_report`. `--sample-modulus` spreads a bounded probe over
stable decision-id hashes; `--month` and `--max-rows` bound the scan. The explicit
`--parity-xor 1 --input-delay-frames 1` combination probes the FBNeo recording
boundary. It does not alter the live host's parity/input contract. New exports
include `held_before_spawn` to distinguish pending fresh presses from carried
holds. Missing prior state and ROM verification remain visible limitations;
an exact stepper match alone is not a ROM certificate.

The private `trainer-input-2026-08-v1` release contains 1,932,789 decisions and
declares both coverage and prior-held contracts. Its manifest SHA-256 is
`6f653c040c8fc4c17008fd58c780c303cdf241f1c3a49227e48c0e3b9e94e9b1`.
The bounded 4,096-row probe matches 4,029 moves exactly, excludes 65 unsupported
windows, and retains two unexplained long-window mismatches. Do not fit those
two rows or promote the old zero-filled sample.

`trainer-execution-control-audit` compares the sampled human targets with exact
native witnesses after 0/4/8 neutral frames, current cadence shaping, and the
held-input movement shaper. Supply `execution_scripts` and
`execution_control_report`; append `-- --heldout-only --seed 20260908` for
evaluation outside the fitted training partition. Input hash and v2 alignment
are mandatory. Statistics weight players equally, games equally within player,
and sampled pills equally within game. The bounded reachability sample balances
rating/speed/high-board cells and is not a population-prevalence estimate.

The historical v10 1,024-state held-out motor comparison spans 53 players and 602 games.
Weighted button changes average 5.78 after shaping versus 19.73 under the
previous matched-duration algorithm and 5.74 in humans. All lock replays and
2,048 paired pace-order checks pass. The held-input route is selected in about
91% of weighted states; all remaining planner routes are retimed at their final
column. Added work averages 2.54 ms on tf3090. These are path diagnostics using
recorded human total durations,
not end-to-end timing-model accuracy. Single-pill windows cannot certify
sustained ten-second operation limits; difficult routes still need constrained
planning. Pace selects from fixed valid wait/drop families while motor
parameters remain constant, preventing route changes from reversing pace.
The first 512-state latency probe preserved all recorded targets at
eight frames, while reducing some alternative candidate sets.

The current local trainer supersedes that post-selection shaper with named
constraints inside reachability. Run `trainer-paced-execution-audit` through
the program launcher with `execution_scripts` pointing to the covered v2
sample and `execution_control_report` to a new report. It checks every returned
legal placement witness, nested feasible sets across all seven presets, and
local CPU latency on held-out boards. This engineering diagnostic does not
promote the corpus-calibrated human execution or product gates. Pair it with
the host's ROM-backed `trainer_smoke` using named pace arguments and `--sparring`.
The v11 engineering audit replays 204,086 witnesses from 1,024 held-out boards
at the live 2,048-frame horizon: zero motor violations and zero feasible-set
inversions. Mac median planner times are 2.5–4.6 ms and 95th percentiles
9.8–15.2 ms. Rare tails remain (up to 275 ms for Sloth, 192 ms for Fast);
the host uses each pace's fixed reaction window for computation and safely
replans after missed deadlines. Seven packaged level-14 MED sparring checks
completed 269 placements with zero missed decisions or placement mismatches;
two observed-state corrections replanned safely. A separate rating-1200
Relaxed check completed 43 placements, with one missed decision and three
corrections, and no placement mismatch. These bounded checks do not certify
long-session latency, human ratings, or sustained human burst distributions.
Additional packaged HI checks completed 40 placements at level 14 Relaxed and
18 at level 20 Fast, with no missed decisions, corrections, or placement
mismatches. The level-20 Fast game topped out after about 4.6 seconds without
clearing a virus: execution acceptance is not a pressure-strength claim.

Planner parity includes the retail 81-entry gravity table and bottle-boundary
DAS behavior. Repeating into a wall preserves the repeat phase; a blocked cell
charges DAS. Python, the standalone C helper, the pinned native pool copy, and
CUDA must all preserve this distinction. Rebuild both native libraries after
changes and run `tests/test_reach_v4_parity.py`; on tf3090 additionally run
`python -m tools.test_reach_cuda_parity --quarks 0 --fuzz 500` to check costs and scripts.
The geometric graph needs room for up to 30 successors and more than 20
incoming variants. The native `make test-reach-bounds` sanitizer check covers
512 roots and the slide/rotation regression. GPU reconstruction accepts only
parent chains reaching the original state at the independently computed cost;
incomplete chains retain an explicit CPU-fallback status.

For local trainer packages, `--device auto` selects CUDA, Metal, then CPU.
Metal uses a small set of padded candidate shapes warmed before the backend
reports readiness. Verify packaged Maximum and regret modes against the ROM;
a warmed microbenchmark alone does not expose first-use shape compilation.

### Trainer planning tournaments

The pace-conditioned strategy experiment uses
`tools.program launch trainer-pace-strategy --set trainer_pace_config=PATH`.
Its JSON names the frozen `checkpoint`, `output`, local `working_db`, native
library, device, seed, `holdout_seeds`, `paces`, `updates`, and even
`games_per_update`. The initial five paces are Sloth through Top Humans.
For frame-budgeted runs, set `target_frames` and
`minimum_decisions_per_pace`; both must be satisfied. `updates` then acts only
as a safety ceiling, and reaching it too early fails the study. Set even
`games_per_pace` counts to balance actual learning coverage, and even
`rollout_games` to limit simultaneous game collection without changing weights
inside an update. `milestone_frames` saves evaluation snapshots as
`adapter-fNNNNNNNNN.pt`; successful completion also writes `adapter-final.pt`.
Sloth training stays at 14 HI; the other paces may use `level20_fraction` for
pressure exposure. Keep complete natural games and exclude time-capped games
from the update. Training win rates include exploration and are not ratings.
Checkpoints contain only the adapter, optimizer, sampling RNG and resume
metadata; the parent remains a separate frozen artifact.
Set `rollout_backend: "events"`, `async_planning: true`, `strict_fp32: true`,
and `planner_workers: 4` for tf3090 training.
This uses exact native controller execution, concurrent planning, and a shared
frozen-core inference batch. `rollout_games: 128` bounds collection; 64 or more
games amortize inference much better than small batches. `frames` retains the
one-frame runner. Disable `async_planning` to measure a barrier after each
batch of planner requests. Keep `strict_fp32` in held-out arena configs too;
it disables TF32 convolution, whose rounding can flip close decisions when
batch sizes differ. Check `training.json`'s `throughput` for actual learner
decisions/s, rollout frames/s, and frames/s including the optimizer and journal,
with phase timings. Checkpoint writes are included in cumulative wall time.

Pace objectives are now explicit config fields: `actor_reduction`,
`value_reduction`, `entropy_reduction`, `parent_kl_reduction` and
`advantage_normalization`. New runs default to `decision_mean` actor credit
(one common batch scale), with the other reductions `episode_mean` and
`episode_center_scale` advantage normalization. The historical actor uses
`episode_mean`. Old checkpoints resume their historical objective; changing it
requires `init_adapter` in a fresh output directory, without `resume`.
`max_update_kl` checks the full categorical behavior distribution after each
epoch, including unchosen actions. Backtracking restores parameters and Adam
state; `max_kl_backtracks` bounds retries. Inspect realized `update_kl`,
`first_step_kl`, `completed_learning_games` and trajectory lengths, not just the
cap. `tools.report_pace_objective_study --config PATH --output PATH` combines
each arm's training directory and stdout log. It separates repeated seed
experiences from distinct seeds, reports realized KL and critic error, and
keeps level/pace outcomes separate. Training curves have no confidence
intervals: the policy changes during collection. The old `independent_games`
log field counted completed learning trajectories, not independent seed pairs.

Launch a two-arm controlled study with `trainer-objective-study` and
`trainer_objective_study=PATH`. Its JSON names two training configs, evaluation
configs, and an output directory. It validates common initialization/exposure
and permits only actor reduction and artifact paths to differ. Training arms
run sequentially; frozen evaluation shards start after both complete.
The September review study is under `trainer-output/review-20260909` on the
mombox overflow mount, with an isolated tf3090 source and 50M additional frames
per arm. Both completed (50,369,235 and 50,201,899 frames), followed by all 16,384
reserved side-swapped games with no censoring. The assessment is in
`runs/review-20260909/objective-arena-assessment.json` and the roadmap.
Independent training-seed finalists still need fresh confirmation games.
The dashboard at `http://127.0.0.1:8098/` retains both arms and their evaluation.

New public native PPO runs set `env.public_observations: true`. This selects
`step_strict` and maintains separate causal opponent snapshots through partial
resets and the direct policy batch path. The historical scheduler is available
for reproduction, but public zero-aux/context training rejects it. Start a new
experiment when changing this observation and scheduling contract; do not
resume an old run as if its data distribution were unchanged.
The loader rejects optimizer/step resumption across this contract change;
weights-only initialization uses the checkpoint's effective EMA actor when
available.
The strict collector counts completed placement transitions separately from
pair-event steps, retains terminal reward through waits, and bootstraps at each
learner's own successor decision. Real native tests exercise a PPO update in
both dictionary and direct-array observation modes, with self-play and a frozen
opponent. Gamma-one outcome training
is supported; censored games, unaligned distillation targets and incomplete
live public-context inputs fail explicitly. Profile its host-side transition
storage before a large CUDA run; the existing pace frame/event trainer keeps
its already measured collection path.

`trainer-controller-core --set controller_core_config=PATH` uses that exact
frame/event path with `training_model: public_core`. It migrates the frozen public
G5 to the live context schema and updates the full network from natural terminal
returns using the corrected episodic actor reduction. Its fixed post-migration
initial policy is the explicit KL reference. `public_replay: true` saves compact,
pickle-free public input shards with the complete frontier and separately named
observed-continuation outcome labels; seeds are split metadata, never model
inputs. These support broader teacher data and policy anchors, not invented
all-action quality labels. `target_frames` and `target_decisions` must both be
met, together with `minimum_decisions_per_pace`. All seven authored paces are
supported. `core-initial.pt`, milestone weights, and `core-final.pt` load in the
ordinary source trainer/arena; `checkpoint_keep_last` bounds resumable update
checkpoints without deleting milestones. Keep outputs on mombox and SQLite on
local storage. New core training does not certify candidate WDL heads, a human
execution profile, device latency, or model promotion.

Full-core PPO preserves every candidate's actual collection log probability.
Parameter/buffer version checks reject network writes during collection and
mixed-update samples before optimization. The independent input/distribution
audit bounds FP32 total variation at `1e-4` and absolute log-probability error
at `1e-3`; a batched outlier is recomputed at the single-row shape under the
same bounds. A failing recheck stops before optimization. Both original
discrepancies and recheck counts are recorded. These numerical bounds limit
probability mass error to 0.01% and likelihood-ratio error to about 0.1%; PPO
still uses the original recorded behavior logs, never the recomputed values.
The 17,980-row Top Humans collection audit on tf3090 found a batched maximum of
`1.3623e-5` (99th percentile `1.1819e-6`), versus at most `5.7154e-6` when its
worst 32 rows were independently evaluated. Inference and gradient-enabled
paths agreed within that bound. A fresh 16,435-decision collection reached
`1.4038e-5` even at single-row shape, showing why `1e-5` is not a reliable FP32
identity test. These audits motivate the separate structural identity check
and numerical bound. They are not evidence of learning or permission to
substitute a newly computed behavior policy.

The active full-core run is `review-20260909/controller-core-live-v4` under
the tf3090 trainer-output mount. Its launch configuration is under the sibling
`configs/` directory, currently `controller-core-live-v4-after-motor-fit.json`; each
checkpoint also carries that configuration. Native code is `19f292c`.
It continues the valid updates in `controller-core-live-v2`;
failed collection attempts remain intact. The budget is 10M learner decisions
and 1B console frames, with at least 500k learner decisions at each pace.
Console frames and learner decisions are distinct counters. Local tf3090 logs
are at `/home/ethan/.cache/drmc-rl/logs/controller-core-live-v4.log` so a storage
reconnection cannot invalidate the open log handle. Progress and checkpoints
remain on mombox. `runs/trainer-pace-v1/sync.json` mirrors only named milestones
and progress, keeping public replay shards on overflow storage.

Training publishes actual in-flight collection frames/decision requests and
optimizer/audit steps every five seconds, plus phase changes. These callbacks
run after real work on the training thread; a timer alone cannot refresh a
stalled worker's timestamp. Budget counters still include completed updates
only. The dashboard shows current activity separately and retains its overdue
and explicit-failure alerts when worker activity or the remote feed stops.

New `drmc-public-controller-replay-v2` shards also retain the exact observed own
controller microstate, BCD pill counter, gravity setting and charged delay as
archival geometry. The actor tensor contract is unchanged. V1 shards cannot
recover this information and are explicitly skipped by
`trainer-motor-opportunity-bank --set motor_opportunity_config=PATH`.
That CPU recipe consumes only replay updates whose checkpoint is complete. Its
JSON names `replay_directory`, `output`, `native_library`, reserved arena
`holdout_seeds`, `seed`, and optional `max_roots`, `per_game`, `per_update` and
`watch`. It samples opening, intermediate and late decisions from whole games,
retains every feasible root move and assigns all occurrences of a reset seed
to one fitting split. `roots.jsonl` and compressed root arrays retain both
spawn-parity branches, exact afterstates/clear effects, next-pill full action
costs, and reachable/clear-enabling cell maps. Costs include the next reaction
and compute delay; they describe valid witnesses, not guaranteed shortest paths
or the frame when the clear animation begins. Incoming garbage is explicitly
excluded. Terminal success is kept separate from losing future access. These
are deterministic auxiliary labels, not all-action match outcomes. Fitting,
independent held-out prediction checks, and controller strength evaluation are
still required before a model uses them in the trainer.

`trainer-motor-auxiliary-fit --set motor_auxiliary_config=PATH` consumes a
completed opportunity `bank`, a public-core `checkpoint`, an independent
`anchor_replay_directory`, and a fresh `output`, plus `seed`, `device`, `epochs`,
`batch_size` and `lr`. It adds versioned effect/access/cost prediction heads and
updates the shared representation. Exact own controller geometry conditions
these auxiliary predictions; it is not silently added to the existing actor
tensor contract. Ordinary policy inference skips the heads entirely. This
phase never turns clear opportunities into match rewards or candidate WDL.
The existing exact effect tokens are targets, with unobserved attack and
uncertainty fields masked out. Future targets keep parity conditions separate
and exclude absorbing root terminals from access losses. Roots and candidates
are normalized within whole-game weights.

Policy preservation includes at least 256 independent unannotated public
anchor games by default; all validation reset seeds and reserved arena seeds
are excluded. Every epoch must satisfy `max_policy_kl` on both annotated
training positions and the broad anchor set. A failing update restores the
model and optimizer before retrying at lower learning rate. Validation errors
and policy drift are measured after accepted epochs and never choose retries.
`fit.json` records the unchanged initial predictor and every accepted epoch;
`progress.json` separates processed examples from examples in accepted epochs.
`core-latest.pt` is the last accepted epoch; `core-final.pt` appears only when
the full fit completes. Both are diagnostic and load through the ordinary
source actor. Fresh prediction confirmation and large controller tournaments
are still necessary; a small fit test is implementation evidence only.

The first `motor-auxiliary-fit-v1` completed all 20 epochs from source
`motor-auxiliary-95d4d08-source` in 414 seconds, with 33,580 accepted root
presentations and no policy-cap rollback. Main training paused after update 56
(66,307,994 frames; 649,637 learner decisions) and resumed its own unchanged
checkpoint and optimizer from source `controller-core-4717a03-source`. The
bounded handoff record is `motor-auxiliary-slot-v1/progress.json`; its preserved
`main-resume.pt` is distinct from the fitted branch. The fitted branch's
`core-final.pt` is the immutable evaluation candidate. Neither branch is promoted.

`trainer-motor-confirmation --set motor_confirmation_config=PATH` checks a
completed auxiliary fit on fresh controller play. Its JSON names
`fit_directory`, the unchanged parent `checkpoint`, the original fitting
`bank`, source `training_config`, reserved `seeds`, explicit level/pace
`conditions`, and an `output`. Every source seed must be excluded from both
outcome training and the fitting bank's policy anchors. The frozen stochastic
parent plays its deterministic policy on swapped sides; full natural games
provide earlier and later public positions, never optimizer updates. Persisted
source replay and exact conditional labels make interrupted conditions resumable.

The audit reconstructs the original auxiliary initialization and first checks
it against the saved fitting baseline. It compares the fitted model with that
initial predictor and a pace/cell prevalence predictor fitted only on training
labels. Report each level/pace separately, with sides and multiple positions
aggregated before a whole-reset-seed bootstrap. This prevents abundant empty
cells or repeated deterministic parity branches from masquerading as evidence
of useful movement prediction. These are prediction diagnostics; full controller
tournaments and actual deployment latency still determine adoption.

Mombox now runs `motor-confirmation-v1` from immutable source
`motor-confirmation-83591c3-source`, using the corresponding JSON in `configs/`.
It reserves the last 64 seeds of the outcome-training exclusion bank for 1,152
natural source games, with up to 1,728 exact labeled roots across seven 14-HI
paces and separate Normal/Top Humans 20-HI conditions. Its `progress.json` and
final `assessment.json` are mirrored to the dashboard. A separate
`motor-auxiliary-arena-v1` uses that same source and compares the immutable
fitted core with the 25M parent: 1,024 games per 14-HI pace and 512 per 20-HI
condition, 8,192 total. Its complete journals, move traces and closed viewer
database feed the common tournament. Both jobs use registered recipes and
have their own `supervisor.pid` and `supervisor.log`; inspect these before any
recovery. The older mombox and Mac evaluators continue their existing schedules.

The Mac's `controller-core-eval-mac-0.json` and `controller-core-eval-mac-1.json`
under `runs/review-20260909/` launch through `trainer-planning-arena` from the
isolated `controller-arena-0c76c0e-source` snapshot. Their 32,768 scheduled games
compare the initial migration with the frozen parent and the 25M-frame core
with initial, parent and corrected E1 adapter. Each 14-HI edge gets 1,024 games
at each of seven paces; separate 20-HI Normal/Top Humans edges get 512 games.
All seeds come from the training exclusion bank. Workers wait for the mirrored
milestone, preserve move traces and use the exact batched event runner.
The four-frame compute charge is an evaluation assumption to be verified on
deployment hardware; this tournament does not certify latency by itself.

An optional `opponent_pool` lists frozen `id`, `weight`, `checkpoint` and
optional `adapter_checkpoint`. A member is sampled per collection update and
recorded in every training journal entry. Frozen member checkpoint/adapter
hashes are saved and checked on resume. The default single parent preserves
the existing sampling RNG sequence. `public_league.empirical_mixture` builds
an entropy-regularized mixture from complete measured paired edges at one
level/speed/pace, rejecting missing, duplicated/conflicting or censored evidence.
Its solver is `entropy-mirror-prox-v1`; old cumulative-softmax temperature was
not a persistent entropy regularizer. A training mixture is not a strength
certificate. Keep population changes outside objective-only comparisons.

For the review's offline diagnostics, launch through these recipes:

- `public-predecessor-bank`: set `competitive_checkpoint`, `bank_device`,
  `bank_states` and `public_predecessor_bank`. It collects clean 14-HI games,
  retaining full causal public views, native restore state, complete reserve
  history and positions 4/8/16 own placements before natural losses.
- `paired-terminal-quality`: set `paired_terminal_config`. The JSON names
  `state_bank`, frozen `members`, weighted `continuations` (actor/opponent),
  `reference`, `states`, `seed`, `device`, `output`, `batch_size`, `max_events`
  and optional level/speed filters and `native_workers` (default 1). Neural
  inference remains ordered and batched; additional workers step independent
  native handles. Progress refreshes after each loop once five seconds have
  elapsed, even when no trajectory has finished. Every root candidate uses the same exact
  reserve/policy panel. Inspect `progress.json`, `targets.jsonl` and
  `rollouts.jsonl`. A capped branch is unknown, not a draw.
- `paired-quality-fit --allow-staged`: set `paired_quality_fit_config` with
  `state_bank`, `targets`, parent `checkpoint`, `mode` (baseline/critic/context/
  combined), `phase` (auxiliary/policy_improvement), `seed`, `device`, `epochs`,
  `batch_size`, `lr`, `max_policy_kl` and fresh `output`. Source-game validation
  is mandatory. A fitted checkpoint may initialize the next phase with the
  same mode/schema; learned heads and effective EMA weights are preserved.
  Architecture changes require a new migration from the frozen core. The
  output is a diagnostic checkpoint, never automatically installed or declared
  calibrated.
- `search-frontier-benchmark`: set `search_frontier_config` with `state_bank`,
  `checkpoint`, `states`, `device`, `output`, depth/beams and batching limit.
  This uses an explicitly uncalibrated leaf link to compare exact searches,
  not to generate quality labels. Both utilities and actual neural call counts
  are retained. Queued search remains opt-in with `--frontier-batch-size` in
  `joint_search_teacher`; no budget-exhausted result supplies training targets.

The corrected bank is `review-20260909/causal-public-bank/states.jsonl.gz`.
The earlier `public-bank` collection was stopped after its observation audit
exposed future warped locks; preserve its audit-status file and do not use it
as public evidence. Historical full-pair banks remain useful privileged
teacher artifacts, but cannot reconstruct missing causal public history.
The expanded review study is `review-20260909/quality64-study` on the mombox
overflow mount, executed on tf3090 from `d5c7c54`. Its `study.json` sequences
registered collection, paired-panel and four fitting recipes; `pipeline.json`
and `stage-0.log` through `stage-5.log` record progress. The local sync mirrors
these small files into `runs/trainer-pace-v1/incoming-quality64`, without copying
the model checkpoints. The plan uses 64 source games, a common 25% holdout and
up to 100 auxiliary epochs per mode, with a 0.02 policy-KL limit. All outputs
are prospective diagnostics and leave promotion gates unchanged.

Whole-game noninferiority uses `tools.confirm_policy_noninferiority --plan
PLAN --games GAMES --output OUTPUT`. The plan declares `baseline`, `candidates`,
`opponents`, `conditions` (level/speed/pace), `confirmation_seeds`, `score_margin`,
`alpha`, and `seed_design`. Each JSONL game names agent/opponent, level, speed,
pace, seed, side and natural `score` (0/0.5/1 or null). All four games per
reference/candidate seed comparison must be present. The comparison family
adjusts confidence; missing/censored games prevent certification. This is a
fixed, predeclared confirmation analysis, not an anytime-valid live ranking.
The independent-seed bound follows [Maurer–Pontil, Theorem 4](https://arxiv.org/pdf/0907.3740);
uniform sampling without replacement uses the conservative range bound in
[Bardenet–Maillard, Proposition 1.2](https://arxiv.org/html/1309.4029v2).
Do not select styles/checkpoints on this set and then call it confirmation.

Use `trainer-pace-throughput --set trainer_throughput_config=PATH` through
`tools.program launch` to compare `reference` and `events` modes on a schedule.
The `async` mode measures planning without the batch barrier. It records
actual frames, decisions, every chosen placement and controller
sequence, and fails on a trajectory/outcome difference. Preserve an old
planner binary and set `reference_planner_library` to compare algorithm changes
too. `planner_roots` captures up to 4,096 real roots; `planner_corpus` plus
`planner_libraries` compares all costs, offsets, lengths, and script bytes
against the first library. Retain benchmark outputs on overflow storage.

On tf3090, a matched strict-FP32 64-game Normal/14-HI benchmark measured:

| Runner | Console frames/s | Relative throughput |
| --- | ---: | ---: |
| Original frame runner and planner | 2,934 | 1.0x |
| Event batches and cost-only feasibility | 21,716 | 7.4x |
| Asynchronous planning and inference batches | 29,856 | 10.2x |

All three produced identical 570,334-frame trajectories, 9,716 decisions and
247,949 validated controller inputs. The planner-only comparison preserved
every output across 4,096 roots and improved from 399 to 898 roots/s. Forty-eight
additional full games across all five trained paces and Normal/20-HI matched
the old runner and planner under the same inference precision. These are
rollout measurements, not strength gains. Five real PPO validation updates
retained another 3,608,269 frames and 32,008 learner decisions with finite
losses; the continuation resumes that optimizer and journal at update 184.
Manual CUDA graphs were slower at the larger batch size; compiled FP32 changed
some decisions and brought no useful warm speedup. Neither prototype is enabled.

Set `resume` to a completed adapter checkpoint to restore optimizer and
sampling state. Resume discards game-journal rows beyond that checkpoint,
including an interrupted final write, while rejecting corrupt completed rows.
Per-pace totals are rebuilt from that journal and checked against the restored
decision total. When resuming into a new output directory, supply
`resume_journal` alongside the checkpoint.

`trainer-pace-study` takes `trainer_pace_study`, a JSON containing `output`,
`training_config`, `evaluation_configs`, and `evaluation_workers`. It launches
the bounded training recipe, then the fixed held-out arena schedules. Failure
status is published alongside training metrics; success requires
every evaluation worker to finish. It never promotes a checkpoint. The arena
accepts variant `adapter_checkpoint` paths on the common frozen parent and
explicit per-comparison `seeds` for reserved evaluation banks. Mixed-policy
evaluation currently requires reaction-covered computation; speculative
preparation remains the separately validated single-policy path.

When using network overflow storage, set the study's `log_directory` to a
local directory on the training host, and redirect the supervisor's own
stdout/stderr there as well. SSHFS reconnects can invalidate long-lived open
log handles while newly opened checkpoint and telemetry files still work.
Logs append across restarts; `pipeline.json` records their directory. Copy
closed logs to overflow storage after the run. Failure telemetry retains the
traceback even if stderr is unavailable. The dashboard prioritizes reported
training/study failures over the plan and warns if running training has no
update for the greater of three minutes or three previous rollout durations.
Completed training is not marked stale while evaluation continues.

For evaluation during training, `trainer-planning-arena` accepts `watch: true`.
For reactive policies, `rollout_backend: events`, `replay_games: 0` and
`planner_workers` select the same causal batched controller runner used by
training. Mixed context-aware and frozen actors have matching complete move
traces, outcomes and shared execution counters against the frame reference.
Per-placement move traces are still retained. Use the default frame runner for
full-frame replay capture or speculative preparation; event mode rejects these
unsupported combinations explicitly.
Predeclare frozen milestone paths in `variants`; the worker skips missing
checkpoints and admits them after their atomic save. It plays one complete
side-swapped batch per matchup, prioritizing the least-covered ready matchup,
then deepens the same reserved-seed schedule. It exits when every scheduled
game is complete. Use separate outputs and local working databases per worker;
split the schedule, not the sides of a seed pair. A variant's `checkpoint`
selects an older public core; `adapter_checkpoint` selects a residual on that
variant's parent. Historical cores with privileged auxiliary contracts are
rejected. Worker state and checkpoint readiness accompany every closed snapshot.
For a reused final-checkpoint filename, add `ready_when` with `path` pointing to
training telemetry, `field: status`, and `equals: Training complete`; an older
file at that path must not be mistaken for the new final model.
The live leaderboard combines tournament phases at the same level and pace;
Elo-reference changes use differences of joint posterior samples, preserving
their covariance. Levels and paces retain separate rating fields.

Controller arena records use the comparison ID as `condition_key`: the same
players, NES seed and port assignment can occur at multiple paces or levels.
The sync worker can recover metadata records omitted by the old unscoped key
from complete, uncensored side-swapped journal pairs, preserving existing
replay references where available. Original controller traces remain in each
worker's `moves/` directory. Database migration retains old rows and IDs;
unscoped native-arena callers retain their previous deduplication behavior.
Published SQLite snapshots use DELETE journaling and explicitly closed handles.
Mirrors exclude `.fuse_hidden*`, `*-wal` and `*-shm`; these are not game artifacts.

The current live schedule adds 36,608 games: eight frozen entrants in a 14 HI
round robin at Sloth, Relaxed, Normal, Fast and Top Humans, with 256 games
(128 side-swapped seeds) per pair and pace. That is 1,792 games per entrant
per pace when all eight are ready. Separate parent-versus-milestone checks
at 20 HI use 128 games per pair at Normal and Top Humans. The live bank is
disjoint from both the pilot and the scaled confirmation bank. This schedule
runs alongside training, split into whole matchups across two mombox CPU
workers and two MacBook workers (Metal and CPU); tf3090 keeps training. The
earlier 960 pilot games remain in the connected standings. The 25M and 50M
milestones and completed final adapter have joined.

The first study is `runs/trainer-pace-v1`, with isolated tf3090 sources and
outputs on `trainer-output/pace-strategy-20260908`. Its 160-game health pilot
completed 997,430 console frames and 9,763 learner decisions. The original
1,600-game budget was too small and allocated only hundreds of decisions to
Sloth. The continuation completed 101,229,824 console frames and at least
100,000 actual learning decisions per pace. After throughput validation, it used
1,024/256/64/64/64 games per update from Sloth through Top Humans, collected in
chunks of 128 with the event runner. It retains the existing weights
and optimizer and saves 25M, 50M and 100M milestones. Pilot and scaled evaluation
use separate slices of a 2,048-seed bank
excluded from this adapter training. Historical parent pretraining exposure
is shared by all candidates. The scaled study queues a 1,920-game 14 HI round
robin among the parent, 25M, 50M and final adapters, plus 256 separate 20 HI
pressure games. The 960-game pilot evaluation was inconclusive. Read live
results before drawing a strength
conclusion; no adapter has been adopted. The dashboard plan can set
`training_file`, `pipeline_file`, and `rating_anchor`; sync excludes checkpoints
and leaves full move archives on the overflow mount.

Use `tools.program launch trainer-planning-latency` with
`competitive_checkpoint`, `trainer_control_checkpoint`, `trainer_planning_roots`
and `trainer_planning_report` to measure complete warm decisions and conditional
preview batches. Measure transport and deadline reliability separately with
the source backend and Professor Pills' `trainer_smoke`; inference timings alone
do not determine a safe frame deadline.

`tools.program launch trainer-planning-arena --set trainer_arena_config=PATH`
runs an explicit JSON schedule. The config supplies checkpoint/device/library
paths, output and working-DB paths, variants with `delay`, optional `anticipation`,
`strict_opponent` and `own_board_only`, measured `reactive_compute_frames` and
`preparation_compute_frames`, and comparison rows with `id`, `a`, `b`, `games`,
`seed`, `level` and optional `pace`. Games must be even: each seed runs on both
physical sides. `seed_exclusions` keeps confirmation seeds separate from the
screen. `memoize` reuses exact immutable planner/policy answers without changing
the simulated compute charge. The arena uses actual controller frames and
aborts on a script/microstate mismatch. Keep level 14 HI primary; report level
20 HI and slower paces separately. The default 60,000-frame cap bounds only
unfinished games; completed rounds stop naturally.

The variant's `delay` is an assumed fresh-decision deadline, not a GPU timing
measurement. Reject a deployment deadline that fails the real host check even
if it wins offline. `trainer-planning-analysis` replays saved `trainer_planning_moves`
through controller frames and writes `trainer_planning_analysis`, comparing
fresh opponent inputs and reachability after four/eight idle frames with the
same `competitive_checkpoint`. Sample whole side-swapped games and keep their
correlation visible. Policy-logit changes are sensitivity evidence, not win-odds regret.

Every game retains public move roots, chosen placements, controller scripts,
cache status and terminal outcome in `moves/` plus `games.jsonl`. Distributed
replay samples retain both bottles and falling pills. Working SQLite must stay
on a local filesystem (on tf3090, `/dev/shm`); output, closed SQLite snapshots
and compressed traces may use the mombox `trainer-output` network mount.

The `trainer-experiment-dashboard` recipe takes `trainer_experiment`, `arena_db`
and `arena_port`. Its plan JSON names the work, goals and variants. Synchronize
closed remote outputs with `python -m tools.trainer_arena_sync --source DIR
--target DIR --feed NAME`; use a different feed per worker output. This imports
into the live DB instead of replacing a SQLite file with active WAL readers.
For live remote runs, launch `trainer-planning-sync` with `trainer_sync_config`:
its JSON contains local `target`, `interval_seconds` (at least ten), and `feeds`
mapping each feed name to an `SSH-host:/output/path` or a local output directory.
It transfers closed snapshots and sampled replays, excluding live `working/`
databases and full move archives. Optional `checkpoint_mirrors` entries contain
`source`, local `target`, and an explicit `files` allowlist; this lets local
evaluators admit new frozen milestones without copying optimizer archives.

The experiment page at `http://127.0.0.1:8098/` presents connected standings
first, with selectable level/pace and any rated player as the Elo reference.
It also shows the head-to-head matrix, all active workers, per-pace training
budgets, checkpoint readiness, a searchable schedule, and paused-by-default
controller replays with playback and scrubbing. Non-experiment arena dashboards
retain their existing page. Ratings use the Davidson/Laplace model and count
each paired seed as one effective observation. The primary view combines
compatible phases; phase-specific fields remain in the data. These are
approximate experiment comparisons, not drmariostats ratings. Small gaps with
wide intervals remain unresolved; inspect the matrix for matchup dependence.

The September 8 experiment lives under `runs/trainer-anticipation-v1/`, with
remote outputs in `trainer-output/anticipation-20260908/`. Professor Pills'
`PROFESSOR_PILLS_AI_PLANNING` permits `baseline`, `reactive` (three frames),
`reactive2`, `anticipatory` (exact opponent), `cached` (older opponent),
`own_reactive`, `own_prepared`, and the measured-budget variants `adaptive`,
`adaptive_exact`, `adaptive_cached`, `adaptive_own`. The app defaults to
`adaptive_cached`. Preparation applies only at Max and the three fastest paces,
and only when the reaction floor does not already cover computation.
Source and package release verification
remain distinct; these execution experiments do not certify human calibration.

### Arena coordinator

Exactly one host owns `arena.sqlite` on a local filesystem. Workers lease
batches through the authenticated coordinator; they never open SQLite over
NFS/SMB/SSHFS. Checkpoint delivery is content-hashed and worker results are
idempotent.

Typical coordinator:

```bash
uv run python -m tools.arena serve \
  --host 0.0.0.0 --port 8097 \
  --worker-token-file ~/.config/drmc-rl/arena-worker.token \
  --replay-dir /data/drmc-arena/replays
```

Typical worker:

```bash
uv run python -m tools.arena worker \
  --coordinator http://coordinator:8097 \
  --token-file ~/.config/drmc-rl/arena-worker.token \
  --worker-id macbook-mps --device mps --threads 2 --batch 12
```

## Gate evidence

A gate report is run output and remains untracked. Record it after reviewing the
actual evidence:

```bash
uv run python -m tools.program gate record timing-action-gate --passed \
  --metric probes=12000 \
  --metric structural_changed_fraction=0.013 \
  --metric clock_divergent_fraction=0.91 \
  --metric beneficial_delay_fraction=0.0002 \
  --artifact /data/drmc/timing/report-v1.json \
  --note "Placement-only retained under the predeclared threshold."

uv run python -m tools.program gate check timing-action-gate
```

The evidence file records time, commit, metrics, artifacts, and notes. The
program registry will not infer a scientific pass merely because a file exists;
`passed` must be explicit.

## Artifact identity

Every candidate entering permanent arena evidence receives a sidecar manifest:

```bash
uv run python -m tools.program artifact \
  runs/campaign/checkpoints/smdp_ppo_step250000000.pt.gz \
  --config drmc_rl/training/configs/campaign.yaml \
  --observation-schema drmc-public-pair-state-v2 \
  --execution-profile unrestricted \
  --search '{"kind":"none"}' \
  --corpus-release human-v3-2026-08 \
  --parent sha256:previous-checkpoint
```

The manifest contains:

- artifact hash and size;
- config hash;
- repository commit and dirty state;
- native submodule revision;
- observation schema;
- execution profile;
- search settings;
- corpus release;
- parent artifacts and additional metadata.

Do not register every autosave in the arena. Register scientifically meaningful
milestones after the file is settled and its manifest exists.

## Timing-action experiment

Prepare one JSON object per line. Each probe records a spawn-time pair reset,
target exact pose, strictly increasing candidate lock frames, and the opponent's
committed lock or spectator marker. Use a stratified state bank covering clear,
pressure, garbage, high-speed, and ordinary states.

```bash
uv run python -m tools.earliest_lock_dominance \
  --input /data/drmc/timing/probes-v1.jsonl \
  --output /data/drmc/timing/report-v1.json
```

The report separates clock divergence from structural next-event divergence;
a later lock is not declared strategically material merely because its clock is
later. Supply a common-scale continuation evaluator when value evidence is
available:

```bash
uv run python -m tools.earliest_lock_dominance \
  --input /data/drmc/timing/probes-v1.jsonl \
  --output /data/drmc/timing/report-v1.json \
  --value-adapter drmc_project.timing_value:score
```

The adapter receives `(snapshot, probe)` and returns a scalar continuation
value from the same frozen policy mixture for every delay.

## Counterfactual and search releases

Both tools require an explicit `module:function` adapter. The adapter returns a
`PairSearchModel` (or ensemble) and a state decoder. This prevents accidental
fallback to the old own-board simulator.

Build a bounded strict-native bank before the first diagnostic pilot:

```bash
uv run python -m tools.build_pair_state_pilot \
  --output runs/counterfactual/pair-state-pilot-v1.jsonl.gz \
  --states 512 --states-per-game 8 --seed 20260816
```

`drmc_rl.search.native_pair:diagnostic_factory` exists only to validate exact
restore, causal branching, full candidate coverage, and release mechanics. Its
public-state heuristic is not a calibrated continuation mixture and its output
must not open `v3-counterfactual-quality`.

```bash
uv run python -m tools.program launch counterfactual-labels --allow-staged \
  --set pair_state_bank=pair-states.jsonl.gz \
  --set counterfactual_release=runs/counterfactual/pilot-v1 \
  --set counterfactual_adapter=drmc_rl.search.strong_league:frozen_strong_league_factory \
  --set counterfactual_root_side=0 \
  --set counterfactual_depth_events=2 \
  --set counterfactual_own_beam=512 \
  --set counterfactual_opponent_beam=1 \
  --set counterfactual_chance_beam=9 \
  --set counterfactual_max_nodes=10000 \
  --set counterfactual_chunk_size=16 \
  --set counterfactual_max_states=512 \
  --set counterfactual_seed=20260818 \
  --set counterfactual_corpus_release=pair-state-bank-v1-sha256:... \
  --set counterfactual_continuation_mixture=strong-league-frozen-mixture-v1 \
  --set counterfactual_mixture_manifest=mixture-manifest.json \
  --set counterfactual_wdl_calibration=wdl-calibration.json \
  --set counterfactual_device=cpu \
  --set counterfactual_native_revision=<native-commit> \
  --set counterfactual_planner_revision=<planner-commit>

uv run python -m tools.joint_search_teacher \
  --states pair-states.jsonl.gz \
  --output search-targets.jsonl.gz \
  --adapter drmc_project.native_adapter:factory \
  --checkpoint checkpoint.pt.gz
```

Full-candidate counterfactual releases must use `own_beam >= legal candidate
count`; omitted candidates raise an error.

Reveal-aware pilots must use `depth-events >= 2` and `chance-beam >= 9` so all
ordered pill colors are integrated. The release rows report `chance_nodes` and
`chance_outcomes`; a state with 32 root candidates should ordinarily report 32
and 288 respectively when each root branch reaches one reveal boundary.

The counterfactual writer produces deterministic, content-addressed gzip
chunks plus verified completion records and an aggregate manifest. Resume only
accepts chunks whose settings and content hashes match. Production releases
reject search-budget exhaustion; `--allow-budget-exhausted` is diagnostic-only.
Single-teacher pilots report uncertainty as unavailable rather than zero.

## Execution profiles and style

Fit a named profile from raw frame-indexed scripts:

```bash
uv run python -m tools.execution_profile fit \
  --input elite-scripts.jsonl \
  --output profiles/elite-p99-v1.json \
  --id elite-p99-v1 \
  --description "Fightcade top-cohort p99 operation envelope" \
  --quantile 0.99

uv run python -m tools.execution_profile validate \
  --input heldout-elite-scripts.jsonl \
  --profile profiles/elite-p99-v1.json
```

Fit a rating-residualized style space from an NPZ containing `features`,
`ratings`, and `player_ids`:

```bash
uv run python -m tools.style_space \
  --input player-style-features.npz \
  --output style-space-v1.json --dimensions 6
```

Profile and style releases are immutable inputs to a trainer artifact; their
hashes belong in its manifest.

## PSRO mixture

Export a square payoff matrix as JSON `{agents, payoff}` or CSV, then run:

```bash
uv run python -m tools.meta_strategy \
  --payoff arena-payoff.json \
  --output meta-strategy.json
```

The output reports row/column/population mixtures, game value, best responses,
and saddle gap. Before this controls opponent sampling, verify the arena graph
is connected and side noise has been audited; the default symmetric path
antisymmetrizes the matrix.

## Recovery and migration

- Checkpoints are immutable once registered.
- Resume optimizer state only when the recipe explicitly declares it.
- Copy an arena database only after pausing workers, checkpointing WAL, and
  verifying checksums, replay hashes, checkpoint paths, and single-writer
  ownership.
- Never repair a failed run by silently changing its embedded objective or
  observation schema.
- A failed full Bayesian rating fit leaves the last accepted posterior visible;
  do not substitute a scalar heuristic and call it the same rating model.

## Pull-request checks

The PR workflow validates the authority manifest, compiles new pure-Python
modules, runs focused architecture tests, and lints. Native/emulator parity and
large training/evaluation jobs remain explicit gate evidence rather than CI
claims.
