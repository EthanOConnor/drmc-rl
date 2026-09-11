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
same bounds. A remaining full-core outlier is checked against an independent
CPU FP64 copy of the unchanged network and inputs, still under those same
bounds. A failure against that reference stops before optimization. The
reference copy never mutates the actor, optimizer, sampling RNG or stored
behavior likelihoods. Raw FP32 discrepancies, single-row rechecks and precise
recheck errors/counts are recorded separately. These numerical bounds limit
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

At update 379, the September 10 run stopped before optimization on an FP32
total-variation discrepancy of 0.000118735, above the unchanged 0.0001 limit.
Its 11,326-decision shard and update-378 model/optimizer remain preserved.
Replaying all rows reproduced a batched maximum of 0.000122210, with 99th
percentile 0.000003517. The worst row's independent FP64 error was 0.000091169
and log-probability error 0.000388528, both within the existing limits. FP64
batch-size variation was below 2e-13; disabling cuDNN or using double group
normalization alone did not remove the FP32 sensitivity. This motivates the
rare accurate-reference check, not a looser audit threshold or an altered
behavior distribution. The preserved numerical reports are
`controller-core-live-v4/failed-collection-379-audit.json` and
`failed-collection-379-precision.json` on the training output mount.
The implemented FP64 fallback then passed all 11,326 preserved decisions with
exact original behavior logs retained; only one row needed the precise check.
Three focused native tests also verify untouched weights, dtypes, RNG and
version counters, plus rejection of real distribution corruption. Source
`controller-core-9154380-source` resumes the unchanged update-378 model and
optimizer through `trainer-controller-core`, configuration
`controller-core-live-v4-after-precision-v1.json`. The recovery directory
`controller-core-precision-recovery-v1` preserves the failed replay, failure
report, and resumable checkpoint (SHA-256
`567d18d23f711c0286c69adb63b9437e757655bdf562389100d7c819d9cc6aa3`).
Its supervisor is 3052699, and its log is the host-local
`controller-core-live-v4-precision-v1.log`. Native libraries remain the original
`19f292c` files in the earlier source snapshot. Update 379 completed after
recovery; counts reached 472,234,760 frames and 4,739,370 learner decisions.

The active full-core run is `review-20260909/controller-core-live-v4` under
the tf3090 trainer-output mount. Its launch configuration is under the sibling
`configs/` directory, currently `controller-core-live-v4-after-precision-v1.json`; each
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
For new heads, `head_initialization: training_cell_prior` starts the output
biases from equal-game-weighted training cell/parity frequencies and conditional
cost means, with zero output weights. This never reads validation or confirmation
labels, and refuses to reset already learned heads. Optional fixed `head_epochs`
fit only the auxiliary network at `head_lr` (default 0.0003), on `head_device`
(default CPU). The shared core's complete public candidate features are cached
in FP32; targets are stored separately and never enter feature extraction.
The cache is discarded before shared-core fitting, and an exact core-state hash
must remain unchanged through warmup. `joint_head_lr` controls the new heads'
learning rate separately from the pretrained core's `lr`. `head-fit.json`
records fixed-epoch development measurements; `post-head-warmup.json` checks
the actual core forward before joint updates. Counters distinguish head-only
and shared-core presentations. Final `condition-metrics.json` compares each
level/pace with training-only pace/cell prevalence; holdout scores never alter
the schedule or select a checkpoint.
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

`motor-confirmation-v1` completed at September 10 11:38:29 UTC: all nine
conditions, 1,152 natural games and 1,728 exact roots, with 46–54 independent
scored reset seeds per condition. Both reach and clear Brier errors were worse
than training-only pace/cell prevalence in every condition's paired interval.
For 14-HI Sloth, reach was 0.08285 versus 0.01555 and clear 0.06070 versus
0.00135; Frame Perfect was 0.14402 versus 0.12034 and 0.07274 versus 0.02047.
The last 20-HI Top Humans condition also lost: reach 0.12263 versus 0.11500,
clear 0.06316 versus 0.01384. This rejects the first fit as useful prediction
evidence despite its improvement over random initialization. Preserve its
fixed final report and the separate ongoing strength arena. Its confirmation
seeds remain excluded from subsequent fitting and new confirmation selection.

The fixed revision `motor-refit-v1`, source `motor-refit-d36716e-source`,
completed on the Mac at September 10 12:14:13 UTC in 837.5 seconds. It used
the same 1,679 training roots, 369 development roots and 1,489 public anchor
games: 40 head-only epochs at 0.001 after training-cell prior initialization,
then 20 shared-core epochs at 0.000003 with head rate 0.0001. Its 67,160
head-only plus 33,580 shared-core root presentations are not outcome-training
frames. The core state was exactly unchanged during head warmup; joint fitting
needed no rollback and final anchor KL was 0.000920. Final development reach
Brier was 0.06444 and clear Brier 0.01443. Reach beat the training-only prior
in all development conditions, while Sloth clear prediction remained slightly
worse; these descriptive results do not establish useful live prediction or
stronger play. The immutable final checkpoint SHA-256 is
`a7db0ca8128b2dc7506e111aff35decfc403f9d6a16e2231c5efbfa01a07c97b`.

`motor-refit-confirmation-v1` now runs through `trainer-motor-confirmation`
on Mac MPS from that same frozen source and the original native `19f292c`
libraries. Its JSON under `runs/review-20260909/` fixes 64 fresh seeds,
disjoint from the first confirmation and all fitting/outcome-training seeds,
with the same nine-condition allocation of 1,152 natural games and up to
1,728 roots. `exclude_confirmation_configs` validates the prior confirmation
exclusion and records its config hash. Inspect its adjacent `.pid`/`.log`
and output progress before recovery. The dashboard shows refitting and fresh
confirmation separately; no product route changed.

The new `motor-refit-confirmation-v1` completed at September 10 12:45:52 UTC:
all nine conditions, 1,152 natural games and 1,728 labeled positions. Its 46–52
scored independent reset seeds per condition support lower reachability Brier
error than the training-only pace/cell prior at every condition. At 14 HI,
Sloth was 0.01020 versus 0.01761, Normal 0.06817 versus 0.09430, Top Humans
0.08766 versus 0.11677, and Frame Perfect 0.09368 versus 0.12062. Clear Brier
also improved in seven conditions; Sloth and 20-HI Top Humans were inconclusive.
`motor-refit-confirmation-family-v1.json` retains 20,000 shared whole-reset-seed
resamples, simultaneous standardized intervals over all 18 reach/clear
contrasts, and the same conclusions after that family correction. These are
conditional no-incoming-garbage predictions, not competitive values or a
demonstrated playing-strength improvement.

The fixed follow-up `motor-refit-arena-v1` completed all 16,384 natural controller
games at September 10 17:35:22 UTC, with zero censoring, on Mac MPS through
`trainer-planning-arena` from frozen
`motor-refit-d36716e-source` with original native `19f292c` libraries and the
unchanged four-frame compute charge. It compares the refit with its 25M parent
at every pace and the current product route: original final Sloth, original
50M Relaxed, corrected E1 Normal/Fast/Top Humans, and public 10M Super Human/
Frame Perfect. Seven 14-HI paces receive 1,024 games per opponent; 20-HI Normal
and Top Humans receive 512 separately. The 768 fixed reset seeds exclude
training, fitting and both motor prediction confirmations. Supervisor 52970,
config, study, journals and log are under the local review root. A live
`motor-refit-arena` feed and `public_core_motor_refit` entrant retain the
complete allocation in the common tournament. Former supervisor 52970 is done;
do not restart the allocation.

`motor-refit-arena-assessment.json` retains 20,000 shared whole-reset-seed
bootstrap resamples and simultaneous intervals across all 18 fixed conditions.
No condition establishes a strength gain. Against product routes at 14 HI,
Sloth scores 43.55% (40.50–46.61%), Relaxed 41.36% (35.64–47.07%), Fast 40.92%
(35.68–46.16%), and Top Humans 41.21% (36.17–46.25%); all intervals are the
simultaneous family intervals. Normal versus its own 25M parent is also below
50% (42.99–49.98%). At 20 HI, Top Humans versus E1 scores 41.89%
(33.89–49.90%); Normal is inconclusive. Natural draws remain half scores.
Reject this refit as a competitive replacement despite its improved conditional
motor prediction. Keep the labels/prediction evidence, existing product routes,
and the independent outcome-training optimizer unchanged. The original motor
auxiliary arena on mombox is a separate still-running study.

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

Both original Mac workers are complete: 18,432 plus 14,336 games, with the last
game at 00:46 UTC September 10. Do not restart either allocation. The audit
`review-20260909/core-initial-25m-assessment.py` checks complete, unique seed/side
pairs, exact schedule/journal counts and score agreement. Its JSON report
resamples whole reset-seed pairs 20,000 times and preserves cross-pace seed
covariance. Pointwise intervals accompany all 36 cells; simultaneous intervals
cover the 14 primary 14-HI comparisons of the 25M core against parent and E1.
The 25M core is inferior to E1 at Sloth through Top Humans even under those
family intervals. Super Human/Frame Perfect use the same frozen E1/parent
behavior; their repeated reference rows are not additional independent evidence.
Sloth's 25M-versus-E1 cell contains 786 natural draws in 1,024 games. Retain
the draw rates and execution details when interpreting the score.

The separate
`controller-core-100m-mac.json` study reuses its MPS capacity for 16,384 games:
the immutable 100M checkpoint against the parent and corrected E1 adapter,
1,024 games per 14-HI pace and 512 per 20-HI Normal/Top Humans condition. It
uses the same reserved seeds and frozen `controller-arena-0c76c0e-source`.
`launch_core100m_mac.py` records both frozen native libraries, including
`DRMARIO_REACH_LIB`; its output has a supervisor PID/log and complete move
journals. This is a fixed milestone comparison without automatic adoption.
The dashboard includes `public_core_100m` and the `core-100m-mac` feed.
This allocation completed at 03:20 UTC September 10 with all 16,384 games and
zero censoring; do not restart it. `core-100m-assessment.py/.json` verifies
journals and uses the same 20,000 shared whole-seed bootstrap method as the
initial/25M assessment. Sloth beats E1 in the primary simultaneous family,
but Fast and Top Humans remain inferior; this is not a replacement across all
speeds. `controller-core-slow-specialists-mac.json` scheduled 2,048 14-HI
games each against `pace_final` at Sloth and `pace_f50m` at Relaxed. Its 1,024
common reset seeds are excluded from main/adapter training, the earlier core
studies and the reserved motor-confirmation tail. It uses the same frozen
controller source/libraries and four-frame compute assumption, through
`launch_core_slow_specialists_mac.py`, with its own output supervisor PID/log.
The matching dashboard feed is `core-slow-specialists-mac`. All 4,096 games
completed at 04:11 UTC September 10, with zero censoring. Do not restart its
supervisor. `core-slow-specialists-assessment.py/.json` verifies complete
seed/side coverage and scores, retaining execution details and 20,000 shared
whole-seed bootstrap draws. Against the selected specialists, Sloth scores
52.17% (two-condition simultaneous 95% interval 50.30–54.05%; 1,383 natural
draws), while Relaxed scores 44.73% (41.60–47.85%; 28 draws). This supports the
core as a Sloth candidate only; actual trainer integration and device latency
remain separate requirements before changing the tester package.

New core migrations use learned zero-initialized side-conditioning residuals
so the network preserves its parent on equal inputs before outcome updates.
Old context checkpoints and resumed references retain their original graph.
The real-parent audit in `review-20260909/context-migration-audit` preserved
all probabilities and values on 288 recorded decisions; the direct migration
changed 12 choices. The active frozen run is unchanged. The next independent
training branch must evaluate the corrected initialization and full controller
behavior; equal-input migration parity alone does not establish match strength.

A fixed initialization-only study used the freed Mac slot:
`controller-residual-initial-mac.json`, launched by
`launch_residual_initial_mac.py` from `controller-residual-07f46eb-source`.
Its 16,384 games compare the corrected initial core with the parent and original
initial core: 1,024 games per 14-HI pace/opponent and 512 per 20-HI Normal/Top
Humans cell. The original parameters are retained exactly; only the two added
zero side-conditioning scales and their graph flag differ. No extra training
is claimed. `residual-initial-v1/manifest.json` records the original and corrected
checkpoint hashes; `prepare_residual_initial_study.py` verifies all shared
tensors and strict controller-core reload. The same frozen native 19f292c
controller libraries and four-frame compute assumption are retained to match
the original study; the separate search-reveal path is not used. Read its
supervisor PID/log and complete journals before recovery. This allocation is
complete: all 16,384 natural games finished at 06:06 UTC September 10 with zero
censoring. Do not restart it. `core-residual-initial-assessment.py/.json` verifies
the complete journal and side pairs, using 20,000 shared whole-seed bootstrap
draws. One Sloth comparison has exactly 0.5 for every seed pair, so this study
uses a common maximum absolute-score bootstrap band across its 14 primary
comparisons instead of dividing by a zero standard error. Its family radius is
4.10 percentage points. No 14-HI comparison resolves an improvement over the
original initial core. Against the parent, Top Humans/Super Human/Frame Perfect
score 43.75%/42.68%/42.58% and all three family intervals stay below 50%.
Normal's pointwise loss does not survive the family check; other paces remain
inconclusive. Level 20 stays separate. Before a new initialization training
branch, examine the remaining live encoding and candidate-frontier changes
using training-only roots. The main run retains its frozen source/checkpoint.

`runs/review-20260909/migration-input-attribution.py` performs the eight-way
factorial input audit without learning or new physics labels. It uses actual
training replay tensors from `context-migration-audit/update-00001/00003/00007.npz`,
four temporal rows per reset seed per shard. These V1 replay files lack exact
motor geometry; never invent it or use this audit to generate movement labels.
The 768-row report reproduces the frozen parent when both encoding and frontier
are retained. Full restoration changes 49 selected controller actions and 30
physical placements. Both bond changes alone alter 11 physical placements;
the frontier change alone alters 25. There are 235 same-color rows and no
new distinct physical placements in this sample; extra orientations still
carry distinct timing/controller witnesses. Report level 20 separately.

For migration fitting use `tools.program launch trainer-public-input-alignment
--set public_alignment_config=PATH`. The JSON requires `parent`, `replays`,
`output` and `source_scope: training-only-controller-replay`; it can set
`device`, `threads`, `seed`, `epochs`, `batch_size`, `roots_per_seed_per_shard`,
`validation_fraction`, `learning_rate` and `support_epsilon`. The teacher uses
its actual legacy public contract. The full-input student targets its behavior
with a small declared uniform probability on every legal action; default
epsilon is 0.0001. This is not WDL/quality supervision. Every shared reset seed
stays together across shards; only training rows enter gradients. The final
fixed epoch is retained regardless of descriptive validation. `progress.json`
records actual supervised root presentations, grouped KL/agreement and source
identities; console frames trained are zero. `core-final.pt` is diagnostic and
must pass a substantial controller comparison before adoption. Focused tests
verify full target support and identical trained weights after changing only
held-out observations/targets.

The first substantive alignment fit is complete at September 10 07:14 UTC:
`public-input-alignment-v2`, frozen source `9d6e38d`, eight fixed epochs on
Metal in 543 seconds. It made 38,616 supervised presentations from 4,827
training positions (153 reset seeds); 1,207 validation positions belong to
38 other seeds. No rows lacked a historical teacher action. Seed-weighted
validation KL fell from 0.14333 to 0.03085 and choice agreement rose from
91.14% to 94.31%. These are behavior-alignment metrics, not stronger play.
The fixed final checkpoint SHA256 is
`66d4327bf3ec6160dd624467da186c6f520f2cdbbc512ef100baff78b4f092a3`.
The initial v1 launch was stopped during source loading before any optimizer
update; v2 fixes repeated NPZ decompression by loading each column once.

`public-input-alignment-arena-v1` completed on the Mac through the
registered `trainer-planning-arena` recipe, former supervisor 68616. Its config
and launch/preparation scripts are under `runs/review-20260909`. The fixed budget
was 16,384 games against the public parent and corrected untrained initialization:
seven 14-HI paces at 1,024 games per opponent, and separate 20-HI Normal/Top
Humans at 512. It uses frozen arena `07f46eb` and the same native `19f292c`
frame/reach libraries, one Metal/Torch worker, three planner workers and four
charged compute frames. Its 768 distinct reserved seeds have no overlap with
any of the 191 alignment source seeds. The bank is reused from the preceding
initialization study, so this is not an independent confirmation study.
The allocation completed without censoring at September 10 11:19:46 UTC.
`public-input-alignment-arena-assessment.py/.json` verifies every seed/side pair
and retains 20,000 shared whole-seed resamples. No primary family comparison
establishes a gain; Sloth versus parent scores 45.21% with simultaneous 95%
40.53–49.90%. Faster 14-HI point estimates are near the parent, while separate
20-HI Normal scores 43.16% versus parent and 42.77% versus corrected initial.
Keep the current pace portfolio and the active outcome-training optimizer.
Do not restart this completed allocation; further alignment work needs broader
training-only pace coverage and a separately evaluated new branch.
The existing single dashboard sync includes this arena, and a separate
alignment card counts example presentations without inflating console frames.

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

- `public-quality-bank`: set `public_quality_bank_config`. The JSON declares
  frozen `members`, two-sided `matchups`, weighted level/speed `conditions`,
  game-count `partitions` (fit/anchor/confirmation), `excluded_reset_seeds`,
  `seed`, `states_per_game`, `max_events`, `batch_size`, `native_workers`,
  `device` and `output`. The catalog assigns distinct reset seeds before play.
  Each game starts from a cold native pair: ordinary cartridge round reset
  preserves attack-color history and is unsuitable for independent source
  games when slot order changes. Causal public inference is batched across
  games. Bounded per-game reservoirs retain opening/middle/late positions and
  tactical diversity, with at most one 4/8/16-placement loss predecessor.
  Every completed game is committed in `games/`; an identical-contract resume
  reuses it. Partition JSONL banks and manifests export on completion or
  failure. Censored games stay explicit and are ineligible for natural-outcome
  labels. This collector accepts legacy public input cores only; motor/history
  cores use controller replay rather than fabricated missing inputs. These
  source-policy lineages provide experience diversity, not independently
  trained uncertainty members.
  Reserve validation caches the seeds compatible with each public initial
  bottle, then filters only those seeds by the reveal history. The cache is
  bounded and preserves the exact posterior; it does not approximate or prune
  possible reserves. Exhaustive 65,536-seed comparisons cover 14/20-HI bottles
  and arbitrary reveal prefixes through all 128 entries.
- `public-predecessor-bank`: set `competitive_checkpoint`, `bank_device`,
  `bank_states` and `public_predecessor_bank`. It collects clean 14-HI games,
  retaining full causal public views, native restore state, complete reserve
  history and positions 4/8/16 own placements before natural losses.
- `paired-terminal-quality`: set `paired_terminal_config`. The JSON names
  `state_bank`, frozen `members`, weighted `continuations` (actor/opponent),
  `reference`, `states`, `seed`, `device`, `output`, `batch_size`, `max_events`
  and optional level/speed filters, `stratum_fields`, `root_batch_size`
  (default 1), `native_workers` (default 1), and `policy_cache_size` (default 0).
  A positive cache size enables bounded per-frozen-member inference memoization
  and deduplication within a batch. Keys cover the actor's complete public
  inputs: legacy zero-context models omit unused clock/age fields, while public
  context models retain their complete history view. Hidden state never keys
  or enters the actor. Inference counts distinguish logical policy decisions,
  cache hits, repeated inputs and actual neural rows. This preserves the
  mathematical frozen policy; as with other batch-shape changes, measure any
  FP32 close-tie trajectory differences explicitly. Neural
  inference remains ordered and batched; additional workers step independent
  native handles. Progress refreshes after each loop once five seconds have
  elapsed, even when no trajectory has finished. Every root candidate uses the same exact
  reserve/policy panel. Batching across roots keeps long continuations from
  leaving the accelerator with a nearly empty last batch at every root.
  Inspect `progress.json`, `targets.jsonl` and `rollouts.jsonl`; utilization
  includes actual neural batch rows, decisions and native/inference time.
  Complete per-root inventory/result files in `roots/` are authoritative for
  resume under the same `contract.json`; partial roots are recomputed and
  aggregate exports rebuilt. Legacy outputs cannot acquire this contract
  after the fact. A capped branch is unknown, not a draw.
  The native reveal query must obey the same P1-before-P2 frame ordering as
  strict stepping. An available reserve entry is not a runnable chance node
  when the other player still owes an earlier input. The regression in
  `test_search_reveal_order.py` catches this on the old library. The original
  one/eight-root and 256-slot benchmark labels predate this fix and are not
  eligible for quality training; retain them as throughput measurements. The
  broad source corpus uses strict stepping and remains usable.
  A `label-validity.json` beside `targets.jsonl` records an execution audit;
  the quality fitter refuses a present marker unless
  `eligible_for_quality_training` is explicitly true. This quarantine does not
  replace the held-out quality or product-promotion gates.
- `paired-quality-fit --allow-staged`: set `paired_quality_fit_config` with
  `state_bank`, `targets`, parent `checkpoint`, `mode` (baseline/critic/context/
  combined), `phase` (auxiliary/policy_improvement), `seed`, `device`, `epochs`,
  `batch_size`, `lr`, `max_policy_kl`, a separate `anchor_bank` and fresh
  `output`. New fits require at least 256 independent anchor games by default
  (`minimum_anchor_games` is explicit for small mechanical tests). An optional
  `confirmation_bank` is read only to check partition isolation. Position ids,
  game ids and reset-seed byte pairs must be disjoint across the complete source,
  anchor and confirmation banks, including source positions without labels.
  Training/validation splitting and loss weighting group repeated reset seeds;
  older banks without seed metadata use whole game ids. Each group's retained
  roots share one unit of weight. Both the labeled training split and independent
  policy anchors constrain epoch acceptance; held-out drift cannot trigger
  rollback, early stopping or a learning-rate change.

  Prepared inputs, initial policy distributions and rollback copies remain in
  host memory. `evaluation_batch_size` bounds reference and validation forwards
  independently of the training batch size. Reports aggregate per-root sufficient
  statistics before normalizing whole-game metrics. They include complete-frontier
  pairwise accuracy, greedy regret/gain and the number of informative roots and
  games; flat outcome panels do not dilute informative-only denominators, and
  predicted ties receive their mean utility. These values describe the frozen
  continuation panel, not optimal game values. Initial and final held-out
  candidate predictions are retained for whole-game comparisons. Layer diagnostics
  use at most `representation_samples` (default 4096) fixed training candidates;
  this sampling does not affect any loss or feasible frontier. `accepted_examples`
  and `examples_processed` distinguish retained updates from rollback attempts.

  Optional `encoder_growth: {"channels": 384, "blocks": 12}` (or 512×12)
  expands a dense G5 bottle encoder after public/head migration. Token width is
  retained, so the reported architecture is bottle channels × residual blocks,
  not a wider transformer. Initialization preserves learned outputs through
  partitioned normalization and an identity projection. Growth cannot shrink
  the parent or discard its learned tensors; fitted expanded models reload
  normally, including a learned projection and repeated growth. Reports retain
  parent/target sizes, partitions, parameter counts and growth seed. The parent
  and caller's CPU/CUDA random state remain unchanged by construction.

  Set a shared `split_seed` across all comparison arms and independent `seed`
  values for teacher members; omitting `split_seed` retains the old `seed`
  behavior. Use the same labels, public mode, anchors and held-out whole games
  when comparing widths. Equal exposure and equal GPU allocation are separate
  studies. Ordinary fitting records epochs, examples, attempted updates and
  elapsed wall time; use the allocation supervisor below for a time-bounded
  comparison. The real-checkpoint initialization audit is under
  `review-20260909/model-growth-audit`; CPU output parity and 41 focused growth,
  streamed-fit, G5 and program checks passed. CUDA evidence is in its
  `assessment-cuda.json` and the remote `model-growth-audit-cuda` directory, run
  from immutable `teacher-growth-2e5b0bf-source`. Both expanded models retained
  all 288 greedy choices within the unchanged 1e-4 total-variation and 1e-3
  log-probability bounds. The stricter value/representation probe failed with
  small FP32 errors; a 16-bottle FP64 comparison matched exactly. Keep these
  numerical diagnostics distinct from strength or inference-latency evidence.

  A fitted checkpoint may initialize the next phase with the
  same mode/schema; learned heads and effective EMA weights are preserved.
  Architecture changes require a new migration from the frozen core. The
  output is a diagnostic checkpoint, never automatically installed or declared
  calibrated.
- `paired-quality-fit-budgeted --allow-staged`: set `paired_quality_budget_config`
  to a JSON object containing a complete nested `fit_config`, a fresh `output`,
  `allocation_seconds`, the full physical NVIDIA `gpu_uuid`, and a fast local
  `scratch_root` (`/dev/shm` on tf3090 when sufficient space is available).
  `checkpoint_interval_seconds` defaults to 30. The supervisor sets the child's
  device to the selected GPU via `CUDA_VISIBLE_DEVICES`, redirects its log, and
  checks that no other compute process uses that GPU before launch or during
  roughly one-second ownership polls. Schedule this separately from main
  training and terminal labeling; their current GPU contexts correctly prevent
  a launch. Other processes are never terminated by this tool.

  An independent watchdog terminates the child at the allowance, escalating
  only that child to a kill if it does not exit within two seconds. Initial and
  accepted-epoch snapshots are atomic, use immutable names and retain two local
  versions. Interrupted writes and checkpoints stored after the cutoff are
  ineligible. The final export copies the last eligible snapshot to
  `diagnostic.pt` after the child exits and records its SHA256. `progress.json`
  is the supervisor's authority; the stopped child's live progress cannot imply
  continued training. Failed runs retain their named scratch directory for
  recovery, including a checked model if GPU monitoring failed.

  Report the allowance, actual elapsed allocation, unused time, cutoff overrun,
  selected checkpoint's age and its accepted examples/steps. A natural early
  stop does not consume the full allowance, and last-reported child counters
  can include work absent from the selected model. Rescore that exact selected
  checkpoint in the common held-out assessment; do not attach predictions from
  a later interrupted child update. Fixed epochs/equal exposure and equal
  maximum allocation are separate arms with common splits/anchors. These
  controls passed 22 focused Mac checks and 11 Linux process/checkpoint checks
  on tf3090 from `quality-budget-07f46eb-source`. The installed NVIDIA query
  correctly reports the main trainer and labeler as existing GPU owners. A
  substantive exclusive-GPU fit and resource comparison remain to be run.
  These tests do not promote a teacher or claim additional GPU training.
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

The broader `review-20260909/public-quality-bank-v1` corpus completed 2,048
independent games: 1,280 fit, 512 policy-anchor and 256 confirmation games, with
up to eight retained positions per game. All games ended naturally, producing
10,232 fit, 4,096 anchor and 2,044 untouched confirmation positions. Three frozen public checkpoints (bootstrap,
2M and 10M) supply all nine matchups. Seven-eighths of games are 14 HI; the
remainder are 20 HI. The 2M input is an inference export of existing weights,
not new training. The initial source `97ea22b` committed 271 games before a
posterior-throughput upgrade. Source `6dfaf54` resumes the identical corpus
contract and retains those games. Its public-bottle factoring passed exhaustive
posterior comparisons on the Mac and tf3090; a 192-query Mac microbenchmark
measured 23x faster matching, which is not an overall training-throughput claim.

The completed `teacher-data-slot-v2` compared one versus eight roots in flight
on the same eight-root 14-HI panels, with all legal actions, nine continuation
pairs and the complete public reserve posterior. It measured 1.032x overall
throughput; all 2,340 terminal outcomes and 260 candidate WDL estimates agreed.
The subsequent v3 comparison increased rollout slots from 64 to 256 and measured
1.174x overall throughput, again with complete label agreement. These three
benchmark label sets predate native reveal-order correction `e0162ed` and are
quarantined by `label-validity.json`; agreement between them does not establish
validity for training. The source corpus and main controller training use
different native paths and are unaffected.

The completed `teacher-data-slot-v4` compared corrected uncached and cached 256-slot
labeling in `terminal-fixed-256-v1` and `terminal-fixed-cache-256-v1`. Both use
immutable `teacher-label-d2cce23-source`, native `e0162ed`, the same eight roots,
four native workers and two Torch threads; the second enables an 8,192-entry
public-input policy cache. Its `benchmark.json` records actual neural rows,
cache hits, throughput and label agreement separately. Cached labeling took
507 seconds versus 578 uncached (1.140x overall throughput), preserved all
2,340 outcomes and 260 candidate WDL estimates, and reduced actual neural rows
from 210,209 to 137,269. The supervisor used a 30-minute GPU cap and resumed the
same main checkpoint and optimizer at 22:40 UTC via
`controller-core-live-v4-after-teacher-data-v4.json`. It preserved update 113
at 140,132,420 console frames and 1,372,391 learner decisions. Read the latest
slot progress, resume configuration and actual supervisor PIDs before recovery;
never rewind a newer main checkpoint to an older slot's saved update.

`terminal-quality-broad-v1` now supervises complete corrected labels for
`terminal-quality-14hi-v1` (1,024 independent source games), followed by
`terminal-quality-20hi-v1` (128 separate games). It selects one temporally and
tactically stratified root per game, retains the full frontier and correlated
reserve panel, and uses that same immutable source with an 8,192-entry cache.
The 256 confirmation games remain untouched. The registered jobs run alongside
main training with one native worker, one Torch thread and nice 15, under a
48-hour wall cap. Monitor combined progress/CPU contention before reallocating
compute; completed root files survive an interruption. The supervisor is
`run_broad_teacher_labels_v1.py`, with current child/PID details in its control
directory's `progress.json`; logs remain local on tf3090. New quality fitting
must include the quarantine guard in `6d7e05a` or later. These studies do not
complete the independent larger-teacher and student-distillation program or
promote a model.

Native `23411a5` adds `drm_vspool_settled_public`, exposed by
`DrMarioVsPoolRunner.settled_public()`. A fresh placement reset records a native
event timeline; `capture_native_state(..., event_public=True)` chooses the new
`causal-settled-pair-v2` contract. Later captures and search branches inherit it.
V2 snapshots append observer history, while old V1 snapshots remain exactly
restorable with the new public API unavailable. Never use an old snapshot's
current raw bottle to initialize this history. Non-strict/forced stepping also
invalidates it; the frame controller uses its own public history.
The isolated Mac library is `drmario-native/build-event-public/libdrmario_pool.dylib`.
Thirty-nine focused Mac checks cover public capture, hidden commitments,
inference, old/new restore, controller physics and the unchanged independent
NES demo. A separate 32-game audit under
`review-20260909/event-public-audit` matches all 964 physics snapshots with
`e0162ed`, including seven one-frame-ahead inputs; every observer roundtrip
matches. Four native snapshot/ABI tests and twelve complete adapter/inference
checks also pass on tf3090 Linux from
`/dev/shm/pp-event-public-native-23411a5` and
`/dev/shm/pp-event-public-rl-1451936`, using CPU only. No running study was migrated and no
complete-reserve shortcut is enabled. V2 corpus collection and selected-reserve
rollout benchmarking must use new experiment identities.

Native `afaf62a` adds atomic `search_set_reserve(pair_index, colors_raw)` with
128 ordered raw-color pairs. It changes only the two private reserve copies;
current/preview colors, clocks, RNG, public observations and output buffers are
unchanged. Python validates shape, integer type and color bounds before the
native call. Terminal rollouts may explicitly select
`reserve_execution="prefilled"`, which requires a genuine V2 observer snapshot
and installs each complete public-posterior hypothesis once. The default remains
`"boundary"`. Prefilled results report zero `boundary_reveal_calls` and null
`reveals`; native-call counts are different work units, not comparable horizons.
Compare natural zero-censored tails and every public decision before claiming a
speedup. Source-bank configuration may explicitly set
`public_observation_schema="causal-settled-pair-v2"`; its default is V1, and
resume rejects a timeline change. The broader running bank and quality labels
remain V1. The isolated Mac library is `build-reserve-v2/libdrmario_pool.dylib`;
do not use the rejected `build-reserve` prototype. The V2 collector/resume checks
and the same-frame earlier-reveal regression pass. The Linux verification and
bounded real-checkpoint comparison are described below.

Source `4d01da3` adds opt-in `trace_decisions` to terminal panels. Each tail
retains a SHA-256 of every complete public input key and chosen action,
including the forced root, plus its decision count. Private reserve and
checkpoint bytes are excluded. Trace time is reported separately; the default
path performs no hashing. Twelve focused Mac and twenty Linux checks pass
with native `afaf62a`. Linux's isolated source/library is
`/dev/shm/pp-reserve-4d01da3`; its source archives are retained under remote
`review-20260909/reserve-v2-benchmark-v1` for recovery.
That directory retains a completed, bounded 40-minute registered comparison: 12 fresh
14-HI V2 source games, four roots from distinct games, full candidate/reserve
and nine-member-pair panels, 256 slots, cache 8,192, one native/Torch worker,
then boundary/prefilled/prefilled/boundary order. All 1,206 tails and complete
targets agreed exactly, including traces of 43,287 public decisions per run,
with no censored tails. Total job seconds were 300.73/273.57/266.71/300.77;
prefilling reduced the mean by 10.2%. The rollout loop fell from 282.5 to
252.3 seconds; native work remained about 188–189 seconds. Scheduler iterations
fell from 508 to 255, and 42,158 boundary reveal calls became 1,206 private
reserve installations. These counts are different work units. `assessment.json`
records every comparison and native-library identity. The result supports
subsequent opt-in V2 research; it does not establish broader speedups, useful
Q ranking or match strength. `supervise.py` advances
only registered child recipes and stops on divergence or its wall cap; it
never terminates another job. Its `progress.json` forwards actual child
progress changes. Main training and broad V1 labels continue, so these are
contended throughput measurements, not an exclusive-GPU allocation comparison.
Do not rerun a completed identity or replace either active study's library.

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

The watch worker imports all feeds before rebuilding standings once per cycle.
Rows are grouped by comparison once; rating mathematics and ordering are
unchanged. A frozen 106,708-game, 305-comparison audit reproduced both rating
views exactly, reducing their combined fit time from 8.0 to 4.4 seconds, in
addition to removing repeated full-field fits after each individual feed.
`runs/review-20260909/dashboard-sync-grouping-assessment.json` retains the
measurement. Run only one sync writer for a target and preserve its existing
SQLite database and feed history when restarting it.

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

The source sidecar accepts `--pace-manifest PATH`, or discovers
`pace_opponents.json` beside its competitive checkpoint. Its
`professor-pills-pace-opponents-v1` manifest names the parent SHA-256,
`own-motor-gravity-v1` context, adapter paths/hashes and all seven pace routes.
Paths are relative to the manifest directory, or absolute for source development.
Super Human and Frame Perfect must route to `null`, retaining the exact parent.
The optional loader shares one parent network, warms every selected residual,
and passes the actual delay/gravity context to reactive and prepared scoring.
With no manifest the prior plain-policy behavior remains. These adapters do not
replace the V3 lower-skill decoder or accept the new history core as their parent.

`tools.package_human_backend --pace-manifest PATH` copies the verified residuals
into portable relative paths and checks the frozen process's loaded portfolio.
Package verification rejects stale or mismatched selections. The four portfolio
and ten backend/context/preparation checks pass, including exact arena scores,
all seven controller witnesses, next-gravity preparation and relocation. This
implements the packaging path; it is not a newly built or validated distributable.
The browser task owns its separate ONNX portfolio. New browser source packages
should use the shared context fields instead of reinserting them during packaging.

`runs/review-20260909/live-portfolio-audit/` retains real-weight source-process
checks on 12 public native roots from four natural 14-HI game prefixes. All
84 portfolio requests and 12 history-core Sloth requests matched CPU arena
choices and controller witnesses exactly. The core's maximum measured JSONL
round trip was 37.2 ms. Another 36 preparation requests matched all 648
conditional branches exactly. Interleaving CPU reference preparation inflated
GPU processing to 80–187 ms; child profiling located that time in inference,
not serialization or geometry. The separate decide-to-prepare flow removes
those interleaved CPU model calls: median decisions were 15–17 ms and median
preparation 32–36 ms, with maxima 24.6/43.5 ms across the three fastest paces.
Retain both workloads in the reports. This small shared-host source check
supports further integration; it does not certify app render/latch timing,
high-stack tails, final packages or a new search compute budget. The Sloth core
still needs per-pace product routing and representative live-game verification.

For source development, query `capabilities.geometry_preparation` before using
`prepare_geometry`. Supply the current public state, committed result, next
pace and `execution_delay_frames`. The reply carries `geometry_token` and two
candidate counts; a null token means preparation was unavailable. Pass that
token with the next ordinary `decide` and its complete actual public state.
Tokens are process-local, single-use and bounded to two outstanding entries.
`result.geometry_preparation.status` distinguishes a hit from execution-profile,
observation-contract, own-state, microstate and unavailable misses. Every miss
uses normal fresh planning with the same requested motor limits. Neither
native buffers nor speculative history are serialized. Existing app/browser
schedulers continue using their prior protocol until explicitly integrated.

`runs/review-20260909/live-geometry-audit.py` retains the bounded real 100M-core
Metal source-process check, with order-balanced fresh/cached requests on twelve
14-HI controller prefixes. All 61 playable decisions matched exactly in scores,
full candidate inventories and controller scripts; three further Sloth states
were unreachable in both modes. Geometry hit 52 times. The nine own-state
misses all coincide with observed incoming volleys. Independent native replay
of all 64 boundaries confirms this in `live-geometry-audit/mismatch-audit.json`.
Geometry preparation medians were 1.5–6.8 ms; hit decision medians remained
21–27 ms. Paired savings were about 0.3/2.7/-0.2/1.5 ms for Sloth/Normal/Top
Humans/Frame Perfect. This small shared-host check does not justify changing
compute charges or certify app scheduling, high-stack tails or strength.

For offline joint-search diagnostics, `--opponent-mode mixed` selects the
complete simultaneous-action matrix in either recursive or frontier-batched
execution. Install the optional `search` extra for the default HiGHS dual
simplex solver. `--matrix-iterations`, `--matrix-time-limit-seconds` and
`--matrix-gap-tolerance` default to 2,048, 0.25 seconds and 0.02 utility units.
`--matrix-solver mirror_prox` retains the earlier diagnostic, whose
`--matrix-temperature` defaults to 0.001. These numerical controls are
not calibrated candidate uncertainty. Mixed mode ignores own/opponent/chance
beam truncation and rejects a partial root inventory. Watch total nodes and
solver milliseconds: the complete matrix can be much larger than a beam.
An unconverged matrix, including an internal one, makes the entire target
unusable. A simultaneous-root consumer must sample `policy_target`; choosing
the reported representative action loses the mixture's protection. The
`SearchResult.select_action(rng)` decoder samples explicitly and rejects
incomplete or unconverged results. The
registered `search-frontier-benchmark` accepts these settings plus an explicit
`native_library` and records both strategies, convergence and causal schema.
It remains a mechanics/throughput probe with an uncalibrated value link.
Its default comparison remains `strict-vector-v1`. New mixed-game probes can
explicitly select `comparison_contract: mixed-game-certificate-v1`, which
checks complete matrix agreement and cross-matrix best-response certificates
while retaining the stricter probability-vector comparison separately. This
handles nonunique equilibria without loosening the neural payoff tolerance.
`alternate_order: true` reverses recursive/queued order on alternating roots;
solver cold-start time is recorded outside those warm comparisons.
The completed `runs/review-20260909/mixed-native-result-v4.json` probe uses
four additional native V2 roots and source `5b0420a`, with matching complete
matrix certificates in both drivers. The `v1`/`v2` mirror-prox failures and
`v3` strict-vector failure remain alongside it. The full matrices, strategies,
cross-matrix gaps, neural rows/calls and timing are retained; none are quality
training labels or controller tournament outcomes.

`trainer-neural-preparation-audit` takes `neural_preparation_config` with a
public controller-core checkpoint, explicit closed `replay_shards`, fresh
output, device, state count and repetition count. It compares fresh forwards
with prepared own/both bottles across all nine conditional previews, preserving
the complete candidate frontier. Preparation and remaining inference times are
separate; include both when reporting total cost. It disables trunk conditioning
only in a temporary diagnostic model and reports policy/value drift against the
unchanged parent. The parent checkpoint is never rewritten or installed.
Within-model reuse parity is not behavior preservation of that architectural
change. Any useful unconditioned student still requires fitting and whole-game
strength/latency evaluation before adoption.
The fixed `runs/review-20260909/neural-preparation-v1` audit is complete
under source `be89516`; former supervisor 17035 is done. Its result/config
and independent `neural-preparation-assessment-v1.json` retain 14 distinct
recorded reset seeds, two per 14-HI pace, 126 conditional preview comparisons
and five timing repetitions per mode. `assess_neural_preparation_v1.py`
groups repetitions at each root and charges preparation plus inference.
All within-variant choices pass; maximum probability/value errors are
9.54e-7/2.39e-7. Nine queries cost 17.90 ms fresh versus a median charged
14.23 ms with preparation. One query with only our bottle cached costs
8.38 versus 9.93 ms fresh; preparing both bottles and then making only one
query costs 12.45 ms in total, despite a shorter 5.42 ms remaining decision.
The untrained architecture changes 12/14 parent choices. Retain the current
models; use this result to inform the planned fast-student training and
subsequent full-game evaluation. No preparation charge, product route or
running training source changed, and this completed audit needs no repeat.

`trainer-adaptive-search-audit` compares complete queued matrices with bounded
root allocation, and complete action vectors for unilateral P1/P2 roots.
Supply `adaptive_search_config` with a frozen checkpoint,
native library, causal state bank, output identity and optional allocation
batch sizes. The diagnostic rotates variant order across roots, records cold
solver initialization separately, and retains every completed variant even if
a later reference or bound check fails. Defaults use depth one, neural batches
of 32, allocation batches of 16/32, a 0.02 response-gap target and a separately
declared 1e-5 numerical evaluator tolerance. It compares complete legal
inventories, checks each retained interval against the full matrix, and measures
the returned strategies' actual full-matrix response gap. Unknown entries stay
explicit and cannot become quality labels. A completed diagnostic is not an
adoption decision: inspect certification, real work savings and wall time for
every root, including budget failures. Launch Metal diagnostics outside the
restricted sandbox, which cannot expose the device, with the virtualenv's bin
directory on PATH for the registered recipe's Python child.
Unilateral reports identify the acting side and `full_vector_regret`; they do
not invent an opponent action or call a root action a joint transition.
`max_root_actions` defaults to 512. Both unilateral allocation modes share
`max_joint_actions` over actual simultaneous descendant transitions.
`policy_temperature` can be set explicitly to zero for pure-choice reference
comparisons; retain the setting in the fixed config. Complete reference values
come from the full-precision W/D/L backups rather than the display utility array.
`allocation_modes` defaults to `["root"]`; use `["root", "nested"]` for a
matched-depth comparison of both mechanisms. Nested mode reports all interior
matrix certificates and counts `max_joint_actions` across actual simultaneous
transitions at every depth. Partial child values retain their real intervals,
with null W/D/L at decision backups. The default root-only contract and frozen
studies remain unchanged. At least four pair-event levels can be needed to
reach a second simultaneous decision after both players reveal a new preview;
depth two does not necessarily exercise nested allocation on native games.
`belief_cache_size` can retain more public posteriors for a larger active
frontier; eviction still fails rather than reconstructing a missing history.
Use the same capacity for all variants and record it in the fixed config.
The completed `runs/review-20260909/adaptive-native-result-v1.json` audit
uses source `142ee1a`, native `afaf62a`, and source-bank SHA-256
`3fbece83ff356bea005501cd9388469f26752002ff7c6b7b14a249149e4f2ff9`.
The four source indices (19, 35, 59, 71) were fixed before evaluation. Both
allocation sizes certified all four roots; all intervals contained the
independent complete matrices. Complete/16/32 allocation took 10.48/4.07/3.57
seconds, with 3,724/1,152/1,184 evaluated joint actions. The largest actual
response gap was 0.016375, below the declared 0.02 target. Keep the raw
matrices, intervals, order and timing; these are development mechanics results
on a shared machine, not an independent strength confirmation or adoption.

The depth-four `nested-native-result-v1.json` audit is complete, from frozen
`a924425` and native `afaf62a`, using one actual V2 root (source index 9,
full 30-by-30 root inventories). `nested-native-config-v1.json` retains the
same five-million-node budget and two-million-entry belief cache for all
variants; adaptive modes share a two-million actual joint-transition cap.
Complete/root-16/nested-16 take 317.48/289.64/135.64 seconds and
330,044/116,992/114,991 nodes. Neural calls are 2,210/12,058/1,434.
Both adaptive modes evaluate 464 of 900 root pairs. Their intervals contain
the complete matrix with zero violation; actual response gaps are
0.01939597/0.01939522 within retained bounds 0.01941518/0.01941461 and the
0.02 target. Nested mode records 60 matrix certificates and 57,120 actual
simultaneous transitions across all depths, without budget exhaustion.
This is one development position with an uncalibrated critic on a shared
machine: the 2.34-fold wall saving does not establish general latency or
strength. Preserve raw matrices, interval certificates and timing. Former
supervisor 79970 is complete; do not rerun this fixed probe. Its dashboard
card reports the completed variants separately.

Unilateral-root comparisons from frozen `3bf1fb3` are complete. Retain
`runs/review-20260909/unilateral-native-{config,result,assessment}-v1.json`
and `assess_unilateral_native_v1.py`: source indices 0/3/30 cover pressured
P2, pressured P1 and quiet P1 at 14 HI, depth one plus up to two tactical
events, including separate unextended controls. All six independent precise
vector/bound checks pass; no root action is pruned. Complete/root/nested costs
are 3.07/3.26/3.19, 3.40/2.17/2.07 and 0.126/0.143/0.141 seconds.
The separate `unilateral-mixed-native-*` depth-four files retain three runs on
index 30 and two passing checks (5.91/5.60/5.51 seconds). Despite its original
filename/aim, that tree never reaches a simultaneous descendant. Record zero
actual joint transitions; analytic mixed-descendant tests remain separate.
`assess_unilateral_native_v1.py --depth-four` checks that saved study.
Both supervisors (99106 and 161) are done; neither study should be repeated.
These are finite-critic mechanics and shared-device costs, not quality labels,
general speedups or strength evidence. The dashboard identifies unilateral
decisions and full-reference checks without fabricating an opponent root.

The 300M milestone completed all 16,384 natural games at September 10
13:10:09 UTC in
`runs/review-20260909/controller-core-300m-mac`, launched by its saved
`launch_core_300m_arena.py` through the registered planning-arena recipe.
It uses the original frozen `controller-arena-0c76c0e-source` and native19f
libraries, four-frame charge, Metal/one Torch thread/three planner workers,
and a fixed 16,384 games against the parent and corrected-credit adapter.
The checkpoint actually contains 301,781,835 frames and 2,988,847 decisions
(update 241). Its 768 reserved seeds are explicitly excluded by the training
configuration and reused from the 100M study for a learning-curve comparison;
this is not independent confirmation. Keep all seven 14-HI paces separate,
with Normal/Top Humans 20 HI reported separately. The complete
`core-300m-assessment.py/.json` audit verifies every side/seed pair, schedule,
score and execution count, with no censoring. At 14 HI, the 300M core scores
59.67% at Super Human and 61.52% at Frame Perfect against the parent. Their
simultaneous primary-family 95% intervals are 54.31–65.02% and 56.28–66.77%.
It also beats the parent at Sloth, Fast and Top Humans. Against corrected E1,
Sloth scores 59.23% (family 55.37–63.08%); Relaxed/Normal/Fast/Top Humans score
50.68/49.32/50.29/51.07%, all inconclusive. E1 falls back to the parent at
Super Human/Frame Perfect, so those duplicate edges are not independent
replications. Separate 20-HI scores against E1 are Normal 56.54% (individual
51.17–61.91%) and Top Humans 48.05% (42.58–53.71%). These findings support a
fastest-mode candidate; direct slow-product comparisons and candidate
confirmation remain. Preserve the completed `core-300m-mac` dashboard feed;
do not restart its former supervisor 92309.

`controller-core-300m-confirmation-mac` is the fixed follow-up, through the
same registered recipe/source/native libraries and unchanged compute charge,
supervisor 55538. Its 8,192 games cover Sloth versus the original final adapter,
Relaxed versus the original 50M adapter, and Super Human/Frame Perfect versus
the parent, 2,048 games per condition. The 1,024 additional training-excluded
reset seeds were never used for this 300M checkpoint or either motor prediction
confirmation. Some were evaluated for other frozen cores: this is candidate
confirmation, not globally untouched seeds or independent training-seed
replication. All 8,192 games completed at September 10 15:18:09 UTC with zero
censoring. `assess_core_300m_confirmation.py` and
`core-300m-confirmation-assessment.json` verify checkpoint identity, schedule,
every seed/side pair, scores and execution totals. Twenty thousand shared
whole-seed bootstrap resamples yield simultaneous 95% intervals over the
four-condition family:

| 14-HI pace | Existing opponent | 300M score | Simultaneous 95% interval |
| --- | --- | ---: | ---: |
| Sloth | Original final adapter | 57.54% | 55.17–59.92% |
| Relaxed | Original 50M adapter | 46.00% | 42.38–49.61% |
| Super Human | Public 10M parent | 61.08% | 57.86–64.31% |
| Frame Perfect | Public 10M parent | 61.72% | 58.46–64.98% |

Each condition contains 2,048 games. Sloth has 1,177 natural draws, Relaxed
12, and the fastest two zero; draws score one half. Retain the original
Relaxed route. The 300M core is the selected candidate for actual-app
Sloth/Super Human/Frame Perfect checks before adoption; other pace routes
and the ongoing outcome optimizer remain unchanged. Config, study, raw play
and launch scripts remain under the local review root. Preserve the completed
`core-300m-confirmation-mac` feed and do not restart former supervisor 55538.
No product route changed or package was built in that evaluation pass.

### Mixed-core live integration, September 10

The v2 live portfolio accepts complete public-history cores alongside existing
parent/residual routes. Selection precedes observation encoding, geometry
preparation and warmup. Public-core routes reject legacy neural anticipation
because they require fresh history; the app respects the sidecar's advertised
pace scope. V1 manifests and lower-skill regret decoding retain their behavior.
Packaging validates and relocates both core and residual artifacts.

The explicit candidate manifest is
`drmc_rl/human/portfolios/tester-300m.json`: 300M at Sloth/Super Human/Frame
Perfect, original 50M at Relaxed, E1 at Normal/Fast/Top Humans. It is **opt-in**;
defaults, browser assets and existing packages remain unchanged. The source app
and sibling `train-versus` launcher accept `--pace-manifest`. Desktop rendered
acceptance is recorded below; this selection file is not a distributable.

Both acceptance tools now supply the causal public history already collected
by the actual app. Six real-time 14-HI checks used actual 300M weights, two
sidecars and the live scheduler: native and ROM runs at each proposed pace.
All finished their existing acceptance checks. Across the six runs there were
378 executed/373 completed placements, zero initial-state or completed-placement
mismatches, 16 missed requests and two guarded replans, both in ROM runs.
Native runs had no corrections. Sloth included physically unreachable late
placements. Maximum round trip was 72.891 ms under concurrent Mac arena load;
the adaptive compute allowance rose from four to five frames where required.
These short operational checks do not replace the 8,192-game strength study;
the two replan causes remain unassigned.

Raw logs, native preview frames and `assessment.json` are retained under
`runs/review-20260909/live-app-300m-v1`. Source bases were RL `e1aba8e` and
app `d7c297a`, plus the mixed-portfolio integration changes. The GUI loaded the
real mixed portfolio into two sidecars but remained in setup. Eight synthetic
presses measured 0.053–5.660 ms at dispatch, seven below 0.2 ms. Its
`gui-frame-perfect.csv` and `gui-menu-timing-report.json` are **setup-only**
diagnostics with focus/occlusion transitions and a bounded trace tail, not
gameplay or physical-input latency evidence. The native loop now emits its
actual input sample, frame start/completion and audio-ready events, matching
the ROM trace contract without inventing an emulated CPU cycle. A separate
12-second setup run verified exactly 722 input samples, completed frames and
audio-ready events, with normal timed shutdown; its trace is retained too.

The app now has an explicit bounded `--trainer-autoplay` acceptance mode. It
waits for both workers, uses ordinary setup controller input, isolates physical
controller sources and leaves saved settings unchanged. It reports the active
play interval separately from loading/results. Normal interactive play retains
its focus and input behavior.

Ten valid 14-HI GUI runs completed 17,984 active frames and 417 executed/412
completed placements, with zero initial-state or completed-placement mismatches.
There were 40 missed requests and five guarded ROM Super Human replans. The
short Sloth top-outs remain evidence; they were not retried until a win. One
earlier native attempt changed to level 15 through physical keyboard input and
was excluded; that finding prompted the explicit test-mode isolation. A separate
short natural Sloth loss and the earlier setup-only attempt are retained too.
`rendered-300m-assessment.json` links the accepted `rendered-300m-v3/v4` traces.
Several fast runs submitted frames with zero actual presentation timestamps;
those intervals are not display-latency evidence. In covered intervals the
per-run 95th percentile from frame completion to first display was 21.3–24.8 ms.
Four short explicitly visible follow-ups completed another 140 placements,
with zero corrections or state/placement mismatches and 4,466 actual
presentations. They cover native/ROM Super Human, native Frame Perfect, and
the revised 45-frame native Sloth preset. Their active-frame interval-error
95th percentiles were 0.174–0.389 ms; frame-complete-to-first-display 95th
percentiles were 24.3–27.3 ms. Missing timestamps before/during visibility
changes remain in `rendered-300m-visible-v1`, rather than being called displayed
frames. These runs
remain operational evidence, not physical input latency or another strength
tournament. Guarded examples include missed horizontal movement and differing
gravity microstates; the underlying ROM cause is not established.

After the fixed old-profile checks, the user requested reaction floors of
45 frames for Sloth (~749 ms) and 30 for Relaxed (~499 ms). App scheduling,
exact reachability and package capability validation now agree on those values.
Edge/motion/overlap limits and all faster presets are unchanged. Nine focused
planner/backend checks passed, including actual delayed decisions at both new
profiles; the app's delayed-execution regression covers both too. Existing
frozen training/arena sources retain Sloth 60 and Relaxed 36. Do not relabel
their strength results as evidence collected at 45/30. Continuous motor context
is supplied to the selected models at the actual new values; outcome confirmation
at the revised slow presets remains distinct work.

The selected tester configuration remains 300M at Sloth/Super Human/Frame
Perfect, original 50M at Relaxed and E1 at Normal/Fast/Top Humans. Use its explicit
manifest for source play; browser assets and distributables still require their
own integration and validation. This does not open certified product gates.
Fourteen focused Python portfolio/public-context checks and the Rust pace-scope
test passed. Release app and acceptance tools built. Portable v2 tests verify
complete choices/scripts at all seven paces and reject mislabeled cores.

### Persistent construction diagnostics

`trainer-target-construction-fit` takes `target_construction_config` with a
completed recurrent proposal `study` and fresh `output`. It reuses the verified
full-precision features and original whole-session split; defaults are eight
fixed epochs, 32 windows per batch and one CPU thread. The prior history-reset
head initializes a route model, while the original root-goal proposer remains
frozen. Training rotates through actual colored payoff cells as explicit
requests. Each entire construction has the same requested cell and a six-turn
cap, not its observed future duration. Report window/action presentations
separately from zero outcome-training frames. Development imitation never
selects a checkpoint or establishes goal completion.

`trainer-target-construction-evaluation` takes `target_evaluation_config` with
the completed fit's `progress.json` under `trained`, fresh `output`, and a fixed
side-balanced `arena`. It runs both actual goal controls and unchanged competitive
controls on every declared seed/side pair, alternating batch order. Both use the
same frozen root proposer, immutable targets, exact motor frontiers and verified
event-driven termination. Retain complete controller journals, original-target
payoffs, true completion lengths, garbage/terminal interruptions and natural
outcomes. The explicit `allow_unadmitted_controller_experiment` flag cannot be
used with PPO learning records, asynchronous planning or another observer.
These are unadmitted diagnostic games, never installed product behavior or a
noninferiority certificate. Report individual whole-seed contrasts by pace;
later independent quality/strength and blind preferences remain necessary.

`target-construction-study-v1-control` completed both registered jobs from
frozen source `2da2fc3` with original controller libraries `19f292c`. Its former
supervisor 92592 has exited. The fixed fit used 304,400 window presentations and
1,069,520 action presentations in 133 seconds, adding no outcome training.
`target-construction-evaluation-v1` completed 512 natural games, 2,642,914 console
frames and zero censoring in 493 seconds. `assess_target_construction_v1.py` and
`target-construction-assessment-v1.json` retain an independent audit of every
journal, original target, observed effect, actor/library identity and complete
side-swapped allocation, with separate 14/20-HI whole-seed families.
The 256 controlled games changed 2,649 of 3,187 decisions and reached none of
802 original targets; the 256 unchanged games reached 354/5,960, including 254
after multiple placements. No target was revised. The severe strength loss
rejects this actor replacement; do not promote or restart the completed study.
Better requested-goal imitation does not supply grounded autonomous routes,
calibrated admission or preference evidence. Those parts of Stage 4 remain open.

`trainer-spatial-expressive-study` takes `spatial_expressive_config` with a
verified sequence `source`, frozen public competitive `checkpoint` and fresh
`output`. It reuses exact own-bottle mean/max features, deduplicated solely by
actual board/pill/preview inputs. Defaults use feature batches of 32, eight
fixed epochs, width 128, 32 windows per batch and learning rate 0.0003. Choose
`feature_device` separately from the small heads' training `device`; one CPU
thread and Metal feature extraction are supported. `prepared.npz` retains
full-precision features and exact spatial target distributions. Two independently
optimized arms start from the same seed and consume the same session-weighted
examples in the same order. The persistent arm's action decoder already sees
both fixed root memory and actual current-state features. Its spatial and
duration plan is predicted from the root under the original `fixed-root-v1`
contract; the stateless control refreshes that plan from each actual current
state. The explicit `plan_update_schema: recurrent-public-v1` instead trains a
GRU on the causal sequence of actual current inputs. Both independently trained
arms have the same GRU capacity, elapsed-placement input and root-selected goal;
the control resets only hidden history. Spatial and remaining-duration plans
are updated at every actual decision. Duration class one is allowed after the
root; the initial 2–6-placement execution budget cannot grow. Repeated runtime
ranking revises the current slot without appending history. Only a unique
completed-placement event commits memory. Existing termination on payoff,
garbage, mismatches, terminal state and exhausted budget still applies.
Both action decoders consume
predicted plans, never payoff labels or true future duration. The fixed-final
checkpoints and paired whole-session descriptive comparisons are retained.
Spatial, duration and intent heads are also compared with smoothed
goal/pill/preview frequency priors fitted only on the training sessions; a
gain over random initialization is not sufficient prediction evidence.
Do not interpret recorded-prefix predicted-goal scores as autonomous play;
plan termination and altered intermediate states require actual execution.
Fresh replay confirmation is separate from these previously used development
sessions. Ordinary competitive weights and product routing remain unchanged.

Set `prepared_from` to a completed study's `progress.json` to reuse its exact
full-precision features and targets without copying or extracting them again.
The source bank, competitive checkpoint and prepared file hashes must match;
the output records the original prepared path and identity. This changes no
examples or split. Each new checkpoint declares its plan-update schema, and
confirmation rejects a schema mismatch before loading its weights.

`trainer-spatial-expressive-confirmation` takes `spatial_confirmation_config`
with the completed fixed study's `progress.json` under `study`, a fresh verified
sequence `source`, and new `output`. It verifies the original data, competitive
checkpoint and both proposal checkpoint hashes, and rejects any session-ID or
blob-content overlap with development. Supply `exclude_sources` for every
earlier confirmation bank that preceded selection of the new architecture;
these session IDs and content hashes are also rejected before output creation.
Shared feature extraction has the same
device/batch settings as fitting. Both heads and their training-only priors are
loaded unchanged; no optimizer or checkpoint selection is available. It reports
paired whole-session intervals and distinct evaluated-window/action counts,
with zero optimizer updates and no new outcome frames. These recorded-prefix
measurements still do not establish autonomous construction completion or
competitive noninferiority.

The completed local spatial study is `runs/review-20260909/spatial-expressive-v3`
(source `fed2bc3`, former supervisor 27144). Attempts v1/v2 failed before any
feature extraction or optimization on path normalization and explicit
`zero_v1_vs` support, respectively; retain their failed records. The completed
study used 126,505 unique public feature inputs and all 48,864 development
windows, with 38,050 training windows from 399 sessions. Each head received
304,400 window and 1,069,520 action presentations over eight fixed epochs.
Total feature preparation and fitting took 541.5 seconds. Persistent and
stateless final checkpoint SHA-256 values are
`1b2586e36d5fd84c58ec1e55874841fdc26b29fb61cb0bfd49d7eb93e32056ee`
and `682249da62d2e0cbce15054a6c8455064e94e6b0d84e3f7232bffd679e3312d3`.

`expressive-confirmation-archive-v1.zip` mirrors 256 reserved replay blobs
from mombox, selected with seed 93517 after excluding all 512 development
sessions and content hashes. Verification ran on the Mac through the registered
sequence recipe and a local read-only catalogue/blob provider, avoiding another
heavy mombox worker. `expressive-confirmation-source-v1` completed 394,902
verified placements and 24,392 constructions/85,199 action examples in 43.3
seconds. Its source SHA-256 is
`2c2049a40bb44ada884b7fdcf68b9713720273f86d41f85d6969404b66c3140d`.
`spatial-confirmation-v1` (source `ff682b1`, former supervisor 29611) completed
in 58.5 seconds at September 10 11:19:27 UTC. All 256 sessions were reserved;
255 yielded scored constructions. Its individual 20,000-resample paired
session intervals confirm worse persistent action NLL (+0.0936, 95%
0.0861–0.1012) and earlier-setup NLL (+0.0634, 0.0514–0.0756) versus the trained
stateless control, despite both spatial predictors beating training-only
frequency priors. Preserve the fixed models, prepared features and complete
per-session measurements; do not retrain on these confirmation sessions or
promote the persistent mechanism. No outcome frames were trained or arena
entrant added. Subsequent actual planning must address changing inputs and
quality admission before strength or preference claims.

`spatial-replanning-v1` (source `991f030`, former supervisor 66581) completed
at September 10 14:08:26 UTC. Both arms used eight fixed epochs, 304,400 window
and 1,069,520 action presentations; cached-feature preparation and fitting took
272.8 seconds, with zero outcome-training frames. Persistent final SHA-256 is
`3c54171d20d7053a1d7b2bc9a5c21fb102fb0181ede48f8928495ef5e97ee974`;
history-reset final is
`789afdad702feaaab9747c96d0a0fa132cf3c224e309d1c1c35ea08001137bb5`.
The original full-precision prepared features are referenced by hash and path;
they were not copied or extracted again.

`expressive-confirmation-archive-v2.zip` reserves 256 new blobs with selection
seed 93617, excluding all 768 earlier session IDs and content hashes. The
source archive on mombox was read only; this bounded copy lives in authorized
overflow storage and locally. Registered source extraction
`expressive-confirmation-source-v2` verified 362,458 placements and retained
24,212 constructions/84,977 actions in 40.0 seconds. All 256 sessions have
scored windows. Source SHA-256 is
`1b0867ae9901ae996fac2527a0593d23ae2be843e790c506ba2f645b1d29a871`.

`spatial-replanning-confirmation-v1` (same frozen source, former supervisor
68003) completed at 14:10:03 UTC in 56.6 seconds, with zero optimizer updates.
Persistent-minus-control action NLL is +0.0071 (individual paired 95%
−0.0014–0.0167); predicted-goal and early-setup differences are inconclusive.
Payoff NLL worsens by +0.0153 (0.0019–0.0298); spatial-anchor hits are
16.05% versus 16.62%, difference −0.57 percentage points (−0.98–−0.17).
Both arms beat all three training-only prediction priors. Retain the complete
per-session reports and fixed models. Neither recurrent strength nor useful
autonomous persistence is established. Future confirmation excludes all
1,024 identities across the original bank and both confirmation banks. The
dashboard contains separate replanning comparison and confirmation cards;
no arena entrant or product route changed.

`trainer-spatial-execution-audit` takes `spatial_execution_config` with the
completed recurrent `study`, a new `output`, and an `arena` using explicit
side-balanced seeds and frozen actor checkpoints. It requires synchronous
event batches and writes progress during each batch. Both fixed proposal heads
observe the same actual 300M-versus-parent moves; they do not control a player.
Full move/controller journals, original and revised targets, complete-frontier
rankings, verified transition counts and abandonment reasons are retained per
game. Conditional payoffs require observed lock/effects/bottle agreement;
garbage and unfinished terminal transitions do not become successful plans.
Report whole-reset-seed descriptive intervals separately by level and pace.
This does not authorize style selection, train a model or establish strength.

The fixed `spatial-execution-v1` allocation completed 256 natural games and
2,465,306 console frames with zero censoring. `assess_spatial_execution_v1.py`
checks all fixed seed/side pairs, model/library identities, actual move journals
and reconstructed payoff counts; `spatial-execution-assessment-v1.json` retains
the first-hit placement and per-condition results. No model was trained.

`trainer-adaptive-search-audit` also accepts `tactical_extension_events` (0–8,
default 0) and `compare_unextended` (default false). All complete/adaptive arms
use the same predicate and path allowance. The optional unextended arm is a
separate shallower comparison, never the reference matrix for extended bounds.
Keep source, predicate, native identity, full matrices, charged work and
independent interval checks. An improved response bound or changed policy is
not evidence that the uncalibrated critic became accurate.
`tactical-native-v1` completed both fixed development positions from source
`5be4aca` with native `afaf62a`. Retain `tactical-native-result-v1.json`, its
configuration and roots, plus `assess_tactical_native_v1.py` and
`tactical-native-assessment-v1.json`. The latter independently checks every
adaptive interval and response bound against the matching extended complete
matrix, and confirms the quiet root stays unchanged. Former supervisor 68101
and worker 68108 are done; do not restart this fixed mechanics study.

`trainer-commentary-alignment` takes `commentary_evidence`,
`commentary_reconstruction` (accepted video-game JSON, optionally gzip), and
`commentary_alignment_output`. It retains unresolved video identities, matches
all candidate feeds within the passage interval plus eight seconds, and links
up to six preceding placements for setup review. Starting from virus-only
initial bottles, it checks each recorded placement and all resulting colored
tiles while preserving internally reconstructed bonds. A sequence gap,
unknown timestamp, invalid lock or mismatched board stops the prefix; it never
resumes from an unknown-bond checkpoint. This is internal physics consistency,
not video verification, preference or a VS-attack label. The September 10
`commentary-alignment-v2/alignment.json` review under `runs/review-20260909`
matches three of nine supplied passages with eight consistent candidates;
two passages retain unresolved video IDs. All candidates remain ineligible
for training until video, commentator referent and event labels are checked.

A separate keyframe review, `commentary-fat-log-video-v1.json`, now records
one verified broadcast example from the April 2026 Gold Speed Monthly,
video `mzox-OAETaU`, at 01:49:44. The right player is atalito, level 16 HI;
the commentator-referenced Fat Log is reconstructed placement 60 in
`apr-2026|4|15|1|atalito|16`: simultaneous blue and red horizontal rows,
zero-based rows 11–12 and columns 4–7, eight cleared cells and three viruses.
Observed keyframes match placement 58's afterstate, placement 59's preceding
vertical clear and falling half, the setup immediately before 60, its payoff,
and placement 61's afterstate. The artifact retains raw bonded states for
placements 55–61 and the exact clear cells. Earlier context and the whole
92-placement consistent prefix are not thereby exhaustively video-verified.
This is one concrete named example for event-driven setup/payoff modeling,
not a universal definition of Fat Log, an outcome target, or a generalized
preference label. Other passages and longer construction windows still need
review; this observation has not been admitted into training.

`trainer-expressive-sequences` takes `expressive_sequences_config`, a JSON with
`db`, `fcr_root`, and a fresh `output`. Defaults are 512 randomly selected
replay sessions, at most 96 windows per session, and seed 20260910. The DB and
blob archive are read-only. `sequences.npz` retains original session/blob
identities, game/player/frame indices, raw boards/pills/previews, canonical
macro actions, levels/speeds, geometric goals and contiguous window offsets.
Source progress counts verified placements and construction windows; none is
reported as new outcome training. Missing blobs and all sequence exclusions
are recorded. The output does not contain reconstructed two-player state or
certified motor traces. Never feed it to outcome/Q training as such.
For fresh confirmation, `exclude_sources` lists earlier verified sequence
banks. Selection excludes both their session IDs and blob hashes, so replay
aliases cannot cross the boundary. Exclusion source hashes and counts remain
in the new bank metadata. Exclude all earlier development sessions, including
the previously inspected validation set, before sampling confirmation.

`trainer-expressive-proposer` takes `expressive_proposer_config` with `source`
pointing at that NPZ and a fresh `output`. Optional settings include `seed`,
`split_seed`, `epochs` (default eight), `batch_windows` (32), `width` (64),
`learning_rate` (0.0003), `device` and `threads` (one). Entire replay sessions
are split before gradients; descriptive validation never selects epochs or
weights. Progress separately counts window and action presentations, with zero
new console-frame training. `proposer-final.pt` is an auxiliary diagnostic,
with quality admission explicitly unavailable. It is not installed in the
backend or admitted to a strength tournament on imitation scores.

`trainer-expressive-study` takes `expressive_study_config`, a JSON containing
its own fresh `output` and both child config paths under their names above.
It invokes the two registered recipes sequentially, stops on failure, and
retains separate extraction/fitting logs. Run it with the existing mombox
evaluation Python, one CPU thread and low scheduling priority while the main
GPU and fixed arenas continue. Do not alter their source snapshots.

The first finite study is complete on mombox: `expressive-study-v1`, frozen source
`expressive-2d7dedc-source`, former supervisor 1475768; configs live under the existing
remote review root's `configs/`. The supervisor PID/log are siblings of the
study output, not inside the fresh directory it creates. Extraction completed
at September 10 08:09:30 UTC in 296 seconds: 512 sessions, 781,583 verified
placements, 48,864 selected constructions and 171,605 stored placement rows.
The corpus SHA256 is
`ffc9c5a477f939aa1dccbc8be1732a58ec69e52a23df1dbfe4c4192df05764a6`.
Goal counts are horizontal 13,971, crossing 3,026, large clear 8,683 and
cascade 23,184. Original levels/speeds remain attached. The source is complete;
do not repeat it. The registered fitter then started on that NPZ, eight fixed
epochs, width 64, 32 windows per batch, one CPU thread, learning rate 0.0003,
model seed 92713 and session split seed 81029. Its actual optimizer counters
are authoritative; extracted placements are not training updates.
The split contains 38,050 training windows from 399 sessions and 10,814
validation windows from 113 other sessions. A separate native `afaf62a`
afterstate check sampled 256 distinct sessions, alternating setup and payoff
positions. Every resulting bottle, clear-wave count and virus-clear count
matched the Python/replay reconstruction exactly. This checks one-placement
physics, not a controller witness or an execution-time allowance. The retained
script/report are `audit_expressive_native.py` and
`expressive-native-assessment.json` under the local review root.

After the fixed final save, `audit_expressive_fit.py` in that directory compares
held-out imitation NLL to training-only categorical priors, with smoothing mass
16 fixed before assessment. Priors condition on goal/current pill/remaining
placements for actions, and root pill/preview for intent. It groups by whole
held-out session, compares root memory with a current-state substitution, and
reports the first action under the model's own selected intent. The memory
check is an input ablation, not a separately trained control. Imitation,
free-intent root choice and actual persistent playing strength are distinct.
The fit completed at 08:23:45 UTC September 10 in 852 seconds: eight fixed
epochs, 304,400 window and 1,069,520 action presentations. Final checkpoint
SHA256 `e1241effa51067d6bd4f28de27f73df952451eaef1551209833c469307a4d325`.
Do not restart this study. The final assessment is retained in
`expressive-fit-assessment.json`: action NLL 4.21252 versus prior 5.83311,
intent NLL 2.74613 versus prior 2.73463, and NLL 4.20028 after replacing root
memory with current features. Shared 20,000 whole-session bootstrap draws
give paired difference intervals respectively [-1.6444,-1.5958],
[0.00271,0.02018] and [0.00906,0.01537]; these are descriptive individual
intervals, not a joint promotion family. Its own selected intent gives 6.12%
first-action agreement; teacher-selected intent gives 7.82% across steps,
both over all 512 action classes rather than a live feasible subset.
Automatic intent and root memory have not demonstrated benefit. Retain the
prototype as a rejected first persistence mechanism, with spatially explicit
goals and common competitive features as the next bounded revision. The
existing core's candidate values/admission, actual persistent games and blind
same-root preferences remain separate required work.

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
