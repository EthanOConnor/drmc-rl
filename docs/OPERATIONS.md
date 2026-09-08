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
mapping each feed name to an `SSH-host:/output/path`. It transfers closed
snapshots and sampled replays; the full move archive stays on remote storage.
The viewer exposes progress, recorded matches, playback and scrubbing, relative
Elo and matchup coverage. Ratings use the existing Davidson/Laplace model,
anchor the baseline at zero, count each paired seed as one effective observation,
and separate `rating_group`, level and pace. They are experiment comparisons,
not drmariostats ratings. Inspect the payoff matrix for matchup dependence.

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
