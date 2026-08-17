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

```bash
git submodule update --init --recursive
uv sync --all-extras
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
