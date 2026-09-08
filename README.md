# drmc-rl

Training, search, evaluation, and live-play software for Dr. Mario agents.

The project now has one governing architecture:

```text
full-pair competitive quality/search core
    -> exact execution planner with selectable operation profiles
    -> quality argmax | human-rate quality argmax | regret/style/cadence decoder
```

The three products are therefore controlled projections of one notion of move
quality rather than unrelated agents:

1. **Unrestricted superhuman:** strongest public-information policy and
   validated joint-event search, using every exact executable placement/timing.
2. **Human-rate superhuman:** the same strategic intelligence, optimized inside
   a named corpus-derived human operation envelope, with no intentional errors.
3. **Human trainer:** the same quality oracle decoded through calibrated
   win-probability regret, explicit style, cadence, form, and plausible motor
   execution.

## Start here

Use Python 3.14.7 and uv 0.12.10 or newer. The locked September 2026 baseline
includes PyTorch 2.14, NumPy 2.5.3 and PyInstaller 6.22.2. Python 3.15 remains a
prerelease and is intentionally outside the supported runtime.

```bash
git submodule update --init --recursive
uv sync --locked --extra rl --extra viz --extra eval --extra corpus
uv run python -m tools.build_drmario_pool
uv run python -m tools.program status
uv run python -m tools.program validate --check-paths
uv run pytest -q
```

Long-running work is launched through the program registry rather than by
selecting an old YAML from the config directory:

```bash
uv run python -m tools.program launch g4-strong-league --dry-run
uv run python -m tools.program launch g4-strong-league
```

Staged work requires `--allow-staged` and open gate evidence. Runtime artifacts
are recorded with immutable provenance:

```bash
uv run python -m tools.program artifact runs/example/checkpoint.pt.gz \
  --config drmc_rl/training/configs/example.yaml \
  --observation-schema drmc-public-pair-state-v2
```

## Core contracts

The local Professor Pills trainer uses the full-corpus V3 epoch-5 model for
adjustable regret and cadence, with the public outcome-trained 10M V5 policy as
the Maximum opponent. This is an experimental local build; the unified
trainer release above remains gated. `tools.human_backend` serves semantic
v1 requests with scheduled execution: `execution_delay_frames` advances neutral
inputs before planning; responses include the start frame, expected microstate,
and replay-verified `controller_states`. The live host uses a fixed reaction
window for each pace (30/20/14/10/8/8/8 frames), checks
the state, and replans on drift. `sample_regret=true` samples V3 regret tails
independently of imitation temperature; median-only selection flattens most of
the rating scale. Capabilities include the checkpoint SHA-256 and rating range.
`strength_control=quality` selects the optional competitive model's best move
without applying V3 regret calibration to its logits. Its auxiliary context is
zero, matching distillation; hidden pending attacks never enter the actor.
The backend selects CUDA, Metal, or CPU according to availability; `--device`
can override it. CPU inference defaults to one thread. Metal candidate arrays
use bounded padded shapes, all warmed before readiness, to avoid compilation
pauses during games.
Maximum play evaluates only the competitive network and cadence model;
coaching also evaluates the human model. Uncomputed human logits and state-win
probabilities are returned as `null` in Maximum play responses.

Named pace (`sloth`, `relaxed`, `normal`, `fast`, `top_humans`, `super_human`,
`frame_perfect`) restricts feasibility before strategic selection. The three
slowest presets use one button at a time; faster modes permit chords. Native
reachability constrains reaction, button-change spacing, actual steering
including DAS, and overlap on every frame. Validated route accelerators and
complete constrained search share the same limits; no unrestricted script
fallback is allowed. `timing.execution_profile` identifies the limits and
`timing.movement` records independent replay validation. These are authored
product presets, not corpus-certified human percentiles. Pace changes actual
playing strength as well as appearance.

`trainer-internal-v20-20260907` preserves the current internal-test baseline:
quite good, and good enough to get testers; Spunky remains an internal tester.
The deployed competitive model SHA-256 is
`805c2b75444baba4325796753564686f1b9c7a5705a7444501b73b61f264c488`;
the V3 human/timing model is
`af8b5eea0a5429b9254318e2b861bd28292a63e6d9e2501a1539eb000af00010`.
These weights remain frozen during expressive-play and slow-pace experiments.
Internal readiness does not open calibrated-strength, human-profile, or
counterfactual-teacher promotion gates.

The slower end is deliberately broad: Sloth, Relaxed, Normal, and Fast use
reaction windows of 60/36/22/12 frames and minimum steering intervals of
24/14/8/5 frames, respectively. Sloth therefore waits about one second and
spaces steering by at least 0.4 seconds. Cartridge gravity continues normally;
these limits can remove placements, never trigger a faster fallback.

The sibling Professor Pills `train-versus --rating 1600 --pace relaxed` launcher
starts this local trainer; `--maximum` selects the competitive ceiling. Default
artifacts live in `runs/human_policy/versus_trainer/`. The V3 conditioning range
is approximately 718–2451 WHR-C; those are requested corpus targets, not measured
achieved ratings. `tools.package_human_backend --competitive-checkpoint ...`
can bundle both models for standalone use.
Use a separate packaging environment so training dependencies do not enter the
download:

```bash
UV_PROJECT_ENVIRONMENT=.venv-package uv sync --locked --no-dev --group package --extra inference
UV_PROJECT_ENVIRONMENT=.venv-package uv run --no-sync python -m tools.package_human_backend \
  --checkpoint runs/human_policy/versus_trainer/v3-full-epoch05.pt.gz \
  --competitive-checkpoint runs/human_policy/versus_trainer/competitive_policy.pt.gz
```

The `inference` and `rl` extras are mutually exclusive: portable inference uses
locked CPU wheels on Linux/Windows and MPS-capable wheels on macOS; `rl` retains
CUDA training wheels on Linux. The packager starts the frozen process, checks
its model/execution capabilities, and records runtime versions, native/source
revisions and library/checkpoint hashes. The release workflow requires a
published model release containing both policies; it no longer guesses a
nonexistent V3 release or silently omits Maximum's checkpoint.

GitHub Actions checks out the native engine at the submodule's exact commit.
Because that repository is private, native CI and backend releases require the
`DRMC_NATIVE_DEPLOY_KEY` Actions secret: an SSH deploy key with read-only access
to `EthanOConnor/drmario-native`. Pure-Python CI does not need this credential.

[PyTorch 2.14](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) brings MPS
reduction/copy optimizations, less allocator fragmentation, and Metal correctness
fixes relevant to local inference. Its clamp/min/max boundary subgradients also
changed: future training resumes must record the new runtime and be evaluated as
a new continuation, not claimed to reproduce a 2.13 trajectory. Existing model
weights and inference information boundaries are unchanged. CUDA graph replay
hooks and reduced cuDNN cold-start cost are useful profiling opportunities on
tf3090; compiled or captured inference must prove a latency benefit across the
trainer's candidate-size buckets before deployment.

Achieved WHR-C calibration, named human operation profiles, and the unified
trainer release remain gated.

- One decision per pill spawn over exact planner-feasible final poses.
- SMDP returns discount over actual elapsed frames.
- `PublicPairState` is the only deployable actor input.
- `PrivilegedPairState` is restricted to critics, parity, search, and teachers.
- `drm_reach_bfs_full` remains the independent reachability oracle.
- The native two-player engine is the throughput simulator; emulator/script
  replay is the independent verification boundary.
- Search improvement is distilled before search controls PPO behavior.
- Match W/D/L is authoritative; tactical signals cannot pay for losing.
- Candidate truncation is a measured failure, not an accepted approximation.

## Repository map

- `drmc_rl/program/`: machine-readable stages, gates, recipes, and products.
- `drmc_rl/game/pair_state.py`: public/privileged pair-state v2 contracts.
- `drmc_rl/planning/`: exact Python/native/CUDA reachability.
- `drmc_rl/envs/backends/`: native and emulator runtime bindings.
- `drmc_rl/models/policy/`: G4/G5 candidate policies and exact effect tokens.
- `drmc_rl/search/joint_event.py`: asynchronous full-pair search algorithm.
- `drmc_rl/teachers/`: counterfactual and policy-improvement target generation.
- `drmc_rl/human/`: exact-afterstate human model, calibrated regret, style,
  unified decoder, timing, and adaptive sparring.
- `drmc_rl/execution/`: named human operation envelopes and script validation.
- `drmc_rl/arena/`: durable W/D/L evidence, ratings, and PSRO meta-strategy.
- `tools/`: guarded launch, corpus, training, arena, and evaluation commands.

## Authority

Read these before architecture or training changes:

- [Design](docs/DESIGN.md)
- [Roadmap](docs/ROADMAP.md)
- [Operations](docs/OPERATIONS.md)
- [Evaluation and release gates](docs/EVALUATION.md)
- [Known risks](docs/RISKS.md)

A legally obtained ROM is required only for independent emulator verification.
ROMs, corpora, checkpoints, run outputs, and operator secrets are not committed.
