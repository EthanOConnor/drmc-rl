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

```bash
git submodule update --init --recursive
uv sync --all-extras
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
