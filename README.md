# drmc-rl

Training, search, evaluation, and live-play software for Dr. Mario agents.

The core abstraction is a placement SMDP: once per pill spawn, the policy
chooses one of 512 final poses. An exact reachability planner supplies the
feasible set and frame cost, and the native engine advances directly to the
next decision. Emulator and live-play paths replay controller scripts when
frame-exact execution matters.

## Products

- **Single player:** curriculum PPO for fast, reliable clears.
- **VS:** native two-player games, human-policy opponents, league training,
  search distillation, tournaments, and a tunable live match agent.
- **Seedlab:** distributed search and a per-seed clear-time catalog.

Normal training does not need a ROM. A legally obtained ROM is required only
for independent emulator verification.

## Setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git submodule update --init --recursive
uv sync --all-extras
uv run python -m tools.build_drmario_pool
```

Validate the default path:

```bash
uv run python -m drmc_rl.training.run --dry_run true
uv run pytest -q
```

Start single-player training:

```bash
uv run python -m drmc_rl.training.run --backend cpp-pool --ui tui
```

Run the continuous VS lineage arena and its browser dashboard:

```bash
uv run python -m tools.arena register arena-roster.json
uv run python -m tools.arena worker
uv run python -m tools.arena serve --host 0.0.0.0
```

Promoted champions remain immutable active entrants after a successor is
promoted, so ratings and promotion evidence continue to cover every era.

The active VS configs are `vs6_tf3090.yaml`, `vsdist2_tf3090.yaml`, and
`vsact_actfromsearch.yaml`. The latter two are staged experiments with explicit
checkpoint placeholders; do not launch them without satisfying their gates.

## Code map

- `drmc_rl/training/algo/ppo_smdp.py`: rollout collection, SMDP returns, PPO, and
  optional search targets.
- `drmc_rl/training/envs/drmario_pool_vec.py`: single-player native vector environment.
- `drmc_rl/training/envs/drmario_vs_vec.py`: VS vector environment and opponent pool.
- `drmc_rl/game/`: backend-independent state, mechanics, and specifications.
- `drmc_rl/planning/`: placement space and Python/native/CUDA reachability.
- `drmc_rl/envs/backends/`: native-pool and libretro runtime bindings.
- `drmc_rl/envs/libretro/`: emulator-only Gymnasium environments and tools.
- `drmc_rl/models/policy/`: candidate policy, packing, and inference-time search.
- `reach_native/`: C source and build output for the native planner.
- `drmc_rl/seedlab/`: seed catalog, workers, and search.
- `tools/live_agent_server.py`: RAM-to-plan live match bridge.
- `vendor/drmario_native/`: pinned `drmario-native` submodule.

## Documentation

- [Design](docs/DESIGN.md)
- [Roadmap](docs/ROADMAP.md)
- [Known risks](docs/RISKS.md)
- [Placement planner](docs/PLACEMENT_PLANNER.md)
- [Placement policy](docs/PLACEMENT_POLICY.md)
- [Search](docs/SEARCH_DESIGN.md)
- [Live match agent](docs/MATCH_AGENT.md)
- [Seed catalog](docs/SEED_CATALOG.md)
- [Verification](docs/VERIFICATION_CHECKLIST.md)

ROMs, cores, checkpoints, run outputs, and large datasets are intentionally
untracked.
