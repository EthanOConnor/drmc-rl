# Contributing

Read `AGENTS.md` before editing this repo. It defines the current architecture
and the notes workflow.

## Ground Rules

- Keep the default mental model on `cpp-pool` placement-SMDP training.
- Do not distribute ROMs; emulator work must use legally obtained local copies.
- Preserve determinism. Timing, RNG, and planner changes need explicit parity or
  regression notes.
- Keep changes focused and avoid unrelated refactors.
- Match existing style and update docs/notes when behavior changes.

## Setup

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl,viz]"
python -m tools.build_drmario_pool
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true
```

If PyTorch requires a platform-specific install, install the correct `torch`
wheel first and then reinstall this package with extras.

## Development Areas

- `training/`: runner, PPO-SMDP, curriculum, diagnostics, vector env factory.
- `training/envs/drmario_pool_vec.py`: active high-throughput env.
- `envs/backends/drmario_pool.py`: native pool ctypes bridge.
- `models/policy/`: placement and candidate-scoring policies.
- `envs/retro/`: active placement planner/reachability plus emulator
  parity/debug wrappers.
- `game_engine/`: `drmario-native` submodule mount.
- `docs/` and `notes/`: current guidance and durable handoff.

## Testing

- Add focused tests near the behavior you change.
- Prefer deterministic seeds and small env counts.
- For native/emulator parity, record action/seed traces and assert board hashes,
  counters, or clear-time invariants.
- Run `pytest -q` when dependencies are installed.
- At minimum, run `python -m training.run --cfg training/configs/smdp_ppo.yaml
  --dry_run true` after runner/config changes.

## Documentation

- Update `docs/DESIGN.md` for architecture changes.
- Update `docs/PLACEMENT_POLICY.md` or `docs/PLACEMENT_PLANNER.md` for SMDP,
  policy, mask, planner, or cost semantics.
- Update `docs/RETRO_CORE_NOTES.md` only for emulator parity/debug changes.
- Record decisions in `notes/MEMORY.md`, risks in `notes/SCRUTINY.md`, and work
  done in `notes/WORKLOG.md`.
