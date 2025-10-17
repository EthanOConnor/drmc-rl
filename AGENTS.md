# Repository Guidelines

## Project Structure & Module Organization
- `envs/retro/` Stable‑Retro wrappers, seed registry, demo runner.
- `envs/pettingzoo/` 2‑player ParallelEnv wrapper (skeleton).
- `envs/specs/` RAM map placeholders and specs.
- `models/` Policy nets, evaluator heads, pixel→state (skeletons).
- `training/` Sample Factory configs and launches.
- `eval/` Evaluation harness and plots.
- `sim-envpool/` Rules‑exact C++ env (later).
- `re/` Reverse‑engineering artifacts (no ROMs).
- `docs/` DESIGN, RNG, CONTRIBUTING. Data lives in `data/` (git‑ignored).

## Build, Test, and Development Commands
- macOS deps: `brew install cmake pkg-config lua@5.1`
- Python env: `uv venv && source .venv/bin/activate`
- Install: `pip install -e .` (then add extras as needed)
- Demo env: `python -m envs.retro.demo --obs-mode pixel --steps 200`
- Eval harness: `python -m eval.harness.run_eval --episodes 100 --obs-mode state`
- Train (SF): `python -m sample_factory.launcher.run --cfg training/sf_configs/state_baseline.yaml`
- Lint/format (optional): `ruff .` and `black .`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, line length 100.
- Type hints required; concise docstrings for public APIs.
- Names: snake_case for functions/vars, CapWords for classes, UPPER_SNAKE for consts.
- Keep modules focused; avoid unrelated refactors.

## Testing Guidelines
- Use `pytest`; tests live in `tests/` and mirror package paths.
- Name tests `test_*.py`; prefer deterministic tests with fixed seeds.
- For env parity, record action/seed traces and assert board hashes/clear times.
- Run: `pytest -q`; target ≥80% coverage for core logic (excluding I/O stubs).

## Commit & Pull Request Guidelines
- Commits: imperative mood; Conventional Commits recommended (e.g., `feat(envs): add state tensor`) .
- PRs: include description, linked issues, how to run, expected metrics/plots, and docs updates.
- Do not commit ROMs, savestates, or large datasets; use paths under `data/` (already git‑ignored).

## Security & Configuration Tips
- Never distribute ROMs (`*.nes` ignored). Keep secrets out of code.
- macOS uses Lua 5.1 for Stable‑Retro; record your libretro core path in run scripts.
