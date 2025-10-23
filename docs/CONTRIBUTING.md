# Contributing

Thanks for helping build Dr. Mario RL. Please read `AGENTS.md` at the repo root for context and constraints that apply across the
codebase.

## Ground Rules
- Do not distribute ROMs; use legally obtained copies only.
- Preserve determinism: changes that affect timing or RNG must note implications and update parity tests.
- Keep changes minimal and focused; avoid unrelated refactors in the same PR.
- Match the existing style and structure; prefer type hints and descriptive names.
- Update docs alongside behavior changes. README, stand-up guides, and design docs must stay in sync with code.

## Setup
- macOS: `brew install cmake pkg-config lua@5.1`
- Python:
  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -e .
  pip install -e ".[retro,dev]"
  pip install -e ".[rl]"  # requires torch installation beforehand on MPS
  ```
- Linux (CUDA): install NVIDIA drivers, create a venv, then install the same extras. Grab torch wheels from the official CUDA
  index (`--index-url https://download.pytorch.org/whl/cu121`).

## Development Areas
- `envs/retro/`: libretro backend glue, placement translator, intent wrapper, seed registry, demo viewer.
- `envs/specs/`: RAM offsets, RNG helpers, timeout tables, observation formatting.
- `models/`: policy/evaluator scaffolding (risk-conditioned PPO + distributional heads).
- `training/`: Sample Factory configs/launchers.
- `eval/`: evaluation harness and plotting.
- `re/`: reverse-engineering artifacts; do not commit ROMs.
- `sim-envpool/`: forthcoming rules-exact high-FPS simulator.

## Testing
- Add focused unit tests near the code you change.
- For parity, record action/seed traces and assert board hashes + clear times.
- Keep tests hermetic; avoid network and large data downloads.
- Run `pytest -q` plus linting (`ruff .`, `black .`) before submitting when possible.

## Documentation
- Update `docs/DESIGN.md`, `docs/RNG.md`, and `docs/STATE_OBS_AND_RAM_MAPPING.md` when specs or RAM understanding change.
- Record decisions and open questions in `DESIGN_DISCUSSIONS.md`.
- Never edit `docs/EOCLabbook.md`; it is a human-authored logbook.
