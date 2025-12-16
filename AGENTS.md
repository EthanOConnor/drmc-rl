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
- `notes/` Inter-session, inter-agent coordination and memory (see below).

## Notes System (`notes/`)

`notes/` is for human + agent memory and coordination across sessions. Files:

- **`notes/MEMORY.md`**
  - Long-lived architectural memory, design decisions, and "why" behind choices.
  - Think ADR-lite: short entries, timestamped, with rationale and trade-offs.

- **`notes/WORKLOG.md`**
  - Chronological log of work done.
  - Use a simple bullet format with date, actor, and brief summary.

- **`notes/BACKLOG.md`**
  - Technical backlog / roadmap with prioritized sections (near-term, medium-term, longer-term).
  - More detailed than top-level docs; use for technical work items.

- **`notes/CHAT.md`**
  - Scratchpad for ideas, hypotheses, sketches—things that might become real work later.
  - Can be informal, but keep it readable and dated.

- **`notes/SCRUTINY.md`**
  - Critical review and risk tracking.
  - Capture concerns about correctness, performance, API contracts, and UX, plus how we'll validate or mitigate them.

**When editing code in this repo:**
- Update `notes/WORKLOG.md` with a short entry for any meaningful change.
- Update `notes/MEMORY.md` when you make or rely on a design decision.
- Prefer adding items to `notes/BACKLOG.md` instead of TODO comments in code.
- When you spot a risk or subtle behavior, add an entry in `notes/SCRUTINY.md`.

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

## Dr. Mario RAM Reference (from `drmarioai/`)
- **Gameplay mode**: RAM `$0046` flips to `0x04` only while a bottle is active. Use it to guard
  gameplay-only logic (virus counters, top-out detection, etc.).
- **Stage transitions**:
  - `$0055` becomes `0x01` once the current stage is cleared.
  - `$0053` stays `0x0A` during normal play and changes once the credits/ending cutscene takes over.
  - The Java bot waits ~`90*60` frames after detecting the ending before tapping START again.
- **Player slots**: `$0727` holds the active player count (1 or 2). Player-relative addresses use
  a `$0080` offset for P2: e.g., `$0305|0x80 = 0x0385` for P2's pill column.
- **Current pill state (per player)**: base addresses `$0305` (column), `$0306` (row from bottom),
  `$0325` (orientation), `$0301/$0302` (left/right colors), and `$0310` (pill spawn counter).
- **Next pill preview**: `$031A/$031B` give upcoming colors.
- **Playfield buffers**: P1 bottle starts at `$0400`, P2 at `$0500`; each row is 8 bytes with the
  high nibble encoding tile type and the low bits encoding color (`00`=yellow, `01`=red, `02`=blue).
  Tile codes mirror `Tile.java`: `0x4` top, `0x5` bottom, `0x6` left, `0x7` right, `0x8` square,
  `0xB/0xD` virus.
- **Gravity/drop timing**: `$0312` counts frames until forced drop; writing `0xFF` stalls the fall
  and `0x01` forces a drop (the bot uses both).
- **Spawn detection**: `$0310` increments whenever a new pill appears. The AI watches for changes
  (or initial values >1) to know when to plan the next placement.
