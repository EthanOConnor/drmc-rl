# Contributing

Thanks for helping build Dr. Mario RL. Please read AGENTS.md at the repo root for context and constraints that apply across the codebase.

## Ground Rules
- Do not distribute ROMs; use legally‑obtained copies only.
- Preserve determinism: changes that affect timing or RNG must note implications and update parity tests.
- Keep changes minimal and focused; avoid unrelated refactors in the same PR.
- Match the existing style and structure; prefer type hints and descriptive names.

## Setup
- macOS: `brew install cmake pkg-config lua@5.1`
- Python: `uv venv && source .venv/bin/activate` (or `python -m venv venv`)
- Install `torch`, `stable-retro`, `gymnasium`, `numpy`, `opencv-python` when network is available.

## Development Areas
- `envs/retro/`: Stable‑Retro wrappers; state extraction; seed registry utilities
- `models/`: policy nets, evaluator heads, pixel→state
- `eval/`: evaluation harness and plots
- `dr-mario-disassembly/`: disassembly + annotations (no ROMs)
- `sim-envpool/`: rules‑exact high‑FPS simulator (later)

## Testing
- Add focused unit tests near the code you change.
- For parity, record action/seed traces and assert board hashes + clear times.
- Keep tests hermetic; avoid network and large data downloads.

## Documentation
- Update `docs/DESIGN.md` and `docs/RNG.md` when changing specs or RNG understanding.
- Record decisions in `notes/MEMORY.md` and open questions/risks in `notes/CHAT.md` / `notes/SCRUTINY.md`.
