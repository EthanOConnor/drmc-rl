# Repository Guidelines

## Current Architecture First

- Treat `cpp-pool` placement-SMDP training as the default path.
- The canonical command shape is:
  `python -m training.run --cfg training/configs/smdp_ppo.yaml --backend cpp-pool`.
- Current default env stack: `DrMarioPlacementEnv-v0`, `ppo_smdp`,
  `bitplane_bottle_mask`, `ln_hop_back` curriculum.
- Do not start new work from Stable-Retro, libretro, EnvPool, per-frame
  controller training, or `cpp-engine` unless the task is explicitly about
  parity, emulator debugging, or historical compatibility.
- `envs/retro/` is a historical package name. Its placement planner/reachability
  modules are active; its emulator wrappers are parity/debug support.
- `cpp-pool` bypasses the emulator backend registry. Look at
  `training/envs/dr_mario_vec.py`, `training/envs/drmario_pool_vec.py`, and
  `envs/backends/drmario_pool.py` for the live training path.
- The pool plans with the costs-only `drm_reach_bfs_v4` planner and warps
  pills to their lock pose (no controller-script replay). The v1 planner
  (`drm_reach_bfs_full`) is the verification oracle; never change one without
  re-running `tests/test_reach_v4_parity.py`. `DRMARIO_POOL_WARP=0` restores
  the legacy replay path for byte-level debugging.
- Roadmap toward top-human play (speedrun + VS league + strength dial):
  `docs/DESIGN_TOP_PLAY.md`. Current measured throughput:
  `docs/BENCHMARKS_2026-06-09.md`.

## Project Structure & Module Organization

- `training/`: runner, configs, PPO-SMDP, curriculum, diagnostics, and vector env
  factory.
- `models/policy/`: placement policy heads, candidate scorer, masked placement
  distribution, and candidate packing.
- `envs/backends/drmario_pool.py`: ctypes bridge to the native pool library.
- `training/envs/drmario_pool_vec.py`: in-process vector env used for current
  training.
- `envs/retro/`: placement planner/reachability plus emulator parity/debug envs.
- `reach_native/`: native reachability helper.
- `game_engine/`: submodule mount for `drmario-native`; engine changes belong in
  the standalone project first, then land here as a submodule SHA update.
- `dr-mario-disassembly/`: pinned disassembly + annotations submodule. The local
  umbrella workspace may also have shared reference checkouts in
  `../disassembly/`.
- `docs/`: current setup, design, placement, reward, and reference docs.
- `notes/`: inter-session memory, worklog, backlog, and scrutiny.
- `data/`, `runs/`, `legal_ROMs/`, cores, checkpoints, and datasets are
  git-ignored runtime artifacts.

## Dr. Mario Umbrella Workspace

- Treat `/Users/ethan/dev/drmario/` as the local metaproject root; read
  `../README.md` for the sibling-project map.
- Keep `dr-mario-disassembly/` as this repo's pinned reproducibility submodule.
- Prefer the shared `../disassembly/` checkouts for ad hoc cross-project
  reference work when available.
- Keep references to sibling projects local and explicit. Do not vendor large
  reference trees into this repo.

## Notes System (`notes/`)

`notes/` is for human and agent memory across sessions.

- `notes/MEMORY.md`: long-lived architectural memory and design decisions.
- `notes/WORKLOG.md`: chronological record of meaningful work.
- `notes/BACKLOG.md`: forward backlog and roadmap.
- `notes/CHAT.md`: scratchpad for ideas and hypotheses.
- `notes/SCRUTINY.md`: correctness, performance, API, and UX risks plus
  validation plans.

When editing code in this repo:

- Update `notes/WORKLOG.md` with a short entry for meaningful changes.
- Update `notes/MEMORY.md` when you make or rely on a design decision.
- Prefer `notes/BACKLOG.md` over TODO comments for future work.
- Add subtle risks or validation gaps to `notes/SCRUTINY.md`.

## Build, Test, and Development Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl,viz]"
git submodule update --init --recursive
python -m tools.build_drmario_pool
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui tui --backend cpp-pool --num_envs 16
```

Focused checks:

```bash
python -m tools.build_drmario_pool --help
python -m tools.bench_multienv --backend cpp-pool --vectorization sync --num-envs 1,2
python -m tools.bench_policy --source cpp-pool --batch-size 16 --repeats 30
pytest -q
```

Emulator parity/debug only:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path /path/to/DrMario.nes \
  --num_envs 1
```

## Coding Style & Naming Conventions

- Python 3.10+, 4-space indent, line length 100.
- Type hints for public interfaces and non-obvious internal contracts.
- Names: `snake_case` for functions/vars, `CapWords` for classes,
  `UPPER_SNAKE` for constants.
- Match the surrounding style. Avoid unrelated refactors.

## Testing Guidelines

- Use `pytest`; tests live in `tests/` and mirror package paths.
- Prefer deterministic tests with fixed seeds and small env counts.
- For parity, record action/seed traces and assert board hashes or clear-time
  invariants.
- If the local venv is stale or missing dependencies, say that directly and
  reinstall from this checkout before claiming test health.

## Commit & Pull Request Guidelines

- Commits: imperative mood; Conventional Commits are fine when useful.
- PRs: include description, commands run, expected metrics/plots when relevant,
  and docs/notes updates.
- Do not commit ROMs, savestates, cores, checkpoints, large datasets, or run
  outputs.

## Security & Configuration Tips

- Never distribute ROMs. `*.nes` and `legal_ROMs/` are ignored.
- `cpp-pool` training needs the native submodule and build artifact, not a ROM.
- Libretro parity/debug needs `DRMARIO_CORE_PATH` and `DRMARIO_ROM_PATH`.

## Dr. Mario RAM Reference (from `drmarioai/`)

- **Gameplay mode**: RAM `$0046` flips to `0x04` only while a bottle is active.
  Use it to guard gameplay-only logic.
- **Stage transitions**:
  - `$0055` becomes `0x01` once the current stage is cleared.
  - `$0053` stays `0x0A` during normal play and changes once the
    credits/ending cutscene takes over.
  - The Java bot waits about `90*60` frames after detecting the ending before
    tapping START again.
- **Player slots**: `$0727` holds the active player count. Player-relative
  addresses use a `$0080` offset for P2.
- **Current pill state**: base addresses `$0305` column, `$0306` row from bottom,
  `$0325` orientation, `$0301/$0302` colors, and `$0310` pill spawn counter.
- **Next pill preview**: `$031A/$031B` give upcoming colors.
- **Playfield buffers**: P1 bottle starts at `$0400`, P2 at `$0500`; each row is
  8 bytes with the high nibble encoding tile type and low bits encoding color.
- **Gravity/drop timing**: `$0312` counts frames until forced drop; `0xFF`
  stalls the fall and `0x01` forces a drop.
- **Spawn detection**: `$0310` increments whenever a new pill appears
  (BCD with manual decimal adjust, `$0311` hundreds; includes the
  currently falling pill — reconciled against the disassembly with the
  fightcadeRatings project, see docs/HUMAN_CORPUS_INTEGRATION.md).

---

## Karpathy-Inspired Coding Guidelines

Source: https://github.com/forrestchang/andrej-karpathy-skills `CLAUDE.md`.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, and clarifying questions come before
implementation rather than after mistakes.
