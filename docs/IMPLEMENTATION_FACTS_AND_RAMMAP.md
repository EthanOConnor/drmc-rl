# Dr. Mario (NES) — Implementation Facts & RAM Map (Consolidated)
**Version:** 2025-10-17

**Purpose:** One-stop reference for agents/humans implementing the faithful Dr. Mario simulator and RL env. Aggregates **confirmed project decisions** and a **RAM map (external candidates → validate)**.

> **Status (2026-05-07):** This is a ROM/RAM-map reference, not the
> current training architecture source of truth. Current training uses
> `ppo_smdp` over the in-process `cpp-pool` backend, `DrMarioPlacementEnv-v0`,
> `bitplane_bottle_conn_mask` observations, and the `ln_hop_back` curriculum. Use
> `README.md`, `docs/DESIGN.md`, `docs/STATE_OBS_AND_RAM_MAPPING.md`,
> `docs/PLACEMENT_POLICY.md`, and `docs/PLACEMENT_PLANNER.md` for the live
> training path.

> ⚖️ **Legal:** Use only a **legally owned** ROM. Do **not** commit ROMs. ROM path via env var `DRMARIO_ROM`.

---
## 0) Target ROM & identity (fill first)
- **Region:** NTSC Dr. Mario (NES)
- **Mapper:** MMC1B family (SEROM board common)
- **Record identity:** dump **CRC32 / SHA1** from emulator and paste:
  - `CRC32 = 0xB1F7E3E9`  `SHA1 = 01de1e04c396298358e86468ba96148066688194`

---
## 1) Project-confirmed details

### 1.1 Observation modes
- **Current training (`cpp-pool`)**: state tensor `(C,16,8)`, normally
  `bitplane_bottle_conn_mask` with twelve channels: red/yellow/blue bottle color
  occupancy, virus mask, four locked-capsule connection-edge planes, and four
  orientation-specific feasible-placement masks.
- **Emulator/debug lane**: pixel and RAM-derived state observations are still
  available for libretro/Stable-Retro parity work.
- See `docs/STATE_OBS_AND_RAM_MAPPING.md` and `envs/specs/ram_to_state.py` for
  the maintained state specifications.

### 1.2 Action model
- **Current training action**: one 512-way macro placement per controllable
  spawn, encoded as `orientation * 16 * 8 + row * 8 + col`.
- **Execution**: the placement planner/native pool maps a legal macro placement
  to the frame-level movement/rotation/down sequence and reports lock cost
  (`tau`, `cost_to_lock`, per-candidate costs).
- **Per-frame controller actions** remain emulator/native-parity primitives, not
  the default RL action space.

### 1.3 Rewards and curriculum
- SMDP-PPO consumes macro-decision rewards plus `placements`, `tau`, and
  `gamma^tau` discounting.
- Current configs use soft time budgets and `ln_hop_back` synthetic curriculum
  stages before real levels. See `docs/PLACEMENT_POLICY.md`,
  `docs/REWARD_SHAPING.md`, and `training/configs/smdp_ppo.yaml`.

### 1.4 Evaluator
- Distributional evaluator code exists, but it is not in the current default
  training loop. Treat `models/evaluator/train_qr.py` as unfinished unless it
  has been refreshed after this document.

### 1.5 Seeds/determinism
- `cpp-pool` training can randomize native RNG state on reset.
- Emulator seed registries under `envs/retro/seeds/` remain a parity/evaluation
  lane, not a prerequisite for current training runs.

### 1.6 Planning hooks
- Current planner-facing data is the legal/feasible mask, per-placement
  `cost_to_lock`/`costs`, and `tau` for the chosen macro action.
- See `docs/PLACEMENT_PLANNER.md` for the contract exposed by `cpp-pool` and
  the compatibility backends.

### 1.7 Backend lanes
- **Default training**: `cpp-pool`, an in-process native pool under
  `game_engine/` / `drmario-native`; no ROM required.
- **Emulator parity/debug**: libretro / Stable-Retro with a legally owned ROM.
- **Compatibility lane**: older `cpp-engine` subprocess/shared-memory backend.

---
## 2) RAM Map — external candidates (must validate on your ROM)
Validated on our ROM (rev0, Mapper 1). Below are confirmed addresses for P1; P2 mirrors at `$0380` block and bottle at `$0500`.

| Address | Meaning (confirmed) | Notes |
|---|---|---|
| $0043 | frameCounter | free-running, incremented in NMI |
| $0051 | waitFrames | general purpose countdowns |
| $0080–$00AF | currentP_* | zero-page mirror of current player block |
| $00F5/$00F7 | P1 buttons pressed/held | bitfields |
| $00F6/$00F8 | P2 buttons pressed/held | bitfields |
| $0300–$032F | p1_RAM | per-player block |
| $0305/$0306 | p1_fallingPillX / p1_fallingPillY | 0-based grid coords |
| $0325 | p1_fallingPillRotation | bit0: 0 vertical, 1 horizontal |
| $0301/$0302 | p1_fallingPill1stColor / 2ndColor | 0=Y,1=R,2=B |
| $031A/$031B | p1_nextPill1stColor / 2ndColor | preview HUD |
| $0322/$0323 | p1_nextPillRotation / Size | size always 2 |
| $0312 | p1_speedCounter | frames at current Y for gravity |
| $0307 | p1_pillPlacedStep | settles/clear/settle micro-step state |
| $0316 | p1_level | virus level index |
| $0324 | p1_virusLeft | remaining viruses |
| $0400 | p1_field (128 bytes) | bottle grid 16×8, stride=8 |
| $0500 | p2_field (128 bytes) | player 2 |

Bottle encoding (per `defines/drmario_constants.asm`):
- Type in high nibble; color in low nibble. Masks: `mask_fieldobject_type=$F0`, `mask_color=$03`.
- Types: topHalf=$40, bottomHalf=$50, leftHalf=$60, rightHalf=$70, single=$80, midVer=$90, midHor=$A0, cleared=$B0, virus=$D0, empty=$FF.

RNG and init:
- RNG state bytes: `$0017=rng0`, `$0018=rng1`.
- RNG update routine: `randomNumberGenerator` at `$B78B` (see `prg/drmario_prg_general.asm`).
- Seeding: at `init` → `@storeRngSeeds`, sets `rng0=$89`, `rng1=$88`.

---
## 3) Hardware I/O (NES)
- $4016: controller read/strobe
- $8000/$A000/$C000/$E000: MMC1 serial-write regs (banking writes)

---
## 4) Confirmation playbook
1) Virus placement → break on bottle writes; trace ±2K ops (addresses in “RNG & Placement” below).
2) Pill RNG/preview → break on preview writes; backtrack RNG to seed moment (see “RNG & Placement”).
3) Game step → input poll, gravity, lock, clear/settle (see “Core Routines”).
4) RAM map → validate against `dr-mario-disassembly/` and update:
   - `envs/specs/ram_offsets.json`
   - `envs/specs/ram_map.py`

---
## 5) Acceptance
- Vectors resolved; mapper writes annotated.
- `envs/specs/ram_offsets.json` updated; `envs/specs/ram_map.py` updated.
- Engine demo parity fixture captured in `data/nes_demo.json` and guarded by unit tests.

---
## 6) References (external — for validation)
- Data Crystal Dr. Mario RAM map: https://datacrystal.tcrf.net/wiki/Dr._Mario_%28NES%29/RAM_map
- FCEUX RAM mapping guide: https://fceux.com/web/help/NESRAMMappingFindingValues.html
- MMC1/SEROM nuance: https://github.com/sanni/cartreader/issues/1060
- Gravity table discussion (validate): https://tetrisconcept.net/threads/dr-mario-virus-placement.2037/page-3
- Dr. Mario AI (context): https://meatfighter.com/drmarioai/
---
## 7) State Observation Spec (current summary)

The maintained state-observation reference is
`docs/STATE_OBS_AND_RAM_MAPPING.md`; use this section only as a quick index.

- `bitplane_bottle`: `4×16×8`, with red/yellow/blue color occupancy plus
  virus mask.
- `bitplane_bottle_mask`: `8×16×8`, extending `bitplane_bottle` with four
  orientation-specific feasible-placement masks. This remains available as a
  no-connection-edge ablation.
- `bitplane_bottle_conn`: `8×16×8`, extending `bitplane_bottle` with
  `connected_{up,down,left,right}` for ordinary locked pill halves.
- `bitplane_bottle_conn_mask`: `12×16×8`, extending `bitplane_bottle_conn` with
  four orientation-specific feasible-placement masks. This is the current
  `cpp-pool` default for `training/configs/smdp_ppo.yaml`.
- `extended`: `16×16×8` RAM-derived emulator/debug representation including
  falling pill planes, preview pill broadcast planes, gravity/lock/level scalar
  planes, and terminal flag planes.

See code: `envs/specs/ram_to_state.py:1` for RAM-derived mapping details, and
`envs/backends/drmario_pool.py` / `envs/retro/placement_env.py` for current
`cpp-pool` observation assembly.

---
## 8) RNG & Placement (from disassembly)
- RNG state bytes: `$0017 (rng0)`, `$0018 (rng1)`; two bytes rotated with carry derived from `bit1` of each.
- Update: `randomNumberGenerator` `$B78B` in `prg/drmario_prg_general.asm:1`.
- Seed moment: `init` → `@storeRngSeeds` in `prg/drmario_prg_game_init.asm:1` sets `rng0=$89`, then `rng1=$88`.
- Virus placement: `addVirus` around `$9D19` in `prg/drmario_prg_game_logic.asm:1`.
  - Chooses random height (masked to 0..15), validates vs level max.
  - Picks column from `rng1 & lastColumn` and combines with field row offset to index bottle.
  - Ensures color distribution across groups of 4 (`virusRndMask`, `virusRndColor` logic) and adjacency constraints (no same color within 2 rows/cols; uses `virusVerCheck`/`virusHorCheck`).
  - Writes color/type value at the chosen field byte when empty, else tries next position/defers.
### Canonical termination flags (used in state mode)
- Fail (P1 top-out): `$0309` (p1_levelFailFlag) non-zero when the newly generated pill cannot be placed.
- Success (stage clear): `$0324 == 0` (p1_virusLeft) and/or zero-page `$0055 == 0x01` (whoWon=player1). The engine sets `$0055` during win handling.
