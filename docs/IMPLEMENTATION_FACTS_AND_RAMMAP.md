# Dr. Mario (NES) — Implementation Facts & RAM Map (Consolidated)
**Version:** 2025-10-17

**Purpose:** One-stop reference for agents/humans implementing the faithful Dr. Mario simulator and RL env. Aggregates **confirmed project decisions** and a **RAM map (external candidates → validate)**.

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
- **pixel**: 128×128 RGB, 4-frame stack.
- **state**: 14×16×8 planes (**visible-only**), 4-stack.

### 1.2 Actions (10 at 60 Hz, latching holds)
- hold LEFT, hold RIGHT, hold DOWN (latched)
- tap A, tap B, tap A+B (one-frame taps)
- NONE (+ release semantics per adapter)

### 1.3 Rewards & shaping
- Base: negative time per step + clear bonus; **per-level T_max** timeouts.
- Potential shaping: r_shape = γ·Φ(s′) − Φ(s), **Φ(terminal)=0**. Evaluator Φ is distributional ETC; scale with `potential_kappa` (≈250).

### 1.4 Evaluator
- Distributional (QR/IQN). Quantiles → mean/τ-quantiles/CVaR. TorchScript runtime; optional bootstrap into PPO-style value targets.

### 1.5 Seeds/determinism
- Seeds by **level-select + frame offset**; catalog under `envs/retro/seeds/`. Eval harness computes E[T], Var[T], CVaR.

### 1.6 Planning hooks
- snapshot/restore/peek_step in env; one-step lookahead wrapper supports mean/quantile/CVaR.

### 1.7 Emulator
- Stable-Retro (guarded); pixel via core frames; state via `retro.get_ram()` → mapper. Env id `DrMarioRetroEnv-v0`.

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
## 7) State Observation Spec (confirmed)
- Shape: `C=14, H=16, W=8`, frame‑stack 4.
- Channels (R/Y/B order within groups):
  - 0..2: viruses (R,Y,B) where type==$D0 and color in {1,0,2}
  - 3..5: fixed pill halves (R,Y,B) where type in {$40,$50,$60,$70,$80,$90,$A0}
  - 6..8: falling pill halves (R,Y,B) — from `p1_fallingPill*`
  - 9: orientation (0=vertical, 1=horizontal), broadcast scalar
  - 10: gravity counter (normalized from `$0312`)
  - 11: settle/lock proxy (normalized from `$0307`)
  - 12: level scalar (normalized from `$0316`)
  - 13: spare/settle flag placeholder (0 for now)

See code: `envs/specs/ram_to_state.py:1` for the exact mapping and masks.

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
