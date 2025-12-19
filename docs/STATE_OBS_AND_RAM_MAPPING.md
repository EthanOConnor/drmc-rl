# State Observation and RAM Mapping

This document explains how the 16×8 state observation is derived from NES RAM for Dr. Mario.

ROM identity
- Dr. Mario (Japan, USA) (rev0), SHA1 `01de1e04c396298358e86468ba96148066688194`, Mapper 1.

Bottle grid
- P1 bottle base: `$0400`, P2 bottle base: `$0500`.
- Layout: `H=16` rows × `W=8` columns, stride `8` bytes per row, linear indexing.
- Encoding per byte:
  - High nibble (type): see `defines/drmario_constants.asm` (`mask_fieldobject_type=$F0`).
  - Low nibble (color): yellow=0, red=1, blue=2 (`mask_color=$03`).

Type codes (subset)
- `0x40` topHalfPill, `0x50` bottomHalfPill, `0x60` leftHalfPill, `0x70` rightHalfPill,
  `0x80` singleHalfPill, `0x90` middleVerHalfPill, `0xA0` middleHorHalfPill,
  `0xB0` clearedPillOrVirus, `0xD0` virus, `0xFF` empty.

Falling pill and preview (P1)
- Falling: X `$0305`, Y `$0306`, Rotation `$0325` (bit0: 0=vertical, 1=horizontal), Size `$0326`.
- Colors: `$0301` (first/left), `$0302` (second/right).
- Preview: colors `$031A/$031B`, rotation `$0322`, size `$0323`.

Scalars
- Gravity counter: `$0312` (frames since last Y advance at current speed).
- Settle/lock proxy: `$0307` (`p1_pillPlacedStep` micro-step state).
- Level: `$0316`.

State representations (channel specs)

The state observation supports multiple channel layouts selected via:
- `DrMarioRetroEnv(..., state_repr=...)`, or
- `envs.specs.ram_to_state.set_state_representation(...)` (process-wide).

All representations emit a `(C, 16, 8)` float32 tensor per frame; `DrMarioRetroEnv`
then provides a fixed 4-frame stack shaped `(4, C, 16, 8)` (some training setups
keep only the last frame).

### `extended` (16 channels)
- Explicit split planes for viruses/static/falling by color, plus scalar broadcasts:
  - `virus_{red,yellow,blue}`
  - `static_{red,yellow,blue}`
  - `falling_{red,yellow,blue}`
  - `orientation`, `gravity`, `lock`, `level`
  - `preview_first`, `preview_second`, `preview_rotation`

### `bitplane` (12 channels)
- Type-blind color planes + entity masks + scalar broadcasts:
  - `color_{red,yellow,blue}`
  - `virus_mask`, `locked_mask`, `falling_mask`, `preview_mask`
  - `clearing_mask`, `empty_mask`
  - `gravity`, `lock`, `level`

### `bitplane_bottle` (4 channels)
- Bottle-only board state (no falling/preview projection):
  - `color_{red,yellow,blue}` (type-blind)
  - `virus_mask`

This is intended for spawn-latched placement policies that receive pill colors
as separate vector inputs (`next_pill_colors` for the current pill and
`preview_pill` for the next/preview pill).

### `bitplane_bottle_mask` (8 channels)
- `bitplane_bottle` plus 4 feasibility-mask channels:
  - `feasible_o0..feasible_o3`

These feasibility planes are *not* derived from RAM; they are injected by the
placement wrapper at decision points and match `info["placements/feasible_mask"]`.

### `bitplane_reduced` (6 channels)
- Minimal decision-time features for spawn-latched placement policies:
  - `color_{red,yellow,blue}` (type-blind)
  - `virus_mask`
  - `pill_to_place` (falling pill mask)
  - `preview_pill` (HUD preview mask projected into the 16×8 grid)

### `bitplane_reduced_mask` (10 channels)
- `bitplane_reduced` plus 4 feasibility-mask channels:
  - `feasible_o0..feasible_o3`

These feasibility planes are *not* derived from RAM; they are injected by the
placement wrapper at true decision points and match `info["placements/feasible_mask"]`.

Implementation
- Code: `envs/specs/ram_to_state.py:1` implements the decoder using the above masks and addresses.
- Offsets source: `envs/specs/ram_offsets.json:1` for the current ROM; `re/out/ram_map.json:1` contains an expanded spec including inputs/timers/RNG for reference.

Termination (state mode)
- The environment uses canonical game flags to end episodes when running with RAM access:
  - Fail: `$0309` (p1_levelFailFlag) non-zero.
  - Success: `$0055 == 0x01` (whoWon) or `$0324 == 0` (p1_virusLeft).
- Pixel mode continues to rely on heuristics (virus deltas, inactivity patterns) as it has no RAM access.
