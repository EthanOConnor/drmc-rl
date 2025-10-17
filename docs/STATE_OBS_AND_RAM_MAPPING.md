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

Channel spec (14 total)
1–3. Viruses (R, Y, B): bytes with type `0xD0` and color 1/0/2 map to channels 0/1/2.
4–6. Fixed pill halves (R, Y, B): bytes with type in `{0x40,0x50,0x60,0x70,0x80,0x90,0xA0}` and color 1/0/2 map to channels 3/4/5.
7–9. Falling halves (R, Y, B): painted at `(row,col)` and its neighbor `(row+1,col)` if vertical else `(row,col+1)` using falling colors.
10. Orientation: broadcast scalar (0 vertical, 1 horizontal).
11–13. Gravity counter, settle/lock proxy, level (normalized planes).
14. Spare (currently 0) reserved for an explicit settle flag if needed.

Implementation
- Code: `envs/specs/ram_to_state.py:1` implements the decoder using the above masks and addresses.
- Offsets source: `envs/specs/ram_offsets.json:1` for the current ROM; `re/out/ram_map.json:1` contains an expanded spec including inputs/timers/RNG for reference.

