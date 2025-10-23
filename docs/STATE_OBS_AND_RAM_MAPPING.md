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
- Lock counter: `$0307` (micro-step state approaching lock).
- Level: `$0316`.

## Channel specs

Two observation layouts are supported and can be selected via `ram_to_state.set_state_representation("extended" | "bitplane")`.

### Extended representation (16 channels)
1-3. Virus color channels (R, Y, B) – only virus tiles are hot in these planes.
4-6. Static pill color channels (R, Y, B) – locked pill halves.
7-9. Falling pill color channels (R, Y, B) – current falling capsule halves.
10. Orientation – scalar plane (`0.0` vertical, `1.0` horizontal).
11. Gravity counter – normalized `[0,1]`.
12. Lock counter – normalized `[0,1]`.
13. Level – normalized by 20.
14. Preview first color – scalar in `{0.0, 0.5, 1.0}` representing `{yellow, red, blue}`.
15. Preview second color – same encoding as channel 14.
16. Preview rotation – scalar in `[0,1]` mapping to NES rotation states `/3`.

### Bitplane representation (12 channels)
1-3. Color bitplanes (all tiles) for red/yellow/blue.
4. Virus mask.
5. Locked pill mask.
6. Falling pill mask.
7. Preview mask (HUD-projected capsule in spawn area).
8. Clearing mask (`0xB0` or `0xF0`).
9. Empty mask (available cells).
10. Gravity counter.
11. Lock counter.
12. Level scalar.

Implementation
- Code: `envs/specs/ram_to_state.py` implements the decoder using the above masks and addresses.
- Offsets source: `envs/specs/ram_offsets.json` for the current ROM; `re/out/ram_map.json` contains an expanded spec including inputs/timers/RNG for reference.
- `envs/retro/state_viz.py` renders the extended representation into RGB overlays for debugging.

Verification tips
- See `docs/VERIFICATION_CHECKLIST.md` for step-by-step validation of offsets and plane semantics.
