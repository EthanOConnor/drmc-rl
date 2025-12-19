# C++ Simulator Notes — Dr. Mario (NES)

Goal: faithful, max-FPS, rules-exact simulator matching the disassembly.
This outlines state layout, per-frame order, and key routines to port.

ROM: Dr. Mario (Japan, USA) (rev0), SHA1 01de1e04c396298358e86468ba96148066688194.

## State layout (per player)
- Field: 16×8 bytes (P1), encoding hi:type, lo:color (yellow=0, red=1, blue=2).
- Falling pill: {x, y, rotation (bit0: 0=vert,1=hor), size=2, colors L/R}.
- Counters: speedCounter, pillPlacedStep, chain flags, combo counters, etc.
- Options: level, speedSetting/index.
- RNG: two-byte state (rng0, rng1).

## Per-frame main loop (level mode)
1) Input polling: sample controllers; derive pressed/held bitfields.
2) Falling pill update:
   - Visual sprite update (for accuracy ties, but sim can compute positions directly).
   - Y move: gravity vs fast drop; anti-piracy check (skip in sim if not needed).
   - X move: auto-repeat timings (hor_accel_speed/hor_max_speed) per constants.
   - Rotate: validate, wall-kick left if blocked, double-left nuance when vertical→horizontal.
3) Lock & settle:
   - When grounded after a gravity step or down press, switch to pillPlaced pipeline.
   - pillPlacedStep micro-steps: check drops, check matches, update field, reset flags.
4) Clear & drop:
   - Match detection (length=4), mark clears; score update; settle, drop remaining halves.
5) Next pill generation:
   - Copy preview→falling, set start X=3, Y=15, reset rotation.
   - Update preview via pill reserve / RNG.
6) End checks:
   - Viruses remaining; level end; speed ups on tens of pill counter.

## RNG
- State: rng0=$0017, rng1=$0018.
- Update: randomNumberGenerator (carry from bit1 EOR) → ROR over two bytes.
- Default seed: rng0=$89, rng1=$88 (engine-only fallback).
- For emulator parity, RNG seed bytes are applied at the `initData_level` boundary (mode==0x03), i.e. "RNG state at level init entry".

## Virus placement
- addVirus: choose valid height vs level cap; choose x; color distribution (cycle every 4) + adjacency constraints; write to field.

## Constants to mirror
- See defines/drmario_constants.asm: rowSize=8, heightSize=16, fieldSize=128.
- Type codes: 0x40..0xA0 (pills), 0xD0 virus, 0xFF empty, masks $F0/$03.
- Timings: hor_accel_speed, hor_max_speed, fast_drop_speed.

## Parity plan
- Use ROM-driven parity fixtures (RAM field snapshots, preview colors, falling pill state) to assert hash of field after N steps given fixed input traces.
- Seed determinism: use the same seed path as the env (initData-level seeding, not menu-time seeding).
- Use `tools/ghost_parity.py` to run libretro (ground truth) and the C++ engine side-by-side and stop on first RAM divergence.

This is enough to scaffold a performant C++ core with a clean state struct and explicit step() ordering.
