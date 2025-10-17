# RNG and Placement — Dr. Mario (NES)

ROM: Dr. Mario (Japan, USA) (rev0)
- SHA1: 01de1e04c396298358e86468ba96148066688194
- Mapper: MMC1 (mapper 1)

## RNG state and update
- State bytes (zero page): `rng0=$0017`, `rng1=$0018`.
- Seed: at init (`prg/drmario_prg_game_init.asm:1` @ `@storeRngSeeds`), `rng0=$89`, `rng1=$88`.
- Update routine: `randomNumberGenerator` at `$B78B` (`prg/drmario_prg_general.asm:1`).
  - Carry source: EOR of `(rng0 & 0x02)` and `(rng1 & 0x02)` mapped to C.
  - Then `Y` bytes are produced by `ROR` over `rng0+X, rng1+X, ...` (always called with `X=$17` and `Y=$02`).
  - Typical callsite macro `generateRandNum` updates both `rng0` and `rng1` per use.

Implication: RNG is deterministic per seed and number of calls; bank switching does not affect it.

## Virus placement (level start)
- Routine: `addVirus` (~`$9D19`) in `prg/drmario_prg_game_logic.asm:1`.
- Summary:
  1) Height: generate random, mask to [`0..15`], reject if above `addVirus_maxHeight_basedOnLvl[level]`.
  2) Field index: map height to starting field offset via `pill_fieldPos_relativeToPillY[height]`, pick column from `rng1 & lastColumn (7)`, add to get linear index.
  3) Color selection: enforce palette cycle every 4 viruses using `currentP_virusToAdd & virusRndMask` with one in four picked randomly from `virusColor_random` table; else rotate colors.
  4) Adjacency constraints: forbid same color within 2 rows/columns (`virusVerCheck = 2*rowSize`, `virusHorCheck = 2`).
  5) Write encoded byte (type=`virus=$D0`, low nibble=`color`) to `currentP_fieldPointer` when empty; otherwise probe next position or retry next frame.

## Pill generation
- Routine: `generateNextPill` (~`$8E9D`) in `prg/drmario_prg_game_logic.asm:1`.
- Summary:
  - Copies preview colors (currentP_nextPill*) into falling pill colors (currentP_fallingPill*), resets rotation and size, sets start X=`$03`, Y=`$0F`.
  - Updates pill counters; preview updated using pill reserve and RNG index logic.

## Useful labels/addresses
- `frameCounter=$0043`, `waitFrames=$0051`.
- P1 falling: X `$0305`, Y `$0306`, Rotation `$0325`, Size `$0326`, Colors `$0301/$0302`.
- Preview (P1): Colors `$031A/$031B`, Rotation `$0322`, Size `$0323`.
- Viruses remaining (P1): `$0324`.
- Bottle: P1 `$0400` (16×8, stride 8), P2 `$0500`.

These details drive the RL state mapping and will later anchor the C++ simulator implementation.
