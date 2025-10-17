# RAM Offsets Verification Checklist

1. Bottle base/stride
   - Advance one frame with no falling pill; grid bytes remain stable.
   - Clear a virus; verify expected cell byte changes.
2. Falling pill row/col/orient
   - Move left/right/down; row/col update as expected.
   - Rotate; orientation toggles {0,1}; secondary half position consistent.
3. Colors
   - On spawn, read left/right color bytes; falling planes (6..8) hot at correct coords.
4. Preview pill
   - Addresses match HUD preview; updates only on spawn.
5. Scalars (gravity/lock/level)
   - Gravity/lock counters move during fall/lock; level normalized to [0,1].
6. No future peeking
   - Do not read hidden RNG; only current + preview pill.

Copy `envs/specs/ram_offsets_example.json` to `envs/specs/ram_offsets.json` with your values once verified.

