# RAM → State Mapper (visible info only)

- Fill `envs/specs/ram_offsets_example.json` with verified addresses from Mesen/FCEUX.
- No future peeking: include only the current falling pill and the next preview pill.
- 14-channel schema over 16×8:
  - 3 virus planes (R/Y/B)
  - 3 fixed pill planes (R/Y/B)
  - 3 falling pill planes (R/Y/B)
  - 1 orientation plane
  - 4 scalar planes (gravity, lock, level, settle_flag[optional])
- Ensure unsigned byte handling; normalize scalars to [0,1].
- Unit tests: `tests/test_ram_to_state_visible_only.py` checks shape and basic invariants.

