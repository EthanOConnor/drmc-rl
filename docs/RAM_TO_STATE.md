# RAM → State Mapper (visible info only)

The RAM→state mapper in `drmc_rl/game/specs/ram_to_state.py` decodes only *currently
visible* gameplay information:
- Bottle tiles (viruses + locked pill halves)
- Current falling pill pose + colors (from RAM registers)
- Next preview pill (HUD preview)
- Optional scalar broadcasts (in the higher-dimensional representations)

Representation selection
- Set via `DrMarioLibretroEnv(..., state_repr="...")` or `drmc_rl.game.specs.ram_to_state.set_state_representation(...)`.
- Supported modes (all emit `(C,16,8)` float32 per frame):
  - `extended` (16ch): explicit virus/static/falling by color + scalars
  - `bitplane_bottle` (4ch): bottle-only type-blind colors + virus mask
  - `bitplane_bottle_conn` (8ch): bottle-only + locked-capsule connection edges
  - `bitplane_bottle_mask` (8ch): bottle-only + feasibility planes (injected by placement env)
  - `bitplane_bottle_conn_mask` (12ch): bottle-only + connection edges + feasibility planes
  - `bitplane` (12ch): type-blind colors + masks + scalars
  - `bitplane_reduced` (6ch): colors + virus + falling + preview
  - `bitplane_reduced_mask` (10ch): reduced + feasibility planes (injected by placement env)

Notes
- No future peeking: only the current falling pill and the next preview pill are included.
- Connection edges come only from ordinary locked pill-half tile codes; viruses,
  singles, and legacy middle-half codes have no edge channel set.
- Scalars are normalized to `[0,1]`.
- Unit tests: `tests/test_ram_to_state_visible_only.py` checks shape and basic invariants.
