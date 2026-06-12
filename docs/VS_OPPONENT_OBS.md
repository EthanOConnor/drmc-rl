# VS Opponent-Board Observability

Opponent-board observability for VS self-play (top-ranked Elo item in
`docs/ARCHITECTURE_REVIEW_2026-06.md`). Everything is observation plumbing on
top of state the native vspool already exports per side (`board_bytes`,
`viruses_rem`, `garbage_pending`, `pill_colors`, `preview_colors`); no engine
changes.

Default OFF: the stock `bitplane_bottle_conn_mask` 12-channel layout and all
existing checkpoints/tests are unaffected.

## Gate

Two config knobs (independent; flip both for the full feature):

```yaml
env:
  state_repr: bitplane_bottle_conn_mask_vs   # opponent board planes
smdp_ppo:
  candidate_board_channels: 16               # own 8 + opponent 8
  aux_spec: "v1_vs"                          # opponent scalars
```

## Plane layout (`bitplane_bottle_conn_mask_vs`, 20 channels)

| channels | content |
| --- | --- |
| 0–7 | own bottle, unchanged: `color_{red,yellow,blue}`, `virus_mask`, `connected_{up,down,left,right}` |
| 8–15 | opponent bottle, same scheme: `opp_color_*`, `opp_virus_mask`, `opp_connected_*` |
| 16–19 | feasible mask planes `feasible_o0..o3` (own pill) |

Notes:

- Side i's opponent planes are exactly side i^1's own planes at the same env
  step, post symmetry reduction — i.e. the opponent board as the opponent
  observes it, including the channel-6/7 zeroing quirk for same-color pills.
  The quirk on the own planes is untouched (1P champion obs parity).
- Feasible planes keep their pre-reduction superset semantics, just moved
  from 8–11 to 16–19.
- The candidate net reads a contiguous `board_channels` prefix, so old
  12-channel nets (`board_channels=8`) run unchanged on 20-channel obs;
  mixed-arch frozen-opponent pools keep working.

## Opponent scalars (`aux_spec: v1_vs`, 72 = 57 + 15)

The first 57 features are bit-identical to `v1`. Appended block (all zeros in
1P envs, where the `vs/*` info keys are absent):

| dims | feature | source info key |
| --- | --- | --- |
| 1 | opponent virus count / 84 | `vs/opponent_viruses_remaining` |
| 1 | garbage pending against me / 4 | `vs/garbage_pending` |
| 1 | garbage pending against opponent / 4 | `vs/garbage_pending_opp` |
| 6 | opponent current pill one-hot (2 halves × R,Y,B) | `vs/opponent_pill_colors` |
| 6 | opponent preview pill one-hot | `vs/opponent_preview_colors` |

## Checkpoint surgery (warm start from the 12-channel champion)

```bash
python -m tools.expand_checkpoint \
    --in  runs/best_agents/smdp_ppo_step535164979.pt.gz \
    --out runs/best_agents/smdp_ppo_step535164979_vsobs.pt.gz \
    --config <new run config with candidate_board_channels: 16, aux_spec: v1_vs>
```

This zero-initializes the stem-conv slices for the new board channels (coord
channels are relocated after the board prefix) and the `aux_encoder` columns
for the new scalars, in both `state_dict` and `ema_state_dict`. Verified
bitwise-identical logits/values vs the original net on random old-shape obs
padded with random new-channel content (`tests/test_vs_opponent_obs.py`, plus
the real champion checkpoint at d192/4-block scale). The optimizer state is
dropped (shapes change); start the new run with `resume_optimizer: false` and
the embedded `cfg.smdp_ppo` is rewritten so the opponent pool / eval tools
rebuild the expanded arch correctly.
