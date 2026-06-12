# Search-Amplified Training Targets (Gumbel-AZ-lite)

Item 2 of `docs/ARCHITECTURE_REVIEW_2026-06.md`: use the depth-2
checkpoint-reset search (`models/policy/search_policy.py`, measured 86.7%
head-to-head win rate over the same weights plain) to produce improved
training targets, so the net banks the search gap at zero inference cost.

## Phase plan

- **Phase 1 (this implementation): distillation alongside PPO.** The executed
  action is still sampled from the behavior policy — PPO's on-policy math is
  untouched. For a sampled fraction of decisions, search produces an improved
  policy target and a search-consistent value; they enter the loss as

  `L = L_PPO + beta * mean_searched KL(pi_target || pi_net)`

  plus (optional, `value_mix > 0`) blending the search value into the value
  regression target of searched decisions.
- **Phase 2 (not implemented): act from search.** Sample executed actions from
  the improved policy and shift loss weight from the PPO term to the
  distillation term as phase 1 proves out. Cross-env batching of the search
  net evals belongs here too (phase 1 searches decisions sequentially).

## Mechanics (phase 1)

Code: `training/algo/search_distill.py` (config, target math, rollout runner),
wired into `training/algo/ppo_smdp.py`; per-decision storage in
`training/rollout/decision_buffer.py`.

- **Rollout.** Each vectorized decision, a Bernoulli(`decision_fraction`)
  sample (per env; sampled, not strided, to avoid phase artifacts) picks the
  searched decisions. For each, `SearchPolicy.decide()` runs on the env's
  `info["board"]` with the stored feasible mask / cost-to-lock, an effectively
  infinite deadline, and a sim budget of `sims` native pool envs (ply-1 beam
  `beam`, ply-2 fan-out `sims // beam` per surviving branch). The search net is
  a copy of the **current** trainer net, refreshed after every PPO update
  (`SearchPolicy.refresh_weights`); root/prior forwards run on CPU, the
  81-combo marginal leaf batch runs on the trainer device (MPS).
- **Improved policy (Gumbel-MuZero-style completed Q).** Over the packed
  candidate slots (identical deterministic packing as the PPO update):
  evaluated candidates keep their search-backup
  `Q = r̂1 + γ^τ1·(r̂2 + γ^τ2·V)` (the training-consistent backup — verified
  necessary; pure value-head backup is anti-clear); unevaluated candidates take
  the net's root value as the value-equivalent baseline. Q is normalized to
  [0, 1] within the decision and
  `improved_logits = prior_logits + sigma_scale * q_norm`;
  `pi_target = softmax` over feasible slots.
- **Value.** `v_search = Σ_a pi_target(a) · Q_completed(a)`; with
  `value_mix = m`, searched decisions regress against
  `(1-m)·G_t + m·v_search`.
- **Loss/metrics.** KL is computed only on searched rows (zero contribution
  otherwise), accumulated on-device, one host sync per update
  (`_UPDATE_METRIC_KEYS` discipline). Emitted scalars:
  `search_distill/searched_fraction_actual`, `search_distill/kl_target_net`,
  `search_distill/q_gap_mean` (mean over searched decisions of
  `max_a Q_completed − Q_completed(executed)`),
  `search_distill/search_ms_p50`.

## Config

```yaml
smdp_ppo:
  search_distill:
    enabled: false        # default OFF; flag-off path is bit-identical
    sims: 12              # native sim envs per searched decision
    beam: 8               # ply-1 beam width K
    decision_fraction: 0.25
    beta: 1.0             # KL term weight
    value_mix: 0.5        # 0 disables value-target blending
    sigma_scale: 1.0      # c in improved_logits = prior + c * norm(Q)
```

Recommended start: the defaults above. `sigma_scale` is the knob to raise
(4–16) if `kl_target_net` is so small the term does nothing; `beta` to lower
if the policy entropy collapses. `decision_fraction` trades target coverage
against throughput linearly.

## Approximations / restrictions

- **VS:** search sims use the **single-player pool for the learner's own
  board** — garbage arrival between decisions and opponent progress are
  ignored at depth 2 (exactly what inference-time `SearchPolicy` does). The VS
  reward is replicated as garbage shaping (volley estimate from cleared lines)
  with training-consistent terminal values ±1 (`win_value`/`loss_value`).
- Requires `policy_type: candidate` and the 12-channel
  `bitplane_bottle_conn_mask` obs with `aux_spec` in `{none, v1}`. The
  opponent-obs stack (`*_vs`, `aux_spec: v1_vs`) is **not** searchable yet —
  the sim pool cannot provide opponent context (enable-time config error).
- `speed_ups` for checkpoint resets is approximated as
  `max(0, level-20) + decisions_in_episode // 10` (same as eval/tournament
  tooling). Synthetic curriculum levels < 0 clamp to 0 inside the sim.
- 1P reward replication uses the global reward config
  (`training/envs/drmario_pool_vec._RewardCfg.load()`); per-run reward
  overrides that bypass it would skew Q magnitudes (ranking is usually
  preserved).

## Measured cost (M-series Mac, 16 envs, 512 decisions, defaults: sims=12 beam=8 fraction=0.25)

| net | search | rollout dec/s | p50 search ms |
| --- | --- | --- | --- |
| tiny (d32, tests) | off | ~1740 | — |
| tiny (d32, tests) | on | ~430 | ~8 |
| production (d192/4-block, MPS leaves) | off | ~260 | — |
| production (d192/4-block, MPS leaves) | on | ~83 | ~39 |

Rollout collection is ~3–4x slower at fraction 0.25 (amortized ≈
`fraction × p50_ms` per decision; search is serial across envs in phase 1).
Update time is unchanged. Scale `decision_fraction` down (or `sims`) if the
run becomes rollout-bound.

## Tests

`tests/test_search_distill.py`: improved-policy math (monotonicity, masking,
normalization, baseline fill, sigma=0 / flat-Q degeneracy), KL masking,
value blending, config validation, flag-off bit-identity regression guards
(no-key vs `enabled:false` vs zero-searched batches), and end-to-end
rollout+update smokes on the 1P pool env and the VS pool env.
