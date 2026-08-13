# Search-Amplified Training Targets (Gumbel-AZ-lite)

Use depth-2 checkpoint-reset search (`drmc_rl/models/policy/search_policy.py`, measured 86.7%
head-to-head win rate over the same weights plain) to produce improved
training targets, so the net banks the search gap at zero inference cost.

## Phase plan

- **Phase 1: distillation alongside PPO.** The executed action is still
  sampled from the behavior policy — PPO's on-policy math is untouched. For a
  sampled fraction of decisions, search produces an improved policy target
  and a search-consistent value; they enter the loss as

  `L = L_PPO + beta * mean_searched KL(pi_target || pi_net)`

  plus (optional, `value_mix > 0`) blending the search value into the value
  regression target of searched decisions.
- **Phase 2 (implemented, all config-gated, defaults = phase-1 behavior):**
  cross-env batched search, opponent-board-obs (`_vs`) search support, an
  optional opponent self-model at ply-2 leaves, and `act_from_search` (execute
  the improved policy's sample at searched decisions). See the phase-2 section
  below.

## Mechanics

Code: `drmc_rl/training/algo/search_distill.py` (config, target math, rollout runner),
wired into `drmc_rl/training/algo/ppo_smdp.py`; per-decision storage in
`drmc_rl/training/rollout/decision_buffer.py`.

- **Rollout.** Each vectorized decision, a Bernoulli(`decision_fraction`)
  sample (per env; sampled, not strided, to avoid phase artifacts) picks the
  searched decisions. They are searched in one `SearchPolicy.decide_batch`
  call on the envs' `info["board"]` with the stored feasible mask /
  cost-to-lock, no wall-clock deadline, and a sim budget of `sims` native pool
  envs per request (ply-1 beam `beam`, ply-2 fan-out `sims // beam` per
  surviving branch). The search net is a copy of the **current** trainer net,
  refreshed after every PPO update (`SearchPolicy.refresh_weights`); the
  batched stage forwards and the 81-combo marginal leaf batch run on the
  trainer device (MPS).
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
  `search_distill/search_ms_p50` (amortized per searched decision), and —
  with `opponent_model: self` — `search_distill/opp_advanced_fraction`.

## Phase 2

### Cross-env batching (`SearchPolicy.decide_batch`)

Phase 1 searched envs serially (one `decide()` per searched env). Phase 2
batches every searched env of a vectorized decision through
`SearchPolicy.decide_batch`: native sims still run per request on the shared
pool (CPU, ~30µs/env-step), but the four net-eval stages — root priors, ply-2
(depth-1) priors, the 81-combo leaf marginalization, and the optional
opponent-model forward — are concatenated across requests and run on the
trainer device, so MPS sees one large batch per stage instead of S small
ones. `decide_batch` has no wall-clock deadline (the sim budget bounds work)
and matches serial `decide()` per request up to float association
(`tests/test_search_distill_phase2.py::test_decide_batch_matches_serial`).

### Opponent-board observability (`_vs` repr)

The 20-channel `bitplane_bottle_conn_mask_vs` observation with
`aux_spec: v1_vs` is searchable. The sims remain 1P (the
learner's own board); the obs handed to the net during search is assembled by
**splicing a frozen opponent context** around the sim planes
(`drmc_rl/models/policy/search_policy.ObsContext` / `splice_obs_context`):

- opponent planes (live obs ch 8..15) and the 15 v1_vs aux-tail scalars are
  captured from the live decision's stored obs/aux;
- every node evaluation uses sim own-board planes (ch 0..7) + frozen opponent
  planes (ch 8..15) + sim feasible planes (ch 16..19), and sim-built v1 aux +
  frozen tail.

This is **exact at ply 1** (the root state is the live state) and a
**frozen-opponent approximation at ply 2** (the opponent has actually placed
~1 pill by then; garbage arrival between plies is likewise ignored, as in
phase 1).

### Opponent model at leaves (`opponent_model: self`, default `none`)

Instead of freezing the opponent planes at ply-2 leaves, advance the opponent
board by ONE placement: at the searched decision, checkpoint-reset a sim env
to the opponent's board/pill/preview (from `info["vs/opponent_board"]` etc.),
evaluate the net (same weights — self-model) on the opponent's **mirrored**
obs (own planes = opponent board sim planes, opponent planes = the learner's
live planes, mirrored aux tail), apply its argmax placement, and use the
resulting planes (+ updated tail: opponent virus count from the sim, opponent
pill = its old preview, opponent preview unknown → zeros, garbage pendings
frozen) for all leaf evaluations. Computed once per searched decision — one
extra pool reset + step + one batched forward per decision batch, independent
of leaf count. Fallbacks (no feasible opponent placement, rejected action, or
the placement ends the opponent's game) keep the frozen context;
`search_distill/opp_advanced_fraction` reports the advance rate.

Known approximations (documented divergences): the opponent is assumed to be
at a fresh spawn of its current pill (it may really be mid-fall); ply-1 evals
keep the frozen (un-advanced) context; garbage the opponent's placement would
send is not applied to the learner's sim boards; the advance is exactly one
placement even though the learner looks two plies ahead.

### Act from search (`act_from_search: false`)

When true, searched decisions **execute** a sample from the improved policy
`pi_target` instead of the behavior sample, and the stored behavior log-prob
is `log pi_target(a)` — i.e. the behavior policy at searched decisions IS the
improved policy, so PPO's importance ratio is computed against the
distribution actually sampled from (ratio against the stored behavior at
step 0 is exactly 1; across optimizer steps the ratio measures
`pi_net / pi_target`, an off-policy correction toward the search policy).
Unsearched decisions are unchanged. Enabling this shifts the algorithm toward
full Gumbel-AZ: raise `beta` / `decision_fraction` as it proves out.

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
    opponent_model: none  # none|self — self advances the opponent board one
                          # placement for ply-2 leaf contexts (_vs repr only)
    act_from_search: false  # execute improved-policy samples (phase-2 AZ)
```

Recommended start: the defaults above. `sigma_scale` is the knob to raise
(4–16) if `kl_target_net` is so small the term does nothing; `beta` to lower
if the policy entropy collapses. `decision_fraction` trades target coverage
against throughput linearly.

## Approximations / restrictions

- **VS:** search sims use the **single-player pool for the learner's own
  board** — garbage arrival between decisions is ignored at depth 2 (exactly
  what inference-time `SearchPolicy` does), and opponent progress is ignored
  unless `opponent_model: self` (one-placement advance, leaves only). The VS
  reward is replicated as garbage shaping (volley estimate from cleared lines)
  with training-consistent terminal values ±1 (`win_value`/`loss_value`).
- Requires `policy_type: candidate` over the 12-channel
  `bitplane_bottle_conn_mask` obs or the 20-channel
  `bitplane_bottle_conn_mask_vs` obs (`aux_spec` in `{none, v1, v1_vs}`); the
  opponent context of the `_vs` stack is frozen-spliced per searched decision
  (phase-2 section above). `opponent_model: self` additionally requires the
  20-channel obs and the VS env (enable-time config errors otherwise).
- `speed_ups` for checkpoint resets is approximated as
  `max(0, level-20) + decisions_in_episode // 10` (same as eval/tournament
  tooling). Synthetic curriculum levels < 0 clamp to 0 inside the sim.
- 1P reward replication uses the global reward config
  (`drmc_rl/training/envs/drmario_pool_vec._RewardCfg.load()`); per-run reward
  overrides that bypass it would skew Q magnitudes (ranking is usually
  preserved).

## Measured cost (M-series Mac, 16 envs, defaults: sims=12 beam=8 fraction=0.25)

Phase 1 (quiet machine, 512 decisions):

| net | search | rollout dec/s | p50 search ms |
| --- | --- | --- | --- |
| tiny (d32, tests) | off | ~1740 | — |
| tiny (d32, tests) | on | ~430 | ~8 |
| production (d192/4-block, MPS leaves) | off | ~260 | — |
| production (d192/4-block, MPS leaves) | on | ~83 | ~39 |

Phase 2 (same config, d192/4-block, MPS; measured 2026-06-12 on a heavily
loaded machine — absolute numbers depressed, the serial/batched comparison is
same-session):

| variant | rollout dec/s | amortized ms/searched decision |
| --- | --- | --- |
| search off | ~730 | — |
| phase-1 serial (`decide()` per env) | ~47 | ~46 |
| phase-2 batched (`decide_batch`) | ~70–96 | ~15–16 (controlled S=4 microbench) |

Controlled microbench (same 4 requests repeated, d192, MPS): batched
65 ms/4-request batch vs serial 185 ms — **~2.8x per searched decision,
~2x rollout throughput at fraction 0.25**. Stage split of the batched 65 ms:
~30 ms device forwards (root + ply-2 priors), ~11 ms leaf marginalization,
~18 ms native sims (per-request, irreducibly serial).

**Opponent-model overhead** (`opponent_model: self`, vs net d192, S=4):
+3.6 ms/decision batch ≈ **+0.9 ms (+5%) per searched decision** (one extra
pool reset+step plus one batched forward; 40/40 advances succeeded on
mid-game boards). `act_from_search` adds only a per-row categorical sample
(unmeasurable).

Update time is unchanged. Scale `decision_fraction` down (or `sims`) if the
run becomes rollout-bound.

## Tests

`tests/test_search_distill.py`: improved-policy math (monotonicity, masking,
normalization, baseline fill, sigma=0 / flat-Q degeneracy), KL masking,
value blending, config validation, flag-off bit-identity regression guards
(no-key vs `enabled:false` vs zero-searched batches), and end-to-end
rollout+update smokes on the 1P pool env and the VS pool env.

`tests/test_search_distill_phase2.py`: obs/aux splice bit-exactness,
batched-vs-serial search equivalence (full ply-2 coverage boards make results
rng-independent), frozen-context flow into net evals, opponent self-model
advance + no-feasible fallback, act-from-search log-prob bookkeeping (ratio 1
against the stored behavior), and rollout+update smokes on the VS env with
the `_vs` repr (base / `opponent_model: self` / `act_from_search: true`).
