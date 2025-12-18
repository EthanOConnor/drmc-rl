# Reward Shaping & Evaluator Integration (Dr. Mario RL)

## 0) Base Environment Reward (`r_env`)

`DrMarioRetroEnv` computes a per-frame reward that is reported in `info` as:

- `r_env`: the “native” environment reward (configured via `RewardConfig`)
- `r_shape`: optional potential-based shaping (evaluator-driven)
- `r_total = r_env + r_shape`: the reward returned by `env.step()`

For macro-action training (`DrMarioPlacementEnv`), each macro step returns the
**sum of `r_total` over all emulated frames** consumed by the macro action
(`placements/tau` frames).

### Configuration

The default reward config is loaded from `envs/specs/reward_config.json`.
You can override it via:

- `DrMarioRetroEnv(..., reward_config_path=...)`, or
- the env var `DRMARIO_REWARD_CONFIG=...`

Key terms in `r_env` (see `envs/retro/drmario_env.py`):

- **Pill lock bonus** when the ROM pill spawn counter advances (previous pill locked):
  `pill_place_base` with quadratic growth `pill_place_growth`.
  Optionally adjusted by a “high placement” heuristic:
  `punish_high_placements`, `placement_height_threshold`, `placement_height_penalty_multiplier`.
- **Virus clear reward**: `virus_clear_bonus * delta_v` where `delta_v` is viruses cleared this frame.
- **Non-virus clear reward**: `non_virus_clear_bonus * cleared_non_virus` (heuristic, state-mode only).
- **Adjacency shaping** (optional): `adjacency_pair_bonus` / `adjacency_triplet_bonus`.
- **Column height penalty** (optional): a dense penalty based on tallest stack vs virus band,
  applied as a delta each frame via `column_height_penalty`.
- **Action penalty** (optional): `action_penalty_scale * action_events` subtracted each frame.
- **Terminal bonuses/penalties**:
  - `terminal_clear_bonus` on stage clear
  - `topout_penalty` on top-out
  - time-to-clear shaping at episode end:
    `time_penalty_clear_per_60_frames` (clear) and `time_bonus_topout_per_60_frames` (top-out)

**Goal:** Use the position evaluator (predicting a distribution of time-to-clear, `T_clear`) to provide dense, low-latency learning signals while preserving the optimal policy.

## 1) Potential-Based Reward Shaping (safe)

Let Φ(s) be a potential over states. If we add shaping reward

> r_shape(s, a, s') = γ Φ(s') − Φ(s)

to the environment reward r_env(s, a, s'), the optimal policy is preserved (Ng, Harada, Russell 1999).

Choice of potential: Φ(s) = − E[T_clear | s] / κ

- E[T_clear | s] is the evaluator’s mean predicted time to clear from state s.
- κ is a positive scale (e.g., 100–500) to keep shaping magnitudes reasonable.
- With this potential, r_shape is ≈ (decrease in expected remaining time), which aligns with speed-running.

Implementation:
1. Train the evaluator on a corpus of (state → Monte Carlo samples of T_clear), using QR/IQN.
2. In the Gym env/wrapper:
   - On each step, compute Φ(s) and Φ(s') via the evaluator (no gradient from policy to evaluator; treat evaluator as fixed for shaping).
   - r_total = r_env + r_shape.
3. Start with small shaping (κ=250) to avoid drowning out terminal rewards.

## 2) Bootstrapped Returns with Evaluator

Use the evaluator prediction as a value bootstrap for n-step returns in PPO:

> G_t^(n) = Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n V_eval(s_{t+n})

- V_eval(s) can be the mean (speedrun) or a quantile/CVaR for risk-aware policy.
- This reduces the need for long rollouts while maintaining correct credit assignment.

## 3) Risk-Aware Action Selection (inference-time planning)

At action time, you may perform a one-step lookahead:
1. For each candidate action a, simulate one step to get s'.
2. Score s' using the evaluator at a chosen risk level (mean / quantile τ / CVaR_α).
3. Select the action minimizing the chosen statistic of T_clear.

## 4) Practical Tips

- Update Φ(s) on a target-network schedule to reduce non-stationarity.
- Start training without shaping; once a baseline clears levels, enable shaping gradually.
- Log r_env, r_shape, and r_total histograms to ensure shaping isn’t overpowering.
- For evaluator training, censor very long episodes at T_max and flag censored targets.
