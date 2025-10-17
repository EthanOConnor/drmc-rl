# Reward Shaping & Evaluator Integration (Dr. Mario RL)

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
