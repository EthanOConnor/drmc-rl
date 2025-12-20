# Training & Evaluation Metrics

- E[T]: mean frames to clear (per seed and aggregate)
- Var[T]: variance of frames to clear
- CVaR_α(T): tail risk at α ∈ {5%, 25%}
- Success rate: clears per N episodes
- Shaping metrics: mean r_env, r_shape, r_total; ratio |r_shape|/|r_env|
- Throughput:
  - Frames/sec: `sum(placements/tau)` per second (macro env “true FPS”)
  - Decisions/sec: macro decisions per second (one per spawn/placement)
- Risk profiles: compare τ ∈ {0.25, 0.5, 0.75}
- Curriculum diagnostics:
  - `curriculum/confidence_lower_bound` vs `curriculum/success_threshold`
  - Time-goal tightening: `curriculum/time_budget_frames`, `curriculum/time_budget_spawns`, `curriculum/time_k`
