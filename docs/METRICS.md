# Training & Evaluation Metrics

- E[T]: mean frames to clear (per seed and aggregate)
- Var[T]: variance of frames to clear
- CVaR_α(T): tail risk at α ∈ {5%, 25%}
- Success rate: clears per N episodes
- Shaping metrics: mean r_env, r_shape, r_total; ratio |r_shape|/|r_env|
- FPS: steps/sec across actors
- Risk profiles: compare τ ∈ {0.25, 0.5, 0.75}
