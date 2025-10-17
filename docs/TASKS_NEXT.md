# Tasks Next (Agent Checklist)

- Stable-Retro binding in `envs/retro/drmario_env.py` (pixel capture + 4-frame stack). Temporary state-mode via RAM→state using `envs/specs/ram_map.py`.
- Wire evaluator-based shaping: add `use_potential_shaping` and `kappa` flags; compute r_shape = γΦ(s') − Φ(s); log r_env, r_shape, r_total.
- PPO bootstrapping: use evaluator mean/quantile/CVaR as n-step bootstrap target in Sample Factory.
- Action-time planning (optional): one-step lookahead scoring via evaluator; toggle with `plan_with_evaluator`.
- Evaluator training: use `models/evaluator/train_qr.py` skeleton; train on RAM-labeled corpus.
- Seed sweeps: extend `eval/harness/seed_sweep.py` and write Parquet/CSV; add risk-conditioned runs at τ∈{0.25,0.5,0.75}.
- Tests: run `pytest`; keep `tests/test_reward_shaping.py`; add determinism tests once RNG is mapped.
