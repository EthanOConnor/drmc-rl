# Sample Factory Configs

These YAML files map high-level training choices to environment args.

- `env: DrMarioRetroEnv-v0` — Registered via `envs/retro/register_env.py`.
- `obs_mode: state|pixel` — Forwarded to env constructor.
- `frame_stack: 4` — Keep consistent with env’s internal stacking.
- Risk conditioning
  - `risk_tau_train_low`/`high`: set the range for τ sampling during training (if used by your pipeline).

Launch examples:
- State mode: `python training/run_sf.py --cfg training/sf_configs/state_baseline_sf2.yaml`
- Pixel mode: `python training/run_sf.py --cfg training/sf_configs/pixel_baseline_sf2.yaml`

Adjust `num_workers`/`num_envs_per_worker` to match your CPU cores.
