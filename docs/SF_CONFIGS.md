# Sample Factory Configs

These YAML files map high-level training choices to environment args.

- `env: DrMarioRetroEnv-v0` — registered via `envs/retro/register_env.py`.
- `obs_mode: state|pixel` — forwarded to the env constructor.
- `frame_stack: 4` — keep consistent with the env’s internal stacking.
- Placement wrapper toggles live under `env.kwargs`: `use_placement_wrapper: true` enables the translator.
- Risk conditioning:
  - `risk_tau_train_low`/`high`: set the range for τ sampling during training (if used by your pipeline).

Launch examples:
- State mode (SF1 launcher):
  ```bash
  python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml
  ```
- Pixel mode (SF2 CLI):
  ```bash
  python -m sample_factory.launcher.run --run training/sf_configs/pixel_baseline_sf2.yaml
  ```
- Distributional evaluator bootstrap:
  ```bash
  python training/run_sf.py --cfg training/sf_configs/state_eval_bootstrap.yaml --timeout 3600
  ```

Adjust `num_workers`/`num_envs_per_worker` to match your CPU cores/GPU throughput. Set `DRMARIO_BACKEND` and core/ROM paths in
your environment before launching.
