This folder contains launch scripts or notes for running Sample Factory or other trainers on local or cluster setups.

Examples:
- Local: `python -m sample_factory.launcher.run --cfg training/sf_configs/state_baseline.yaml`
- Cluster: SLURM/Ray scripts can be added here as needed.

Note: These configs expect the `DrMarioRetroEnv` entrypoint to be registered with the chosen framework. Add the appropriate env factory hooks in your training script.
