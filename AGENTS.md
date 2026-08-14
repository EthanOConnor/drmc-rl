# Repository guidance

- Supported learning paths are placement-level SMDP PPO over the native pools
  and measured offline afterstate learning from `HumanCorpus`. Keep one stack
  per objective; do not add framework-level trainer duplication.
- Preserve the policy/planner contract: the policy selects a feasible final
  pose and the planner supplies timing or an executable script.
- `drm_reach_bfs_full` is the planner oracle. Changes to v4 or CUDA planning
  require parity tests.
- `vendor/drmario_native/` is the `drmario-native` submodule; commit engine work
  in the standalone repo first, then update its pinned revision here.
- Libretro is an independent verification path, not the training backend.
- Keep changes small, deterministic, and covered by focused tests. Do not commit
  ROMs, cores, datasets, checkpoints, or run outputs.
- Update current docs when a contract changes. Do not add session logs,
  historical retrospectives, or archived plans to this repository.
