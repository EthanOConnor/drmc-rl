# Known risks

- **Planner assurance:** v4 exactness is enforced by oracle parity and fuzzing,
  not a formal proof. Keep `drm_reach_bfs_full` independent.
- **Warp timing:** relative training behavior is well tested, but absolute
  seedlab records require controller-script or emulator audits.
- **VS objective:** self-play can optimize for mutual ceiling attrition. Require
  clear-win and human-opponent gates before interpreting Elo gains.
- **Search approximation:** current VS search simulates the learner board and
  does not model simultaneous opponent decisions.
- **Checkpoint state:** curriculum scheduler state is not yet restored with the
  policy checkpoint.
- **Configuration:** staged configs contain deliberate checkpoint placeholders.
  Configuration parsing rejects unknown environment keys to prevent silent
  experiment drift.
