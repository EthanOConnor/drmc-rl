# League Roles for VS Self-Play

League training extends the frozen-opponent pool with explicit roles:
exploiter agents vs the champion, not just PFSP over the run's own history.
Config-gated under `env.opponent_pool.league`; default off — the stock PFSP
pool (`drmc_rl/training/envs/vs_opponents.py`) is unchanged.

## Config

```yaml
env:
  opponent_pool:
    enabled: true
    league:
      mode: exploiter          # pfsp (default) | exploiter | mixed
      main_agents:             # fixed target checkpoints
        - runs/best_agents/vs_champion_smdp_ppo_step530046434.pt.gz
      exploiter_fraction: 0.3  # mixed only
```

## Modes

- `pfsp` — today's behavior: PFSP over frozen snapshots of the learner's own
  history (EMA snapshot every `snapshot_every_matches`).
- `exploiter` — the learner trains exclusively against `main_agents` and never
  snapshots itself into the pool. With multiple targets, PFSP weighting
  (`(p(1-p))² + 0.05` over per-target win rates) hammers the targets the
  learner is closest to cracking. Trains an exploiter that finds a fixed
  champion's weaknesses.
- `mixed` — league depth for a main-agent run: `exploiter_fraction` of pair
  assignments sample from `main_agents`, the rest from the normal PFSP
  self-history pool (snapshots continue as usual).

## Mechanics

- Targets are copied into the pool dir, flagged `league_target` in
  `manifest.json`, and protected from eviction. Per-target win/game counts
  live in the manifest, so they persist across restarts; re-seeding on
  restart is idempotent (matched by checkpoint filename).
- Mixed-arch targets (8-channel vs2, 12-channel, 16-channel vs3) work
  unchanged: each entry rebuilds its net from the checkpoint's embedded cfg
  with its own `aux_spec`/`candidate_max` (`OpponentPool.ensure_loaded`).

## Metrics

League modes add to `get_vs_metrics()` (dashboard: `tools/vs_dashboard.py`
shows a "league wr (targets)" row; existing keys unchanged):

- `vs/league_targets` — number of fixed targets in the pool.
- `vs/league_win_rate` — cumulative learner win rate pooled over all targets.
- `vs/league_win_rate_min` / `vs/league_win_rate_max` — per-target extremes.
- `vs/league_wr_<target_id>` — per-target win rate.

## Tests

`tests/test_vs_league.py`: config validation, exploiter never samples self
snapshots, mixed fraction statistics, PFSP weighting over targets, manifest
round-trip, and an exploiter smoke vs the real champion checkpoint on the
native vspool.
