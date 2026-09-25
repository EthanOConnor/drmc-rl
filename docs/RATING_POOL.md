# Rating pool

`tools.rating_pool` runs one continuously running, anchored rating arena for every
drmc-rl and Professor Pills computer player. Workers join and leave freely; the
coordinator fills them with seed-paired, side-swapped batches from a priority
list, and every rating is recomputable from the game journal alone.

It extends the distributed study transport (`tools.trainer_arena_distributed`,
docs/ARENA_HOSTS.md): the same bearer-token HTTP, checkpoints fetched by SHA-256,
exact source-revision pinning, numerics classes with `trust`, calibration replays
and sampled replicate audits with tolerant/strict fidelity, SIGTERM release and
idempotent uploads. Games are played by the same `ArenaRuntime.play` call as the
single-host arena, so a pool game is an arena game.

## Design

| Piece | File | Rule |
|---|---|---|
| Entrants | `drmc_rl/pool/store.py` | id, checkpoint (and adapter) by sha256 with known paths, loader (`plain`, `pace_adapter`), player settings, era, lineage (parent, recipe, run, step), status (`active` scheduled; `benched` jobs only; `retired` rated, never scheduled), tags, notes |
| Conditions | `drmc_rl/pool/conditions.py` | key = backend, engine commit, level, speed, pace + execution-profile key, decision contract (delay, decision point, early preview, preview input), movement |
| Ratings | `drmc_rl/pool/ratings.py` | anchored Bradley–Terry per condition, MAP + Laplace, widened to seed-pair sandwich variance |
| Scheduler | `drmc_rl/pool/scheduler.py` | priority list below |
| Intentions | `drmc_rl/pool/intentions.py` | planned experiments and their dependencies; auto-submits their job |
| Coordinator | `drmc_rl/pool/coordinator.py` | torch-free; leases, journal, fidelity, refits |
| Worker | `drmc_rl/pool/worker.py` | ArenaRuntime per backend, policy LRU, host budget, disk guard |
| Reports | `drmc_rl/pool/report.py` | JSON export, CLI summary, LAN page, stop rule |

### Journal (crash-safe, replayable)

The data directory holds `pool.json` (settings), `games.jsonl` (one compact row per
game, the only rating input), `registry.jsonl` (typed events: entrant, condition,
condition_set, job, intention, admission, block, unblock), `batches.jsonl`,
`audit.jsonl`, sampled `traces/`, content-addressed `artifacts/` and `spool/`
(audit references). Every file is append-only JSON lines written with one
`write` + `fsync`; a torn last line is truncated on open. A game's id is
`condition/a/b/seed/side` with `a < b`, so resubmissions and re-imports are
idempotent. Restarting the coordinator replays the files; workers ride out the
restart (their submissions carry the batch spec).

Move traces are kept for one seed pair in `trace_every` (16) up to
`trace_cap_mb` (2 GB); rows are about 400 bytes. Calibration uses traced games.

### Conditions and comparability

Both entrants of a pool game play under one condition. Settings that change wall
clock or numerics but not the game (device, threads, planner workers, memoization,
asynchronous planning, max game frames) are runtime settings, not part of the key;
replicate audits cover them. Studies whose two sides played different decision
contracts or movement models, or used diagnostic ablations (anticipation,
own-board-only, context/planning pace), are not imported: those are not symmetric
games under one condition. Nothing mixes silently: a different engine commit, pace
profile or backend is a different key.

A **condition set** names conditions, one anchor, a background weight and whether
it is primary. The pooled view of a set is the pace-weighted mean of an entrant's
per-condition ratings (below), only for entrants rated under every condition of
the set (no imputation). Sets registered at bootstrap
(`runs/rating-pool-v1/bootstrap.json`):

| Set | Conditions | Weight |
|---|---|---|
| `l14-spawn` (primary) | events, engine 19f292c, level 14 HI, spawn decision delay 4, exact movement, 7 paces (`ev-L14-<pace>-spawn4`) — the afterstate panel/tournament conditions | 1.0 |
| `l20-spawn` | the same at level 20 | 0.3 |
| `l14-shipped` | frames, lock_safe early decision, repeat preview (`fr-L14-<pace>-lock_safe4-repeat`) — the shipped v17/v18 contract; needs public-context actors | 0.3 |
| `l14-human-movement` | events, spawn 4, human movement, 6 human paces — blocked until the pool engine includes the movement model | 0.2 |

### Anchor

`champion-retention-mixed-v2` (sha256 `fae04a59…`) is fixed at **1500** in every
set. It is the shipped champion and immutable, it plays under every condition
(spawn and lock_safe, all paces, both levels), it is the opponent of almost every
pre-registered panel, tournament and guard (so imported history connects to it
directly), and it sits in the middle of the current strength range, where games
against it carry the most information about new entrants.

### Rating model

Per condition: P(i beats j) = sigmoid(θi − θj), θanchor = 0, θ ~ N(0, 4²) (only
keeps separated records finite). A draw scores ½. The fit is the posterior mode by
damped Newton (deterministic). Uncertainty is the Laplace curvature, widened per
entrant to the cluster-robust sandwich variance with each side-swapped seed pair
as a cluster, so correlated pairs are not over-counted. Only complete seed pairs
enter: a pair with a timed-out game is dropped whole, and games from a numerics
class that failed an audit leave the ratings. Entrants not connected to the anchor
under a condition are reported unanchored. Display scale: 400/ln 10 per logit.
`tools.rating_pool rate --data COPY` recomputes everything offline from the files.

### Seeds

Three fixed seed sets, persisted in `pool.json` when first drawn (so a refreshed
frequency table never changes them), each played in order, both sides of a seed
on one worker, reused across pairings, conditions and levels but never within a
pairing (at most 1,024 games per pairing, condition and set):

| Set | Seeds | Role |
|---|---|---|
| `reserve` | reserve allocation `rating-pool-v1` (512, games 96–607 of the reserve) | never trainable; the memorization reference |
| `uniform` | 512 non-reserve console seeds, uniform (`draw_mixture_seeds`, seed_mix 0) | the fair "seen" comparison for memorization |
| `mixture` | 512 non-reserve seeds from the training mixture (seed_mix 0.5: half Fightcade play frequency, half uniform) | its plain mean is real-play-weighted strength |

Background pairs rotate among the sets by share (`background_seed_shares`:
mixture 0.5, reserve 0.25, uniform 0.25), always the set furthest below its share.
Jobs with `--seeds bank` use the reserve set; confirmatory studies keep fresh
allocations (`--seeds allocation:STUDY`), and job seeds outside the reserve are
refused. Imported studies keep their own registered seeds.

**Which games the ratings use (settled).** The default ranking uses every
comparable game, whatever its seeds: pool games on all three sets plus the
imported history (which carries most of the anchor connections). This is a
permanent decision (user, 2026-09-25), independent of how many games the mixture
set accumulates. "Real-play seeds" (the same fit on mixture-set games, the
real-play-weighted strength) and "Uniform seen seeds" are secondary views shown
alongside. The one exception: an entrant whose memorization check is flagged is
ranked by its reserve or real-play view instead of the default.

**Memorization check.** Per entrant, over pool games on the reserve and uniform
sets: seen-minus-reserve score gap in points with a seed-clustered 95% interval
(`memorization_report`), pooled over conditions, with the smallest detectable gap
(`detectable_gap`) for the seeds played so far. Flagged when the interval
excludes 0 by more than 2 points.

### Pace weighting

The pooled ranking of a condition set is the pace-weighted mean of its
per-condition ratings (confirmed weights: Frame Perfect 3, Super Human 3, Top
Humans 2, Fast 1.5, Normal 1, Relaxed 0.5, Sloth 0.5; `pace_weights` in
pool.json), se = sqrt(Σ w² se²)/Σw, and LOS uses the same weights with the fitted
covariances. The equal-weight view is shown alongside. The pool stop rule of new
runs uses the weighted view (`stop-rule --equal` for equal weights); runs with a
pre-registered rule keep their own.

### Scheduler priority list

1. Replicate audits, and calibration of a new numerics class.
2. Focused jobs with priority above `background_priority` (10), by priority,
   deadline, submission; within a job the (pairing, condition) furthest below its
   target first.
3. Background fill over the weighted sets: for each active entrant e and candidate
   opponent o (the anchor; once e has games also each era's best and the 4
   nearest-rated), value = weight × boost(e) × p(1−p) × (var_e + var_o) / (1 +
   in-flight leases), with boost = 1 + 8·[games < 64] + 3·max(0, 1 − games/512).
   An unplayed entrant plays the anchor first.
4. Jobs at or below background priority.

The report shows integer Elo with 95% bounds and, per table, the likelihood of
superiority (LOS) of each row over the next: P(rating_i > rating_i+1) under the
normal approximation with the fitted covariance of the two estimates (per
condition from the sandwich-scaled Laplace covariance; pooled, the per-condition
difference variances summed and divided by k²). Each condition set has a per-pace
drill-down: an entrant × pace matrix marking paces whose 95% interval excludes
the pooled rating, and per-pace tables sorted by that pace's rating.

Every 10th lease tries background first (`background_min_share`), so ratings stay
fresh while long jobs run. A pairing has at most 2 leases in flight. A batch that
fails twice with a deterministic error (e.g. a checkpoint without a public
auxiliary contract) blocks the entrant under that condition (never the anchor);
the block and its reason appear in the report.

### Intentions (roadmap)

An intention records a planned experiment before its entrants or engine exist:
hypothesis, entrant patterns, conditions, metrics, decision rule, dependencies
(`intention:ID`, `capability:CAP`, `entrant:PATTERN`, `external:TEXT` — met when
listed in `resolved`), status (`planned`, `blocked`, `running`, `done`,
`dropped`), owner, due date and an optional job spec. When everything is met the
coordinator submits the job (id = intention id) and marks it running; people mark
done or dropped. The report lists planned, blocked (with what each waits on),
running and overdue items. Seeded from `runs/rating-pool-v1/intentions.json`.

### Lineages (training runs)

Snapshots with the same `lineage.run` (the `watch-run --run` id) form a lineage.
While the run is active every snapshot stays active and visible, but the lineage
shares one entrant's background budget: only the newest snapshot gets the
new-entrant and underplayed boosts and serves as an opponent for others; each
older snapshot gets a maintenance share (`maintenance_share` 0.1) until its
per-condition 95% half-width is below `maintenance_ci` (75 Elo, about ±28 pooled
over seven paces), then only occasional games (`maintenance_idle` 0.01). Focused
jobs are unaffected.

A run concludes with `lineage conclude RUN [--best E]`, when a lineage that opted
in with `lineage set RUN --stop-set SET --step-every N --auto` (or `watch-run
--auto-conclude`) has its pool stop rule fire, or when `watch-run --final-marker
GLOB` sees the run's final marker. Its best snapshot (stop-rule selection) and its
final snapshot stay active; intermediates become retired (rated, never scheduled).

Reports show one row per lineage: the best snapshot once concluded, otherwise the
newest rated one, labelled `run · frames`, with an expandable trajectory (every
snapshot's rating, CI and games against frames, retired ones included) and a
sparkline. In the per-pace drill-down a run expands into its snapshots: matrix sub-rows (retired ones muted; ▲/▼ relative to each snapshot's own pooled rating) and, per pace, each snapshot's Elo, CI, games and LOS vs the next snapshot. Expanded rows survive the 60 s refresh. Other entrants are unchanged.

Snapshot cadence and stop rules: runs may snapshot every 25M frames for the
trajectory. Stop-rule panels and decisions stay on their registered marks:
`watch-run --panel-step-every 50000000` limits the panel job to 50M multiples, and
`stop-rule --step-every 50000000` (or the lineage's `step_every`) evaluates only
those snapshots, so the extra snapshots never change a decision.

## Operations

Coordinator: **mombox** (always on, 458 GB free), `~/drmc-rl-pool/{src,venv,data}`,
API `http://192.168.157.190:8097` (bearer token `~/.config/drmc-rl/study-worker.token`),
read-only report `http://192.168.157.190:8098/` (LAN only; not proxied by the site
or Cloudflare). No workers or model inference on mombox: the coordinator imports
no torch. It runs at low priority inside a memory/CPU-capped user scope:

```bash
cd ~/drmc-rl-pool/src && git fetch && git checkout <commit>      # the workers' commit
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=50% nice -n 15 ionice -c3 \
  ~/drmc-rl-pool/venv/bin/python -m tools.rating_pool serve --data ~/drmc-rl-pool/data \
  --host 192.168.157.190 --port 8097 --report-port 8098
```

mombox's firewall (ufw) admits only ssh from the LAN. Until a LAN rule for ports
8097/8098 exists (`sudo ufw allow from 192.168.157.0/24 to any port 8097,8098 proto tcp`,
an operator decision), hosts reach the coordinator through an ssh tunnel:
`ssh -N -L 127.0.0.1:8097:192.168.157.190:8097 -L 127.0.0.1:8098:192.168.157.190:8098 mombox`
(`drmc-rl-pool-data/tunnel.sh` on the Mac) with `DRMC_POOL_URL=http://127.0.0.1:8097`.

It runs as the user systemd service `rating-pool` (`systemctl --user restart rating-pool`;
it resumes from the journal). Checkpoint
downloads are limited to 2 concurrent streams at 20 MB/s each; uploads from
`entrant add`/`bootstrap`/`watch-run` are throttled to 40 MB/s by the client.

Workers must run the coordinator's exact commit (and native libraries built from
engine `19f292c`). Upgrading the pool code means: commit, update mombox `src`,
restart the coordinator, restart workers.

Mac (3 MPS workers while the Mac is shared; they yield to other arena workers such
as a stop-rule panel and pause below 5 GB free disk):

```bash
cd /Users/ethan/dev/drmario/drmc-rl-pool
N=/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries
for slot in 0 1 2; do
  /Users/ethan/dev/drmario/drmc-rl/.venv/bin/python -m tools.rating_pool worker --device mps --slot $slot --host-budget 3 --min-free-gb 5 \
    --native-library $N/libdrmario_pool.dylib --reach-library $N/libdrm_reach_full.dylib
done
```

(`~/dev/drmario/drmc-rl-pool-data/start-mac-workers.sh` starts these with nohup;
`start-watchers.sh` starts the arm A and arm C snapshot watchers.) Green (CUDA): the ARENA_HOSTS.md bootstrap, then
`python -m tools.rating_pool worker --coordinator http://192.168.157.190:8097
--device cuda --slot N` from a checkout of the pool commit; a new CUDA class
replays `calibration_games` (8) traced games before it contributes.

Status and control:

```bash
python -m tools.rating_pool summary                  # CLI summary
python -m tools.rating_pool summary --json out.json  # full JSON export
python -m tools.rating_pool release BATCH_KEY        # hung worker
curl -H "Authorization: Bearer $(cat ~/.config/drmc-rl/study-worker.token)" \
  http://192.168.157.190:8097/api/v1/pool/status
```

## Adding entrants

```bash
python -m tools.rating_pool entrant add my-core-v1 --checkpoint PATH --era afterstate \
  --parent champion-retention-mixed-v2 --recipe RECIPE --notes "..."
python -m tools.rating_pool entrant set old-thing --status retired
```

`entrant add` uploads the checkpoint to mombox if it lacks it. Pace adapters use
`--adapter PATH` with `--checkpoint PARENT`.

## Submitting a focused experiment

A job is a priority spec: entrants (ids or globs), a mode (`vs` opponents, default
each condition's anchor; `round_robin`; `vs_parent`; `explicit` pairings),
conditions (names, keys or `set:NAME`), games per pairing and condition, priority,
optional deadline and seeds.

```bash
python -m tools.rating_pool job submit armC-vs-armA --entrants 'armC-*' --opponents 'armA-ppo-v1-*' \
  --conditions set:l14-spawn --games 128 --priority 70 --deadline 2026-10-01
python -m tools.rating_pool job submit confirm-x --entrants X --conditions set:l14-shipped --games 128 \
  --priority 90 --seeds allocation:confirm-x-v1     # after seed_reserve allocate confirm-x-v1 N
python -m tools.rating_pool job set armC-vs-armA --status paused
```

The results feed the ordinary per-condition ratings. Pre-registered analyses keep
their own scripts; import the finished journal with `import-study` so its games
also enter the ratings where conditions match.

## How a training run hooks in

1. Record the plan: `intention add RUN --title ... --entrants 'RUN-*' --job '{...}'`.
2. Run the watcher where snapshots land (stdlib only; any host that reaches mombox):

   ```bash
   python -m tools.rating_pool watch-run --dir OUTPUT_DIR --pattern 'core-f*.pt' --run RUN \
     --era afterstate --parent PARENT_ENTRANT --recipe RECIPE --panel-set l14-spawn --panel-games 128
   ```

   Each stable new snapshot is uploaded, registered as `RUN-fNNNNNNNNNNN` (active,
   lineage run/step/parent/recipe), and the open job `stop-panel-RUN` plays every
   snapshot against the anchor on the set at priority 60. A trainer can instead
   call `drmc_rl.pool.client.register_snapshot` after writing a snapshot.
3. The stop rule queries pool ratings:

   ```bash
   python -m tools.rating_pool stop-rule --run RUN --set l14-spawn --min-games 128 --exit-code
   ```

   A snapshot improves when its pooled rating over the set exceeds the best of the
   parent and every earlier snapshot; the rule fires after two consecutive
   non-improving snapshots (exit 10) and names the best snapshot. Snapshots with
   fewer than `--min-games` games in any condition of the set are pending.

Runs with a pre-registered stop rule or tournament keep it as written; the pool
rates their snapshots in addition (arm A's snapshots are registered by
`watch-run`, without a stop-panel job).

## Importing history

`python -m tools.rating_pool import-study STUDY.json [--register-unknown --era E]`
maps each variant to an entrant by checkpoint/adapter sha256 and player settings
(unknown ones are skipped, or registered as retired with `--register-unknown`),
maps each schedule row to a condition (creating it if new), and imports its games
and a sample of its move traces. Old league systems (the G4 Strong League sqlite
arena, `tools.arena`) are not imported: they ran a different environment and
their players cannot run under today's public-information arena (below).

## Era-best entrants

Included (all smoke-tested through the arena loader): `bc-gt2000` (behaviour-cloned
humans, floor), `v5-public-bootstrap`, `public-outcome-10m` (the pre-online v20
core), `pace-adapter-final` (trainer-pace-v1 adapter on outcome 10M),
`public-core-300m`, `public-core-final`, `retention-control-v1`, the champion, and
the afterstate arm A student and PPO snapshots.

Not runnable in today's arena: the G2/G3/G4 Strong League checkpoints (including
+900M, +1.0B and `g4-strongest`) read privileged pending-attack auxiliary inputs
(`v1_vs`) and were trained on the old warp-buffer timeline; early VS PPO,
`capsule_prime` and vs1p/vs4 (`v1`, opponent-blind) fail the arena's public
auxiliary contract and would play off-distribution with zero-filled inputs; V3
afterstate and `human_policy_v2` have architectures `PlainPolicy` cannot load;
anticipation (`prepared4`) cannot play a different checkpoint. Making any of them
fair would need new public-feature builders or loaders.
