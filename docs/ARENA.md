# Continuous arena

## Live ratings and snapshots

Match production never waits for rating inference. The live table rebuilds the
hierarchical Davidson W/D/L posterior with a fast Laplace approximation after
128 new games. Rating/LOS/rank summaries update on that cadence; the more
expensive scheduler information matrix refreshes every 1,024 games. Full
multi-chain HMC is an explicit offline calibration, not a lease gate. The live
backlog remains visible as `scheduler.rating_pending_games`.

The browser also never computes a dashboard view in an HTTP request thread.
One background producer materializes an immutable snapshot every five seconds;
polling clients only receive the last complete byte buffer. This keeps browser
traffic, rating work, leases, and result submissions isolated from each other.

Multiple unsettled entrants share one bounded new-entry multiplier; their
pairing no longer compounds two 6x boosts into a 36x new-vs-new preference.

The arena is the durable evaluation loop around VS training. It records every
game in `runs/arena/arena.sqlite`, computes one connected Bayesian rating
posterior, promotes candidates through an SPRT against the current champion,
and keeps every prior champion active as a `lineage` entrant. The browser UI
refreshes from that same database; it has no separate state.

Agent identity is immutable and distinct from a filename. Give experiments
short recognizable names such as `Capsule Prime G3`, `Searchlight G1`, and
`Human 2000`; use `family` for the comparable lineage and `generation` for its
position. The supported statuses are:

- `candidate`: receives concentrated games against its family's champion.
- `champion`: current promoted model for a family.
- `lineage`: a former champion, retained in tournament scheduling forever.
- `anchor`: a frozen human-rating or search configuration.

Register a JSON manifest:

```json
{
  "agents": [
    {
      "id": "capsule-prime-g0",
      "name": "Capsule Prime G0",
      "family": "central",
      "generation": 0,
      "checkpoint": "runs/arena/checkpoints/capsule-prime-g0.pt.gz",
      "status": "champion"
    },
    {
      "id": "human-2000-v2",
      "name": "Human 2000 · V2",
      "family": "human",
      "generation": 2,
      "checkpoint": "runs/arena/checkpoints/human-policy-v2.pt.gz",
      "status": "anchor"
    }
  ]
}
```

Then run the worker and dashboard as separate supervised processes:

```bash
uv run python -m tools.arena register runs/arena/roster.json
uv run python -m tools.arena worker --device cuda --batch 8
uv run python -m tools.arena serve --host 0.0.0.0 --port 8097
```

`serve` owns the lightweight rating thread by default. It checks every five
seconds and publishes a fresh Laplace posterior after 128 new games. Run an
offline HMC audit with
`uv run python -m tools.arena rate --once --rating-full-hmc`; use `--no-ratings`
on `serve` only when another supervised `rate` process owns live updates.

## Distributed coordinator and workers

SQLite always belongs to exactly one host and stays on that host's local
filesystem. Never put `arena.sqlite` on SSHFS, SMB, NFS, OneDrive, or another
network-synchronized directory. Remote machines lease deterministic match
batches over HTTP and submit immutable results; they never open the database.

Create a private worker token outside the repository on the coordinator, then
start the dashboard/coordinator on a trusted encrypted network such as
Tailscale:

```bash
umask 077
openssl rand -hex 32 > ~/.config/drmc-rl/arena-worker.token
uv run python -m tools.arena serve \
  --host 0.0.0.0 --port 8097 \
  --worker-token-file ~/.config/drmc-rl/arena-worker.token \
  --replay-dir /data/drmc-arena/replays
```

Copy the token through an authenticated channel to each worker and launch it
without any `--db` access:

```bash
uv run python -m tools.arena worker \
  --coordinator http://green:8097 \
  --token-file ~/.config/drmc-rl/arena-worker.token \
  --worker-id macbook-mps \
  --device mps --threads 2 --batch 12
```

The coordinator reserves globally unique game serials, issues expiring leases,
and reassigns abandoned work. Workers renew long-running leases in the
background. Each claim has a fresh secret, each match has a deterministic
SHA-256 ID, and an identical retried submission is acknowledged without
creating duplicate games or telemetry. Checkpoints are delivered by the
authenticated coordinator and cached by content hash on each worker.

New replay samples are gzip-compressed into content-addressed files and the
database stores only their relative hashes. Before migrating an existing
database, operate on a local copy while the old arena remains authoritative:

```bash
uv run python -m tools.arena --db /data/staging/arena.sqlite \
  externalize-replays --replay-dir /data/drmc-arena/replays --vacuum
```

Verify the copied database, replay hashes, checkpoint paths, API integration,
and remote worker throughput before cutover. The final cutover is a brief
worker pause, WAL checkpoint, checksummed copy, and single-writer ownership
switch; never run Mac and Green coordinators against the same history.

## Rating model

Every checkpoint is treated as an immutable player with one fixed latent
skill. The complete W/D/L history is fit with a hierarchical Davidson model:

- decisive win odds are logistic in the two checkpoints' skill difference;
- every agent has a separate draw tendency, so horizon survival is not scored
  as half a win or confused with strength;
- a child checkpoint's skill prior is centered on its parent, while the common
  lineage transition width is learned from the arena;
- independent roots receive broad weakly informative priors;
- P1/P2 advantage is fixed at exactly zero. Sides are alternated by the arena,
  and a measured side split is a protocol diagnostic rather than a parameter
  the rating model is allowed to conceal.

Fresh fits use four adaptive HMC chains. Nothing is published unless there are
zero sampling divergences, rank-normalized split R-hat is at most 1.01, and
both bulk and tail effective sample sizes pass a 400 minimum. The SQLite cache retains the posterior
draws as well as summaries. Between full fits, new matches update those draws
by their exact incremental likelihood (sequential importance sampling), which
takes milliseconds. A roster change, 512 new games, or importance ESS below
50% forces a new HMC fit; a failed fit leaves the last accepted posterior
visible.

The dashboard reports posterior mean Elo, asymmetric 95% credible intervals,
probability of being best, and a 95% rank interval. `LOS ↓ next` is the joint
posterior probability that a row is stronger than the next visible row;
`LOS ↳ parent` compares it with its registered lineage parent. Elo is only the
familiar display scale (`400 / ln(10)` times latent log-odds); inference is
performed on the full W/D/L posterior. Fit method, staleness, R-hat, sampling
ESS, online importance ESS, and the observed side split are exposed in
`/api/snapshot`.

Each posterior update also computes the expected Bayesian information gain of
one more W/D/L result for every possible pair. The scheduler samples matchups
in proportion to that expected uncertainty reduction, with explicit weights
for candidates/provisional entrants and an approximate compute-cost discount
for search. Low-game and uncertain comparisons therefore rise naturally;
already-certain mismatches fall away. Before the first posterior, a count-based
bootstrap policy establishes a connected graph. Promotion remains a separate
0-vs-10 Elo SPRT with 5% errors and at most 400 games. A promoted model changes
the old champion to `lineage`; no checkpoint is moved or deleted.

Training remains the existing placement-SMDP PPO stack. Point each campaign's
checkpoint directory at the registrar or explicitly register selected
milestones. Do not register every autosave: candidates are scientific
identities, while autosaves are recovery artifacts. A discovery config is:

```json
{
  "campaigns": [{
    "id": "central-az",
    "family": "central",
    "name": "Capsule Prime G{generation} · {step}",
    "root": "/home/ethan/drmario/drmc-rl",
    "glob": "runs/central_az/checkpoints/*.pt.gz",
    "settle_seconds": 60
  }]
}
```

Run `uv run python -m tools.arena discover campaigns.json`; it watches by
default, registers each settled checkpoint once, links it to the current
family champion, and assigns a human-readable generation.
