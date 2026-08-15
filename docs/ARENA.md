# Continuous arena

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

`serve` owns a non-blocking rating thread by default. It publishes a fast
sequential update every 16 new games and a fresh HMC fit every 512 games. Run a
one-off fit with `uv run python -m tools.arena rate --once`; use `--no-ratings`
on `serve` only when another supervised `rate` process owns that work.

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
