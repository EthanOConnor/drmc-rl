# Continuous arena

The arena is the durable evaluation loop around VS training. It records every
game in `runs/arena/arena.sqlite`, computes one connected Elo ladder, promotes
candidates through an SPRT against the current champion, and keeps every prior
champion active as a `lineage` entrant. The browser UI refreshes from that same
database; it has no separate state.

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

The scheduler prioritizes candidate/champion evidence, then matchup edges with
the fewest games. Historical champions therefore receive fewer games as the
lineage grows but never age out. Promotion defaults to a 0-vs-10 Elo SPRT with
5% errors and at most 400 games. A promoted model changes the old champion to
`lineage`; no checkpoint is moved or deleted.

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
