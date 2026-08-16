# Continuous arena

The arena is the durable evidence system around competitive training. It owns:

- immutable agent identities and lineage;
- every full-game result and replay reference;
- the connected W/D/L payoff graph;
- Bayesian rating/posterior summaries;
- information-gain scheduling;
- candidate promotion tests;
- PSRO/meta-strategy inputs;
- live training and worker telemetry.

SQLite belongs to exactly one coordinator on local storage. Remote workers lease
deterministic match batches over authenticated HTTP and never open the database.

## Agent identity

Autosaves are recovery artifacts. Arena entrants are scientific identities with:

- stable `id`, family, generation, role, and parent;
- immutable checkpoint and artifact manifest;
- mode/decoder, execution profile, and search parameters;
- status: candidate, champion, lineage, or anchor.

A promoted champion becomes permanent lineage when replaced. Human, style,
execution, and search configurations are anchors; requested-rating clones are
not accepted as distinct strength evidence unless independently calibrated.

## Ratings and payoff matrix

The displayed Elo scale is derived from the hierarchical Davidson W/D/L model,
including separate draw tendency and lineage priors. Side advantage remains a
protocol diagnostic rather than a parameter that conceals bias.

The full pairwise payoff matrix is equally authoritative. Cycles and exploiters
must remain visible even when scalar ratings look ordered.

## Scheduling

Before a posterior exists, count-based scheduling connects the graph. Afterwards,
expected Bayesian information gain prioritizes uncertain and decision-relevant
matchups, with compute-cost adjustment for expensive search agents.

The PSRO layer consumes a square payoff export and returns a regularized
population mixture, best responses, and saddle gap. The intended role set is:

- main;
- main exploiter;
- league exploiter;
- human/execution exploiter.

## Promotion

Promotion requires more than head-to-head Elo:

- sequential evidence versus the current main;
- improved expected value against the active meta-strategy;
- no catastrophic regression against permanent anchors or active exploiters;
- clean-start, tactical-suite, side, seed, horizon, and objective health gates;
- complete artifact provenance.

## Operations

Coordinator and worker commands are maintained in [Operations](OPERATIONS.md).
The database, replay store, and checkpoints are migrated only with workers
paused, WAL checkpointed, checksums verified, and one authoritative writer.
