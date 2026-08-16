# Population and PSRO roles

PFSP over frozen self-history remains useful but is not the final population
algorithm. The program trains explicit approximate best responses to a
regularized empirical meta-strategy.

## Roles

- **Main:** maximizes value against the active population mixture.
- **Main exploiter:** searches for weaknesses in the current main.
- **League exploiter:** best-responds to the historical/meta mixture.
- **Human/execution exploiter:** targets human styles, cadence profiles, and
  mechanically constrained agents absent from ordinary self-play.

Each role has its own immutable lineage. Exploiters do not silently replace the
main objective or poison the main's opponent pool with collapsed snapshots.

## Mixture

`drmc_rl.arena.meta_strategy` solves the finite payoff game by averaged
entropy-regularized multiplicative weights. It reports:

- row, column, and symmetric population mixtures;
- game value;
- row and column best responses;
- unregularized saddle gap;
- per-opponent regression values.

The arena adapter should antisymmetrize side-balanced pairwise estimates,
maintain a small exploration floor, and persist the exact payoff version and
solver parameters used for a training campaign.

## Promotion

A candidate is promoted only when it improves mixture value and stays above
predeclared floors against all permanent anchors and active exploiters. Latest-
self win rate or scalar Elo alone is insufficient.

## Existing opponent pool

The current frozen-policy/human-afterstate opponent pool remains the execution
mechanism while the arena-to-mixture adapter is completed. New code should add
explicit mixture weights and roles rather than another independent sampling
heuristic.
