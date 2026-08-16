# Known risks and required controls

- **Timing abstraction:** earliest lock may not dominate in an asynchronous
  pair game. Resolve through the strict timing-action gate.
- **Partial observability:** settled bottles and coarse scalars may omit the
  opponent's committed in-flight intent. Complete PairState v2 and public event
  belief before final deployment claims.
- **Teacher contamination:** current quality can inherit human-choice and
  handcrafted tactical priors. Mature quality requires full-pair
  counterfactual W/D/L labels.
- **Search model error:** own-board depth-2 search ignores causal opponent and
  garbage interactions. Do not promote it as the final search architecture.
- **Reward corruption:** clear/garbage/progress shaping can change optimal match
  behavior. Final promotion uses W/D/L.
- **Population cycles:** scalar Elo can hide catastrophic matchups. Retain the
  payoff matrix, exploiters, and PSRO mixture gates.
- **Candidate truncation:** a fixed packed width can discard a rare best move.
  Require zero drops or dynamic expansion.
- **Human-rate overclaim:** average APM can hide impossible bursts and chords.
  Claims name a signed execution profile with zero violations.
- **Strength/style leakage:** a style latent can encode rating and alter
  strength. Residualize against rating and recalibrate each style.
- **Independent mistakes:** iid noise looks unlike humans. Use regret tails and
  slowly varying form; preserve intent in motor errors.
- **Simulator self-verification:** optimized native/planner paths cannot be
  their own sole oracle. Keep full planner, emulator, traces, and script replay.
- **Artifact ambiguity:** checkpoint filenames are not identities. Require
  hashes, config, revisions, schemas, profiles, search, corpus, and gate
  evidence.
- **Checkpoint state:** any scheduler/opponent/curriculum state not restored must
  be declared in resume provenance.
