# Architecture Review — June 2026

Question asked: what next for Elo, and is SMDP-PPO over placement
candidates even the right architecture for Dr. Mario VS / speedrun?

## Verdict

The decision-level abstraction (placement candidates + warp execution +
exact native simulator) is correct and is the asset. What sits on top of
it — pure model-free PPO — is no longer the best use of a 30µs exact
simulator. Dr. Mario is a stochastic game with afterstate structure
(deterministic placement → stochastic next pill from a known 9-pair
distribution), exactly the regime where search-based policy improvement
(AlphaZero/MuZero family, Gumbel variant for small sim budgets)
dominates model-free RL.

## Endpoint architecture

Gumbel-AZ-lite over the existing candidate net:

- Keep the candidate-scoring net (d192/4-block, EMA) as prior + value.
- At each decision during training, run small search (8–16 simulations,
  depth 2 with exact preview-pill marginalization — the machinery
  already exists in `models/policy/search_policy.py`).
- Policy target: the search action distribution (Gumbel improved
  policy), not the PPO clipped-ratio gradient.
- Value target: search-backup Q (training-consistent backup
  r̂₁+γ^τ(r̂₂+γ^τV) — verified necessary; pure value-head backup is
  anti-clear).
- Phase in: add a distillation term alongside PPO first
  (L = L_PPO + β·KL(π_search‖π_net)), then shift weight as it proves
  out. This avoids a cold-turkey optimizer change on a live lineage.

Measured basis: depth-2 beam-8 search wins 86.7% (52/60, 95% CI
75.8–93.1) head-to-head vs the same weights plain. The net trained to
match search output banks a large fraction of that gap at zero
inference cost, and search on top of the improved net compounds.

## Biggest VS gap: opponent-board observability

The net sees only its own bottle. Humans scout the opponent board
constantly (attack timing, kill confirmation, defensive hold). The
vspool already exports both boards; the fix is observation plumbing:
opponent board planes + scalars (opponent virus count, garbage pending
both directions, opponent pill queue) into the candidate-net encoder.
This is the single largest expected Elo item for VS.

## Ranked next-Elo items

1. Opponent-board observability (obs plumbing, retrain).
2. Search-amplified training targets (Gumbel-AZ-lite above).
3. Ponder — search during fall/stun/dead time
   (`PonderingSearchPolicy`, probe in flight).
4. League depth: exploiter agents vs the champion, not just PFSP over
   own history.
5. BC human-style opponents from the fightcadeRatings corpus
   (style diversity the self-play league lacks).
6. Go-Exploit style start-state sampling from corpus positions
   (train from real mid-game crises, not only clean starts).
7. Opponent model at search leaves (predict opponent placement instead
   of assuming static board).

## What we are NOT changing

- Candidate/placement action space (strictly better than per-frame
  buttons for this game; verified by the whole 1P campaign).
- Native engine + warp execution + planner v4 (exactness oracle: v1).
- Win-prob credit (gamma=1.0) for VS terminal structure.
