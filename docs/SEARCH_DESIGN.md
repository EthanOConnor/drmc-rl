# Joint-event search

## Authority

The target search is `drmc_rl.search.joint_event.JointEventSearch`. It searches
the asynchronous full pair process. The existing own-board depth-2
`SearchPolicy` may remain for diagnostics, live fallback, and historical
reproduction, but it is not the permanent competitive model.

## PairSearchModel contract

A backend adapter provides:

- stable state/transposition key;
- decision boundary: P1, P2, both, deterministic advance, or terminal;
- exact legal actions and policy priors per side;
- strict joint action application to the next pair event;
- newly revealed chance outcomes;
- terminal W/D/L;
- calibrated leaf W/D/L.

The native adapter must restore full pair state, including both clocks, phases,
committed actions, attacks, and animation state, while excluding future RNG
from public deployment search.

## Backup

- Root-side decision: maximize backed-up utility.
- Opponent decision: integrate a declared opponent policy or use minimax stress.
- Simultaneous decision: evaluate joint actions and integrate/minimize the
  opponent branch for each root action.
- Chance: probability-weighted W/D/L.
- Leaf: calibrated `P(win), P(draw), P(loss)`.

Priors select expansion order and beam coverage; they do not directly decide
the backed-up root action. Full-candidate counterfactual releases use a beam
large enough to cover every legal root action.

## Training role

Search first produces offline policy/value targets. It controls live behavior
only after paired same-weight full-game evaluation opens the gate. It controls
training behavior only under an algorithm that records the actual behavior
policy and corrects off-policy learning explicitly.

## Performance

Measure depth in pair events, not learner plies. Report nodes, cache hits,
opponent and chance coverage, deadline completion, W/D/L gain, and strength per
millisecond. Pondering is an implementation optimization and never changes the
state/information contract.
