# Match agent and three product decoders

Training and match play share the competitive policy, public state schema,
planner semantics, and artifact identity. Match play additionally requires
frame-exact scripts, deadlines, desynchronization checks, and one declared
product decoder.

## Unrestricted

- public/belief pair state;
- strongest competitive policy;
- validated joint-event search when available;
- unrestricted exact execution;
- deterministic quality maximization.

Unrestricted refers to controller precision, not hidden information.

## Human-rate

- same competitive core and search;
- candidate scripts filtered by a signed `ExecutionProfile`;
- quality maximization among remaining candidates;
- no intentional strategic errors.

Average APM is not the constraint. The profile covers reaction, edge intervals,
bursts, chords, corrections, holds, and soft drop.

## Trainer

`UnifiedDecisionDecoder` applies:

1. rating-independent win probability;
2. execution feasibility;
3. calibrated win-logit regret for requested strength;
4. explicit rating-residualized style inside the regret envelope;
5. temporally correlated form;
6. decision cadence;
7. exact profile-valid script selection.

Temperature and beam width can remain diagnostic knobs but do not define
strength.

## Runtime boundary

The intelligence backend remains a supervised nonblocking subprocess/service.
Hosts maintain at most one pending request, discard stale frame IDs, continue
play on failure, and never block emulation/audio/render threads. Responses carry
chosen action, exact script, quality/regret/style diagnostics, execution-profile
identity, and deadline health.
