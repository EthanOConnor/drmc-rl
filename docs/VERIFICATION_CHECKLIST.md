# Verification checklist

## Public state and RAM semantics

- bottle bases/strides and tile encoding;
- visible falling pill pose, colors, current/preview transitions;
- speed, lock, animation, and decision boundaries;
- opponent snapshot age and semantic event ordering;
- no hidden/future RNG or internal-only attack fields in `PublicPairState`;
- exact provenance for every public scalar.

## Reachability

- Python/full native planner agreement;
- v4 and CUDA mask/cost parity against `drm_reach_bfs_full`;
- exact script replay to every sampled pose;
- same-color canonical equivalence;
- zero silently dropped legal candidates;
- execution-profile validation and Pareto scripts.

## Pair dynamics

- replay init/checkpoint restoration;
- both side clocks and decision flags;
- attack creation, release frame, colors, and columns;
- simultaneous clear/topout ordering;
- strict forced-lock advancement versus recorded locks;
- throughput SMDP approximation quantified against strict advancement;
- terminal W/D/L and horizon behavior.

## Timing-action gate

- earliest and all sampled delayed scripts reach the same intended pose;
- forced lock frames are exact pair-clock values;
- next-event state hashes and value deltas are recorded;
- results stratified by pressure, clear, garbage, speed, and ordinary states;
- architecture decision and threshold recorded as gate evidence.

## Search and teachers

- complete restorable pair-state key;
- single-side, simultaneous, deterministic, and chance event unit fixtures;
- opponent prior/mixture identity;
- full root candidate coverage for counterfactual releases;
- W/D/L calibration and uncertainty;
- no privileged field in deployed search;
- same-weight paired search acceptance.

## Products

- unrestricted: exact scripts, deadline/desync, human cohort protocol;
- human-rate: signed profile and zero violations;
- trainer: monotone strength, style independence, cadence/motor distributions,
  temporal form, and pedagogy evidence;
- complete artifact manifest for every released identity.
