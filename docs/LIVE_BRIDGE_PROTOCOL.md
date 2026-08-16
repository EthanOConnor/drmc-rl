# Live bridge protocol

The bridge transports semantic public state from the emulator/console host to a
nonblocking match backend and returns an exact frame-indexed controller script.
It does not expose hidden RNG or internal pair checkpoints to a deployed actor.

## State message

At visible decision/event boundaries, send:

- monotone frame and spawn identifiers;
- `PublicPairState` v2;
- exact own falling-pill controller microstate needed by reachability;
- host deadline and product controls;
- latest script/desync acknowledgement.

Opponent state carries an explicit age. A stale settled-board snapshot is never
represented as frame-synchronized.

## Plan message

Return:

- source frame/spawn IDs;
- chosen macro action and intended final pose;
- exact per-frame NES button masks;
- predicted lock frame and resolved next-event key when available;
- product, execution profile, quality/regret/style/cadence diagnostics;
- planner and replay validation status;
- search completion/deadline status;
- artifact identity.

The host rejects stale source IDs, invalidates pondering/search caches on
rollback or divergence, and verifies predicted versus observed pose.

## Rollback and timing

Scripts are indexed against the emulator's canonical frame clock and replayed
idempotently under rollback. The backend must leave serialization/scheduling
headroom inside the host deadline and provide the plain policy fallback before
search begins.

## Fair-play boundary

The bridge may read only information visible on the game presentation or
required from the player's own controller state. Any debug/native extension
that transmits privileged pair state uses a distinct schema and cannot be used
for a public superhuman claim.
