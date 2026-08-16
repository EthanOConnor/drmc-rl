# Human player and coach backend protocol

The supported host boundary is a supervised nonblocking JSONL subprocess or an
isomorphic local service. Model loading, reachability, quality, human decoding,
search, timing, and script validation remain outside gameplay threads.

Protocol schema: `drmc-human-backend-v2`.

## Host rules

- Start and warm the backend off real-time threads.
- Keep at most one pending request and one latest result.
- Use monotone `request_id` and emulator `frame_id`; discard stale results.
- Send `cancel` when work is obsolete, but still reject stale responses.
- Treat EOF, timeout, or error as loss of optional AI service; gameplay
  continues and the host may restart it.
- Send semantic `PublicPairState`; never send duplicated WRAM offsets or a
  privileged engine checkpoint to a deployed actor request.

## Hello and health

```json
{"schema":"drmc-human-backend-v2","type":"hello","request_id":0,"frame_id":0}
```

Capabilities report:

- model/artifact identity;
- public-state and execution-profile schemas;
- supported products;
- rating/style ranges;
- search profile and deadline behavior;
- corpus releases and held-out calibration;
- health and latency percentiles.

## Decide request

```json
{
  "schema":"drmc-human-backend-v2",
  "type":"decide",
  "request_id":42,
  "frame_id":9182,
  "deadline_ms":90,
  "product":"trainer",
  "controls":{
    "target_rating":1750,
    "style":[0.6,-0.2,0.0,0.3],
    "cadence_scale":1.0,
    "execution_profile":"rating-conditioned-human"
  },
  "state":{"schema":"drmc-pair-state-v2","...":"PublicPairState payload"}
}
```

Products:

- `unrestricted`: quality argmax, unrestricted exact scripts;
- `human_rate`: quality argmax after the named profile filter;
- `trainer`: calibrated regret, style, cadence, form, and profile-valid
  execution.

Rating is ignored for competitive quality. Search breadth and temperature are
not strength controls.

## Decide response

The response includes:

- selected action and exact frame-indexed button script;
- best and chosen calibrated win probabilities;
- win-logit regret and requested target regret;
- candidate W/D/L/uncertainty summaries;
- style score and form state;
- requested/realized cadence;
- execution-profile identity, script metrics, and zero/explicit violations;
- search depth/nodes/deadline status;
- artifact manifest identity;
- planner replay verification and desync key.

The host schedules the script and checks predicted versus observed pose. It does
not reinterpret the backend's strength or style semantics.

## Coach request

A coach request adds the observed `chosen_action`. The response reports:

- feasibility and exact consequences;
- human probability/rank and surprisal;
- competitive W/D/L/rank/regret;
- common strength-consistent alternatives;
- deterministic explanation facts from exact afterstates/events.

Human frequency is never labeled move quality. Language presentation may
summarize deterministic facts but may not invent causal explanations.

## Timing and execution

Decision latency and motor execution are distinct outputs. The backend chooses
only scripts that replay to the selected pose and satisfy the active profile.
For trainer motor mistakes, future implementations must preserve recognizable
intent through late correction/failure models; random placement substitution is
not permitted.

## Compatibility

V1 hosts can be supported by an explicit adapter that constructs
`PublicPairState` and maps the old request to one product. V1 raw logit-gap
strength is retired and must not be silently mapped to a claimed human rating.
