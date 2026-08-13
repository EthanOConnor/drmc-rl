# Human player and coach backend

`python -m tools.human_backend` is the supported intelligence boundary for
Professor Pills. It is a supervised JSONL subprocess: one request per stdin
line, one response per stdout line. Model loading, inference, reachability,
and coaching remain outside the host process and therefore cannot stall or
crash emulation, rendering, input, or audio.

Protocol schema: `drmc-human-backend-v1`.

## Host behavior

- Start the backend off all real-time threads and send `hello` while showing a
  warming-up state.
- Keep at most one pending request and one latest result. Never wait for a
  response on a gameplay thread.
- Use monotonically increasing `request_id` and emulator `frame_id` values.
  Discard results for frames older than the current decision.
- Send `cancel` when a pending result is no longer useful. Cancellation is
  cooperative; stale-result rejection is still mandatory.
- Treat EOF, timeout, or an error response as loss of optional AI service.
  Gameplay continues and the host may restart the subprocess.
- Send semantic state produced from the shared Dr. Mario specification. Do not
  reproduce WRAM addresses in the host/backend protocol.

## Discovery and health

```json
{"schema":"drmc-human-backend-v1","type":"hello","request_id":0,"frame_id":0}
```

The `capabilities` response reports request types, state conventions, model
schema, checkpoint identity, corpus release, supported WHR-C range, and held-out
model metrics. `health` additionally reports readiness, uptime, model-load time,
request/error counters, and inference latency percentiles.

## Semantic decision state

`decide` and `coach` share this envelope:

```json
{
  "schema": "drmc-human-backend-v1",
  "type": "decide",
  "request_id": 42,
  "frame_id": 9182,
  "deadline_ms": 90,
  "target_rating": 1750,
  "temperature": 1.0,
  "state": {
    "board_planes": "8 x 16 x 8 nested 0/1 values",
    "opponent_board_planes": "8 x 16 x 8 nested 0/1 values",
    "opponent_state_age_frames": 12,
    "pill": [0, 1],
    "preview": [2, 0],
    "speed": 2,
    "speed_ups": 0,
    "falling": {
      "x": 3,
      "y": 0,
      "rotation": 0,
      "speed_counter": 0,
      "horizontal_velocity": 0,
      "hold_dir": 0,
      "rotation_hold": 0,
      "frame_parity": 0
    }
  }
}
```

Colors are canonical `0=red, 1=yellow, 2=blue`; row zero is the bottle top.
Each set of eight board planes is color `[3]`, virus `[1]`, and pill connectivity
`[up, down, left, right]`. The backend computes every reachable placement and
its exact controller script. The response declares the chosen placement,
frame-indexed NES button masks, resolved/clamped rating, timing distribution,
candidate actions, and human-policy logits. The opponent board is the corpus-
compatible latest known spawn state; its explicit age prevents it from being
misrepresented as frame-synchronized. Professor Pills decides how to
schedule or present those declarations; the backend never issues UI commands.

The timing model predicts human slack beyond the planner-minimal controller
script. It is metadata until the host implements an execution scheduler that
can add delay without invalidating reachability.

## Coaching

A `coach` request may include `chosen_action` and `alternative_limit`. Its
response reports how typical the choice is for comparable humans, its rank,
surprisal, and common alternatives. Human frequency is explicitly not called
move quality. A competitive-policy score will be added as a separate axis;
until then the backend provides faithful behavioral comparison, not claims of
optimality or strategic causation.

## Training

```sh
uv run python -m tools.train_human_policy extract --planner cuda --sample-modulus 32
uv run python -m tools.train_human_policy train --device cuda --epochs 8
uv run python -m tools.human_backend --checkpoint runs/human_policy/human_policy_v1.pt.gz
```

Extraction reads only immutable `HumanCorpus` releases, joins each decision to
the release's continuously interpolated WHR-C trajectory, recomputes exact
feasible candidates, and deterministically samples by decision ID. One model
replaces rating buckets. Rating-density weights prevent the middle of the
population from drowning out the tails. Validation holds out both replay
splits and a complete player fold.
