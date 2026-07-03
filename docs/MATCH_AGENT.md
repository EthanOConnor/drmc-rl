# The match agent: playing real games (vs the training architecture)

Status 2026-07-02. This documents the "designed to play real matches" stack —
what runs when the agent sits across from a human — and how it deliberately
differs from the "designed to train" stack. Companions:
docs/LIVE_BRIDGE_PROTOCOL.md (transport), docs/SEARCH_DESIGN.md (search),
docs/DESIGN_REVIEW_2026-07.md (why these pieces).

## Two architectures, one policy

Training and match play share the policy net, the planner semantics, and the
feasibility/planner contract (policy sees board + feasible placements; the
planner turns the choice into inputs). Everything around that differs:

| Dimension          | Training (drmario-vs pool)         | Match agent (live bridge)             |
|--------------------|------------------------------------|---------------------------------------|
| Execution          | warp (teleport pill to lock pose)  | frame-exact input scripts on a real console/emulator |
| Planner output     | costs only (v4 / reach_cuda costs) | costs + per-pose input script (reach_cuda scripts mode, CPU fallback) |
| Decision budget    | none (throughput matters)          | hard: plan must be written ~6 frames (~100 ms) after spawn |
| Search             | off in rollouts (distill is a training-signal path) | depth-2 beam expectimax + ponder |
| Opponent           | frozen pool / league               | a human; opponent board ignored by obs (8ch) |
| Strength           | always full                        | dialable (see below)                  |
| State source       | native engine truth                | RAM read via Lua each frame; desync-verified |

The asymmetry is intentional: training wants millions of cheap, exact
decisions; match play wants ~1 excellent decision per second under a latency
deadline, executed frame-perfectly, at a chosen strength.

## The stack (tools/live_agent_server.py)

1. **State**: `fc_live_agent.lua` reads RAM every frame, emits state lines on
   spawn edges (protocol in docs/LIVE_BRIDGE_PROTOCOL.md).
2. **Plan**: the server rolls the observed micro-state forward `--margin`
   neutral frames, then solves reachability *from that exact mid-fall state*:
   - `--planner cuda` (new): one `reach_cuda.solve_scripts` call returns exact
     costs AND a replay-verified optimal input script per pose. Nonzero status
     (no greedy-matched script / overflow / parity alarm) falls back to the
     CPU planner for that decision. Every script is re-simulated through the
     frame-exact twin before it ships; mismatches are logged as desyncs.
   - `--planner cpu`: the original `drm_reach_bfs_full` path.
3. **Choose**: plain policy argmax, `--temperature` sampling, or
   `--search [BEAM]` depth-2 expectimax (+ `--ponder`: search the next
   decision during fall dead-time; cache hits commit with a 2-frame margin).
   With `gpu_planner` search (SearchPolicy(gpu_planner=True)), all ply-1/ply-2
   sim replans are batched on the GPU — biggest effect exactly where the CPU
   BFS is slowest (sparse boards, i.e. endgames).
4. **Execute**: the plan is a frame-indexed button script; the Lua side plays
   it byte-per-frame, robust to GGPO rollbacks, and the server verifies
   predicted pose vs observed every few frames.

## The strength dial (`--strength`, new)

One scalar in [0,1], applied at the decision layer:

- **Plain path**: value-gap rule over policy logits (same as
  `tools/eval_policy.py --strength`): sample uniformly among candidates whose
  logit is within `(1-strength) * logit_range` of the best.
- **Search path**: the same rule over the *searched Q values* of the beam's
  root candidates. Beam preselection keeps even strength=0 picks plausible —
  the agent makes believable suboptimal placements, not random ones.

Composable notches, weakest to strongest:
`--strength 0..1` (choice quality) x `--temperature` (plain-path entropy) x
`--search`/`--ponder` + beam width (lookahead depth). Calibration of dial
values to human Elo bands is R11 in the design review: fit
`strength -> (rank1 rate, value_gap distribution)` against the annotated
corpus per WHR band, validate ~50% win rate vs the matching BC band net.

## Gaps / next steps

- **Reaction-time model** (R12): the dial degrades *choice*, not *execution* —
  scripts are still frame-perfect. For believable low-Elo play, filter
  frame-perfect-only candidates (cost outliers vs a relaxed replan) and/or
  inject decision latency. Exhibition polish, not strength.
- **Dial calibration** (R11) — needs the corpus join; mechanism is in place.
- **Fightcade injection**: stock Fightcade disables netplay Lua; a patched
  build or virtual-controller path is required for ranked play (etiquette
  requirements in LIVE_BRIDGE_PROTOCOL.md).
- Search still simulates the own board only (1P approximation); opponent
  modeling in live search is gated behind the opponent-obs re-test (R6).
