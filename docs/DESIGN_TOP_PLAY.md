# Top-Play System Design — Beating Strong Humans at Dr. Mario (June 2026)

Status: design + partial implementation. The throughput substrate (warp
execution + v4 planner + sync-free training loop) landed 2026-06; this
document specifies the rest of the system: what it takes for this codebase to
produce an agent with a realistic shot at outplaying top humans in both
**speedrun** (fastest virus clears) and **VS** (head-to-head battle) modes,
with **adjustable strength** for human play.

It is grounded in what actually works in 2025–2026 practice: large-batch PPO
with exact action masking and shaped curricula for single-agent play;
prioritized fictitious self-play (PFSP) leagues with exploiters for
adversarial play (the AlphaStar/OpenAI-Five recipe, scaled down to a game
whose decision space is tiny by comparison); and behavioral strength
conditioning rather than post-hoc noise injection.

---

## 1. Why this game is winnable

Dr. Mario's decision problem is small but deep:

- ~30–100 legal placements per spawn (we enumerate them exactly, with exact
  frame costs, in ~0.1–4 ms).
- Episode horizon: tens to low hundreds of decisions.
- Perfect information except pill sequence beyond the preview (and the
  128-entry pill reserve cycles deterministically per seed — the agent can in
  principle learn distributional regularities humans cannot track).
- Human constraints the agent does not share: ~200 ms reaction time, imperfect
  frame timing for tucks/kicks, and limited lookahead under speed pressure.
  The planner gives the agent *every* frame-perfect maneuver (wall-charged DAS
  tucks, double-left kicks, parity-timed soft drops) as primitive actions.

Top humans clear high levels with near-optimal placement chains. The agent's
edge must come from (a) frame-perfect execution for free, (b) globally better
placement sequencing learned from millions of games, (c) exact speed/risk
tradeoffs via cost-aware candidate scoring.

## 2. Architecture (current + planned)

```
┌────────────────────────────────────────────────────────────┐
│ native pool (C++, in-process, N envs, thread-parallel)     │
│  • exact NES rules engine (GameLogic)                      │
│  • v4 reachability: feasible set + exact frame costs       │
│  • warp execution: decision → decision, no fall frames     │
│  • [VS] two boards + attack/garbage rules + speed/combo    │
└──────────────┬─────────────────────────────────────────────┘
               │ batched arrays (obs, masks, costs, events)
┌──────────────┴─────────────────────────────────────────────┐
│ SMDP-PPO trainer (torch, MPS/CUDA)                         │
│  • candidate scorer: per-feasible-placement embeddings     │
│  • γ^τ discounting over exact frame costs                  │
│  • curriculum (ln_hop_back) → level/speed ladder           │
│  • [VS] league: main agents + exploiters + past selves     │
└──────────────┬─────────────────────────────────────────────┘
               │ checkpoints
┌──────────────┴─────────────────────────────────────────────┐
│ eval harness                                               │
│  • tools/eval_policy: fixed-seed level sweeps              │
│  • [VS] round-robin Elo vs frozen pool + scripted bots     │
│  • human-play bridge (libretro parity lane) for exhibition │
└────────────────────────────────────────────────────────────┘
```

## 3. Speedrun track (single player)

Objective: minimal frames to clear level L at speed HI, across all RNG seeds.

1. **Reward**: current virus-clear normalization + terminal time-goal reward
   (already implemented as soft task budgets). Once clear rates saturate at a
   level, the curriculum's mastery-gated time budgets tighten toward
   best-known times (BestTimesDB floors).
2. **Capacity schedule**: the 363k-param candidate CNN is a throughput-tuned
   starter. Plan: widen to d_model 256 / 4–6 residual blocks (~2–4M params)
   once curriculum reaches positive levels; throughput math (19k dec/s policy
   at batch 128 vs ~3k dec/s env) leaves a ~6× capacity margin before
   inference becomes the binding constraint.
3. **Preview + reserve exploitation**: the policy sees current + preview pill.
   Add a small recurrent state (GRU over decision embeddings) so it can carry
   pill-sequence statistics; cheap at SMDP cadence and known-useful for
   planning two pills ahead.
4. **Search at eval time (optional, big lever)**: the warp engine simulates a
   placement in ~30 µs, so depth-2 expectimax over (placement × next-pill
   distribution) with the value net as leaf evaluator costs ~1–3 ms per
   decision — viable even live at 60 Hz with the decision latched at spawn.
   This is the classic "policy for training, shallow search for matches"
   pattern and is how the system should play exhibitions.
5. **Targets**: (a) clear rate ≈ 1.0 at level 20 / speed HI; (b) median
   clear-time within 5% of TAS-known times on fixed seeds; (c) beat top human
   PBs on standard speedrun categories simulated seed-for-seed.

## 4. VS track (two player)

### Engine work (native, exact)

Port the ROM's 2P rules into the pool (the disassembly submodule has the
attack tables):

- Combo detection at clear time → attack queue (number + colors of garbage
  determined by simultaneous clears).
- Garbage drop scheduling on the receiving board (columns from the ROM RNG).
- Win/loss: opponent top-out, or own clear; speed/level handicaps per ROM.
- Pool ABI: envs become *pairs*; step takes two action vectors, returns both
  sides' decision contexts. Reuse warp execution per side (each side's fall
  is still board-static; garbage lands at spawn boundaries).

### Training (league, PFSP)

Small-scale AlphaStar recipe — the action/observation spaces are tiny, so a
single workstation league is realistic:

1. **Seed** the league with the speedrun agent (clearing skill transfers; VS
   adds attack/defense timing).
2. **Main agents** (1–2): train vs a PFSP mixture weighted toward opponents
   with ~50% win rate (hardest informative opponents).
3. **Exploiters**: periodically spawn from current main, train *only* vs main
   to find degenerate strategies (e.g., garbage-timing cheese); fold their
   counters back via continued main training.
4. **Past-self snapshots** every N updates form the frozen pool; round-robin
   Elo over fixed seed sets tracks non-transitivity.
5. **Reward**: win/loss terminal ±1, small shaping on viruses cleared and
   garbage sent *early in league training only* (anneal to pure win/loss).
6. **Architecture additions**: opponent-board planes (full information per
   ROM rules — both boards are visible to humans too), attack-queue scalars
   in aux, and the same recurrent state.

### Targets

- Elo curve vs frozen pool monotone; main agent beats all scripted baselines
  and the speedrun agent adapted to VS.
- Exhibition: beat strong human players in Bo7 on stream-able libretro parity
  setup (the parity lane exists for exactly this).

## 5. Adjustable strength (human-facing)

Strength knobs that degrade *believably*, unlike ε-random:

1. **Value-gap sampling**: at each decision compute candidate logits; sample
   among candidates within Δ of the best, with Δ and temperature mapping to a
   strength dial calibrated against the Elo ladder.
2. **Reaction-time model**: latch the decision k frames after spawn (k ~ human
   distribution) and forbid frame-perfect-only maneuvers (filter candidates
   whose minimal script requires >X taps/sec) at lower strengths — both are
   trivial given exact costs/scripts.
3. **Persona conditioning (optional)**: condition the policy on a strength
   embedding trained with capped-KL distillation from intermediate
   checkpoints, so each strength level is a coherent policy rather than a
   noised expert.

Calibration: bucket strengths, measure win rates vs the frozen ladder, fit the
dial to target Elo spacings (~150–200 Elo per notch).

## 6. Throughput budget (measured, M3 Max 16-core)

| Component | Measured | Headroom |
|---|---:|---|
| Planner (v4, mid-game boards) | ~0.5 ms/spawn | NEON + factored expansion ≈ 2× if needed |
| Planner (worst sparse boards) | ~4–9 ms/spawn | horizon option / level mix amortizes |
| Pool env-only (64 envs, random) | 182k FPS / 3.1k dec/s | linear in cores |
| Policy forward (MPS, batch 128) | 19k dec/s | compile, bigger batches |
| Full training loop | 64k FPS / ~950 dec/s | async rollout/update overlap ≈ 1.5–2× |

A 5M-frame run (~80 s) already clears level 0 at 12.5% from scratch; the
sample budget for the full curriculum (est. 1–5 B frames) is days, not weeks,
on this machine — and the stack ports to CUDA unchanged for league training.

## 7. Sequencing

1. ✅ Substrate: planner v4, warp execution, sync-free training loop, eval CLI.
2. Speedrun push: capacity bump + recurrence + long runs through the level
   ladder; track eval sweeps per checkpoint; tighten time budgets.
3. VS engine port (attack/garbage tables from disassembly) + 2P pool ABI +
   parity tests vs emulator 2P mode.
4. League infra: snapshot pool, PFSP sampler, Elo harness (extend
   tools/eval_policy to head-to-head).
5. Strength dial + exhibition bridge.

Risks and open questions tracked in notes/SCRUTINY.md; throughput numbers in
docs/BENCHMARKS_2026-06-09.md.
