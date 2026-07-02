# Design review: the path to a strong & tunable VS player (2026-07-02)

Scope: top-to-bottom review of every layer of the VS-player approach, written
as training moves from the Mac to tf3090 (RTX 3090, 4 CPU cores, 15 GB RAM).
One architectural commitment is FIXED by decision: the feasibility/planner
split — the policy sees board state plus a planner-derived list of feasible
placements, and the planner turns the chosen placement into emulator inputs.
Everything else is up for grabs.

Companion docs: docs/ARCHITECTURE_REVIEW_2026-06.md (previous review, still
largely valid), docs/ABLATION_PLAN.md (the empirical record this review leans
on), docs/DESIGN_TOP_PLAY.md (roadmap).

## 0. What changed since the June review

Two ground facts shifted; most recommendations below fall out of them.

1) **Compute moved.** The Mac had many CPU cores and MPS; tf3090 has 4 cores
   and a 3090. Measured on the running vs6 config: the pool trains at ~12k
   frames/s, ~223 decisions/s, with the GPU at ~3% utilization — and CPU
   planning (drm_reach_bfs_v4, measured 11.9 ms/solve on realistic mid-game
   boards at HI speed) costs ~2.7 of the 4 cores. **Planning is the dominant
   CPU cost of training.** Meanwhile reach_cuda solves the same instances
   bit-exactly at 0.43 ms/solve wall (batch 256) for ~zero CPU. The design
   should treat the GPU as the planner, the search engine, and the net — the
   CPU only steps the emulator core.

2) **Data grew ~30×.** The annotated corpus is now 51.7M moves / 29,162
   quarks (planner-optimal costs, per-move rank and value_gap, repaired lock
   poses), fully mirrored on this box. The June-era corpus assets (BC nets
   from 98 quarks, start banks from 894) sampled a sliver of what now exists.

## 1. Layer-by-layer review

### 1.1 Action space & planner (FIXED, but not frozen in implementation)

The placement-SMDP candidate action space stays — it is also the best part of
the design: it converts a 60 Hz reflex game into ~1 decision/25 frames with
exact feasibility, and both the 1P champion and the human corpus annotations
validate it end to end.

Within the fixed architecture, three implementation recommendations:

- **R1 (done 2026-07-02): rollout planning moved to the GPU.** Deferred-plan
  mode in DrMarioVsPool: sides park at decision boundaries without running
  the CPU BFS; Python batches all parked sides through CudaReach and injects
  costs back (`drm_vspool_inject_plans`); bit-exact parity gated by
  tests/test_vs_gpu_planner.py. MEASURED RESULT, honest version: throughput-
  neutral at 16 pairs (11.9k vs 12.0k fps) — the 12 ms/solve estimate came
  from synthetic dense boards; typical in-play solves are much cheaper, and
  the loop turns out to be latency-bound in the serial Python decision path
  (obs build, candidate packing, opponent forwards), not planner-bound. The
  change still frees ~15-20% CPU (330% vs saturated 400%), scales to bigger
  pair counts (+13% fps at 32 pairs), and is the enabler for GPU search
  (R8), where solve batches are orders of magnitude larger. Keep it on.
- **R2 (revised): the next throughput lever is the per-decision Python
  overhead**, not planning: profile and vectorize the vec-env decision path
  (obs build + packing are numpy-per-side today; opponent forwards are
  CPU-torch). num_pairs scan measured: 16→32 pairs = 12.0k→13.6k fps
  (+13%); diminishing beyond (Python serial section is the wall).
- **R3: keep warp execution for training; scripts only at the edges.**
  reach_cuda scripts mode is replay-verified — use it for the live bridge
  and any frame-exact exhibition, not in the training loop.

### 1.2 Environment & reward (the metagame problem — highest design risk)

The empirical record (ABLATION_PLAN forensics + correction) is unambiguous:
pure self-play converged to attrition-at-the-ceiling — 40/40 matches decided
by inherited top-out, nobody below 5 viruses, median 20 min. Clearing is NOT
hard (the 1P lineage clears L15 ~73%); self-play *eroded* it. Against humans
who cure to win, this is the single biggest exposure. The vs4→vs6 fix bundle
(clear_win_bonus + BC league + near-clear start bank) is the right shape and
is retained in the tf3090 config. Additions now unlocked by the corpus:

- **R4: extract REAL human clear-endgames** as a start bank (the earlier
  clear_practice bank thins viruses artificially; the corpus contains actual
  human closing sequences — the sampler just stopped before them). One new
  strata pass over spawn events with viruses_rem ≤ 8 does it. Enable
  start_bank at fraction ~0.35 in the next config rev.
- **R5: keep terminal ±1 + clear_win_bonus; resist dense shaping.** The
  forensics show chronic mutual pressure decides matches; garbage-differential
  shaping would reward exactly the camping equilibrium we are trying to leave.
  The bonus (0.25, reordering win types only) plus endgame reps is the
  evidence-backed lever. Revisit only with anchored tournament evidence.
- The draw/truncation path and volley accounting in drmario_vs_vec are sound;
  no changes recommended.

### 1.3 Observation

Opponent-board observation FAILED its ablation decisively in the attrition
metagame (18% vs 49% control; reliance probe inverted) and was rightly
dropped. But the test is confounded by the metagame itself: when neither side
races, opponent state carries little decision-relevant signal.

- **R6: re-run the opponent-obs ablation only after the clear-race metagame
  is established** (post-R4, once clear-wins are a nontrivial fraction of
  endings). The `_vs` plumbing (20ch obs, aux v1_vs, expand_checkpoint
  surgery) is built and tested; this is a cheap re-test with the existing
  harness. Until then, 8-channel own-board obs stays.

### 1.4 Model

The candidate transformer (d192, 4 blocks, ~48 MB ckpt) was sized for Mac
inference. On the 3090 the net is nearly free (0.25 ms/decision average).
But: capacity was NOT the binding constraint in any recorded failure — the
metagame was. So:

- **R7: hold the architecture until the metagame fix is validated, then do
  one deliberate capacity bump** (e.g. d256-d320, 6 blocks) as its own gated
  ladder step, warm-started via expand_checkpoint. Don't bundle it with other
  changes; the 1P lesson was that bundled interventions are unreadable.
- GRU recurrence (roadmap item) stays parked: the placement SMDP with full
  preview is near-Markov; recurrence buys little until opponent-modeling
  (which is gated behind R6 anyway).

### 1.5 Algorithm

PPO-SMDP with γ=1.0/GAE 0.98 at low LR (3.5e-5 warm-start) is stable and its
implementation is battle-tested. The validated Elo headroom is in
**search-based policy improvement**: depth-2 beam search beats the plain
policy 81.7% head-to-head with the SAME net. The distillation path
(search_distill, phase 1) and act_from_search (phase 2) are implemented and
config-gated but unproven — the ladder stalled when the Mac era ended, partly
because search was CPU-expensive (39 ms/searched decision, and VS search runs
on a 1P approximation of the own board).

- **R8: after R1, wire reach_cuda into SearchPolicy's per-branch replans and
  raise the sims budget.** Search cost is planner-dominated; on GPU the
  vsdist config's sims=12/frac=0.1 becomes cheap enough to run at
  frac=0.25-0.5 or deeper beams. This is the highest-Elo algorithmic step on
  the books, and it composes with everything else.
- **R9: the vsdist → vsact ladder stays the plan**, gated exactly as
  ABLATION_PLAN specifies (anchored tournaments, SPRT, bc-gt2000 gate). The
  vsact config's init placeholder must be filled from the vsdist verdict.
- Offline pretraining from the corpus (BC or offline RL on 51.7M annotated
  human moves, planner-optimality labels included) is now feasible at scale.
  Not recommended as a replacement for the warm-start lineage — but as the
  *skill-restoration* tool if the clear-retention probe fails (ABLATION's
  open question), a corpus-BC phase targeted at endgame decisions is the
  cheapest fix. Keep in reserve.

### 1.6 Opponents & league

PFSP + league roles (pfsp/exploiter/mixed) and pool persistence are solid and
tested. The BC opponents are the weakest asset: trained on 50k moves/band
from a 98-quark June corpus slice — now 3 orders of magnitude more data
exists.

- **R10: retrain the BC band nets on the full corpus** (e.g. 1-2M moves/band,
  more bands — the WHR spread supports lt1400/1400-1700/1700-2000/gt2000),
  possibly at d128 rather than d96. Better human-style league seeds AND
  better strength-dial anchors (below). Cheap on this box (CUDA extraction
  already built; the 3-band set took 40 s + minutes of training).
- vs6's league anchor on this box is best-535m (the Mac's vs_champion_530m
  and vs1p_gatebest are not here). **Sync the Mac's runs/best_agents/ and
  tournaments.sqlite over when convenient** — they are the historical Elo
  anchors; without them every new tournament floats relative to old numbers.

### 1.7 Strength dial (the "tunable" half of the goal)

Built: value-gap dial (`--strength`, sample within Δ of best logit),
temperature, search beam notches, and BC rating-band opponents. Missing: any
*calibration* of dial settings to human ratings, and the reaction-time model.

The corpus annotation makes calibration direct — this is new since June:

- **R11: calibrate the dial against the annotated corpus.** Every human move
  carries (rank, value_gap) vs the planner-optimal move, joined to a WHR
  rating. Compute per-band distributions of value_gap/rank1-rate, then fit
  the dial (Δ, temperature, beam) so the AGENT's induced value_gap/rank
  distribution matches each band's — validated by playing the dialed agent
  against the matching BC band and checking ~50% win rate. Deliverable: a
  monotone strength→Elo curve, which is exactly what "tunable" should mean.
- **R12 (later, for believability): reaction-time/latch model** for human-like
  weaker settings (frame-perfect maneuvers filtered at low dial values). The
  value-gap dial degrades *choice*, not *execution*; low-Elo humans differ in
  both. Only matters for exhibition realism, not for training.

### 1.8 Evaluation & ops

Tournaments/SPRT/forensics/dashboard are in good shape; standing rules
(anchored tournaments only, ≥200 games or SPRT, behavioral metrics recorded)
stay. Ops notes for this box:

- Checkpoint retention now exists (`train.checkpoint_keep_last`, implemented
  2026-07-02) — vs6_tf3090 keeps 6 × 25M-step checkpoints ≈ 300 MB/run.
  Promote keepers to runs/best_agents/ by hand; retention never touches them.
- Disk is 7 GB free until the bigger drive arrives: no videos (opencv not
  installed → placeholder bytes), TB kept, metrics.jsonl.gz is the compact
  record. The fightcade extractors on this box are stopped while training
  owns the CPU.
- `envs/pettingzoo/` is a non-functional mock — ignore or delete.

## 2. Recommended sequence

1. **R1** GPU planner in the VS rollout loop (+ parity gate + R2 pair scan) —
   in progress.
2. **R4** real clear-endgame bank from the corpus; enable start_bank; **R10**
   full-corpus BC bands. (Both are hours of work, CPU-light after R1.)
3. Continue the vs6-lineage run through the metagame gate from ABLATION_PLAN
   (clear-win rate > 0 sustained, BC-band gate, anchored tournament vs
   best-535m).
4. **R8/R9** GPU-powered search distillation (vsdist rev B), then
   act-from-search, each SPRT-gated.
5. **R6** opponent-obs re-test in the new metagame; **R7** capacity bump.
6. **R11** dial calibration → publishable strength→Elo curve; **R12** if/when
   exhibition realism matters.

Everything above keeps the fixed feasibility/planner contract; nothing in the
record argues against it, and both the champion lineage and the human corpus
pipeline are built on it.
