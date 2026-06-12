# Human VS Corpus Integration (fightcadeRatings ↔ drmc-rl)

Status: spec, 2026-06-10. Counterpart doc:
`../fightcadeRatings/docs/DRMC_RL_INTEGRATION.md`.

`../fightcadeRatings` is building a corpus of ranked Fightcade `nes_drmario`
VS games: raw GGPO replay streams (content-addressed), local re-emulation
through a headless fcadefbneo harness that emits JSONL RAM-change events,
crown/game segmentation with DrMC metrics, and Whole-History Ratings per
player. This document specs how drmc-rl consumes that corpus and what we
provide back. Division of labor in one line: **they own replay acquisition
and re-emulation truth; we own game-rules intelligence (planner, native
engine, RAM semantics) layered on top.**

## What the corpus is for (drmc-rl side)

Priority order:

1. **2P engine parity ground truth.** The native pool needs the ROM's VS
   rules (combo→attack tables, garbage volley scheduling, win/loss). The
   re-emulated replays are frame-exact traces of real 2P games — the same
   role `data/nes_demo.json` played for the 1P port. Every attack volley in
   the event stream (already emitted: columns + fall depths) becomes a parity
   assertion against our implementation.
2. **Strength-dial calibration.** WHR gives a rating for each side of every
   game. Mapping our value-gap dial / league Elo onto the human WHR scale
   makes "strength 6" externally meaningful.
3. **League seeding & diversity.** Behavior-cloned "human-like" opponents at
   rating buckets (e.g., quartiles of WHR) join the PFSP opponent pool so
   exploiters train against human styles, not just self-play conventions.
4. **Held-out eval.** Game-start and mid-game positions with known outcomes
   and known-strength players: "win from here vs a 1700-rated human style"
   probes before any exhibition.

A few thousand games (~1-2M decisions) is far too small to be the primary
training signal and that is fine — self-play remains the strength engine.

## Event schema v2 (harness additions, fightcadeRatings side)

The current harness emits scalar watches + garbage volleys. Decision-level
extraction needs four additions (all cheap: a handful of extra
`M6502CheatRead`s on existing change-detection paths):

1. **Game-init record** — on `$0046` entering gameplay / virus-count reset:
   `{"t":"init","f":N,"rng":[$0017,$0018],"lvl":[$0316,$0396],"spd":[$030B,$038B],
   "field1":b64(128B),"field2":b64(128B)}`
   The RNG bytes + level/speed are what our engine needs to replay a game
   from init (the 1P parity reset path already consumes exactly these).
2. **Spawn record** — per player, when the pill spawn counter changes
   (P1 `$0310` family / P2 `+$80`; reserve index `$0327/$03A7`):
   `{"t":"spawn","p":1,"f":N,"pill":[$0301,$0302],"prev":[$031A,$031B],
   "field":b64(128B),"spd":...,"spdups":...}`
   Field snapshot at spawn = the decision-time observation.
3. **Lock record** — per player, on leaving the pill-falling state:
   `{"t":"lock","p":1,"f":N,"x":$0305,"y":$0306,"rot":$0325}`
   (P2 mirrors at `+$80`.) Spawn→lock frame delta = the human's τ.
4. **Schema/version field** on every line (`"v":2`) and in
   `processed_replay.method`.

Everything else (crowns, whoWon, combos, volleys) stays as-is. Fields are
b64 to keep lines compact; 128 bytes/snapshot ≈ 172 chars.

## drmc-rl ingestion (new tools, this repo)

1. `tools/ingest_fc_corpus.py` — reads `fightcadeRatings/data/drmario.sqlite`
   (`processed_replay` blobs with v2 events + `crown` rows) and the WHR
   output; joins per-game player ratings; emits
   `data/human_vs/decisions.parquet` with rows:
   `(quarkid, game_idx, player, whr, opp_whr, frame, field[128], pill, preview,
   lock_x, lock_y, lock_rot, tau_frames, garbage_pending, speed, outcome)`.
2. `tools/annotate_replay_events.py` — runs the v4 planner on every spawn
   record: feasible set, minimal cost to the chosen pose, minimal cost to the
   best pose. Derived per-move metrics:
   - **execution slack**: human τ − planner-minimal τ for the chosen pose
   - **placement rank**: value-net rank of the chosen pose among feasible
   - **tuck rate**: chosen poses unreachable without kicks/wall-charges
   These flow BACK to fightcadeRatings as new per-player skill axes (see
   counterpart doc) and serve us as BC targets + dial calibration anchors.
3. BC pipeline: rating-bucketed behavior cloning on `decisions.parquet`
   (the candidate policy net consumes (field, pill, preview, feasible-set)
   tuples natively — no architecture changes needed).

## Parity workflow (both repos)

1. fightcadeRatings extracts v2 events for a replay.
2. drmc-rl replays the same game on the native 2P engine (once ported):
   init from the `init` record (RNG bytes, level, speed), drive both players
   with the lock-pose sequence (warp execution), assert: virus counts, combo
   counters, attack sizes, volley columns/depths, crown transitions match
   the event stream frame-for-frame (modulo render-only state).
3. Mismatch triage: their RAM map vs our disassembly-derived semantics
   (e.g., reconcile `$0310` "pills-thrown BCD" vs our 1P note "pill spawn
   counter" — both repos cite the same disassembly; one description is
   stale).

This is mutual testing: their extraction validates our 2P port; our
rules-exact engine flags any re-emulation/segmentation bugs in their
pipeline.

## Sequencing

1. Now (corpus still building): land schema v2 in the harness — cheap, and
   every replay processed before it exists must be re-run later (blobs are
   kept, so re-extraction is local and free, but do it once, early).
2. drmc-rl: `ingest_fc_corpus.py` against v2 events; planner annotation tool.
3. 2P engine port (BACKLOG P0 item 2) with the replay parity workflow as its
   acceptance test — this replaces the "build 2P parity fixtures by hand"
   plan; the corpus IS the fixture set.
4. WHR join → dial calibration; BC opponents → league seeding (P0 items 3-4).

## Skill grading: metrics → WHR regression (`tools/skill_grade.py`)

Since the agent cannot play on Fightcade, we estimate its position on the
fightcadeRatings WHR scale by regression from DrMC play metrics to human
ratings. `tools/skill_grade.py fit` joins each valid `crown` row (both
sides) in `../fightcadeRatings/data/drmario.sqlite` to that player's
corrected WHR-C rating (`eC` in `data/out/players.json` trajectories,
nearest day), filtering known extraction artifacts (length < 30 s or < 5
pills per side). Features per player-side: CPM, CUR, SPD, SALT/min,
pills/min, garbage/min — outcome (won/lost) is deliberately excluded.
Model: numpy-only weighted ridge on standardized degree-2 features
(squares + interactions), samples weighted 1/n_crowns(player), CV grouped
by player. `grade` mode maps agent per-game metrics (JSON/JSONL) to an
estimated rating with the CV residual std as uncertainty and flags
out-of-human-range features as extrapolation.

Fit quality (2026-06-10, 329 crowns → 658 samples, 45 rated players,
target range 704–2751 Elo):

- grouped CV MAE **289 Elo** (constant-prediction baseline: 403), residual
  std 362 Elo, ridge alpha 10.
- direction sanity checks (weighted r vs rating): CPM +0.65,
  SALT/min +0.68, garbage/min +0.67, SPD +0.51, CUR +0.49,
  pills/min +0.18 — all positive, as expected.

Caveats (see module docstring): this grades playstyle, not head-to-head
strength; an ~300-Elo blur is inherent; superhuman metric values are
extrapolation (grade mode flags them); self-play metric distributions are
not identical to human-vs-human ones, so stage-over-stage comparisons are
more meaningful than absolute levels. Refit as the corpus grows.

## BC human-style opponents (`tools/train_bc_opponent.py`, 2026-06-12)

Behavior-cloned (state -> placement) opponents at three WHR bands, for league
style diversity. Extraction reads the corpus read-only (v2 spawn/lock events
+ `raw_quark` player names + `players.json` WHR-C trajectories, nearest-day
join as in `tools/skill_grade.py`); the spawn/lock pairing and planner
machinery is reused from `tools/annotate_replay_events.py`. Per move we keep
the decision-time field, pill/preview colors, the planner-feasible candidate
set (packed exactly like the training envs, `sort_by_cost=True`, K=128), and
the slot of the human's placement. Same-color pills mirror the envs'
symmetry reduction (orientations 2/3 masked, chosen action canonicalized).
Moves whose chosen pose is planner-infeasible (~0.24%) are dropped.

Rebuild:

    nice -n 19 .venv/bin/python -m tools.train_bc_opponent extract \
        --max-moves-per-band 50000          # -> data/human_vs/bc_dataset_v1.npz
    nice -n 19 .venv/bin/python -m tools.train_bc_opponent train \
        --epochs 4                          # -> runs/bc_opponents/bc_<band>.pt.gz

Bands: `lt1600` (<1600), `1600to2000`, `gt2000` (>=2000 WHR-C). The nets are
small candidate-policy nets (d_model 96, 2 CNN blocks + 2 transformer layers,
aux_spec `none`); the architecture config is embedded in each checkpoint's
`cfg`, so `OpponentPool.ensure_loaded` -> `_build_net_from_cfg` reconstructs
them with no code changes. Per-band metrics land in
`runs/bc_opponents/bc_summary.json` (val = held-out quarks).

Enable in a VS run by seeding the opponent pool with the BC checkpoints,
e.g. in the run config:

    env:
      opponent_pool:
        enabled: true
        seed_paths:
          - runs/best_agents/vs_champion_smdp_ppo_step530046434.pt.gz
          - runs/bc_opponents/bc_lt1600.pt.gz
          - runs/bc_opponents/bc_1600to2000.pt.gz
          - runs/bc_opponents/bc_gt2000.pt.gz

(or copy the files into an existing `<logdir>/opponent_pool/` and add
`{"id": "bc_<band>", "file": "bc_<band>.pt.gz", "protected": true,
"wins": 0, "games": 0}` entries to its `manifest.json`).

## Go-Exploit start-state bank (`tools/build_start_bank.py`, 2026-06-12)

Mid-game two-board positions sampled from corpus games, loaded as native
VS-pool checkpoint resets so training can start from real human crises
instead of only clean level starts. Requires the VS checkpoint reset added
to the engine (`DrmVsResetSpec.checkpoint_*`, applied via
`GameLogic::loadCheckpoint` per side; rebuild with
`make -C game_engine libdrmario_pool`).

Reconstructibility (documented in the tool docstring): boards / pills /
counters are exact from spawn-record snapshots; the two sides' snapshots are
within one move of each other (the partner's board comes from its latest
spawn <= the sampled frame); mid-game RNG is NOT reconstructible from the
event stream, so the loader randomizes the seed — realistic boards,
stochastic continuations, which is all Go-Exploit needs. In-flight stored
attacks are dropped (volleys only release between moves, so the boards
already contain all landed garbage).

Strata per row: `early` (spawn 8-20), `mid` (21-45), `late` (46+),
`crisis` (receiver's first spawn within 90 frames of a garbage volley).
Filters: both sides >= 1 virus, spawn cells free.

Rebuild + validate (plays N rows to completion in the native pool):

    nice -n 19 .venv/bin/python -m tools.build_start_bank --validate 64
    # -> runs/start_bank/start_bank_v1.npz (+ .json build report)

Enable in a VS run (`training/envs/start_bank.py` loader, sampled in
`DrMarioVsPoolVecEnv._build_reset_specs`):

    env:
      start_bank:
        enabled: true
        path: runs/start_bank/start_bank_v1.npz
        fraction: 0.25      # of episode resets drawn from the bank

`enabled: false` (or absent) is bit-identical to the previous reset path —
no extra RNG draws.
