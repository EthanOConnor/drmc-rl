# VS Tournaments and SPRT Change Gates

`tools/tournament.py` compares VS agents with statistically grounded
infrastructure: resumable round-robin tournaments, a sqlite results store,
Wilson/Elo reporting, and a sequential SPRT gate for "is change X better?"
questions. Matches run on the native VS pool
(`drmc_rl/training/envs/drmario_vs_vec.DrMarioVsPoolVecEnv`) at level 14 / speed HI by
default, reusing the head-to-head machinery from `tools/vs_head_to_head.py`.

## Defining a roster

A roster is a yaml file with an `entries` list. Each entry is a named agent:

```yaml
entries:
  - name: vs2-tip
    checkpoint: runs/vs2_02/checkpoints/smdp_ppo_step540020887.pt.gz
    mode: plain                      # deterministic argmax (live-bridge default)
  - name: vs2-tip-search
    checkpoint: runs/vs2_02/checkpoints/smdp_ppo_step540020887.pt.gz
    mode: search                     # depth-2 SearchPolicy
    params: {beam: 8, deadline_ms: 60}
  - name: vs2-tip-ponder
    checkpoint: runs/vs2_02/checkpoints/smdp_ppo_step540020887.pt.gz
    mode: ponder                     # PonderingSearchPolicy (simulated dead time)
    params: {beam: 8, deadline_ms: 60, ponder_budget_s: 1.0}
```

Checkpoints are repo-relative (absolute paths also work); the plain policy
prefers `ema_state_dict`, matching `tools/eval_policy.py`. Search/ponder
entries take `device` in `params` (default: the runner device, mps if
available). Ponder entries share one ponder slot per policy instance, so run
ponder rosters with `--pairs 1` or the hit rate craters.

## Running a tournament

```bash
.venv/bin/python -m tools.tournament run \
    --roster my_roster.yaml --games-per-pair 200 --level 14 \
    [--name my_tourney] [--pairs 4] [--seed 12345] \
    [--db runs/tournaments/tournaments.sqlite]
```

- Full round-robin over the roster: every unordered pair plays
  `--games-per-pair` games, alternating physical sides (game 0: A on side 0;
  game 1: A on side 1; ...) with per-game NES RNG seeds derived
  deterministically from `--seed`, the pair names, and the game index.
- Every finished game is one sqlite row, committed immediately, so a killed
  run resumes exactly where it stopped (`run` again with the same name/seed;
  recorded game indices are skipped). Re-running a complete tournament is a
  no-op. Raising `--games-per-pair` later extends every pair's series.
- `--pairs N` plays N games of the current matchup concurrently (plain
  policies are fast; search policies serialize on `decide`, so high pair
  counts mostly help plain rosters).

## Reading the report

```bash
.venv/bin/python -m tools.tournament report --tournament my_tourney
```

Outputs (plain text):

1. **Pairwise W-L-D matrix** (row's record vs column) and per-pair win rates
   over decisive games with **Wilson 95% CIs** (draws shown separately, same
   convention as `tools/vs_head_to_head.py`).
2. **Elo table**: maximum-likelihood Bradley-Terry/logistic ratings over all
   recorded games (draws scored 0.5), anchored to mean 0, sorted descending.
   The ±95 column is 1.96× the standard error from the observed Fisher
   information (pseudo-inverse of the rating-graph Laplacian, i.e. the
   covariance in the mean-zero gauge). An entry that never loses has a
   divergent MLE — its rating is capped by iteration damping and its CI blows
   up; play more games.

## SPRT change gating

To gate a change ("does candidate beat baseline by ≥ elo1?"):

```bash
.venv/bin/python -m tools.tournament sprt \
    --a cand=runs/exp/checkpoints/new.pt.gz,mode=search,beam=8,deadline_ms=60 \
    --b base=runs/vs2_02/checkpoints/smdp_ppo_step540020887.pt.gz,mode=plain \
    --elo0 0 --elo1 5 --alpha 0.05 --beta 0.05 --max-games 400
```

Entry specs are `NAME=CHECKPOINT[,mode=plain|search|ponder][,k=v...]`. Games
run strictly sequentially (one env pair) with alternating sides; after each
game the runner prints the running trinomial log-likelihood ratio and stops at

- `LLR ≥ log((1-β)/α)` → **accept H1** (elo ≥ elo1; ship it),
- `LLR ≤ log(β/(1-α))` → **accept H0** (elo ≤ elo0; reject),
- `--max-games` → inconclusive.

The LLR is the fishtest-style BayesElo trinomial model: the draw rate is
estimated from the observed W/D/L (zero cells regularized at 0.5 games — VS
draws are rare and must not stall the test) and the W/D/L probabilities under
elo0 vs elo1 enter a three-term likelihood ratio. Games are recorded in the
same sqlite store under `sprt_<a>_vs_<b>_<date>` (override with `--name`);
re-running the same name resumes from the recorded counts.

## Statistical conventions and recommended game counts

- **Wilson 95% CI** on win rate over decisive games; draws reported but not
  folded into the win rate.
- **Elo MLE**: logistic model `P(A beats B) = 1/(1+10^(-(Ra-Rb)/400))`, draws
  = half points, mean-zero anchor, Fisher-information SEs.
- **SPRT** with `alpha = beta = 0.05` gives LLR bounds ±2.944.

Rough planning numbers (near 50% win rates, low draw rates):

- Elo SE per pair ≈ `1/(2·k·√(N·p(1-p)))` ≈ `350/√N` per entry; so **~±10 Elo
  (95%) resolution needs ~1000 games per pair**, ~±20 Elo needs ~300.
- An SPRT at (elo0=0, elo1=5) typically resolves in a **few hundred games**
  when the true difference sits at or beyond one of the hypotheses; expect the
  worst case (true elo midway) to hit `--max-games`. Wider hypothesis gaps
  (e.g. elo1=10) resolve in well under 200 games.
- Plain-vs-plain games at level 14 take a few seconds of wall time each on the
  native pool; search entries cost ~60 ms per decision, so budget accordingly
  (a few hundred search games is hours, not minutes).

## Files

- `tools/tournament.py` — CLI (`run` / `report` / `sprt`), stats, store,
  match runner.
- `tests/test_tournament.py` — stats references, resumability with a stub
  runner.
- Store: `runs/tournaments/tournaments.sqlite` (tables `tournaments`,
  `games`; one row per game).
