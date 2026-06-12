# Ablation & Characterization Plan (endpoint phase, June 2026)

Pre-committed methodology so conclusions aren't drawn from whatever
happened to be measured. All results land in the tournament store
(`runs/tournaments/tournaments.sqlite`); verdicts use the conventions in
docs/TOURNAMENTS.md (Wilson CIs for trends, ≥200-game or SPRT for
verdicts).

## Anchors

- Primary: vanilla 530M VS champion
  (`runs/best_agents/vs_champion_smdp_ppo_step530046434.pt.gz`) — the
  round-robin-verified strongest agent (vs_lineage_rr_2026-06-12).
- Secondary (sensitivity, occasional): vs2-515m and the 1P champion
  535M. A change that helps only against one anchor is suspect.

## Active experiment: opponent-board observability (vs3)

Hypothesis under test includes the NULL: opponent info may not matter at
current skill, and the extra input pathway may cost playing ability.
Three measurements, all automated by the evalwatch
(`runs/vs3_evalwatch.sh`):

1. **Treatment curve**: vs3_01 tip vs champion, every checkpoint
   (60 games/pair trend points).
2. **Control curve**: vs3ctl_01 — the un-surgered 8-channel champion
   trained under the byte-identical regime (same PFSP pool config, level,
   hyperparams; only the obs gate differs — see
   training/configs/vs3_control.yaml vs vs3_opponent_obs.yaml). The
   opponent-obs effect is the DIFFERENCE of the curves at equal frames;
   either curve alone conflates obs effect with resume transient +
   continued-self-play effect.
3. **Reliance probe**: each vs3 checkpoint also fights AS ITSELF with
   opponent planes + vs aux zeroed (`params: {mask_opponent: true}`).
   - masked ≈ unmasked → the net isn't using opponent info (null
     supported regardless of Elo trajectory).
   - masked << unmasked AND treatment ≤ control → it uses the info but
     the info doesn't pay at this level (capacity/interference cost).
   - treatment > control with growing reliance gap → obs validated.

Decision points (no action before): first read at ~50M frames, verdict
at ~100-150M or plateau. Verdict = ≥200-game tournament of best
treatment vs best control checkpoint (equal frames) + SPRT confirmation.

## vs4 component ladder

Components enter ONE AT A TIME on top of the last accepted
configuration, each gated by SPRT (elo0=0, elo1=5) or a ≥200-game
tournament against the predecessor's best checkpoint:

1. opponent obs (vs3, running — may be REJECTED per above; ladder
   continues from whichever of vs3/vs3ctl wins)
2. search distillation (fraction 0.1; phase 2)
3. league mixed mode (champion targets, exploiter_fraction 0.2)
4. BC style seeds in the pool
5. start-bank resets (fraction 0.25)
6. opponent_model: self at search leaves
7. act_from_search (full Gumbel-AZ)

Order rationale: expected Elo per review ranking; cheap-to-revert last.
If a step fails its gate, drop the component (recording the result) and
continue the ladder without it. Interactions (e.g., obs × distillation)
are tested only after the one-at-a-time pass, and only for accepted
components.

## Standing rules

- Never judge from training-time win rate (it's vs the PFSP pool, a
  moving target); only anchored tournaments count.
- Trend points 60-100 games; verdicts ≥200 games or SPRT.
- High-N fresh-seed verification before promoting any "best" checkpoint
  (the 12-ep/fixed-seed lesson from the 1P campaign).
- Behavioral metrics (cpm, garbage/min, match length, WHR grade) are
  recorded alongside Elo to explain WHY a change wins or loses, not as
  acceptance criteria.
