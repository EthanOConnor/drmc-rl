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

## Opponent-obs verdict (2026-06-12, CLOSED — dropped from vs4)

50M-frame read was decisive against: treatment 18% vs control 49%
against the champion anchor (non-overlapping CIs), reliance probe
INVERTED (masked beat unmasked at 20M and 40M), and the 200-game
head-to-head verdict (vs3@65M vs vs3ctl@50M) was 96-104 — parity
despite a 15M-frame treatment advantage. Conclusion: at this skill
level, in the attrition metagame, opponent-board obs is at best
useless and the extra input pathway interferes with training. Caveat
recorded for the re-test: this metagame barely uses opponent state;
inside a clear-race metagame (vs4+) the opponent board is race
progress — re-queue obs as a later ladder step there. Tournaments:
vs3_eval_step*, vs3ctl_eval_step*, vs3_verdict_obs_vs_ctl.

## Live pipeline to the endpoint (2026-06-12)

The "fully realized endpoint" = one agent with the full stack (clearing
base + Gumbel-AZ search distillation + re-tested obs + act_from_search)
that beats the bc-gt2000 human gate. It's gated on training hours, run as
a sequenced pipeline (each stage = tens of M frames):

1. **A/B warm-start (RUNNING)**: vs5 (camped vs4-best) vs vs1p (1P-clearer
   535M). Both BC-league + bonus 0.25, no bank. Gate = win rate vs
   bc-gt2000. Early signal (<1M frames): vs1p clears 2x viruses/ep (24 vs
   13) and sends half the garbage — the 1P clearing skill survives the
   warm-start; vs5 stays attrition-style. Hypothesis: camping is a
   fine-tune trap of the camped champion. Verdict at ~50M frames.
2. **Distillation (STAGED, training/configs/vsdist_distill.yaml)**:
   Gumbel-AZ phase-1 search distillation (fraction 0.1, beta 1.0) on the
   A/B winner. Both A/B nets are 12-channel + aux v1 = distillation-
   compatible. Launch when a trainer slot frees (init_checkpoint =
   REPLACE_WITH_AB_WINNER_CHECKPOINT). Gate: SPRT vs the A/B-winner best.
3. **Opponent-obs re-test** inside the clear-race metagame (surgery +
   control twin) — only meaningful once an agent actually clears/scouts.
4. **act_from_search** (full Gumbel-AZ) as the final amplification.
5. **Fallback if vs1p also camps**: BC-imitation distillation of the
   clearing skill (supervised, then RL) rather than relying on warm-start.

## vs4 component ladder

Components enter ONE AT A TIME on top of the last accepted
configuration, each gated by SPRT (elo0=0, elo1=5) or a ≥200-game
tournament against the predecessor's best checkpoint. REORDERED
2026-06-12 after the forensics + BC-band + obs results:

1. metagame-fix bundle (vs4_01, running): clear_win_bonus 0.25 +
   start-bank resets 0.25 + BC league seeds + mixed league vs the
   champion. Bundled deliberately — co-dependent (the bonus is only
   discoverable from late-strata starts; BC opponents punish ignoring
   the clear race). Gates: champion tournament AND the BC-band gate
   (must beat bc-* convincingly) AND clear-win rate > 0 in eval games.
2. search distillation (fraction 0.1; 12-channel phase-1 path)
3. opponent obs RE-TEST inside the clear-race metagame (surgery +
   control twin again)
4. opponent_model: self at search leaves
5. act_from_search (full Gumbel-AZ)
6. exploiter runs against whatever champion emerges

Order rationale: expected Elo per review ranking; cheap-to-revert last.
If a step fails its gate, drop the component (recording the result) and
continue the ladder without it. Interactions (e.g., obs × distillation)
are tested only after the one-at-a-time pass, and only for accepted
components.

## Match-ending forensics (2026-06-12)

How do VS matches actually end? `tools/vs_death_forensics.py`: 40
plain-vs-plain matches (champion 530M both sides, deterministic argmax,
fresh seeds, level 14 HI), logging per-decision stack height,
viruses_rem, and garbage-release events per side
(`runs/forensics/vs_death_forensics_40.json`). All 40 decisive, no
draws; matches are long wars of attrition (median 20 min emulated, max
56 min, ~11 garbage half-pills/min received per side).

- **40%** of losses had ZERO garbage land on the loser in the final
  15 s (self-burial at the moment of death); only **10%** show an
  acute pressure spike (final-15 s garbage ≥ 2x the match average).
- Loser viruses_rem at death: mean 19.2, median 14 (of 60); winner
  mean 18.4, never 0 — no clear-win occurred in any match.
- Death is not a sudden tail collapse: in 37/40 losses the loser's
  final 10 decisions were all at stack height ≥14/16; mean tail slope
  in the zero-garbage losses is +0.06 rows/decision (9/16 exactly
  flat). Both sides live pinned at the bottle ceiling for most of the
  match.
- Cumulative garbage received still predicts losing: the loser
  received more total garbage than the winner in **31/40 (78%)**
  matches, but by a thin margin (mean 213 vs 203 half-pills).

Verdict: wins are predominantly *inherited* at the moment of death —
the loser tops itself out placing pills into a saturated board, with
no acute garbage spike in 90% of endings — but the saturation itself
is mutual chronic pressure (the steady ~11/min garbage stream keeps
both boards at the ceiling, and receiving slightly more of it over the
whole match is what eventually decides). The agent has learned
attrition-at-the-ceiling, not clear-to-win: in 40 matches nobody got
below 5 viruses. Against humans who clear to win this is a real
exposure — a 20-minute median match leaves ample time for a human to
hit the cure win condition the agent never contests. Caveats: volley
timestamps are garbage *release* (landing) events, so a volley
released just before the 15 s window can settle inside it;
deterministic argmax on both sides may lengthen matches relative to
the sampled-action training regime; decision-time stack heights can
include just-released garbage still falling.

### Correction (2026-06-12, important — earlier framing was wrong)

Do NOT read "no clear-win in 40 matches" as "clears are hard/rare." Two
things were conflated and corrected here:
- **Clearing is very achievable.** The 1P speed champion clears L15 ~73%
  / L20 ~58%. The capability lives in the same architecture and the VS
  champion's own warm-start lineage. The VS policy not clearing is a
  *learned pathology of the self-play reward* (camp-and-outlast beats
  race-to-clear when BOTH sides camp), not a property of the task. VS
  self-play eroded a clearing skill the lineage demonstrably had.
- **Humans clear constantly.** Win-by-clear is common in the
  fightcadeRatings corpus. The start bank's lack of near-clear positions
  is a SAMPLING artifact of tools/build_start_bank.py (it sampled at
  volley events + fixed pill-count strata, which stop well before the
  end-of-game clearing sequence), NOT a fact about human play. The real
  clear endgames are in the corpus; we just didn't extract them.

Consequences: (1) the clear_win_bonus is sound but had no gradient
because the agent never *completes* a clear to receive it (cur=0 across
all training windows is real, not a grading artifact — the grader uses
the bank-seeded training games). (2) Fix = give the agent reps at the
closing-out skill via near-clear start states. First cut:
tools/build_clear_practice_bank.py (real boards, viruses thinned to
2-8/side, median 5). Better follow-up: extract REAL near-clear endgames
from the corpus. (3) Open empirical question being probed before any
restart: does the champion still RETAIN enough clearing skill to finish
a clear from a 5-virus board? If yes, the bank bootstraps the bonus; if
no, we need 1P-distillation / BC-imitation to restore the skill first.

## Standing rules

- Never judge from training-time win rate (it's vs the PFSP pool, a
  moving target); only anchored tournaments count.
- Trend points 60-100 games; verdicts ≥200 games or SPRT.
- High-N fresh-seed verification before promoting any "best" checkpoint
  (the 12-ep/fixed-seed lesson from the 1P campaign).
- Behavioral metrics (cpm, garbage/min, match length, WHR grade) are
  recorded alongside Elo to explain WHY a change wins or loses, not as
  acceptance criteria.
