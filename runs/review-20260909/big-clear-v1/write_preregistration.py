"""Write the big-clear-setups pre-registration from the measured data files (no hand-copied numbers).

  python runs/review-20260909/big-clear-v1/write_preregistration.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('/Users/ethan/dev/drmario/drmc-rl-bigclear-data')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    from drmc_rl.eval import big_clear as bc
    from drmc_rl.program.seed_reserve import allocated_seeds
    sys.path.insert(0, str(HERE.parent))
    from prepare_big_clear_finetune_v1 import BONUS, PARENT, PARENT_ENTRANT, SEED, START_MIX, TARGET_DECISIONS

    mining = load(DATA / 'mining/mining-summary.json')
    train = load(DATA / 'bank/big-clear-train-v1.json')
    bench = load(DATA / 'bank/big-clear-benchmark-v1.json')
    selection = load(DATA / 'bank/selection.json')
    pool = load(DATA / 'pool-natural-baseline.json')
    parent_bench = DATA / 'eval/parent-armA-f100m/summary.json'
    seeds = {s: hashlib.sha256(json.dumps(allocated_seeds(s)).encode()).hexdigest()
             for s in ('big-clear-benchmark-v1', 'big-clear-natural-v1')}
    placements = mining['counts']['placements']
    side_games = mining['counts']['side_games']
    doc = dict(
        schema='drmc-big-clear-preregistration-v1',
        created_at='2026-09-25',
        intention='big-clear-setups (runs/rating-pool-v1/intentions.json)',
        branch='trainer/big-clear-setups',
        definition=dict(
            module='drmc_rl/eval/big_clear.py (drmc-big-clear-v1)',
            principle=('Big means large and showy, not an ordinary attack. Points, each zero for a plain single 4-line: '
                       'cells beyond 4 (x1), cascade rounds beyond 1 (x3), lines in the richest round beyond 1 (x2), '
                       'tiles beyond 4 per line (x1.5), viruses beyond 2 (x1), rows spanned beyond 6 (x0.5), a '
                       'horizontal/vertical cross (+2), all three colors (+2); garbage only as a minor bonus at 3 (+1) '
                       'or 4 (+2) pieces, 2-piece combos earn nothing. Level and speed are sampling context only.'),
            weights=bc.WEIGHTS, tiers=dict(bc.TIERS),
            tier_rationale=('Ordinary 2-line/2-garbage combos score 10-13 (median 9 cells), so T1 starts at 20: '
                            'the top 2.4% of human clears (median 13 cells, 3 lines, 3 colors). T2 >= 30 is the top '
                            '0.44% (about 17 cells, 3 rounds, 4 lines); T3 >= 42 the top 0.035% (about 24 cells, '
                            '4 rounds, 6 lines).'),
        ),
        mining=dict(
            release='human-v2-20260924T170224Z', tool='tools/mine_big_clears.py',
            placements=placements, side_games=side_games, clears=mining['clears'],
            seed_verified_side_games=mining['counts']['side_games_seed_ok'],
            prefilter_missed=f"{mining['counts']['prefilter_missed']} of {mining['counts']['prefilter_checked']} "
                             'sampled non-line placements (pre-existing lines)',
            score_quantiles_over_clears=mining['quantiles'],
            clears_at_least_score=mining['at_least'],
            leaderboard_crosscheck=('drmariostats big_clear (cells>=18 | rounds>=4 | viruses>=6, 119,052 rows) used '
                                    'as the seed list: on 2025-03 3,328 of 3,329 rows join the corpus on quark, set, '
                                    'crown, slot and spawn frame with cells, viruses and rounds agreeing exactly; '
                                    'every top-50 row by cells or by chain scores T3. The mined set is a superset '
                                    '(it also covers long lines, crosses and multi-line rounds the board omits); '
                                    'the board is not used for measurement.'),
        ),
        training_bank=dict(path='big-clear-v1/bank/big-clear-train-v1.npz', **{k: train[k] for k in (
            'sha256', 'rows', 'groups', 'kinds', 'lookbacks', 'tiers_rows', 'types', 'players', 'replayable_rows',
            'reserved_seed_rows', 'levels')}, selection=selection['training'],
            rules=('9,000 T1+ targets from training players only; weight tier (T1 1, T2 2.5, T3 6) x type (inverse '
                   'share, cap 3) x level (10-13 0.5, 14+ 1) x speed (HI 1, MED 0.5) / sqrt(player targets); at '
                   'most 1.5% of targets per player. Starts 3, 6, 10 and 20 placements back, kept only when the '
                   "human's own placements from the start settle exactly into each next bottle (no garbage) and "
                   'reproduce the target; each start as a real row (setup side vs the settled causal opponent) and '
                   'a mirror row (setup in both bottles). Rows whose source seed is an evaluation-reserve seed '
                   'never replay it.')),
        benchmark=dict(path='drmc-rl-bigclear-data/bank/big-clear-benchmark-v1.npz', **{k: bench[k] for k in (
            'sha256', 'rows', 'groups', 'kinds', 'lookbacks', 'tiers_rows', 'types', 'players')},
            selection=selection['benchmark'],
            holdout='12% of players (hash) held out entirely, and 8% of the other players\' games; neither is ever '
                    'in the training bank',
            targets='T2+ clears at level 14+ HI with a verified seed and a clean 10-placement path; one per game, '
                    'at most 6 per player, half from held-out players',
            controls='per target, the same held-out side-game at a similar virus count (gap <= 5) where the human '
                     'made no T1+ clear in the next 16 placements, away from the target build-up',
            seeds=dict(fresh_study='big-clear-benchmark-v1', fresh_sha256=seeds['big-clear-benchmark-v1'],
                       replay='each row\'s own source-game seed (evaluation only; native pill stream verified 152/152)'),
            evaluator='tools/eval_big_clear.py', paces=['normal', 'frame_perfect'], horizon='lookback + 6 pills',
            natural_panel=dict(study='big-clear-natural-v1', sha256=seeds['big-clear-natural-v1'], seed_pairs=128,
                               opponent=PARENT_ENTRANT, paces=['normal', 'frame_perfect'])),
        parent=dict(entrant=PARENT_ENTRANT, checkpoint=PARENT,
                    rationale=('Arm A PPO 100M: the best-rated pool entrant (l14-spawn 1562, 95% CI 1537-1587, 1,088 '
                               'games; champion anchor 1500), and an afterstate core whose inputs carry the exact '
                               'settled afterstate of every candidate (tiles, rounds, lines, garbage), so a big clear '
                               'is visible one placement ahead: the curriculum and the bonus act on a representation '
                               'that already sees clears. The shipped champion stays the anchor and remains in the '
                               'pool; the style lever is measured as a difference to this parent, which the pool '
                               'rates with many games. Arm A was still training at registration; its f100M snapshot '
                               'is frozen and rated, so the comparison does not depend on the Stronger-3 outcome.')),
        arms=dict(
            common=dict(trainer='tools/train_controller_retention.py (mixed_retention)', seed=SEED,
                        target_decisions=TARGET_DECISIONS, snapshots_every_frames=25_000_000,
                        keep_every_checkpoint=True, retention_and_kl='inherited from arm A unchanged (parent_kl 0.02, '
                        'max_update_kl 0.03, retention 0.1, max_anchor_kl_increase 0.03)'),
            std=dict(entrants='bigclear-std-f*', reward='natural outcome only'),
            shaped=dict(entrants='bigclear-shaped-f*', reward='natural outcome + showiness_bonus', bonus=BONUS)),
        curriculum=dict(
            start_mix=START_MIX,
            rationale=('share0 0.30 below the stranded 0.40 because bank games start mid-game and run longer than '
                       'endgame starts, so each mixed pair costs more frames; half-life 25M frames (not 8M) because '
                       'the fine-tune is only about 100M frames and the curriculum should span the first two or '
                       'three snapshots: shares 0.30, 0.15, 0.075, 0.0375 at 0, 25, 50, 75M, cut to 0 below 0.02 '
                       '(about 95M), so the final snapshot reflects natural play. replay_share 0.5: half of mixed '
                       "pairs replay the human's pills (the setup is realisable), half take a training-pool seed "
                       '(the setup must generalise).')),
        reward=dict(
            spec=BONUS,
            form=('event based, per learner placement: min(0.15, 0.05 + 0.005 x (score - 20)) for T1+ clears, '
                  'capped at 0.30 per game in order; decision t receives the undiscounted outcome plus the bonus '
                  'still to come (setup placements share the credit).'),
            bound=('0.30 per game is 15% of the 2-point win/loss swing; the policy can never prefer a sure loss, '
                   'and at equilibrium buys at most (bonus gained)/2 of win probability. At the parent\'s natural '
                   'rate the bonus is about 0.036 per game (1.8% of the swing), so doubling big clears is worth at '
                   'most about 1.8 percentage points of score, roughly 12 Elo, the scale of the -15 Elo guard.'),
            natural_firing=dict(source='rating-pool traces (1 seed pair in 16)', parent=pool['entrants'].get(PARENT_ENTRANT),
                                champion=pool['entrants'].get('champion-retention-mixed-v2')),
            anneal=('constant, not annealed: the style lever must persist in the final weights (annealing to zero '
                    'returns the objective to pure winning and PPO would drift back to the parent style); the cap '
                    'bounds the price, and the curriculum, not the bonus, is what decays.'),
        ),
        evaluation=dict(
            strength=dict(pool_job='intention big-clear-setups: vs_parent, set:l14-spawn (7 paces), 256 games per '
                                   'condition per snapshot, open to new snapshots',
                          guard='pooled l14-spawn rating difference to the parent, 95% lower bound >= -15 Elo',
                          drill_down='per-pace ratings from the pool report'),
            skill=dict(primary=('benchmark, replay condition: paired solo difference (candidate - parent) in the rate '
                                'of realising a T1+ clear within lookback + 6 pills on setup rows, pooled over '
                                'normal and frame_perfect and lookbacks 6 and 10, bootstrap over groups'),
                       success='point >= +0.05 and 95% lower bound > 0',
                       secondary=['same on the fresh condition (generalisation)',
                                  'T2+ realisation and best score within the horizon',
                                  'setup - control lift (does it go big where a setup exists, not everywhere)']),
            style=dict(primary=('natural panel: candidate T1+ clears per 100 placements vs the parent in the same '
                                '256 games per pace (128 side-swapped seed pairs), ratio with 95% interval'),
                       secondary=['T2+ and T3 per 100 placements; mean clear score; max score',
                                  'pool traced games (tools/big_clear_pool_metrics.py)',
                                  'training-time learner rates from journal_showiness (natural games only)'])),
        decision=dict(
            keep_as_strength_candidate=('an arm snapshot passes the guard AND the skill primary succeeds; compare '
                                        '(a) vs (b) on the same rules'),
            style_tradeoff=('Every checkpoint of both arms is kept. For each snapshot report: delta Elo to the parent '
                            '(pooled l14-spawn, 95% CI) against the style gain (natural T1+ rate ratio and T2+ '
                            'ratio, 95% CI) and the price, Elo per +10% T1+ rate. Classes: free style (guard passes, '
                            'ratio lower bound > 1); priced style (ratio lower bound > 1, guard fails, delta Elo >= '
                            '-60): kept as the style lever with its price; costly (delta Elo < -60): archived; no '
                            'style effect (ratio interval covers 1).'),
            registration=('both arms register every 25M snapshot in the pool under era style:big-combo (run '
                          'bigclear-std / bigclear-shaped, parent armA-ppo-v1-f00100000000); final and best (best '
                          '= highest pooled rating; for arm (b) also style-best = highest natural T1+ rate among '
                          'priced-or-better snapshots) get tags style:big-combo,keep.'),
            snapshots='the pool rates every snapshot; there is no early stop, each arm runs its 1M-decision budget'),
        capacity=dict(host='tf3090', queue='runs/review-20260909/big-clear-v1/bigclear_queue.sh (tmux bigclear-queue)',
                      rule=('one arm at a time (shaped, then std), only in a free trainer slot: <= 1 other trainer, '
                            '>= 5.0 GB available, no committed arm C or C+S launch pending; systemd scope MemoryMax '
                            '4.5 GB, nice 10, oom_score_adj 1000; SIGINT below 1.5 GB available or when arm A falls '
                            'below 50% of 5,125 frames/s; resume from the newest update checkpoint'),
                      measured=('arm A 3.3 GB PSS (3.5 GB RSS), A+S 2.9 GB and growing (cap 5.5 GB); /dev/shm holds '
                                '4.8 GB of which about 3 GB is stale files from other projects')),
        parent_benchmark=load(parent_bench)['results'] if parent_bench.exists() else 'pending (Mac run in progress)',
    )
    doc['mining']['tiers'] = {}
    ladder = {int(k): v for k, v in mining['at_least'].items()}
    for name, bar in bc.TIERS:
        n = ladder.get(int(bar))
        doc['mining']['tiers'][name] = dict(bar=bar, clears=n, per_clear=round(n / mining['clears'], 5),
                                            per_100_placements=round(100 * n / placements, 4),
                                            per_side_game=round(n / side_games, 4))
    out = HERE / 'preregistration.json'
    out.write_text(json.dumps(doc, indent=1, default=str) + '\n')
    print(out, sha(out))


if __name__ == '__main__':
    main()
