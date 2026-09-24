"""Assess an afterstate-core arena (quick read, checkpoint panel or the pre-registered tournament).

Whole reset-seed pairs are the unit: each seed is played twice with sides
swapped and scored as the mean of the two games. Per-pace 95% intervals come
from 20000 whole-seed bootstrap resamples; the seven-pace simultaneous band
uses the max-|studentized deviation| method of assess_retention_fresh_v1.py.
The pooled score is the equal-weight mean of pace scores. Only a config with
a ``preregistration`` block receives a PROMOTE/PARITY/REJECT verdict.

Usage: python assess_afterstate_tournament_v1.py afterstate-core-v1/tournament.json
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PACES = ('sloth', 'relaxed', 'normal', 'fast', 'top_humans', 'super_human', 'frame_perfect')


def elo(score):
    return float(400 * np.log10(score / (1 - score))) if 0 < score < 1 else None


def assess(config_path, *, allow_partial=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    worker = Path(config['output'])
    seed_file = Path(config['seed_file'])
    assert hashlib.sha256(seed_file.read_bytes()).hexdigest() == config['seed_file_sha256'], 'seed file changed'
    for path, digest in config['model_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'model changed: {path}'
    schedule = {m['id']: m for m in config['schedule']}
    by_pace, seen = defaultdict(list), set()
    with (worker / 'games.jsonl').open() as handle:
        for line in handle:
            game = json.loads(line)
            match = schedule[game['comparison']]
            key = (game['comparison'], game['seed'], game['side'])
            assert key not in seen and game['side'] in (0, 1) and game['score'] in (0., .5, 1.)
            assert game['level'] == 14 and game['pace'] == match['pace'] and game['execution_key'] == match['execution_key']
            assert game['seed'] in match['seeds']
            seen.add(key)
            by_pace[match['pace']].append(game)
    expected = sum(m['games'] for m in schedule.values())
    complete = len(seen) == expected
    assert complete or allow_partial, f'arena not finished: {len(seen)}/{expected} games'
    rng = np.random.default_rng(20260924)
    paces = [p for p in PACES if by_pace[p]]
    summaries, draws = [], {}
    for pace in paces:
        pairs = defaultdict(list)
        reasons, execution = Counter(), {'a': Counter(), 'b': Counter()}
        for game in by_pace[pace]:
            pairs[game['seed']].append(game)
            reasons[game['reason']] += 1
            for side in ('a', 'b'):
                execution[side].update(game[side + '_stats'])
        seeds = tuple(sorted(s for s, p in pairs.items() if len(p) == 2 and {g['side'] for g in p} == {0, 1}))
        assert not any(v for side in execution.values() for k, v in side.items()
                       if k in ('dropped_feasible_candidates', 'candidate_overflow')), 'candidate truncation'
        scores = np.asarray([np.mean([g['score'] for g in pairs[s]]) for s in seeds])
        draws[pace] = scores[rng.integers(len(seeds), size=(20000, len(seeds)), dtype=np.int32)].mean(-1)
        counts = Counter(g['score'] for s in seeds for g in pairs[s])
        point = float(scores.mean())
        summaries.append(dict(
            pace=pace, games=2 * len(seeds), independent_reset_seeds=len(seeds),
            wins=counts[1.], losses=counts[0.], draws=counts[.5], censored=reasons.get('timeout', 0),
            score=point, marginal_score_ci95=np.quantile(draws[pace], [.025, .975]).tolist(),
            elo_difference=elo(point), seed_standard_error=float(scores.std(ddof=1) / np.sqrt(len(scores))),
            reasons=dict(reasons), candidate_execution=dict(execution['a']),
        ))
    deviations = np.stack([(draws[r['pace']] - r['score']) / max(r['seed_standard_error'], 1e-12) for r in summaries])
    critical = float(np.quantile(np.max(np.abs(deviations), axis=0), .95))
    for row in summaries:
        radius = critical * row['seed_standard_error']
        row['simultaneous_score_ci95'] = [max(0., row['score'] - radius), min(1., row['score'] + radius)]
    pooled_draws = np.mean(np.stack([draws[p] for p in paces]), axis=0)
    pooled = float(np.mean([r['score'] for r in summaries]))
    pooled_ci = np.quantile(pooled_draws, [.025, .975]).tolist()
    report = dict(
        schema='drmc-afterstate-arena-assessment-v1', config=str(config_path), complete=complete,
        games=len(seen), expected_games=expected, candidate=config['variants']['candidate'],
        champion=config['variants']['champion'], pooled_score=pooled, pooled_ci95=pooled_ci,
        pooled_elo=elo(pooled), simultaneous_critical_value=critical, comparisons=summaries,
        censored_games=sum(r['censored'] for r in summaries),
    )
    pre = config.get('preregistration')
    if pre is not None:
        assert complete, 'a pre-registered verdict requires the complete schedule'
        clear_regression = any(r['simultaneous_score_ci95'][1] < .45 for r in summaries)
        if pre['schema'] == 'drmc-afterstate-shipped-timing-confirmation-v1':
            verdict = 'FAIL' if pooled_ci[0] < .45 else 'PASS'
        elif pooled_ci[0] > .50 and not clear_regression:
            verdict = 'PROMOTE'
        elif pooled_ci[0] <= .50 <= pooled_ci[1] and not clear_regression:
            verdict = 'PARITY'
        else:
            verdict = 'REJECT'
        report.update(verdict=verdict, decision_rule=pre['decision'], preregistered_candidate=pre['candidate'])
    return report


def main():
    config = Path(sys.argv[1])
    report = assess(config, allow_partial='--partial' in sys.argv)
    out = config.with_name(config.stem + '-assessment.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    for r in report['comparisons']:
        lo, hi = r['simultaneous_score_ci95']
        print(f"{r['pace']:>13} score {r['score']:.4f} marginal95 [{r['marginal_score_ci95'][0]:.3f}, "
              f"{r['marginal_score_ci95'][1]:.3f}] sim95 [{lo:.3f}, {hi:.3f}]  W/L/D {r['wins']}/{r['losses']}/{r['draws']}")
    print(f"pooled {report['pooled_score']:.4f} 95% [{report['pooled_ci95'][0]:.4f}, {report['pooled_ci95'][1]:.4f}] "
          f"games {report['games']}/{report['expected_games']}", report.get('verdict', ''))


if __name__ == '__main__':
    main()
