"""Assess the pre-registered Stronger-3 three-way tournament (arm A vs arm B vs champion).

Implements exactly the analysis and decision rules in
``afterstate-core-human-v1/threeway-preregistration.json``. Whole reset-seed
pairs are the unit; all pairings share one set of bootstrap resamples so the
paired difference B-vs-champion minus A-vs-champion is computed per seed.

Usage: python assess_afterstate_threeway_v1.py afterstate-core-human-v1/threeway.json [--partial]
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PACES = ('sloth', 'relaxed', 'normal', 'fast', 'top_humans', 'super_human', 'frame_perfect')
PAIRINGS = ('b-vs-champion', 'a-vs-b', 'a-vs-champion')


def _games(config, seeds):
    """pairing -> pace -> seed -> {side: candidate score}; arm A's own tournament may be imported."""
    schedule = {m['id']: m for m in config['schedule']}
    out = defaultdict(lambda: defaultdict(dict))
    sources = [(Path(config['output']) / 'games.jsonl', None)]
    if config.get('arm_a_tournament_games'):
        sources.append((Path(config['arm_a_tournament_games']), 'a-vs-champion'))
    for path, imported in sources:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            game = json.loads(line)
            if imported:
                assert game['comparison'].startswith('tournament-'), 'not an arm A tournament journal'
                pairing, pace = imported, game['pace']
            else:
                match = schedule[game['comparison']]
                pairing, pace = game['comparison'].removeprefix('threeway-').split('-b')[0], match['pace']
                assert game['pace'] == pace
            assert game['seed'] in seeds[pace], 'game outside the registered tournament seeds'
            assert game['level'] == 14 and game['side'] in (0, 1) and game['score'] in (0., .5, 1.)
            cell = out[pairing][pace].setdefault(game['seed'], {})
            assert game['side'] not in cell, 'duplicate game'
            cell[game['side']] = game['score']
    return out


def assess(config_path, allow_partial=False):
    config = json.loads(Path(config_path).read_text())
    pre = config['threeway_preregistration']
    for path, digest in config['model_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'model changed: {path}'
    assert hashlib.sha256(Path(config['seed_file']).read_bytes()).hexdigest() == config['seed_file_sha256']
    seeds = json.loads(Path(config['seed_file']).read_text())['tournament']
    games = _games(config, {p: set(v) for p, v in seeds.items()})
    complete = all(len(games[p][pace]) == len(seeds[pace]) and all(len(v) == 2 for v in games[p][pace].values())
                   for p in PAIRINGS for pace in PACES)
    assert complete or allow_partial, 'three-way tournament not finished'
    rng = np.random.default_rng(20260924)
    report = dict(schema='drmc-afterstate-threeway-assessment-v1', config=str(config_path), complete=complete,
                  pairings={})
    resamples = {pace: rng.integers(len(seeds[pace]), size=(20000, len(seeds[pace])), dtype=np.int32)
                 for pace in PACES}
    per_seed = {}
    for pairing in PAIRINGS:
        rows, draws = [], {}
        for pace in PACES:
            ordered = seeds[pace]
            score = np.asarray([np.mean(list(games[pairing][pace][s].values())) if len(games[pairing][pace].get(s, {})) == 2
                                else np.nan for s in ordered])
            per_seed[(pairing, pace)] = score
            ok = np.isfinite(score)
            if not ok.any():
                continue
            filled = np.where(ok, score, np.nanmean(score))
            draws[pace] = filled[resamples[pace]].mean(-1)
            point = float(score[ok].mean())
            se = float(score[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else 0.0
            rows.append(dict(pace=pace, seeds=int(ok.sum()), score=point, standard_error=se,
                             marginal_ci95=np.quantile(draws[pace], [.025, .975]).tolist()))
        if not rows:
            continue
        deviations = np.stack([(draws[r['pace']] - r['score']) / max(r['standard_error'], 1e-12) for r in rows])
        critical = float(np.quantile(np.abs(deviations).max(0), .95))
        for r in rows:
            r['simultaneous_ci95'] = [max(0., r['score'] - critical * r['standard_error']),
                                      min(1., r['score'] + critical * r['standard_error'])]
        pooled_draws = np.mean(np.stack([draws[r['pace']] for r in rows]), axis=0)
        pooled = float(np.mean([r['score'] for r in rows]))
        ci = np.quantile(pooled_draws, [.025, .975]).tolist()
        entry = dict(pooled_score=pooled, pooled_ci95=ci, simultaneous_critical_value=critical, paces=rows)
        clear_low = any(r['simultaneous_ci95'][1] < .45 for r in rows)
        if pairing == 'a-vs-b':
            # arm A is side "a" (candidate); the registered primary is stated with arm B as candidate.
            b_pooled, b_ci = 1 - pooled, [1 - ci[1], 1 - ci[0]]
            b_low = any(1 - r['simultaneous_ci95'][0] < .45 for r in rows)
            b_high = any(1 - r['simultaneous_ci95'][1] > .55 for r in rows)
            verdict = ('B_STRONGER' if b_ci[0] > .50 and not b_low else
                       'A_STRONGER' if b_ci[1] < .50 and not b_high else 'NO_DIFFERENCE_SHOWN')
            entry.update(arm_b_pooled_score=b_pooled, arm_b_pooled_ci95=b_ci, verdict=verdict)
        else:
            entry['verdict'] = ('PROMOTE' if ci[0] > .50 and not clear_low else
                                'PARITY' if ci[0] <= .50 <= ci[1] and not clear_low else 'REJECT')
        report['pairings'][pairing] = entry
    diffs = []
    for pace in PACES:
        b, a = per_seed.get(('b-vs-champion', pace)), per_seed.get(('a-vs-champion', pace))
        if b is None or a is None:
            continue
        d = b - a
        ok = np.isfinite(d)
        if ok.any():
            diffs.append(np.where(ok, d, np.nanmean(d))[resamples[pace]].mean(-1))
    if diffs:
        d_draws = np.mean(np.stack(diffs), axis=0)
        report['paired_difference_b_minus_a_vs_champion'] = dict(
            mean=float(np.mean(d_draws)), ci95=np.quantile(d_draws, [.025, .975]).tolist())
    primary = report['pairings'].get('a-vs-b')
    if primary and diffs and primary['verdict'] != 'NO_DIFFERENCE_SHOWN':
        sign = np.sign(primary['arm_b_pooled_score'] - .5)
        if np.sign(report['paired_difference_b_minus_a_vs_champion']['mean']) != sign:
            primary['verdict'], primary['consistency'] = 'NO_DIFFERENCE_SHOWN', 'contradicted by paired difference'
    if complete:
        verdicts = {k: v['verdict'] for k, v in report['pairings'].items()}
        a_ok, b_ok = verdicts['a-vs-champion'] == 'PROMOTE', verdicts['b-vs-champion'] == 'PROMOTE'
        if a_ok and b_ok:
            winner = {'B_STRONGER': 'arm_b', 'A_STRONGER': 'arm_a'}.get(verdicts['a-vs-b'])
            if winner is None:
                winner = 'arm_b' if (report['pairings']['b-vs-champion']['pooled_score']
                                     > report['pairings']['a-vs-champion']['pooled_score']) else 'arm_a'
        else:
            winner = 'arm_a' if a_ok else 'arm_b' if b_ok else 'champion'
        report.update(adoption=winner, decision_rule=pre['decision'])
    return report


def main():
    config = Path(sys.argv[1])
    report = assess(config, allow_partial='--partial' in sys.argv)
    config.with_name(config.stem + '-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, entry in report['pairings'].items():
        print(f"{name:>14} pooled {entry['pooled_score']:.4f} 95% [{entry['pooled_ci95'][0]:.4f}, "
              f"{entry['pooled_ci95'][1]:.4f}] {entry['verdict']}")
        for r in entry['paces']:
            print(f"    {r['pace']:>13} {r['score']:.4f} sim95 [{r['simultaneous_ci95'][0]:.3f}, {r['simultaneous_ci95'][1]:.3f}]")
    if 'paired_difference_b_minus_a_vs_champion' in report:
        d = report['paired_difference_b_minus_a_vs_champion']
        print(f"paired B-A vs champion {d['mean']:+.4f} 95% [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print('adoption:', report.get('adoption', '(incomplete)'))


if __name__ == '__main__':
    main()
