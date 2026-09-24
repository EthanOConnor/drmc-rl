"""Timing vs rating on the sampled corpus with fastest-route times (output of slack.py).

Metrics per placement (processed NES buttons R=1 L=2 D=4 U=8 B=64 A=128, spawn->lock window):
  reaction    frames to the first NEW press of L/R/A/B relative to the held byte before spawn
  tau         recorded spawn->lock frames
  fastest     fewest frames to the same lock pose from the same start state (native planner)
  lost        tau - fastest (time lost to thinking, cautious movement and not soft-dropping)
  lost_per_row lost / rows fallen (lock_y_top + 1)
  gap, taps, rot, das, down_share as in timing_stats.py
Equal-player weighting: 'player' = median over players (>=30 placements in the band) of each
player's median; 'all' = pooled placement percentiles.
"""
import json, struct, sys
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

src, ratings_src, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = pq.read_table(src).to_pylist()
curves = defaultdict(list)
for r in pq.read_table(ratings_src).to_pylist():
    curves[r['player']].append((r['day'], r['skill_elo'], r['skill_sd']))
curves = {p: tuple(np.array(sorted(v), dtype=float).T) for p, v in curves.items()}
MOVE = 0xC3
rows, players = [], {}
for r in t:
    if r['fastest_frames'] is None or r['player'] not in curves:
        continue
    days, elo, sd = curves[r['player']]
    e, s = float(np.interp(r['day'], days, elo)), float(np.interp(r['day'], days, sd))
    if s > 150:
        continue
    prev, pos, reaction, first_down, presses, taps, rot, down, das = r['held_before_spawn'], 0, -1, -1, [], 0, 0, 0, 0
    hold = {1: None, 2: None}
    for n, b in struct.iter_unpack('<HB', r['rle']):
        new = b & ~prev
        if new & MOVE:
            presses.append(pos)
            if reaction < 0:
                reaction = pos
        taps += bool(new & 1) + bool(new & 2); rot += bool(new & 64) + bool(new & 128)
        if new & 4 and first_down < 0:
            first_down = pos
        for bit in (1, 2):
            if b & bit:
                hold[bit] = pos if hold[bit] is None else hold[bit]
                das |= pos + n - hold[bit] >= 16
            else:
                hold[bit] = None
        down += n * bool(b & 4)
        prev, pos = b, pos + n
    field = r['field_bytes']
    height = next((16 - y for y in range(16) if any(field[y * 8 + c] != 0xFF for c in range(8))), 0)
    rows_fallen = int(r['lock_y_top']) + 1
    lost = r['tau_frames'] - r['fastest_frames']
    gaps = np.diff(presses)
    pid = players.setdefault(r['player'], len(players))
    dx = abs(int(r['lock_x']) - 3)
    min_rot = {0: 0, 1: 1, 2: 2, 3: 1}[int(r['lock_rotation']) & 3]
    corrected = int(r['h_reversals'] > 0 or r['h_overshoot'] > 0 or r['h_rot_changes'] > min_rot + 1 or r['h_late_lateral'] > 0)
    rows.append((e, r['speed'], r['speed_ups'], r['tau_frames'], r['fastest_frames'], lost, lost / rows_fallen,
                 reaction, float(np.median(gaps)) if len(gaps) else -1, taps, rot, int(das),
                 down / max(r['tau_frames'], 1), first_down, dx, int(r['lock_rotation']),
                 height, rows_fallen, pid, r['day'],
                 r['h_lateral'] - dx, max(0, r['h_rot_changes'] - min_rot), r['h_reversals'], r['h_overshoot'],
                 r['h_late_lateral'], r['h_idle_max'], r['h_edges'], corrected))
names = ['elo', 'speed', 'speed_ups', 'tau', 'fastest', 'lost', 'lost_per_row', 'reaction', 'gap', 'taps', 'rot',
         'das', 'down_share', 'first_down', 'dx', 'lock_rot', 'height', 'rows_fallen', 'player', 'day',
         'extra_lateral', 'extra_rot', 'reversals', 'overshoot', 'late_lateral', 'idle_max', 'edges', 'corrected']
a = np.array(rows, dtype=float)
c = {n: i for i, n in enumerate(names)}
metrics = ['reaction', 'tau', 'fastest', 'lost', 'lost_per_row', 'gap', 'taps', 'rot', 'das', 'down_share', 'first_down',
           'extra_lateral', 'extra_rot', 'reversals', 'overshoot', 'late_lateral', 'idle_max', 'edges', 'corrected']
valid = {'reaction': lambda b: b[:, c['reaction']] >= 0, 'gap': lambda b: b[:, c['gap']] >= 0,
         'first_down': lambda b: b[:, c['first_down']] >= 0}
report = {'placements': len(a), 'players': len(players), 'days': [int(a[:, c['day']].min()), int(a[:, c['day']].max())],
          'bands': {}}
for speed in (0, 1, 2):
    s = a[a[:, c['speed']] == speed]
    table = []
    for lo in range(1000, 2600, 100):
        b = s[(s[:, c['elo']] >= lo) & (s[:, c['elo']] < lo + 100)]
        if len(b) < 300:
            continue
        ids = np.unique(b[:, c['player']])
        entry = {'band': f'{lo}-{lo + 100}', 'placements': int(len(b)), 'players': int(len(ids))}
        for m in metrics:
            v = b[valid[m](b)] if m in valid else b
            x = v[:, c[m]]
            per = [float(np.median(x[v[:, c['player']] == p])) for p in ids if np.count_nonzero(v[:, c['player']] == p) >= 30]
            entry[m] = {'p10': float(np.percentile(x, 10)), 'p25': float(np.percentile(x, 25)), 'p50': float(np.median(x)),
                        'p75': float(np.percentile(x, 75)), 'p90': float(np.percentile(x, 90)), 'mean': float(x.mean()),
                        'player': float(np.median(per)) if per else None, 'players': len(per)}
        table.append(entry)
    report['bands'][f'speed{speed}'] = table

def ols(y, terms, mask):
    X = np.column_stack([np.ones(mask.sum())] + [terms[k][mask] for k in terms])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    res = y[mask] - X @ beta
    return {'terms': ['const'] + list(terms), 'beta': [round(float(b), 4) for b in beta],
            'r2': round(float(1 - res.var() / y[mask].var()), 4), 'n': int(mask.sum())}

hi = a[:, c['speed']] == 2
z = (a[:, c['elo']] - 1800) / 100
geo = {'elo100': z, 'fastest': a[:, c['fastest']], 'rows_fallen': a[:, c['rows_fallen']], 'dx': a[:, c['dx']],
       'lock_rot': a[:, c['lock_rot']], 'height': a[:, c['height']], 'speed_ups': a[:, c['speed_ups']]}
report['regressions_hi_speed'] = {
    'tau ~ elo + geometry': ols(a[:, c['tau']], geo, hi),
    'lost ~ elo + geometry': ols(a[:, c['lost']], geo, hi),
    'lost ~ geometry only': ols(a[:, c['lost']], {k: v for k, v in geo.items() if k != 'elo100'}, hi),
    'log1p(lost) ~ elo + geometry': ols(np.log1p(a[:, c['lost']]), geo, hi),
    'reaction ~ elo': ols(a[:, c['reaction']], {'elo100': z}, hi & (a[:, c['reaction']] >= 0)),
    'log1p(reaction) ~ elo + height': ols(np.log1p(a[:, c['reaction']]), {'elo100': z, 'height': geo['height']}, hi & (a[:, c['reaction']] >= 0)),
    'gap ~ elo': ols(a[:, c['gap']], {'elo100': z}, hi & (a[:, c['gap']] >= 0)),
    'das ~ elo': ols(a[:, c['das']], {'elo100': z}, hi),
    'down_share ~ elo + geometry': ols(a[:, c['down_share']], geo, hi),
}
# Between-player vs within-player: how much of the spread in time lost is the player, not rating.
m = hi & np.isfinite(a[:, c['lost']])
pl = a[m, c['player']]
lost = a[m, c['lost']]
means = {p: lost[pl == p].mean() for p in np.unique(pl) if np.count_nonzero(pl == p) >= 200}
report['variance_hi_speed'] = {'lost_total_sd': float(lost.std()),
                               'player_mean_sd': float(np.std(list(means.values()))), 'players_200plus': len(means)}
# Player profiles (Hi speed, >=300 placements): one row per player, standardized, then styles.
profile_metrics = ['reaction', 'lost', 'gap', 'das', 'down_share', 'extra_lateral', 'extra_rot', 'corrected', 'idle_max']
pm = a[hi]
profiles, prof_ids = [], []
for p in np.unique(pm[:, c['player']]):
    v = pm[pm[:, c['player']] == p]
    if len(v) < 300:
        continue
    row = [float(np.median(v[:, c['elo']]))]
    for m in profile_metrics:
        x = v[valid[m](v)][:, c[m]] if m in valid else v[:, c[m]]
        row.append(float(x.mean()) if m in ('das', 'down_share', 'extra_lateral', 'extra_rot', 'corrected', 'idle_max') else float(np.median(x)))
    profiles.append(row); prof_ids.append(int(p))
P = np.array(profiles)
report['player_profiles'] = {'metrics': ['elo'] + profile_metrics, 'players': len(P)}
if len(P) >= 12:
    # Variance explained by rating for each metric across players, and residual spread.
    explained = {}
    for j, m in enumerate(profile_metrics, 1):
        X = np.column_stack([np.ones(len(P)), (P[:, 0] - 1800) / 100])
        beta, *_ = np.linalg.lstsq(X, P[:, j], rcond=None)
        res = P[:, j] - X @ beta
        explained[m] = {'per_100_elo': round(float(beta[1]), 4), 'r2_rating': round(float(1 - res.var() / P[:, j].var()), 3),
                        'residual_sd': round(float(res.std()), 4), 'player_sd': round(float(P[:, j].std()), 4)}
    report['player_profiles']['rating_explains'] = explained
    # Styles: k-means on rating-residualized, standardized profiles (style independent of level).
    R = []
    for j in range(1, P.shape[1]):
        X = np.column_stack([np.ones(len(P)), P[:, 0]])
        beta, *_ = np.linalg.lstsq(X, P[:, j], rcond=None)
        res = P[:, j] - X @ beta
        R.append(res / (res.std() or 1))
    R = np.array(R).T
    for k in (2, 3, 4, 5):
        rng = np.random.default_rng(k)
        best = None
        for _ in range(30):
            centers = R[rng.choice(len(R), k, replace=False)]
            for _ in range(100):
                labels = np.argmin(((R[:, None] - centers[None]) ** 2).sum(-1), 1)
                centers = np.array([R[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(k)])
            inertia = float(((R - centers[labels]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, labels.copy(), centers.copy())
        inertia, labels, centers = best
        report['player_profiles'][f'styles_k{k}'] = {
            'sizes': [int(np.sum(labels == i)) for i in range(k)],
            'centers_sd_units': [[round(float(x), 2) for x in centers[i]] for i in range(k)],
            'median_elo': [float(np.median(P[labels == i, 0])) if np.any(labels == i) else None for i in range(k)],
            'inertia': round(inertia, 1)}
json.dump(report, open(out, 'w'), indent=1)
print(json.dumps({k: report[k] for k in ('placements', 'players', 'days')}))
for e in report['bands']['speed2']:
    print(f"{e['band']} n={e['placements']:>6} pl={e['players']:>3} react {e['reaction']['p50']:>4.0f}/{e['reaction']['player'] or 0:>4.1f}"
          f" tau {e['tau']['p50']:>4.0f} fastest {e['fastest']['p50']:>3.0f} lost {e['lost']['p50']:>4.0f}/{e['lost']['player'] or 0:>4.1f}"
          f" lost/row {e['lost_per_row']['p50']:.2f} gap {e['gap']['p50']:>4.1f} das {e['das']['mean']:.2f} down {e['down_share']['mean']:.2f}")
for k, v in report['regressions_hi_speed'].items():
    print(k, v)
print(report['variance_hi_speed'])
