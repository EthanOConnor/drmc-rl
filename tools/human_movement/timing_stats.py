"""Human placement timing vs rating from trainer-input corpus releases.

Per placement (spawn->lock input window, processed NES buttons R=1 L=2 D=4 U=8 B=64 A=128):
  reaction   frames from spawn to the first NEW press of L/R/A/B (relative to the held state
             before spawn; placements without any such press are counted separately)
  tau        spawn -> lock frames (whole placement)
  gap        median frames between consecutive new L/R/A/B presses (placements with >=2)
  taps       new L/R presses; rotations = new A/B presses
  das        any L or R held continuously >= 16 frames (auto-repeat)
  down_share frames with Down held / tau; first_down = frame of the first Down press
Ratings: WHR-C skill_elo trajectories interpolated linearly by day (release ratings table).
Weighting: every table reports placements and players; "player median" = median over players
of each player's own median in that band (equal players).
"""
import glob, json, struct, sys
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

roots = sys.argv[1:-1]
out_path = sys.argv[-1]
CAP_PER_FILE = 600_000
rng = np.random.default_rng(20260924)
MOVE, LATERAL, ROTATE, DOWN = 0xC3, 0x03, 0xC0, 0x04

ratings = defaultdict(list)
for root in roots[:1]:
    for f in glob.glob(f'{root}/ratings/*.parquet'):
        for r in pq.read_table(f).to_pylist():
            ratings[r['player']].append((r['day'], r['skill_elo'], r['skill_sd']))
curves = {p: tuple(np.array(v, dtype=float).T) for p, v in ((p, sorted(v)) for p, v in ratings.items())}

def rating(player, day):
    c = curves.get(player)
    if c is None:
        return None, None
    days, elo, sd = c
    return float(np.interp(day, days, elo)), float(np.interp(day, days, sd))

cols = ['player', 'day', 'speed', 'speed_ups', 'tau_frames', 'input_frames', 'held_before_spawn',
        'input_rle_u16_u8', 'lock_x', 'lock_rotation', 'lock_y_top', 'field']
rows = []
files = sorted(f for root in roots for f in glob.glob(f'{root}/decisions/*/*/*.parquet'))
for f in files:
    table = pq.read_table(f, columns=cols)
    n = table.num_rows
    idx = np.sort(rng.choice(n, min(n, CAP_PER_FILE), replace=False))
    table = table.take(idx)
    for r in table.to_pylist():
        if not r['input_frames'] or r['held_before_spawn'] is None:
            continue
        elo, sd = rating(r['player'], r['day'])
        if elo is None or sd > 150:
            continue
        prev, t = r['held_before_spawn'], 0
        reaction = first_down = None
        presses, taps, rot, down, das = [], 0, 0, 0, False
        hold_start = {1: None, 2: None}
        for length, b in struct.iter_unpack('<HB', r['input_rle_u16_u8']):
            new = b & ~prev
            if new & MOVE:
                presses.append(t)
                if reaction is None:
                    reaction = t
            taps += bool(new & 1) + bool(new & 2)
            rot += bool(new & 64) + bool(new & 128)
            if new & DOWN and first_down is None:
                first_down = t
            for bit in (1, 2):
                if b & bit:
                    if hold_start[bit] is None:
                        hold_start[bit] = t
                    if t + length - hold_start[bit] >= 16:
                        das = True
                else:
                    hold_start[bit] = None
            if b & DOWN:
                down += length
            prev, t = b, t + length
        field = r['field']
        height = next((16 - row for row in range(16) if any(field[row * 8 + c] != 0xFF for c in range(8))), 0)
        gaps = np.diff(presses)
        rows.append((elo, r['speed'], r['speed_ups'], r['tau_frames'], -1 if reaction is None else reaction,
                     float(np.median(gaps)) if len(gaps) else -1, taps, rot, int(das),
                     down / max(r['tau_frames'], 1), -1 if first_down is None else first_down,
                     abs(r['lock_x'] - 3), r['lock_rotation'], height, hash(r['player']) & 0x7FFFFFFF))
    print(f, len(rows), flush=True)

a = np.array(rows, dtype=float)
names = ['elo', 'speed', 'speed_ups', 'tau', 'reaction', 'gap', 'taps', 'rot', 'das', 'down_share', 'first_down',
         'dx', 'lock_rot', 'height', 'player']
col = {n: i for i, n in enumerate(names)}
report = {'placements': int(len(a)), 'players': int(len(np.unique(a[:, col['player']]))), 'bands': {}}
bands = [(lo, lo + 100) for lo in range(1100, 2500, 100)]
for speed in (0, 1, 2):
    s = a[a[:, col['speed']] == speed]
    out = []
    for lo, hi in bands:
        b = s[(s[:, col['elo']] >= lo) & (s[:, col['elo']] < hi)]
        if len(b) < 500:
            continue
        players = np.unique(b[:, col['player']])
        entry = {'band': f'{lo}-{hi}', 'placements': int(len(b)), 'players': int(len(players))}
        for metric, mask in (('reaction', b[:, col['reaction']] >= 0), ('tau', slice(None)), ('gap', b[:, col['gap']] >= 0),
                             ('taps', slice(None)), ('rot', slice(None)), ('das', slice(None)),
                             ('down_share', slice(None)), ('first_down', b[:, col['first_down']] >= 0)):
            v = b[mask]
            x = v[:, col[metric]]
            per_player = [np.median(x[v[:, col['player']] == p]) for p in players
                          if np.count_nonzero(v[:, col['player']] == p) >= 30]
            entry[metric] = {'p10': float(np.percentile(x, 10)), 'p25': float(np.percentile(x, 25)),
                             'p50': float(np.median(x)), 'p75': float(np.percentile(x, 75)),
                             'p90': float(np.percentile(x, 90)), 'mean': float(x.mean()),
                             'player_median': float(np.median(per_player)) if per_player else None,
                             'players_30plus': len(per_player)}
        entry['no_move_share'] = float(np.mean(b[:, col['reaction']] < 0))
        out.append(entry)
    report['bands'][f'speed{speed}'] = out

def fit(y, xs, mask):
    X = np.column_stack([np.ones(mask.sum())] + [a[mask, col[x]] for x in xs])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y[mask] - pred) ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2)
    return {'terms': ['1'] + xs, 'beta': [float(v) for v in beta], 'r2': float(r2), 'n': int(mask.sum())}

hi = a[:, col['speed']] == 2
e100 = (a[:, col['elo']] - 1800) / 100
a = np.column_stack([a, e100]); col['elo100'] = a.shape[1] - 1
report['regressions_speed2'] = {
    'log_reaction ~ elo100': fit(np.log1p(a[:, col['reaction']]), ['elo100'], hi & (a[:, col['reaction']] >= 0)),
    'log_tau ~ elo100 + dx + lock_rot + height + speed_ups': fit(np.log(a[:, col['tau']].clip(1)),
        ['elo100', 'dx', 'lock_rot', 'height', 'speed_ups'], hi),
    'log_gap ~ elo100': fit(np.log(a[:, col['gap']].clip(1)), ['elo100'], hi & (a[:, col['gap']] >= 1)),
    'das ~ elo100': fit(a[:, col['das']], ['elo100'], hi),
    'down_share ~ elo100 + height': fit(a[:, col['down_share']], ['elo100', 'height'], hi),
}
json.dump(report, open(out_path, 'w'), indent=1)
print('placements', report['placements'], 'players', report['players'])
