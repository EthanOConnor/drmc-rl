"""Summarize timing-contract v2 arenas: scores, request kinds, lead frames and mismatch reasons."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent/'timing-contract-v2'
RESAMPLES, RNG_SEED, FPS = 20000, 20260924, 60.0988
KINDS = ('commit', 'lock', 'settled')


def journal_rows(directory):
    """Rows of an arena journal (committed gzipped; live runs write plain JSONL)."""
    plain, packed = directory/'games.jsonl', directory/'games.jsonl.gz'
    text = plain.read_text() if plain.exists() else gzip.open(packed, 'rt').read() if packed.exists() else ''
    return [json.loads(line) for line in text.splitlines()]


def seed_means(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row['score'])
    return {s: np.mean(v) for s, v in by_seed.items() if len(v) == 2 and None not in v}


def summarize(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row['score'])
    pairs = np.array([np.mean(v) for v in by_seed.values() if len(v) == 2 and None not in v])
    boot = np.random.default_rng(RNG_SEED).choice(pairs, (RESAMPLES, len(pairs))).mean(axis=1)
    s = Counter()
    for row in rows:
        s.update(row['a_stats'])
    decisions = max(s['decisions'], 1)
    out = dict(games=2*len(pairs), score=float(pairs.mean()),
               ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
               censored=sum(r['score'] is None for r in rows), decisions=s['decisions'],
               mean_frames_spawn_to_first_input=s['spawn_wait_frames']/decisions,
               spawn_fallback_share=(s['early_mismatch'] + s['early_unavailable'])/decisions,
               mismatch_rate=s['early_mismatch']/max(s['early_accepted'] + s['early_mismatch'], 1),
               mismatch_reasons={k[len('early_mismatch_'):]: v for k, v in s.items() if k.startswith('early_mismatch_')},
               unsafe_skips={k: s[k] for k in ('early_commit_unsafe', 'early_lock_unsafe') if s[k]})
    for kind in KINDS:
        n = s[f'early_accepted_{kind}']
        if n:
            lead = s[f'early_lead_{kind}_frames']/n
            out[kind] = dict(share_of_decisions=n/decisions, mean_lead_frames=lead, mean_lead_ms=1000*lead/FPS,
                             lead_histogram={k.rsplit('_', 1)[1]: v for k, v in s.items()
                                             if k.startswith(f'early_lead_{kind}_') and not k.endswith('_frames')})
    return out


def main():
    groups = defaultdict(list)
    for journal in sorted(d for d in HERE.iterdir() if d.is_dir() and d.name.startswith('P')):
        for row in journal_rows(journal):
            groups[row['comparison'].rsplit('-b', 1)[0]].append(row)
    out = {key: summarize(rows) for key, rows in sorted(groups.items())}
    # Paired by seed against v1 'settled' (same seeds, same spawn-contract base opponent).
    settled = defaultdict(list)
    for journal in (HERE.parent/'timing-contract-v1').glob('B*'):
        for row in journal_rows(journal):
            if row['comparison'].rsplit('-b', 1)[0].endswith('-settled'):
                settled[row['pace']].append(row)
    contrasts = {}
    for key, rows in sorted(groups.items()):
        reference = seed_means(settled.get(rows[0]['pace'], []))
        mine = seed_means(rows)
        common = sorted(set(mine) & set(reference))
        if common:
            diff = np.array([mine[s] - reference[s] for s in common])
            boot = np.random.default_rng(RNG_SEED).choice(diff, (RESAMPLES, len(diff))).mean(axis=1)
            contrasts[f'{key}-minus-v1-settled'] = dict(seed_pairs=len(common), difference=float(diff.mean()),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
    (HERE/'assessment.json').write_text(json.dumps(dict(comparisons=out, contrasts=contrasts), indent=1)+'\n')
    for key, c in contrasts.items():
        print(f"{key:44s} pairs={c['seed_pairs']:3d} diff={c['difference']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]")
    for key, s in out.items():
        kinds = ' '.join(f"{k}:{s[k]['share_of_decisions']:.2f}@{s[k]['mean_lead_frames']:.0f}f"
                         for k in KINDS if k in s)
        print(f"{key:30s} n={s['games']:4d} score={s['score']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
              f"wait={s['mean_frames_spawn_to_first_input']:.2f} fallback={s['spawn_fallback_share']:.3f} "
              f"mismatch={s['mismatch_rate']:.3f} {kinds} {s['mismatch_reasons']}")


if __name__ == '__main__':
    main()
