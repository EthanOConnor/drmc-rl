"""Summarize timing-contract arenas: per-comparison paired scores with 95% CIs over seed pairs."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent/'timing-contract-v1'
JOURNALS = [HERE/d for d in ('A', 'B', 'B2', 'B-mac', 'B-mac2', 'C', 'C-mac')]
RESAMPLES, RNG_SEED = 20000, 20260924


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
    complete = [v for v in by_seed.values() if len(v) == 2 and None not in v]
    pairs = np.array([np.mean(v) for v in complete])
    boot = np.random.default_rng(RNG_SEED).choice(pairs, (RESAMPLES, len(pairs))).mean(axis=1)
    stats, base = Counter(), Counter()
    for row in rows:
        stats.update(row['a_stats'])
        base.update(row['b_stats'])
    decisions = max(stats['decisions'], 1)
    result = dict(games=2*len(pairs), score=float(pairs.mean()),
                  ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                  censored=sum(r['score'] is None for r in rows),
                  split_pairs=int(np.sum(pairs == .5)),
                  a_wait_frames=stats['spawn_wait_frames']/decisions,
                  base_wait_frames=base['spawn_wait_frames']/max(base['decisions'], 1))
    requests = stats['early_accepted'] + stats['early_mismatch']
    if stats['early_requests']:
        result.update(early_requests=stats['early_requests'], early_accepted=stats['early_accepted'],
                      early_mismatch=stats['early_mismatch'], early_unavailable=stats['early_unavailable'],
                      mismatch_rate=stats['early_mismatch']/max(requests, 1),
                      fallback_share_of_decisions=(stats['early_mismatch']+stats['early_unavailable'])/decisions,
                      mean_lead_frames=stats['early_lead_frames']/max(stats['early_accepted'], 1))
    return result


def main():
    groups, first = defaultdict(list), defaultdict(list)
    for journal in JOURNALS:
        for row in journal_rows(journal):
            key = row['comparison'].rsplit('-b', 1)[0]
            groups[key].append(row)
            if row['comparison'].endswith('-b0'):
                first[key].append(row)
    out = {key: dict(all=summarize(rows), first_read=summarize(first[key]) if first[key] else None)
           for key, rows in sorted(groups.items())}
    # Timing effect net of preview marginalization: early point minus the spawn-time marginal
    # control, paired by seed (both measured against the same spawn-contract base).
    for pace in ('frame_perfect', 'super_human', 'top_humans'):
        control = seed_means(groups.get(f'C-{pace}-premarg', []))
        for point in ('settled', 'lock'):
            key = f'B-{pace}-{point}'
            early = seed_means(groups.get(key, []))
            common = sorted(set(early) & set(control))
            if not common:
                continue
            diff = np.array([early[s] - control[s] for s in common])
            boot = np.random.default_rng(RNG_SEED).choice(diff, (RESAMPLES, len(diff))).mean(axis=1)
            out[f'{key}-minus-premarg'] = dict(seed_pairs=len(common), difference=float(diff.mean()),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
    (HERE/'assessment.json').write_text(json.dumps(out, indent=1)+'\n')
    for key, value in out.items():
        if 'difference' in value:
            print(f"{key:36s} pairs={value['seed_pairs']:3d} diff={value['difference']:+.3f} "
                  f"[{value['ci95'][0]:+.3f},{value['ci95'][1]:+.3f}]")
            continue
        s = value['all']
        extra = ('' if 'early_requests' not in s else
                 f" mismatch={s['mismatch_rate']:.3f} fallback={s['fallback_share_of_decisions']:.3f}"
                 f" lead={s['mean_lead_frames']:.1f}")
        print(f"{key:28s} n={s['games']:4d} score={s['score']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
              f" wait={s['a_wait_frames']:.2f}/{s['base_wait_frames']:.2f} split={s['split_pairs']}{extra}")


if __name__ == '__main__':
    sys.exit(main())
