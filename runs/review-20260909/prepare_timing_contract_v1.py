"""Select review-unseen seeds and write the timing-contract arena configs (experiments A and B)."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MAIN = Path('/Users/ethan/dev/drmario/drmc-rl')                 # read-only inputs (checkpoints, audits)
HERE = Path(__file__).resolve().parent
OUT = HERE/'timing-contract-v1'
FRESH = MAIN/'runs/review-20260909/controller-retention-fresh-v1'
AUDIT_NOW = OUT/'audit/seed-audit-local-20260923.json.gz'      # tools: FRESH/audit/seed_audit_local.py, rerun now
SELECTION_SEED = 20260924
PACES = ('frame_perfect', 'super_human', 'top_humans', 'fast', 'normal')
SEEDS_PER_PACE, FIRST = 192, 64                                # first read 128 games; confirmation 384 games
CONFIRM_A = ('frame_perfect', 'super_human', 'top_humans')
CORE = MAIN/'runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt'
CORE_SHA = 'fae04a599635cbf600914e5dede2f5dcdfeff7e5436dd53053c7d9fc1c85b3a3'
OUTCOME = MAIN/'runs/trainer-baseline-v1/public-outcome-10m-inference.pt.gz'
MAC_NATIVE = MAIN/'runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
SHM = Path('/dev/shm/timing-contract')

sys.path.insert(0, str(HERE.parents[1]))
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.arena.experiment import execution_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def journal_seeds(audit, predicate=lambda path: True):
    return set().union(*[set(v['seeds']) for p, v in audit['journals'].items() if predicate(p)])


assert sha(CORE) == CORE_SHA
remote = json.load(gzip.open(FRESH/'audit/seed-audit-tf3090.json.gz', 'rt'))
local_then = json.load(gzip.open(FRESH/'audit/seed-audit-local.json.gz', 'rt'))
local_now = json.load(gzip.open(AUDIT_NOW, 'rt'))
review = lambda p: 'review-20260909' in p
fresh_seeds = {s for v in json.loads((FRESH/'seeds.json').read_text()).values() for s in v}
excluded = dict(
    review_journals_local=journal_seeds(local_now, review) | journal_seeds(local_then, review),
    review_journals_tf3090=journal_seeds(remote, review),
    core_training_holdout_anchor=set(remote['mixed_v2_holdout']) | journal_seeds(
        remote, lambda p: 'controller-retention-mixed-v2/training-games' in p or 'controller-retention-bank-v1' in p),
    retention_fresh_v1=fresh_seeds,
)
every = set().union(*excluded.values())
available = np.asarray(sorted(set(range(1, 65536)) - every))
chosen = np.random.default_rng(SELECTION_SEED).choice(available, SEEDS_PER_PACE*len(PACES), replace=False)
seeds = {p: [int(s) for s in chosen[i*SEEDS_PER_PACE:(i+1)*SEEDS_PER_PACE]] for i, p in enumerate(PACES)}
OUT.mkdir(exist_ok=True)
(OUT/'seeds.json').write_text(json.dumps(seeds, indent=1)+'\n')
all_journals = journal_seeds(local_now) | journal_seeds(remote) | journal_seeds(local_then)
(OUT/'seed-audit.json').write_text(json.dumps(dict(
    schema='drmc-timing-contract-seed-audit-v1', created_at=datetime.now(timezone.utc).isoformat(),
    method='Exclude every seed in any seed-bearing journal under runs/review-20260909 (local audit rerun now plus '
           'the Sep 23 local and tf3090 audits), the mixed-v2 core training/holdout/anchor seeds, and the '
           'retention-fresh-v1 seeds; draw without replacement with numpy default_rng(selection_seed); '
           'split in pace order (192 per pace, first 64 = first read).',
    selection_seed=SELECTION_SEED, excluded={k: len(v) for k, v in excluded.items()}, excluded_total=len(every),
    available_after_exclusion=int(len(available)),
    overlap_with_older_non_review_journals=len(set(chosen.tolist()) & all_journals),
    globally_unseen_remaining=65535 - len(all_journals | every),
    limitation='Not globally unseen: too few seeds remain outside every audited journal for 960 seeds. '
               'Chosen seeds may appear in pre-review journals of other models; none were used to train or '
               'evaluate the mixed-v2 core in the review program.',
), indent=1)+'\n')


A_VARIANTS = dict(
    base=dict(delay=4),
    c5=dict(delay=5), c6=dict(delay=6), c8=dict(delay=8),
    c6pin=dict(delay=6, compute_input_frames=4), c8pin=dict(delay=8, compute_input_frames=4),
)
B_VARIANTS = dict(
    base=dict(delay=4),
    settled=dict(delay=4, decision_point='settled'),
    lock=dict(delay=4, decision_point='lock'),
)


def schedule(names, paces, blocks, prefix):
    rows = []
    for block, (lo, hi) in blocks:
        for pace in paces[block]:
            for name in names:
                profile = resolve_pace(pace).to_dict()
                rows.append(dict(id=f'{prefix}-{pace}-{name}-b{block}', a=name, b='base', games=2*(hi-lo),
                                 seeds=seeds[pace][lo:hi], level=14, pace=pace, phase='Timing contract',
                                 rating_group=f'{prefix} {name}', execution_profile=profile,
                                 execution_key=execution_key(profile)))
    return rows


def config(output, native, reach, device, core, outcome, variant_params, rows, backend):
    return {
        'checkpoint': str(outcome), 'device': device, 'native_library': str(native), 'threads': 1,
        'max_game_frames': 120000, 'output': str(output), 'working_db': str(output/'working/arena.sqlite'),
        'variants': {k: dict(name=f'Mixed-v2 core · {k}', checkpoint=str(core), **v) for k, v in variant_params.items()},
        'reactive_compute_frames': 4, 'preparation_compute_frames': 6, 'memoize': True, 'pairs': 32 if backend == 'events' else 256,
        'replay_games': 0, 'watch': False, 'strict_fp32': True, 'rollout_backend': backend,
        'planner_workers': 3, 'async_planning': backend == 'events',
        'source': 'drmc-rl trainer/pace-conditioned-strategy 6703d7d + trainer/timing-contract knobs',
        'native_commit': '19f292c', 'reach_library': str(reach),
        'model_sha256': {str(core): CORE_SHA}, 'seed_file': str(OUT/'seeds.json'), 'schedule': rows,
    }


names_a = ('c5', 'c6', 'c8', 'c6pin', 'c8pin')
blocks_a = [(0, (0, FIRST)), (1, (FIRST, SEEDS_PER_PACE))]
rows_a = schedule(names_a, {0: PACES, 1: CONFIRM_A}, blocks_a, 'A')
# B runs on the frame runner; split across hosts by pace for wall time (seeds are disjoint by pace).
rows_b_remote = schedule(('settled', 'lock'), {0: CONFIRM_A[:2], 1: CONFIRM_A[:2]}, blocks_a, 'B')
rows_b_mac = schedule(('settled', 'lock'), {0: CONFIRM_A[2:], 1: CONFIRM_A[2:]}, blocks_a, 'B')
rows_b = rows_b_remote + rows_b_mac
remote_out = SHM/'A'
configs = {
    'config-A-tf3090.json': config(remote_out, SHM/'native/libdrmario_pool.so', SHM/'native/libdrm_reach_full.so',
                                   'cuda', SHM/'ckpt/core-final-inference.pt', SHM/'ckpt/public-outcome-10m-inference.pt.gz',
                                   A_VARIANTS, rows_a, 'events'),
    'config-B-tf3090.json': config(SHM/'B', SHM/'native/libdrmario_pool.so', SHM/'native/libdrm_reach_full.so',
                                   'cuda', SHM/'ckpt/core-final-inference.pt', SHM/'ckpt/public-outcome-10m-inference.pt.gz',
                                   B_VARIANTS, rows_b_remote, 'frames'),
    'config-B-mac.json': config(OUT/'B-mac', MAC_NATIVE/'libdrmario_pool.dylib', MAC_NATIVE/'libdrm_reach_full.dylib',
                                'mps', CORE, OUTCOME, B_VARIANTS, rows_b_mac, 'frames'),
    # Diagnostic: spawn-time decisions that marginalize the visible preview, as the early points must.
    'config-C-tf3090.json': config(SHM/'C', SHM/'native/libdrmario_pool.so', SHM/'native/libdrm_reach_full.so',
                                   'cuda', SHM/'ckpt/core-final-inference.pt', SHM/'ckpt/public-outcome-10m-inference.pt.gz',
                                   dict(base=dict(delay=4), premarg=dict(delay=4, preview_input='marginal')),
                                   schedule(('premarg',), {0: CONFIRM_A, 1: CONFIRM_A}, blocks_a, 'C'),
                                   'frames'),
}
for name, value in configs.items():
    if not (OUT/name).exists():   # never rewrite a launched (possibly pruned) config
        (OUT/name).write_text(json.dumps(value, indent=1)+'\n')
print('available', len(available), 'excluded', len(every), 'A games', sum(r['games'] for r in rows_a),
      'B games', sum(r['games'] for r in rows_b))
