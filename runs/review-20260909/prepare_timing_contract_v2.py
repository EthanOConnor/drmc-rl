"""Timing-contract v2 configs: lock mismatch reasons, lock_safe, commit_safe and early preview handling.

Reuses the v1 per-pace seeds (same spawn-contract base opponent) so v2 variants pair with v1 results.
Each process gets its own config; rerunning never rewrites a launched config.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V1 = HERE/'timing-contract-v1'
OUT = HERE/'timing-contract-v2'
SHM = Path('/dev/shm/timing-contract')
MAIN = Path('/Users/ethan/dev/drmario/drmc-rl')
MAC_NATIVE = MAIN/'runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
sys.path.insert(0, str(HERE.parents[1]))
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.arena.experiment import execution_key

seeds = json.loads((V1/'seeds.json').read_text())
FIRST, FULL = 64, 192
PACES = ('frame_perfect', 'super_human', 'top_humans', 'fast')
VARIANTS = dict(
    base=dict(delay=4),
    lock=dict(delay=4, decision_point='lock'),
    lock_safe=dict(delay=4, decision_point='lock_safe'),
    commit_marg=dict(delay=4, decision_point='commit_safe', early_preview='marginal'),
    commit_repeat=dict(delay=4, decision_point='commit_safe', early_preview='repeat'),
    commit_branch=dict(delay=4, decision_point='commit_safe', early_preview='branches'),
)
remote = json.loads((V1/'config-A-tf3090.json').read_text())
mac = json.loads((V1/'config-B-mac.json').read_text())


def rows(names, paces, block):
    lo, hi = (0, FIRST) if block == 0 else (FIRST, FULL)
    out = []
    for pace in paces:
        for name in names:
            profile = resolve_pace(pace).to_dict()
            out.append(dict(id=f'V2-{pace}-{name}-b{block}', a=name, b='base', games=2*(hi-lo),
                            seeds=seeds[pace][lo:hi], level=14, pace=pace, phase='Timing contract v2',
                            rating_group=f'V2 {name}', execution_profile=profile, execution_key=execution_key(profile)))
    return out


def config(name, schedule, host='tf3090'):
    template = remote if host == 'tf3090' else mac
    output = SHM/name if host == 'tf3090' else OUT/name
    drop = ('schedule', 'pruned_after_first_read', 'variants', 'rebalanced')
    return {**{k: v for k, v in template.items() if k not in drop},
            'output': str(output), 'working_db': str(output/'working/arena.sqlite'),
            'rollout_backend': 'frames', 'async_planning': False, 'pairs': 128,
            'variants': {k: dict(name=f'Mixed-v2 core · {k}', checkpoint=template['variants']['base']['checkpoint'], **v)
                         for k, v in VARIANTS.items()},
            'source': 'drmc-rl trainer/pace-conditioned-strategy 6703d7d + trainer/timing-contract v2 knobs',
            'schedule': schedule}


OUT.mkdir(exist_ok=True)
configs = {
    # First read: 128 games (64 seed pairs) per comparison.
    'config-P1-tf3090.json': config('P1', rows(('commit_marg',), PACES, 0) + rows(('lock',), ('frame_perfect',), 0)),
    'config-P2-tf3090.json': config('P2', rows(('commit_branch',), PACES, 0)),
    'config-P4-tf3090.json': config('P4', rows(('commit_repeat',), PACES, 0)),
    'config-P3-mac.json': config('P3', rows(('lock_safe',), PACES, 0), host='mac'),
    # Extension to 384 games (another 128 seed pairs) for the comparisons that decide the contract.
    'config-P5-tf3090.json': config('P5', rows(('commit_marg',), PACES[:3], 1)),
    'config-P6-tf3090.json': config('P6', rows(('commit_branch',), PACES[:3], 1)),
    'config-P7-tf3090.json': config('P7', rows(('commit_repeat',), PACES[:3], 1) + rows(('lock_safe',), PACES[1:2], 1)),
    'config-P8-mac.json': config('P8', rows(('lock_safe',), PACES[:1], 1), host='mac'),
}
for name, value in configs.items():
    if not (OUT/name).exists():
        (OUT/name).write_text(json.dumps(value, indent=1)+'\n')
print({k: sum(r['games'] for r in v['schedule']) for k, v in configs.items()})
