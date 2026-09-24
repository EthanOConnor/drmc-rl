"""Pre-register the afterstate-core evaluation: quick read, checkpoint panel and seven-pace tournament.

Stage 1 (``seeds``): audit every seed-bearing journal reachable on this Mac and
tf3090 (read-only), then draw three disjoint globally unseen seed sets:

* ``quick``: 64 seeds each at normal and frame_perfect (128 side-swapped games
  per pace) for the first distilled checkpoint;
* ``panel``: 64 seeds per pace at all seven paces (896 games) for the outcome-PPO
  stop rule, reused unchanged for every checkpoint;
* ``tournament``: 256 seeds per pace at all seven paces (3584 games), played
  exactly once, by exactly one pre-registered candidate.

All three sets must also be excluded from training (``holdout_seeds`` of any
PPO continuation). Stage 2 (``quick``/``panel``/``tournament``) writes arena
configs for a named candidate checkpoint versus the champion; the tournament
refuses a second candidate.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from drmc_rl.arena.experiment import execution_key  # noqa: E402
from drmc_rl.execution.pace import resolve_pace  # noqa: E402

OUT = HERE / 'afterstate-core-v1'
MAIN = Path('/Users/ethan/dev/drmario/drmc-rl')
CHAMPION = MAIN / 'runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt'
CHAMPION_SHA = 'fae04a599635cbf600914e5dede2f5dcdfeff7e5436dd53053c7d9fc1c85b3a3'
OUTCOME = MAIN / 'runs/trainer-baseline-v1/public-outcome-10m-inference.pt.gz'
NATIVE = MAIN / 'runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
PACES = ('sloth', 'relaxed', 'normal', 'fast', 'top_humans', 'super_human', 'frame_perfect')
SELECTION_SEED = 20260924
SIZES = dict(quick=dict(normal=64, frame_perfect=64), panel={p: 64 for p in PACES},
             tournament={p: 256 for p in PACES})
LOCAL_ROOTS = [MAIN / 'runs', *sorted(Path('/Users/ethan/dev/drmario').glob('drmc-rl-*/runs')),
               *sorted(MAIN.glob('.claude/worktrees/*/runs')),
               *sorted(Path('/Users/ethan/dev/drmario/professorPills/.claude/worktrees').glob('*/drmc-rl*/runs'))]
LINEAGE = ('controller-core-live-v4/', 'controller-retention-mixed-v2/')
REMOTE_ROOTS = ['/home/ethan/.cache/drmc-rl/trainer-output', '/dev/shm']
AUDIT = r'''
import json, os, gzip, sys
out = {"journals": {}, "errors": []}
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.endswith("-source") and d not in ("public-replay", ".git", "vendor", "replays", "moves", "node_modules", "__pycache__")]
        for f in fn:
            if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")):
                continue
            p = os.path.join(dp, f)
            try:
                op = gzip.open if f.endswith(".gz") else open
                s = set(); n = 0
                with op(p, "rt") as h:
                    for i, line in enumerate(h):
                        if i == 0 and '"seed"' not in line:
                            break
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        n += 1
                        if isinstance(r.get("seed"), int):
                            s.add(r["seed"])
                if s:
                    out["journals"][p] = dict(rows=n, seeds=sorted(s))
            except Exception as e:
                out["errors"].append([p, repr(e)[:200]])
json.dump(out, sys.stdout)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    local = subprocess.run([sys.executable, '-c', f'ROOTS={[str(r) for r in LOCAL_ROOTS if r.exists()]!r}\n' + AUDIT],
                           check=True, capture_output=True, text=True).stdout
    remote = subprocess.run(['ssh', 'tf3090', 'python3', '-'], input=f'ROOTS={REMOTE_ROOTS!r}\n' + AUDIT,
                            check=True, capture_output=True, text=True).stdout
    return json.loads(local), json.loads(remote)


def stage_seeds():
    OUT.mkdir(exist_ok=True)
    seed_file = OUT / 'seeds.json'
    assert not seed_file.exists(), 'seeds already registered'
    local, remote = audit()
    for name, value in (('local', local), ('tf3090', remote)):
        with gzip.open(OUT / f'seed-audit-{name}.json.gz', 'wt') as handle:
            json.dump(value, handle)
    earlier = json.load(gzip.open(MAIN / 'runs/review-20260909/controller-retention-fresh-v1/audit/seed-audit-tf3090.json.gz', 'rt'))
    journals = {**earlier['journals'], **remote['journals'], **local['journals']}
    fresh = json.loads((MAIN / 'runs/review-20260909/controller-retention-fresh-v1/seeds.json').read_text())
    union = lambda paths: set().union(set(), *[set(journals[p]['seeds']) for p in paths])
    training = {p for p in journals if 'training-games' in p}
    lineage = {p for p in training if any(k in p for k in LINEAGE)}
    # The 16-bit reset-seed space is nearly exhausted by training journals, so
    # "fresh" means: never in any evaluation/arena/anchor journal, never in the
    # champion lineage's or the distillation source's training games, and not a
    # registered holdout. Globally unseen seeds are used first; the remainder
    # appear only in training games of unrelated study arms.
    excluded = union(set(journals) - training) | union(lineage)
    excluded |= {s for v in fresh.values() for s in v} | set(earlier['mixed_v2_holdout'])
    other_training = union(training - lineage)
    unseen = sorted(set(range(1, 65536)) - excluded - other_training)
    reused = sorted(set(range(1, 65536)) - excluded - set(unseen))
    need = sum(sum(v.values()) for v in SIZES.values())
    rng = np.random.default_rng(SELECTION_SEED)
    take_unseen = list(rng.permutation(unseen)[:need])
    take_reused = list(rng.choice(reused, need - len(take_unseen), replace=False)) if need > len(take_unseen) else []
    chosen = [int(s) for s in rng.permutation(np.asarray(take_unseen + take_reused, dtype=np.int64))]
    assert len(chosen) == need
    seeds = {}
    for group, per_pace in SIZES.items():
        seeds[group] = {}
        for pace, count in per_pace.items():
            seeds[group][pace] = sorted(int(chosen.pop()) for _ in range(count))
    flat = [s for g in seeds.values() for v in g.values() for s in v]
    assert len(flat) == len(set(flat)) == need and not set(flat) & excluded
    seed_file.write_text(json.dumps(seeds, indent=1) + '\n')
    (OUT / 'seed-audit.json').write_text(json.dumps(dict(
        schema='drmc-afterstate-seed-audit-v1', created_at=datetime.now(timezone.utc).isoformat(),
        selection_seed=SELECTION_SEED, journals_audited=len(journals),
        errors=dict(local=len(local['errors']), tf3090=len(remote['errors'])),
        local_roots=[str(r) for r in LOCAL_ROOTS if r.exists()], remote_roots=REMOTE_ROOTS,
        rule='exclude every evaluation/arena/anchor journal seed, champion-lineage and distillation-source training '
             'seeds (controller-core-live-v4, controller-retention-mixed-v2), fresh-v1 seeds and the mixed-v2 holdout; '
             'use globally unseen seeds first, then seeds seen only in unrelated study-arm training games',
        excluded=len(excluded), globally_unseen_available=len(unseen), unrelated_training_only_available=len(reused),
        selected_globally_unseen=len(take_unseen), selected_unrelated_training_only=len(take_reused),
        champion_lineage_training_journals=sorted(lineage),
        unrelated_training_journals=sorted(training - lineage), sizes=SIZES,
        seed_file_sha256=sha(seed_file),
        limitation='Only journals reachable on this Mac and tf3090 are audited; deleted logs cannot be ruled out.',
    ), indent=1) + '\n')
    print('excluded', len(excluded), 'unseen', len(unseen), 'reused', len(reused), 'seed sha', sha(seed_file))


CONFIRMATION = dict(per_pace=64, rng=20260926)
SHIPPED_TIMING = dict(decision_point='lock_safe', early_preview='repeat')


def stage_confirmation_seeds():
    """Addendum seeds: disjoint from every registered list and every training journal (fresh audit)."""
    path = OUT / 'confirmation-seeds.json'
    assert not path.exists(), 'confirmation seeds already registered'
    local, remote = audit()
    for name, value in (('local', local), ('tf3090', remote)):
        with gzip.open(OUT / f'confirmation-seed-audit-{name}.json.gz', 'wt') as handle:
            json.dump(value, handle)
    earlier = json.load(gzip.open(MAIN / 'runs/review-20260909/controller-retention-fresh-v1/audit/seed-audit-tf3090.json.gz', 'rt'))
    journals = {**earlier['journals'], **json.load(gzip.open(OUT / 'seed-audit-tf3090.json.gz', 'rt'))['journals'],
                **json.load(gzip.open(OUT / 'seed-audit-local.json.gz', 'rt'))['journals'],
                **remote['journals'], **local['journals']}
    union = lambda paths: set().union(set(), *[set(journals[p]['seeds']) for p in paths])
    training = {p for p in journals if 'training-games' in p}
    registered = set(training_holdout()) | set(earlier['mixed_v2_holdout'])
    for seed_list in Path('/Users/ethan/dev/drmario').glob('drmc-rl*/runs/review-20260909/**/*.json'):
        if seed_list.name in ('seeds.json',) or seed_list.name.startswith('early-preview'):
            try:
                value = json.loads(seed_list.read_text())
            except ValueError:
                continue
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    if item and all(type(x) is int for x in item):
                        registered |= set(item)
                    else:
                        stack.extend(item)
    related = {p for p in training if any(k in p for k in (*LINEAGE, 'afterstate'))}
    blocked = registered | union(related)
    trained, evaluated = union(training - related), union(set(journals) - training)
    unseen = sorted(set(range(1, 65536)) - blocked - evaluated - trained)
    evaluated_only = sorted(evaluated - blocked - trained)
    unrelated_training = sorted(trained - blocked)
    need = CONFIRMATION['per_pace'] * len(PACES)
    rng = np.random.default_rng(CONFIRMATION['rng'])
    chosen, tiers = [], {}
    for name, pool in (('unseen', unseen), ('evaluation_only', evaluated_only), ('unrelated_training_only', unrelated_training)):
        take = [int(x) for x in rng.permutation(np.asarray(pool, dtype=np.int64))[:need - len(chosen)]]
        tiers[name] = dict(available=len(pool), selected=len(take))
        chosen += take
    assert len(chosen) == need
    chosen = [int(x) for x in rng.permutation(np.asarray(chosen, dtype=np.int64))]
    seeds = {'confirmation': {p: sorted(chosen[i * CONFIRMATION['per_pace']:(i + 1) * CONFIRMATION['per_pace']])
                              for i, p in enumerate(PACES)}}
    assert len({s for v in seeds['confirmation'].values() for s in v}) == need
    path.write_text(json.dumps(seeds, indent=1) + '\n')
    (OUT / 'confirmation-seed-audit.json').write_text(json.dumps(dict(
        created_at=datetime.now(timezone.utc).isoformat(), rng=CONFIRMATION['rng'], need=need,
        rule='exclude every registered seed list under drmc-rl*/runs/review-20260909 (seeds.json files and early-preview '
             'configs), the mixed-v2 holdout, and training games of the champion lineage and every afterstate arm '
             '(fresh audit of both hosts, including running PPO); then take never-seen seeds, seeds seen only in '
             'evaluation journals, and finally seeds seen only in unrelated study arms\' training games',
        tiers=tiers,
        seed_file_sha256=sha(path)), indent=1) + '\n')
    print('unseen', len(unseen), 'evaluated-only', len(evaluated_only), 'seed sha', sha(path))


def training_holdout():
    """Every registered evaluation seed: pass as holdout_seeds to any continuation."""
    seeds = json.loads((OUT / 'seeds.json').read_text())
    return sorted({s for g in seeds.values() for v in g.values() for s in v})


def arena_config(group, candidate, label, device, output):
    seed_file = OUT / ('confirmation-seeds.json' if group == 'confirmation' else 'seeds.json')
    seeds = json.loads(seed_file.read_text())[group]
    blocks = 4 if group == 'tournament' else 1
    schedule = []
    for b in range(blocks):
        for pace in PACES:
            if pace not in seeds:
                continue
            chunk = len(seeds[pace]) // blocks
            profile = resolve_pace(pace).to_dict()
            schedule.append(dict(id=f'{group}-b{b}-{pace}', a='candidate', b='champion', games=2 * chunk,
                                 seeds=seeds[pace][b * chunk:(b + 1) * chunk], level=14, pace=pace,
                                 phase=f'Afterstate core {group}', rating_group=f'Afterstate core {group}',
                                 execution_profile=profile, execution_key=execution_key(profile)))
    assert sha(CHAMPION) == CHAMPION_SHA
    return {
        'checkpoint': str(OUTCOME), 'device': device, 'native_library': str(NATIVE / 'libdrmario_pool.dylib'),
        'reach_library': str(NATIVE / 'libdrm_reach_full.dylib'),
        'threads': 1, 'max_game_frames': 120000, 'output': str(output), 'working_db': str(output / 'working/arena.sqlite'),
        'variants': {
            'candidate': dict(name=f'Afterstate core · {label}', delay=4, checkpoint=str(candidate)),
            'champion': dict(name='Mixed-pace retention v2 · 3M placements', delay=4, checkpoint=str(CHAMPION)),
        },
        'reactive_compute_frames': 4, 'preparation_compute_frames': 6, 'memoize': True, 'pairs': 32,
        'replay_games': 0, 'watch': False, 'poll_seconds': 20, 'strict_fp32': True,
        'rollout_backend': 'events', 'planner_workers': 3, 'async_planning': True,
        'source': f'drmc-rl trainer/afterstate-core {subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}',
        'native_commit': '19f292c',
        'model_sha256': {str(candidate): sha(candidate), str(CHAMPION): CHAMPION_SHA, str(OUTCOME): sha(OUTCOME)},
        'seed_file': str(seed_file), 'seed_file_sha256': sha(seed_file),
        'schedule': schedule,
    } | (CONFIRMATION_BACKEND if group == 'confirmation' else {})


# The shipped browser contract (v17/v18) decides early on both sides; early
# decision points need the frame-accurate runner, as in early-preview-v1.
CONFIRMATION_BACKEND = dict(rollout_backend='frames', async_planning=False, pairs=16)


PREREGISTRATION = dict(
    schema='drmc-afterstate-tournament-preregistration-v1',
    champion=dict(path=str(CHAMPION), sha256=CHAMPION_SHA, label='retention-mixed v2 (public_pair_context_v3)'),
    level='14 HI', decision_point='spawn (default contract); decision delay max(4, reaction_frames)',
    score='candidate score per game: win=1, draw=0.5, loss=0; per-seed mean of the two side-swapped games',
    analysis='20000 whole-seed bootstrap resamples (numpy default_rng(20260924)) per pace; seven-pace simultaneous 95% '
             'intervals by max-|studentized deviation|; pooled score = equal-weight mean of the seven pace scores with '
             'its own 95% bootstrap interval from the same resamples',
    decision=dict(
        PROMOTE='pooled 95% lower bound > 0.50 AND no pace with simultaneous 95% upper bound < 0.45',
        PARITY='not PROMOTE AND pooled 95% interval contains 0.50 AND no pace with simultaneous 95% upper bound < 0.45',
        REJECT='otherwise (pooled upper bound < 0.50, or any pace clearly below 0.45)',
    ),
    censoring='timeouts are reported and scored as recorded; any censored game flags the report',
    single_candidate='exactly one candidate plays the tournament seeds: the snapshot with the best pooled panel score '
                     'among the distilled student (snapshot 0) and the outcome-PPO 50M-frame snapshots, selected when '
                     'the stop rule fires (two consecutive non-improving snapshots) or PPO ends; panel seeds only',
    assessment_script=str(HERE / 'assess_afterstate_tournament_v1.py'),
)


CONFIRMATION_PREREGISTRATION = dict(
    schema='drmc-afterstate-shipped-timing-confirmation-v1',
    purpose='Addendum (registered before any tournament game): before any core ships, the tournament winner plays the '
            'champion under the shipped browser timing contract. The spawn-contract tournament remains the primary test.',
    champion=dict(path=str(CHAMPION), sha256=CHAMPION_SHA),
    timing=dict(SHIPPED_TIMING, sides='both candidate and champion', source='early-preview-v1 variant lock_safe_repeat; '
                'browser v17/v18'),
    level='14 HI', seeds='confirmation-seeds.json: 64 per pace at all seven paces, sides swapped (896 games)',
    analysis='as the tournament: whole-seed bootstrap, pooled equal-pace score with 95% interval',
    decision=dict(FAIL='pooled 95% lower bound < 0.45', PASS='otherwise'),
    applies_to='whichever arm (A, B or C) wins its tournament; the same seeds and rule for every arm',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('seeds', 'confirmation-seeds', 'quick', 'panel', 'tournament', 'confirmation', 'holdout'))
    parser.add_argument('--candidate')
    parser.add_argument('--label')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    if args.stage == 'seeds':
        return stage_seeds()
    if args.stage == 'confirmation-seeds':
        return stage_confirmation_seeds()
    if args.stage == 'holdout':
        return print(json.dumps(training_holdout()))
    candidate = Path(args.candidate).resolve()
    name = f'{args.stage}-{args.label}' if args.stage not in ('tournament', 'confirmation') else args.stage
    path = OUT / f'{name}.json'
    assert not path.exists(), f'{path.name} already registered'
    config = arena_config(args.stage, candidate, args.label, args.device, OUT / name)
    if args.stage == 'confirmation':
        for variant in config['variants'].values():
            variant.update(SHIPPED_TIMING)
        config['preregistration'] = dict(CONFIRMATION_PREREGISTRATION, created_at=datetime.now(timezone.utc).isoformat(),
                                         candidate=dict(path=str(candidate), sha256=sha(candidate), label=args.label),
                                         total_games=sum(m['games'] for m in config['schedule']))
    if args.stage == 'tournament':
        config['preregistration'] = dict(PREREGISTRATION, created_at=datetime.now(timezone.utc).isoformat(),
                                         candidate=dict(path=str(candidate), sha256=sha(candidate), label=args.label),
                                         total_games=sum(m['games'] for m in config['schedule']),
                                         schedule='4 interleaved blocks of 64 seeds per pace; sides swapped per seed')
    path.write_text(json.dumps(config, indent=1) + '\n')
    print(path, sum(m['games'] for m in config['schedule']), 'games')


if __name__ == '__main__':
    main()
