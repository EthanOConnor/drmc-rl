"""Pre-register the Stronger-3 three-way tournament: arm A final vs arm B final vs the champion.

Arm A initializes the afterstate core by distilling the champion
(retention-mixed v2); arm B initializes it by imitating strong humans. Both
then run the identical outcome-PPO continuation (same config apart from
``checkpoint``/``output``). Each arm's *final* is its stop-rule selection: the
snapshot with the best pooled 896-game panel score versus the champion (ties
to the earlier snapshot), chosen on panel seeds only.

Seeds. The 16-bit reset-seed space is exhausted by training journals, and
arm A's PPO samples every seed outside its holdout. The only seeds that
neither arm (nor the champion lineage) trained on and that no selection has
touched are arm A's pre-registered ``tournament`` seeds (256 per pace), held
out by both continuations. This design plays all three pairings on exactly
those seeds, which also makes B-vs-champion and A-vs-champion a paired
comparison. The A-vs-champion pairing is arm A's own pre-registered
tournament (same seeds, schedule and settings); if it has already been played
its games are imported unchanged, otherwise it is played here. Arm A's
"exactly one candidate" clause is preserved in spirit: one candidate per arm
is fixed by a rule registered before either final exists, and neither arm's
selection sees these seeds. Using arm A's seeds needs the main session's
approval; the fallback (only if refused) is recorded below.

Stages:
    register   write afterstate-core-human-v1/threeway-preregistration.json (now)
    config --arm-a FINAL_A --arm-b FINAL_B [--arm-a-tournament-games PATH]
               write the arena config once both finals exist
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from drmc_rl.arena.experiment import execution_key  # noqa: E402
from drmc_rl.execution.pace import resolve_pace  # noqa: E402

OUT = HERE / 'afterstate-core-human-v1'
SEEDS = HERE / 'afterstate-core-v1' / 'seeds.json'
SEEDS_SHA = 'recorded at registration'
MAIN = Path('/Users/ethan/dev/drmario/drmc-rl')
CHAMPION = MAIN / 'runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt'
CHAMPION_SHA = 'fae04a599635cbf600914e5dede2f5dcdfeff7e5436dd53053c7d9fc1c85b3a3'
OUTCOME = MAIN / 'runs/trainer-baseline-v1/public-outcome-10m-inference.pt.gz'
NATIVE = MAIN / 'runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
PACES = ('sloth', 'relaxed', 'normal', 'fast', 'top_humans', 'super_human', 'frame_perfect')
PAIRINGS = (('b-vs-champion', 'arm_b', 'champion'), ('a-vs-b', 'arm_a', 'arm_b'),
            ('a-vs-champion', 'arm_a', 'champion'))

PREREGISTRATION = dict(
    schema='drmc-afterstate-threeway-preregistration-v1',
    question='Does initializing the afterstate core from strong-human imitation (arm B) instead of champion '
             'distillation (arm A) produce a stronger core after identical outcome PPO?',
    arms=dict(
        arm_a='trainer/afterstate-core: distilled from retention-mixed v2, then afterstate-core-v1/ppo-ppo-v1.json',
        arm_b='trainer/afterstate-core-human: human-corpus imitation, then afterstate-core-human-v1/ppo-*.json '
              '(arm A config with only checkpoint/output/source_commit changed)',
        champion=dict(path=str(CHAMPION), sha256=CHAMPION_SHA, label='retention-mixed v2'),
    ),
    final_selection='each arm: the PPO snapshot with the best pooled panel score versus the champion on the '
                    'registered panel seeds (ties to the earlier snapshot); stop after two consecutive '
                    'non-improving snapshots. No tournament seed is used for selection.',
    seeds='afterstate-core-v1/seeds.json group "tournament": 256 reset seeds per pace, held out by both '
          'continuations; each seed played twice with sides swapped in every pairing',
    pairings={name: f'{a} (candidate) vs {b}' for name, a, b in PAIRINGS},
    games=dict(per_pairing=3584, total=10752, new_if_arm_a_tournament_imported=7168),
    settings='level 14 HI; spawn decisions; delay max(4, reaction); reactive compute 4, preparation 6; events '
             'backend; strict FP32; 4 interleaved blocks of 64 seeds per pace; same as arm A tournament',
    score='candidate score per game: win=1, draw=0.5, loss=0; per-seed mean of the two side-swapped games',
    analysis='per pairing and pace: 20000 whole-seed bootstrap resamples (numpy default_rng(20260924)); '
             'seven-pace simultaneous 95% intervals by max-|studentized deviation|; pooled score = equal-weight '
             'mean of the seven pace scores with its 95% bootstrap interval from the same resamples. Paired '
             'difference D = score(B vs champion) - score(A vs champion) per seed, same resampling.',
    decision=dict(
        primary='a-vs-b, arm B as candidate: B_STRONGER if pooled 95% lower bound > 0.50 and no pace with '
                'simultaneous 95% upper bound < 0.45; A_STRONGER if pooled 95% upper bound < 0.50 and no pace '
                'with simultaneous 95% lower bound > 0.55; otherwise NO_DIFFERENCE_SHOWN',
        consistency='the paired difference D must have the same sign as (primary pooled score - 0.50) when '
                    'the primary verdict is not NO_DIFFERENCE_SHOWN; a contradiction is reported and '
                    'downgrades the primary verdict to NO_DIFFERENCE_SHOWN',
        versus_champion='b-vs-champion and a-vs-champion each receive arm A\'s rule: PROMOTE if pooled lower '
                        'bound > 0.50 and no pace simultaneous upper bound < 0.45; PARITY if not PROMOTE, pooled '
                        'interval contains 0.50 and no clear pace regression; REJECT otherwise',
        adoption='the browser core becomes the arm with PROMOTE versus the champion; if both PROMOTE, the '
                 'primary winner; if both PROMOTE and NO_DIFFERENCE_SHOWN, the higher pooled score versus the '
                 'champion; if neither PROMOTEs, the champion stays',
    ),
    censoring='timeouts are reported and scored as recorded; any censored game flags the report',
    fallback_if_seed_reuse_refused='after both continuations finish, rerun the prepare_afterstate_tournament_v1 '
        'seed audit (selection seed 20260925) additionally excluding both PPO training journals and arm A\'s '
        'tournament seeds, draw 256 seeds per pace (fewer if unavailable, equal per pace), and apply the same '
        'rules; the A-vs-champion pairing is then played on those seeds too',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    OUT.mkdir(exist_ok=True)
    path = OUT / 'threeway-preregistration.json'
    assert not path.exists(), 'three-way tournament already registered'
    seeds = json.loads(SEEDS.read_text())['tournament']
    arm_a_config = json.loads((HERE / 'afterstate-core-v1' / 'ppo-ppo-v1.json').read_text())
    assert all(set(v) <= set(arm_a_config['holdout_seeds']) for v in seeds.values())
    record = dict(PREREGISTRATION, created_at=datetime.now(timezone.utc).isoformat(),
                  seed_file=str(SEEDS.relative_to(REPO)), seed_file_sha256=sha(SEEDS),
                  seeds_per_pace={p: len(v) for p, v in seeds.items()},
                  source_commit=subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'],
                                               capture_output=True, text=True).stdout.strip())
    path.write_text(json.dumps(record, indent=1) + '\n')
    print(path, record['seed_file_sha256'])


def config(args):
    registration = json.loads((OUT / 'threeway-preregistration.json').read_text())
    assert sha(SEEDS) == registration['seed_file_sha256'], 'seed file changed since registration'
    assert sha(CHAMPION) == CHAMPION_SHA
    seeds = json.loads(SEEDS.read_text())['tournament']
    path = OUT / 'threeway.json'
    assert not path.exists(), 'three-way arena already configured'
    arm_a, arm_b = Path(args.arm_a).resolve(), Path(args.arm_b).resolve()
    imported = args.arm_a_tournament_games is not None
    schedule = []
    for block in range(4):
        for pace in PACES:
            chunk = seeds[pace][block * 64:(block + 1) * 64]
            profile = resolve_pace(pace).to_dict()
            for name, a, b in PAIRINGS:
                if name == 'a-vs-champion' and imported:
                    continue
                schedule.append(dict(id=f'threeway-{name}-b{block}-{pace}', a=a, b=b, games=2 * len(chunk),
                                     seeds=chunk, level=14, pace=pace, phase='Afterstate three-way',
                                     rating_group='Afterstate three-way', execution_profile=profile,
                                     execution_key=execution_key(profile)))
    output = OUT / 'threeway'
    result = {
        'checkpoint': str(OUTCOME), 'device': args.device,
        'native_library': str(NATIVE / 'libdrmario_pool.dylib'), 'reach_library': str(NATIVE / 'libdrm_reach_full.dylib'),
        'threads': 1, 'max_game_frames': 120000, 'output': str(output), 'working_db': str(output / 'working/arena.sqlite'),
        'variants': {
            'arm_a': dict(name='Afterstate core A · distilled + PPO', delay=4, checkpoint=str(arm_a)),
            'arm_b': dict(name='Afterstate core B · human imitation + PPO', delay=4, checkpoint=str(arm_b)),
            'champion': dict(name='Mixed-pace retention v2 · 3M placements', delay=4, checkpoint=str(CHAMPION)),
        },
        'reactive_compute_frames': 4, 'preparation_compute_frames': 6, 'memoize': True, 'pairs': 32,
        'replay_games': 0, 'watch': False, 'poll_seconds': 20, 'strict_fp32': True,
        'rollout_backend': 'events', 'planner_workers': 3, 'async_planning': True,
        'source': f'drmc-rl trainer/afterstate-core-human {registration["source_commit"]}', 'native_commit': '19f292c',
        'model_sha256': {str(arm_a): sha(arm_a), str(arm_b): sha(arm_b), str(CHAMPION): CHAMPION_SHA,
                         str(OUTCOME): sha(OUTCOME)},
        'seed_file': str(SEEDS), 'seed_file_sha256': registration['seed_file_sha256'],
        'arm_a_tournament_games': str(Path(args.arm_a_tournament_games).resolve()) if imported else None,
        'threeway_preregistration': dict(registration, configured_at=datetime.now(timezone.utc).isoformat(),
                                         finals=dict(arm_a=dict(path=str(arm_a), sha256=sha(arm_a)),
                                                     arm_b=dict(path=str(arm_b), sha256=sha(arm_b)))),
        'schedule': schedule,
    }
    path.write_text(json.dumps(result, indent=1) + '\n')
    print(path, sum(m['games'] for m in schedule), 'new games')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'config'))
    parser.add_argument('--arm-a')
    parser.add_argument('--arm-b')
    parser.add_argument('--arm-a-tournament-games', help="arm A tournament games.jsonl, when already played")
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    register() if args.stage == 'register' else config(args)


if __name__ == '__main__':
    main()
