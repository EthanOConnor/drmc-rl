"""Outcome-PPO continuation of the distilled afterstate core with the mixed-pace retention recipe.

Copies the controller-retention-mixed-v2 contract unchanged except: the
initialization is the distilled afterstate student; the frozen opponent pool
adds the champion (retention-mixed v2) itself; every registered afterstate
evaluation seed (quick, panel, tournament) joins the training holdout; and an
inference snapshot is written every 50M natural-game frames for the stop rule
(evaluate each snapshot on the fixed 896-game panel versus the champion; stop
after two consecutive snapshots that do not improve the best pooled panel
score). Writes the config next to this script and prints its tf3090 path.
"""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output'
BASE = f'{REMOTE}/review-20260909'
STUDENT = sys.argv[1]  # tf3090 path of the distilled student inference checkpoint
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else f'{REMOTE}/afterstate-core-v1/ppo-v1'
CHAMPION = f'{BASE}/controller-retention-mixed-v2/core-final-inference.pt'

mixed = json.loads(subprocess.run(['ssh', 'tf3090', 'cat', f'{BASE}/controller-retention-mixed-v2/config.json'],
                                  check=True, capture_output=True, text=True).stdout)
seeds = json.loads((HERE / 'afterstate-core-v1/seeds.json').read_text())
registered = {s for group in seeds.values() for pace in group.values() for s in pace}
pool = {entry['id']: entry for entry in mixed['opponent_pool']}
config = dict(mixed)
config.update(
    checkpoint=STUDENT,
    output=OUTPUT,
    opponent_pool=[
        dict(id='champion', weight=0.4, checkpoint=CHAMPION),
        dict(pool['core300m'], weight=0.3),
        dict(pool['pace_corrected'], weight=0.3),
    ],
    holdout_seeds=sorted(set(mixed['holdout_seeds']) | registered),
    native_library='/dev/shm/afterstate-core/native/libdrmario_pool.so',
    planner_workers=3,
    source_commit=subprocess.run(['git', '-C', str(HERE.parents[1]), 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True).stdout.strip(),
    native_commit='19f292c',
    milestone_decisions=[],
    checkpoint_every_frames=50_000_000,
    arm='mixed_retention',
)
path = HERE / 'afterstate-core-v1' / f'ppo-{Path(OUTPUT).name}.json'
assert not path.exists(), f'{path.name} already written'
path.write_text(json.dumps(config, indent=1) + '\n')
print(path)
