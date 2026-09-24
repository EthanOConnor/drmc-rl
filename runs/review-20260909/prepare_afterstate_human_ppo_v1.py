"""Outcome-PPO continuation of the human-imitation afterstate core (Stronger-3 arm B).

Arm B must differ from arm A only in its initialization. This copies arm A's
registered continuation config (``afterstate-core-v1/ppo-ppo-v1.json``: the
controller-retention-mixed-v2 recipe, the champion in the opponent pool, every
registered afterstate evaluation seed in ``holdout_seeds``, the same training
seed and therefore the same collection schedule, a snapshot every 50M frames)
and replaces only ``checkpoint`` (the arm-B imitation inference checkpoint),
``output`` and ``source_commit``. The stop rule is arm A's: evaluate each
snapshot on the fixed 896-game panel versus the champion and stop after two
consecutive snapshots that do not improve the best pooled panel score.

Usage (paths are tf3090 paths):
    python prepare_afterstate_human_ppo_v1.py STUDENT [OUTPUT]
"""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output'
ARM_A = HERE / 'afterstate-core-v1' / 'ppo-ppo-v1.json'
STUDENT = sys.argv[1]
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else f'{REMOTE}/afterstate-core-human-v1/ppo-v1'

config = json.loads(ARM_A.read_text())
seeds = json.loads((HERE / 'afterstate-core-v1' / 'seeds.json').read_text())
registered = {s for group in seeds.values() for pace in group.values() for s in pace}
assert registered <= set(config['holdout_seeds']), 'arm A holdout must contain every registered evaluation seed'
changed = dict(
    checkpoint=STUDENT,
    output=OUTPUT,
    source_commit=subprocess.run(['git', '-C', str(HERE.parents[1]), 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True).stdout.strip(),
)
config.update(changed)
out = HERE / 'afterstate-core-human-v1'
out.mkdir(exist_ok=True)
path = out / f'ppo-{Path(OUTPUT).name}.json'
assert not path.exists(), f'{path.name} already written'
path.write_text(json.dumps(config, indent=1) + '\n')
(out / f'ppo-{Path(OUTPUT).name}.provenance.json').write_text(json.dumps(dict(
    schema='drmc-afterstate-arm-b-ppo-provenance-v1',
    copied_from=str(ARM_A.relative_to(HERE.parents[1])),
    changed_keys=sorted(changed),
    unchanged='every other key, including seed, opponent_pool, anchor_banks, holdout_seeds, paces, '
              'target_decisions, lr, parent_kl, retention and checkpoint_every_frames',
), indent=1) + '\n')
print(path)
