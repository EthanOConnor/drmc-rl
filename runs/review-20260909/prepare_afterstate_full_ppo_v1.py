"""Arm C outcome PPO: arm A's registered config with only the initialization changed.

Copies ``afterstate-core-v1/ppo-ppo-v1.json`` (the mixed-v2 retention recipe,
opponents champion 0.4 / core300m 0.3 / pace_corrected 0.3, every registered
evaluation seed held out, a snapshot every 50M frames, the same seed) and
changes: ``checkpoint`` (the champion plus a zero-output afterstate branch,
tools/build_afterstate_full_init.py), ``output``, ``native_library`` and
``source_commit`` (paths of this run), and adds the new-branch learning-rate
schedule. Every champion tensor keeps arm A's rate (3e-6); the 0.32M new
``afterstate.`` parameters ramp linearly to 10x over the first three updates.
Justification: they are random or zero, not a converged solution to protect,
and at 3e-6 Adam moves the zero projection by at most ~3e-6 per step
(~0.14 over the whole ~47k-step run), so the branch could barely contribute;
the policy-level guards (max update KL, parent KL, retention veto, backtracks)
still bound how far the policy moves regardless of which parameters move it.

``python prepare_afterstate_full_ppo_v1.py`` writes the tf3090 config;
``--smoke DIR`` writes a one-update Mac config with local paths and tiny
per-pace collections (pipeline check only, never evidence).
"""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'afterstate-core-full-v1'
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output/afterstate-core-full-v1'
BRANCH = dict(new_branch_prefix='afterstate.', new_branch_lr_multiplier=10.0, new_branch_warmup_updates=3)
MAIN = Path('/Users/ethan/dev/drmario/drmc-rl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    base = json.loads((HERE / 'afterstate-core-v1/ppo-ppo-v1.json').read_text())
    commit = subprocess.run(['git', '-C', str(HERE.parents[1]), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    config = dict(base, **BRANCH)
    config.update(checkpoint=f'{REMOTE}/init-arm-c.pt', output=f'{REMOTE}/ppo-v1',
                  native_library=f'{REMOTE}/native/libdrmario_pool.so', source_commit=commit)
    if args.smoke is None:
        OUT.mkdir(exist_ok=True)
        path = OUT / 'ppo-full-v1.json'
        assert not path.exists(), f'{path.name} already written'
    else:
        smoke, native = args.smoke.resolve(), MAIN / 'runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
        champion = str(MAIN / 'runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt')
        outcome = str(MAIN / 'runs/trainer-baseline-v1/public-outcome-10m-inference.pt.gz')
        config.update(
            checkpoint=str(smoke.parent / 'init-arm-c.pt'), output=str(smoke / 'out'), device='mps',
            opponent_parent=outcome, native_library=str(native / 'libdrmario_pool.dylib'),
            opponent_pool=[dict(id='champion', weight=0.4, checkpoint=champion),
                           dict(id='pace_corrected', weight=0.6, checkpoint=outcome,
                                adapter_checkpoint=str(smoke / 'adapter-final.pt'))],
            anchor_banks=[str(smoke / 'banks' / Path(p).name) for p in base['anchor_banks']],
            games_per_update=2, games_per_pace={}, rollout_games=2, planner_workers=2, updates=1,
            target_decisions=1, minimum_decisions_per_pace=1, checkpoint_every_frames=1)
        path = smoke / 'smoke.json'
    path.write_text(json.dumps(config, indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    main()
