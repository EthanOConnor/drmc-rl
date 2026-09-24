"""Write the human-movement fine-tuning config for a chosen champion core.

    python prepare_human_movement_training_v1.py --core mixed_v2 OUT.json
    python prepare_human_movement_training_v1.py --core afterstate OUT.json

Both variants share the recipe: the retention-trainer mixed arm, every entrant
executing its placements through the human movement generator at the six paces
with a profile, Frame Perfect exact and guarded by its own teacher-retention
bank, and the champion's frozen pre-fine-tune weights as the main opponent.
The afterstate variant needs a source that also contains trainer/afterstate-core.
"""
import argparse, json
from pathlib import Path

BASE = '/home/ethan/.cache/drmc-rl/trainer-output'
REVIEW = BASE + '/review-20260909'
CORES = {
    'mixed_v2': REVIEW + '/controller-retention-mixed-v2/core-final-inference.pt',
    'afterstate': BASE + '/afterstate-core-v1/ppo-v1/core-final-inference.pt',
}
HERE = Path(__file__).parent


def config(core, output, checkpoint=None):
    template = json.loads((HERE / 'controller-retention-mixed-v2-config.json').read_text())
    plan = json.loads((HERE / 'evaluation-plan-v1.json').read_text())
    champion = checkpoint or CORES[core]
    holdout = sorted(set(template['holdout_seeds']) | set(plan['reserved_seeds']))
    pool = [dict(id='champion', weight=0.5, checkpoint=champion),
            dict(id='core300m', weight=0.3, checkpoint=REVIEW + '/controller-core-live-v4/core-f300000000.pt'),
            dict(id='pace_corrected', weight=0.2,
                 checkpoint=BASE + '/public-outcome-v1/public-outcome-10m-inference.pt.gz',
                 adapter_checkpoint=REVIEW + '/training-decision_mean/adapter-final.pt')]
    if core == 'afterstate':
        # Keep the incumbent browser core in the pool beside the new champion.
        pool.insert(1, dict(id='mixed_v2', weight=0.2, checkpoint=CORES['mixed_v2']))
        pool[0]['weight'], pool[2]['weight'] = 0.4, 0.2
    return {**template,
            'checkpoint': champion, 'opponent_pool': pool, 'holdout_seeds': holdout,
            'movement': 'human', 'retention_paces': ['frame_perfect'],
            'anchor_banks': [REVIEW + '/controller-retention-bank-v1/frame_perfect.pt'],
            'seed': 924601, 'lr': 5e-6, 'planner_workers': 3,
            'target_decisions': 1500000, 'minimum_decisions_per_pace': 90000,
            'milestone_decisions': [500000, 1000000],
            'native_library': REVIEW + '/controller-core-4717a03-source/vendor/drmario_native/build/libdrmario_pool.so',
            'source_commit': 'FILLED_AT_LAUNCH', 'output': output}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', choices=sorted(CORES), required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--checkpoint', default=None, help='champion inference export (default per core)')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.output or f'{REVIEW}/human-movement-{args.core}-v1'
    args.out.write_text(json.dumps(config(args.core, out, args.checkpoint), indent=1) + '\n')
