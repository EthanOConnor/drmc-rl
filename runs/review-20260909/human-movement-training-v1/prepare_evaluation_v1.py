"""Distributed-arena study config for the pre-registered evaluation (evaluation-plan-v1.json).

    python prepare_evaluation_v1.py CANDIDATE.pt INCUMBENT.pt OUTPUT_DIR STUDY.json [--families primary,...]

Frame Perfect has no human profile, so ``movement: human`` there is exact execution.
Serve with tools.trainer_arena_distributed (frames backend, frozen native19f libraries).
"""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).parent
NATIVE = '/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries'
HUMAN = ['sloth', 'relaxed', 'normal', 'fast', 'top_humans', 'super_human']


def study(candidate, incumbent, output, families):
    plan = json.loads((HERE / 'evaluation-plan-v1.json').read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    variants = {
        'candidate_human': dict(name='Fine-tuned core · human movement', checkpoint=candidate, delay=4, movement='human'),
        'incumbent_human': dict(name='Incumbent core · human movement', checkpoint=incumbent, delay=4, movement='human'),
        'incumbent_motor': dict(name='Incumbent core · motor-limit pace', checkpoint=incumbent, delay=4),
    }
    schedule = []
    def add(id, a, b, pace, games, seeds, group):
        schedule.append(dict(id=id, a=a, b=b, games=games, level=plan['level'], pace=pace,
                             phase='Human movement training v1', rating_group=group, seeds=seeds[:games // 2]))
    if 'primary' in families:
        for pace in HUMAN:
            add(f'P-{pace}', 'candidate_human', 'incumbent_human', pace, 512, plan['primary_seeds'], 'primary')
    if 'frame_perfect_guard' in families:
        add('FP-guard', 'candidate_human', 'incumbent_human', 'frame_perfect', 512, plan['primary_seeds'], 'guard')
    if 'gap' in families:
        for pace in plan['families']['gap']['paces']:
            add(f'G-{pace}', 'candidate_human', 'incumbent_motor', pace, 256, plan['secondary_seeds'], 'gap')
    return {'checkpoint': incumbent, 'device': 'mps', 'native_library': NATIVE + '/libdrmario_pool.dylib',
            'reach_library': NATIVE + '/libdrm_reach_full.dylib', 'native_commit': '19f292c',
            'threads': 1, 'max_game_frames': 120000, 'reactive_compute_frames': 4, 'preparation_compute_frames': 6,
            'memoize': True, 'replay_games': 0, 'watch': False, 'strict_fp32': True, 'rollout_backend': 'frames',
            'planner_workers': 3, 'async_planning': False, 'pairs': 32,
            'model_sha256': {p: sha(p) for p in {candidate, incumbent}},
            'output': str(output), 'working_db': str(Path(output) / 'working/arena.sqlite'),
            'source': 'drmc-rl trainer/human-movement-training evaluation-plan-v1',
            'variants': variants, 'schedule': schedule}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate'); parser.add_argument('incumbent'); parser.add_argument('output')
    parser.add_argument('study', type=Path)
    parser.add_argument('--families', default='primary,frame_perfect_guard,gap')
    args = parser.parse_args()
    args.study.write_text(json.dumps(study(args.candidate, args.incumbent, args.output,
                                           args.families.split(',')), indent=1) + '\n')
