"""Ablation arenas: which part of human movement costs strength at Fast and up."""
import json, sys
from pathlib import Path
base = json.loads(Path('/Users/ethan/dev/drmario/drmc-rl-movement-data/arena/config-strength-B.json').read_text())
seeds = {m['pace']: m['seeds'] for f in 'ABC' for m in json.loads(
    Path(f'/Users/ethan/dev/drmario/drmc-rl-movement-data/arena/config-strength-{f}.json').read_text())['schedule']}
core = base['variants']['motor']['checkpoint']
H = dict(checkpoint=core, delay=4, movement='human')
variants = {
  'motor': dict(name='motor-limit pace', checkpoint=core, delay=4),
  # R: human reaction only (planner steering, prompt descent, no hesitation)
  'R': dict(H, name='human reaction only', movement_ablation=dict(steering='planner', descent='prompt')),
  # RD: human reaction + human descent, exact planner steering
  'RD': dict(H, name='human reaction+descent, planner steering', movement_ablation=dict(steering='planner')),
  # RS: human reaction + human steering (incl. corrections), prompt descent
  'RS': dict(H, name='human reaction+steering, prompt descent', movement_ablation=dict(descent='prompt')),
  # SD: human steering + descent at the pace's reaction
  'SD': dict(H, name='human steering+descent, pace reaction', movement_ablation=dict(reaction='pace')),
  'H': dict(H, name='human-like movement (full)'),
  'NC': dict(H, name='human, corrections off', movement_ablation=dict(corrections=False)),
  'CTX': dict(H, name='human, core told Normal context', context_pace='normal'),
}
def build(name, paces, entrants, games=128, pairs=64):
    cfg = {k: v for k, v in base.items() if k not in ('variants', 'schedule')}
    out = Path(__file__).parent / name
    cfg.update(output=str(out), working_db=str(out / 'working/arena.sqlite'), pairs=pairs,
               source='drmc-rl trainer/human-movement-training (ablations)',
               variants={k: variants[k] for k in ['motor', *entrants]}, schedule=[])
    for pace in paces:
        for e in entrants:
            cfg['schedule'].append(dict(id=f'{e}-{pace}', a=e, b='motor', games=games, level=14, pace=pace,
                                        phase='Human movement ablation', rating_group='ablation vs motor',
                                        seeds=seeds[pace][:games // 2]))
    Path(f'config-{name}.json').write_text(json.dumps(cfg, indent=1))
if __name__ == '__main__':
    build('smoke', ['fast'], ['R', 'RD', 'RS', 'SD', 'NC', 'CTX'], games=2, pairs=2)
    build('abl-1', ['fast'], ['H', 'R', 'RD', 'RS'])
    build('abl-2', ['fast'], ['SD', 'NC', 'CTX'])
    build('abl-3', ['top_humans'], ['R', 'RD', 'RS', 'SD'])
    import json as _j
    c=_j.loads(Path('config-abl-2.json').read_text())
    c['variants']['H']=variants['H']
    c['schedule'].append(dict(c['schedule'][0], id='H-top_humans', a='H', pace='top_humans', seeds=seeds['top_humans'][:64]))
    Path('config-abl-2.json').write_text(_j.dumps(c, indent=1))
    variants['FS'] = dict(H, name='human, Normal feasible set', planning_pace='normal')
    build('abl-4', ['fast', 'top_humans'], ['FS'])
