"""Diagnose lock_safe: public history behind garbage mismatches and a sample of each unsafe reason."""
from collections import Counter
import json
from pathlib import Path
import sys

import tools.trainer_planning_arena as arena
from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.human import early_decision as ed
from tools.vs_head_to_head import PlainPolicy

config = json.load(open(sys.argv[1]))
match = config['schedule'][0]
reasons, dumps, samples = Counter(), [], {}
original_resolve, original_on_lock = ed.EarlyRequests.resolve, ed.EarlyRequests.on_lock


def events(public):
    return [(e.kind.value, e.frame_id, e.side, dict(e.public_payload)) for e in public.recent_events[-12:]]


def resolve(self, side, point, frame, current, opponent, stats):
    request = self.request[side]
    result = original_resolve(self, side, point, frame, current, opponent, stats)
    if isinstance(request, dict) and result is None and int(opponent.garbage_sent_total) != request['incoming']:
        dumps.append(dict(request_frame=request['frame'], kind=request['kind'], spawn_frame=frame,
                          viewer=request['public'].viewer_side, events=events(request['public'])))
    return result


def on_lock(self, side, point, frame, current, opponent, pool, stats):
    if point in ('lock_safe', 'commit_safe') and not isinstance(self.request[side], dict):
        public = pool.public_state(side)
        risk = ed.garbage_risk(public, own_clears=bool(resolve_cascade(bytes(current.board)).steps))
        opp = public.sides[1 - public.viewer_side]
        reasons[f'{risk}:{opp.animation_phase}'] += 1
        if risk and risk not in samples:
            board = opp.board
            samples[risk] = dict(frame=frame, viewer=public.viewer_side, opponent_phase=opp.animation_phase,
                                 native_phase=(opponent.phase, opponent.subphase),
                                 predicted_cells=resolve_cascade(board).cells_cleared, events=events(public),
                                 board=[' '.join(f'{board[r*8+c]:02x}' for c in range(8)) for r in range(16)])
    return original_on_lock(self, side, point, frame, current, opponent, pool, stats)


ed.EarlyRequests.resolve, ed.EarlyRequests.on_lock = resolve, on_lock
policy = PlainPolicy(Path(config['variants']['base']['checkpoint']), config['device'], public_only=True)
planner = arena.NativeReachabilityRunner()
arena.run_batch(config, match, arena.paired_jobs(config, match)[:int(sys.argv[2])], policy, planner, None)
print(json.dumps(dict(reasons=reasons, mismatches=len(dumps)), indent=1))
for d in dumps[:4]:
    print(json.dumps(d))
for k, v in samples.items():
    print(k, json.dumps(v, indent=0))
