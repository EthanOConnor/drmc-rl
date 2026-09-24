"""CPU cost of scoring nine after-next previews versus one, with and without reusing the opponent bottle.

For each sampled real spawn decision (retention mixed-v2 core, frame_perfect pace) this times
  one:      batch 1 (the spawn-time contract)
  nine:     batch 9, one row per possible preview (marginal / branches)
  nine_reuse: batch 9 with the opponent bottle encoded once and passed as prepared_bottles
and checks that reusing the opponent encoding is exact (it is only if the opponent trunk
condition does not depend on the own preview). Wall times are medians over repeats.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.backend import plan_candidates
from drmc_rl.human.controller_context import controller_policy_inputs
from drmc_rl.human.early_decision import PREVIEWS, with_own_preview
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.vs_head_to_head import PlainPolicy

CORE = '/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt'


class Cached:
    """A prepared bottle: fixed features broadcast to the batch."""
    def __init__(self, features):
        self.features = features

    def resolve(self, model, obs):
        return self.features.expand(len(obs), *self.features.shape[1:])


def sample_states(count, seed):
    pace, planner, states = resolve_pace('frame_perfect'), NativeReachabilityRunner(), []
    with FrameVsPool(1, lib_path=os.environ['DRMC_FRAME_LIBRARY']) as pool:
        pool.reset([seed], level=14)
        was = False
        while len(states) < count and not pool.states[0].terminal:
            s = pool.states[0]
            if s.falling and not was:
                state = pool.semantic(0, public_context=True)
                states.append((state, plan_candidates(planner, state, 4, pace)))
            was = s.falling
            pool.step([0x04 if len(states) % 2 else 0, 0])
    planner.close()
    return pace, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--positions', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    policy = PlainPolicy(Path(CORE), 'cpu', public_only=True)
    net = policy.net.eval()
    captured = []
    hook = net.bottle_projection.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    pace, states = sample_states(args.positions, 4242)
    records = []
    for state, candidate in states:
        rows = [controller_policy_inputs(policy, candidate, state, pace, 4, 4,
                                         public=with_own_preview(state['public_pair_state'], p)) for p in PREVIEWS]
        obs9 = np.concatenate([r[0] for r in rows])
        infos9 = [r[1][0] for r in rows]
        in1, aux1, _, _ = policy.model_inputs(obs9[:1], infos9[:1])
        in9, aux9, _, _ = policy.model_inputs(obs9, infos9)
        with torch.inference_mode():
            captured.clear()
            net(*in9, aux=aux9)
            own9, opp9 = captured[-2], captured[-1]
            captured.clear()
            net(*in1, aux=aux1)
            opponent = captured[-1]
            exact = float((opp9 - opponent.expand_as(opp9)).abs().max())
            own_spread = float((own9 - own9[:1]).abs().max())
            full = net(*in9, aux=aux9)[0]
            reused = net(*in9, aux=aux9, prepared_bottles=(None, Cached(opponent)))[0]
            output_diff = float((full - reused).abs().max())

            def timed(call):
                values = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    call()
                    values.append(1000 * (time.perf_counter() - start))
                return float(np.median(values))
            hook.remove()
            for _ in range(3):
                net(*in1, aux=aux1), net(*in9, aux=aux9)
            times = dict(one_ms=timed(lambda: net(*in1, aux=aux1)),
                         nine_ms=timed(lambda: net(*in9, aux=aux9)),
                         nine_reuse_ms=timed(lambda: net(*in9, aux=aux9, prepared_bottles=(None, Cached(opponent)))),
                         opponent_encode_ms=timed(lambda: net.bottle_projection(net.bottle(in1[0][:, 8:16], torch.zeros(1, net.d_model)))))
            hook = net.bottle_projection.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
        records.append(dict(legal=int(np.count_nonzero(infos9[0]['placements/feasible_mask'])),
                            opponent_reuse_max_abs_diff=exact, own_bottle_preview_spread=own_spread,
                            logits_max_abs_diff_with_reuse=output_diff, **times))
    summary = {k: float(np.median([r[k] for r in records])) for k in records[0] if k.endswith('_ms')}
    report = dict(schema='drmc-preview-branch-cost-v1', checkpoint=CORE, threads=args.threads,
                  host=platform.node(), machine=platform.machine(), load_average=os.getloadavg(),
                  positions=len(records), repeats=args.repeats, median_ms=summary,
                  opponent_reuse_exact=max(r['opponent_reuse_max_abs_diff'] for r in records) == 0.0,
                  max_opponent_reuse_diff=max(r['opponent_reuse_max_abs_diff'] for r in records),
                  max_logit_diff_with_reuse=max(r['logits_max_abs_diff_with_reuse'] for r in records),
                  note='opponent_encode_ms uses a zero condition and is only a rough cost of one bottle trunk pass',
                  records=records)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=1))


if __name__ == '__main__':
    main()
