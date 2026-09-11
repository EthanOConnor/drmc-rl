"""Test actual requested-goal controls against same-seed competitive continuations.

An explicitly unadmitted experiment: no motif reward, quality certificate,
training return or automatic product promotion is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.execution.pace import PACES
from drmc_rl.human.target_construction import SCHEMA, TargetConstruction
from drmc_rl.human.target_execution import ConstructionController
from drmc_rl.planning.native_reach import resolve_library_path
from tools.audit_spatial_execution import load_models
from tools.build_expressive_sequences import write_progress
from tools.fit_spatial_expressive import sha256
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import paired_jobs, variant_policy
from tools.vs_head_to_head import PlainPolicy


def summaries(rows):
    output = {}
    for arm in ('controlled','shadow'):
        selected = [r for r in rows if r['arm'] == arm]
        plans = [p for r in selected for p in r['plans']]
        decisions = [d for r in selected for d in r['selections']]
        output[arm] = dict(games=len(selected), plans=len(plans),
            verified_original_payoffs=sum(p['root_goal_observed'] for p in plans),
            multi_placement_payoffs=sum(p['root_goal_observed'] and p['completed_placements'] >= 2 for p in plans),
            anchor_revisions=sum(p['anchor_revisions'] for p in plans),
            terminations=dict(Counter(p['reason'] for p in plans)),
            payoff_lengths=dict(Counter(str(p['completed_placements']) for p in plans if p['root_goal_observed'])),
            controlled_decisions=sum(d['selected_action'] != d['incumbent_action'] for d in decisions),
            decisions=len(decisions))
    return output


def comparisons(rows):
    rng = np.random.default_rng(94027)
    results = []
    for condition in sorted({r['condition'] for r in rows}):
        selected = [r for r in rows if r['condition'] == condition]
        seeds = sorted({r['game']['seed'] for r in selected})
        delta = []
        for seed in seeds:
            arms = []
            for arm in ('controlled','shadow'):
                games = [r for r in selected if r['game']['seed'] == seed and r['arm'] == arm]
                if len(games) != 2 or {r['game']['side'] for r in games} != {0,1}:
                    raise ValueError('complete side-swapped common-seed comparisons required')
                arms.append(np.mean([[r['game']['score'],
                    sum(p['root_goal_observed'] for p in r['plans'])/max(1,len(r['plans'])),
                    sum(p['root_goal_observed'] and p['completed_placements'] >= 2 for p in r['plans'])/max(1,len(r['plans']))]
                    for r in games],axis=0))
            delta.append(arms[0]-arms[1])
        delta = np.asarray(delta)
        draws = delta[rng.integers(len(seeds),size=(20000,len(seeds)))].mean(1)
        results.append(dict(condition=condition, reset_seeds=len(seeds),
            contrasts={name:dict(controlled_minus_shadow=float(delta[:,i].mean()),
                individual_ci95=np.quantile(draws[:,i],[.025,.975]).tolist())
                for i,name in enumerate(('match_score','original_payoff_rate','multi_placement_payoff_rate'))}))
    return results


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(int(config.get('threads',1)))
    started = time.monotonic()
    trained_path = Path(config['trained'])
    trained = json.loads(trained_path.read_text())
    if trained.get('schema') != SCHEMA or trained.get('status') != 'Complete':
        raise ValueError('completed fixed-final target fitting required')
    model_path = trained_path.parent/'target-final.pt'
    if sha256(model_path) != trained['checkpoint_sha256']:
        raise ValueError('trained route model changed')
    models, encoder, study = load_models(trained['config']['study'],config.get('feature_device','cpu'))
    if (study['source_sha256'] != trained['source_sha256']
            or study['competitive_sha256'] != trained['competitive_sha256']
            or study['arms']['stateless']['checkpoint_sha256'] != trained['root_proposer_sha256']):
        raise ValueError('root proposal or training identities changed')
    payload = torch.load(model_path,map_location='cpu',weights_only=False)
    if payload['schema'] != SCHEMA or payload['request_budget'] != 6:
        raise ValueError('unknown target execution contract')
    model = TargetConstruction(payload['feature_dim'],payload['width'])
    model.load_state_dict(payload['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    arena = {**config['arena'],'allow_unadmitted_controller_experiment':True}
    if arena.get('async_planning') or arena.get('replay_games') or arena.get('mixed_core_actor'):
        raise ValueError('unadmitted evaluation requires synchronous non-training controller games')
    report = dict(schema='drmc-target-construction-evaluation-v1',status='Running',config=config,
        trained_sha256=sha256(trained_path), checkpoint_sha256=sha256(model_path),
        root_proposer_sha256=trained['root_proposer_sha256'], native_sha256=sha256(arena['native_library']),
        reach_sha256=sha256(resolve_library_path()), motor_profiles=[p.to_dict() for p in PACES],
        games=0,target_games=2*sum(m['games'] for m in arena['schedule']),censored_games=0,
        simulated_console_frames=0,outcome_training_frames=0,optimizer_updates=0,
        quality_admission=False,product_gates_passed=False,
        scope='Actual proposal-controlled and unchanged competitive games on common side-balanced seeds. Unadmitted diagnostic; no calibrated regret, noninferiority certification or preference claim.')
    rows, planner = [], None
    write_progress(output,report)
    try:
        parent = PlainPolicy(Path(arena['checkpoint']),arena.get('device','cpu'),public_only=True)
        policies = {name:variant_policy(arena,params,parent) for name,params in arena['variants'].items()}
        report['actor_sha256'] = {name:sha256(params.get('checkpoint',arena['checkpoint']))
                                 for name,params in arena['variants'].items()}
        planner = ParallelPlanning(arena.get('planner_workers',2))
        for match_index,match in enumerate(arena['schedule']):
            jobs = paired_jobs(arena,match)
            size = int(arena.get('pairs',16))
            for start in range(0,len(jobs),size):
                batch = jobs[start:start+size]
                # Balance ordering without changing either seed/side allocation.
                arms = ('controlled','shadow') if (match_index+start//size)%2 == 0 else ('shadow','controlled')
                for arm in arms:
                    controller = ConstructionController(model,models['stateless'],encoder,batch,
                        control=arm == 'controlled',budget=6,feature_device=config.get('feature_device','cpu'))
                    report.update(phase='playing',current_condition=match['id'],current_arm=arm)
                    write_progress(output,report)
                    def activity(value):
                        report.update(current_batch=value,elapsed_seconds=time.monotonic()-started)
                        write_progress(output,report)
                    result,seconds = run_event_batch(arena,match,batch,parent,planner,None,
                        policies=policies,controller=controller,activity=activity)
                    controller.close()
                    for pair,(game,moves,_) in enumerate(result):
                        side = 2*pair+game['side']
                        row = dict(condition=match['id'],level=match['level'],pace=match['pace'],arm=arm,game=game,
                            plans=[p for p in controller.plans if p['side'] == side],
                            selections=[d for d in controller.selections if d['side'] == side],
                            transitions=[t for t in controller.transitions if t['side'] == side],
                            observation_counts=dict(controller.states[side]['counters']))
                        rows.append(row)
                        with gzip.open(output/f"{match['id']}-{arm}-{game['index']:04d}.json.gz",'wt') as handle:
                            json.dump(dict(**row,moves=moves),handle)
                    report.update(games=len(rows),arms=summaries(rows),last_batch_seconds=seconds,
                        censored_games=sum(r['game']['reason']=='timeout' for r in rows),
                        simulated_console_frames=sum(r['game']['frames'] for r in rows),
                        elapsed_seconds=time.monotonic()-started)
                    report.pop('current_batch',None)
                    write_progress(output,report)
                    if report['censored_games']:
                        raise ValueError('censored games do not supply outcome comparisons')
        report.update(status='Complete',phase='complete',comparisons=comparisons(rows))
    except BaseException as error:
        report.update(status='Failed',error=str(error))
        raise
    finally:
        if planner is not None:
            planner.close()
        report['elapsed_seconds'] = time.monotonic()-started
        write_progress(output,report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(json.loads(parser.parse_args().config.read_text()))
