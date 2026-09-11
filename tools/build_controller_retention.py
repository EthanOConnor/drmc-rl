"""Collect independent, full-frontier public anchors from the frozen portfolio."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.training.controller_retention import RetentionRecorder, save_anchor_bank, select_game_anchors
from drmc_rl.training.public_league import PublicOpponentPool
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import variant_policy
from tools.vs_head_to_head import PlainPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    torch.set_num_threads(config.get('threads',1))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    device=config.get('device','cuda')
    encoder=PlainPolicy(Path(config['encoder_checkpoint']),device,public_only=True)
    parent=PlainPolicy(Path(config['checkpoint']),device,public_only=True)
    opponents=PublicOpponentPool(config['opponent_pool'],parent,config['checkpoint'],device)
    identities=opponents.identities()
    files={config['encoder_checkpoint'],config['checkpoint']}
    for p in config['references'].values():
        files.update(p[k] for k in ('checkpoint','adapter_checkpoint') if k in p)
    contract=dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  model_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
                  opponent_identities=identities,
                  execution_profiles={p:resolve_pace(p).to_dict() for p in config['paces']})
    from drmc_rl.planning.native_reach import resolve_library_path as reach_path
    from drmc_rl.envs.backends.drmario_pool import resolve_library_path as pool_path
    contract['native_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (reach_path(),pool_path(config.get('native_library')))}
    contract['torch_version']=str(torch.__version__)
    contract_path=output/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('retention recovery changed its frozen contract')
    else: dump(contract_path,contract)
    config.update(replay_games=0,mixed_core_actor=None,reactive_compute_frames=4,
                  preparation_compute_frames=6,variants={'learner':{'delay':4},**{p:{'delay':4} for p in opponents.names}})
    progress=dict(schema='drmc-controller-retention-collection-v1',status='Running',
                  games=0,frames=0,anchor_rows=0,optimizer_updates=0,training_decisions=0,
                  target_games=sum(2*len(config['seeds'][p]) for p in config['paces']),conditions=[])
    def report(**values):
        progress.update(**values,updated_at=datetime.now(UTC).isoformat())
        dump(output/'progress.json',progress)
    planner=ParallelPlanning(config.get('planner_workers',3))
    try:
        for pace in config['paces']:
            seeds=config['seeds'][pace]
            if len(set(seeds))!=len(seeds) or set(seeds)&set(config['holdout_seeds']):
                raise ValueError('anchor seeds overlap evaluation or repeat within a pace')
            reference=variant_policy(config,config['references'][pace],parent)
            totals=Counter(); retained=[]; journal=[]
            pairs=int(config.get('pairs',32))
            if pairs<2 or pairs%2:
                raise ValueError('anchor chunks require paired sides')
            rng=np.random.default_rng(config['seed']+config['paces'].index(pace))
            for start in range(0,len(seeds),pairs//2):
                opponent_id=opponents.choose(rng)
                selected=seeds[start:start+pairs//2]
                shard=output/f'{pace}-{start:05d}.pt'
                metadata=dict(pace=pace,seeds=selected,opponent=opponent_id,contract=contract)
                if shard.exists():
                    saved=torch.load(shard,map_location='cpu',weights_only=True)
                    if saved['metadata']!=metadata:
                        raise ValueError('retention shard differs from fixed allocation')
                    rows=saved['records']
                    games=json.loads(shard.with_suffix('.json').read_text())
                else:
                    match=dict(id=f'anchor-{pace}-{start}',a='learner',b=opponent_id,pace=pace,
                               level=14,games=2*len(selected))
                    jobs=[(int(seed),side,2*(start+i)+side) for i,seed in enumerate(selected) for side in (0,1)]
                    report(current_pace=pace,phase='collecting',current_start=start)
                    batch,_=run_event_batch(config,match,jobs,None,planner,None,
                        policies={'learner':reference,opponent_id:opponents.load(opponent_id)},
                        anchor_recorder=RetentionRecorder(encoder,planner.planner,
                                                         [2*i+side for i,(_,side,_) in enumerate(jobs)]),
                        activity=lambda w: report(activity=w))
                    if any(g['reason']=='timeout' for g,_,_ in batch):
                        raise ValueError('anchor allocation contains a censored game')
                    rows=select_game_anchors(batch,pace=pace,seed=config['seed']+start,
                                             rows_per_game=config.get('rows_per_game',4))
                    games=[g for g,_,_ in batch]
                    # Sidecar precedes the atomic shard: a completed shard always
                    # has its journal; an interrupted sidecar is safely replaced.
                    dump(shard.with_suffix('.json'),games)
                    save_anchor_bank(shard,rows,metadata)
                retained.extend(rows); journal.extend(games)
                totals.update(games=len(games),frames=sum(g['frames'] for g in games),anchor_rows=len(rows))
                report(**{k:progress[k]+v for k,v in dict(games=len(games),frames=sum(g['frames'] for g in games),anchor_rows=len(rows)).items()})
            independent={int(r['game_seed']) for r in retained}
            if len(independent)<config.get('minimum_seeds_per_pace',128):
                raise ValueError('too few independent playable anchor seeds')
            save_anchor_bank(output/f'{pace}.pt',retained,dict(pace=pace,contract=contract))
            report(conditions=progress['conditions']+[dict(pace=pace,independent_reset_seeds=len(independent),**totals)])
        report(status='Complete',phase='complete',activity=None)
    except BaseException as error:
        report(status='Failed',error=str(error)); raise
    finally:
        planner.close()


if __name__=='__main__':
    main()
