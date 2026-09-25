"""Matched 300M continuations: cycling PPO versus mixed-pace PPO with retention.

Both arms use natural controller games, the same per-pace game schedules and
frozen opponent population. Only the revised arm mixes a cycle before updating
and applies independent teacher retention. Outcome targets remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.models.policy.controller_core import ControllerCorePolicy, write_public_replay
from drmc_rl.program.seed_reserve import require_training_seeds, training_seed_pool
from drmc_rl.training.controller_retention import PaceRetention
from drmc_rl.training.episodic_objective import objective_contract
from drmc_rl.training.public_league import PublicOpponentPool
from tools.train_pace_strategy import (
    TrainingActivity, add_game_totals, restore_game_journal, terminal_samples, update_adapter,
)
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.vs_head_to_head import PlainPolicy


class StartMix:
    """Replace a share of each collection's seed pairs with start-bank positions.

    The mix draws from its own random stream, so the natural seeds, opponents and
    levels of a collection are exactly those of the same run without a mix. A
    mixed pair keeps its training-pool seed (it supplies the pill stream after
    the bank position's preview), plays both side assignments from the bank
    row, and is journaled with ``start_row``. Outcomes stay natural terminal
    results from that position; no scenario-specific reward is added.

    The share is either constant (``fraction``) or decays with the frames the
    run has simulated since the mix began: ``share0 * 0.5 ** (t / half_life_frames)``,
    never below ``floor``, and exactly 0 once it falls under ``cutoff``
    (default 0.01) with a zero floor, leaving only natural occurrences.
    """
    def __init__(self, spec):
        from drmc_rl.training.envs.start_bank import StartBank
        self.spec=dict(spec); self.bank=StartBank(spec['bank'])
        if ('fraction' in spec)==('share0' in spec):
            raise ValueError('start_mix needs exactly one of fraction or share0')
        self.share0=float(spec.get('fraction',spec.get('share0')))
        self.half_life=spec.get('half_life_frames')
        self.floor=float(spec.get('floor',0.)); self.cutoff=float(spec.get('cutoff',.01))
        if 'share0' in spec and not (self.half_life and self.half_life>0):
            raise ValueError('a decaying start_mix needs a positive half_life_frames')
        if not (0<self.share0<1 and 0<=self.floor<=self.share0 and 0<=self.cutoff<1):
            raise ValueError('start_mix shares must satisfy 0 <= floor <= share0 < 1')
        self.paces=set(spec.get('paces') or []); self.levels=set(spec.get('levels',[14]))
        self.sha256=hashlib.sha256(Path(spec['bank']).read_bytes()).hexdigest()
        if spec.get('bank_sha256') not in (None,self.sha256):
            raise ValueError('start_mix bank bytes differ from the declared bank_sha256')

    def share(self, frames_since_start):
        if self.half_life is None:
            return self.share0
        value=self.share0*0.5**(max(0,frames_since_start)/self.half_life)
        if value<self.cutoff and self.floor==0.:
            return 0.
        return max(self.floor,value)

    def starts(self, config, cycle, index, match, pairs, share=None):
        share=self.share0 if share is None else share
        if share<=0 or (self.paces and match['pace'] not in self.paces) or match['level'] not in self.levels:
            return None
        rng=np.random.default_rng([config['seed'],cycle,index,0x5E])
        rows=[int(rng.integers(len(self.bank))) if rng.random()<share else None for _ in range(pairs)]
        if all(r is None for r in rows): return None
        return [(r,None if r is None else self.bank.spec_kwargs(r)) for r in rows for _ in (0,1)]


# Keys a fork may change relative to the run it branches from.
FORK_KEYS=('resume','fork','start_mix','output','source_commit','seed_reserve')


def fork_contract(config):
    return {k:v for k,v in config.items() if k not in FORK_KEYS}


def collection_schedule(config, update, available, opponents, start_mix=None, share=None):
    paces=config['paces']
    if config['arm']=='mixed_retention':
        cycle=update-1; selected=list(enumerate(paces))
    elif config['arm']=='cycling_control':
        cycle,index=divmod(update-1,len(paces)); selected=[(index,paces[index])]
    else:
        raise ValueError('unknown continuation arm')
    result=[]
    for index,pace in selected:
        rng=np.random.default_rng(config['seed']+cycle*len(paces)+index)
        opponent=opponents.choose(rng)
        level=20 if pace!='sloth' and rng.random()<config.get('level20_fraction',.15) else 14
        count=config['games_per_pace'].get(pace,config['games_per_update'])
        if count<2 or count%2:
            raise ValueError('collections require complete seed pairs')
        seeds=rng.choice(available,count//2,replace=False)
        jobs=[(int(seed),side,2*i+side) for i,seed in enumerate(seeds) for side in (0,1)]
        match=dict(id=f'train-{update}-{pace}',a='learner',b=opponent,games=count,pace=pace,level=level)
        starts=None if start_mix is None else start_mix.starts(config,cycle,index,match,len(seeds),share)
        result.append((match,jobs,starts))
    return result


def optimizer_groups(net,config):
    """One group, or trunk plus a separately scheduled newly initialized branch."""
    prefix=config.get('new_branch_prefix')
    if not prefix: return [dict(params=list(net.parameters()))]
    named=list(net.named_parameters())
    branch=[p for n,p in named if n.startswith(prefix)]
    if not branch: raise ValueError(f'no parameters under the new branch prefix {prefix!r}')
    return [dict(params=[p for n,p in named if not n.startswith(prefix)],name='trunk'),
            dict(params=branch,name='new_branch')]


def set_learning_rates(optimizer,config,update):
    """Per-update base rates: the branch ramps linearly to its multiplier, never below the trunk."""
    for group in optimizer.param_groups:
        rate=config['lr']
        if group.get('name')=='new_branch':
            ramp=min(1.,update/max(1,config.get('new_branch_warmup_updates',1)))
            rate*=max(1.,config.get('new_branch_lr_multiplier',1.)*ramp)
        group['update_lr']=group['lr']=rate


def target_met(progress,config):
    return (progress['decisions']>=config['target_decisions'] and
            all(progress['paces'].get(p,{}).get('learning_decisions',0)>=config['minimum_decisions_per_pace']
                for p in config['paces']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    config['objective']=objective_contract(config)
    if config['arm'] not in ('cycling_control','mixed_retention'):
        raise ValueError('unknown continuation arm')
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    if (output/'training.json').exists() and not config.get('resume'):
        raise ValueError('new continuations require a fresh output')
    torch.set_num_threads(config.get('threads',1)); torch.set_num_interop_threads(1)
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=config.get('device','cuda')
    fork=config.get('fork')
    if fork and config.get('resume') is None and fork.get('sha256') != hashlib.sha256(Path(fork['checkpoint']).read_bytes()).hexdigest():
        raise ValueError('fork checkpoint bytes differ from the declared sha256')
    actor=ControllerCorePolicy(config['checkpoint'],device,seed=config['seed'],
                               resume=config.get('resume') or (fork['checkpoint'] if fork else None))
    parent=PlainPolicy(Path(config['opponent_parent']),device,public_only=True)
    opponents=PublicOpponentPool(config['opponent_pool'],parent,config['opponent_parent'],device)
    retention=PaceRetention(actor,config['anchor_banks'],excluded_seeds=config['holdout_seeds'],
        paces=config['paces'],max_kl_increase=config.get('max_anchor_kl_increase',.03),
        coefficient=config.get('retention_coefficient',.1),batch_size=config.get('retention_batch_size',64),
        pressure_strength=config.get('retention_pressure_strength',0.))
    require_training_seeds(retention.seeds,config=config,what='retention anchor seeds')
    available=training_seed_pool(set(config['holdout_seeds'])|retention.seeds,config=config)
    start_mix=StartMix(config['start_mix']) if config.get('start_mix') else None
    identities=dict(opponents=opponents.identities(),anchors=retention.identities,
                    initialization=actor.parent_sha256,
                    execution_profiles={p:resolve_pace(p).to_dict() for p in config['paces']})
    if start_mix is not None:
        identities['start_mix_bank']=start_mix.sha256
    from drmc_rl.planning.native_reach import resolve_library_path as reach_path
    from drmc_rl.envs.backends.drmario_pool import resolve_library_path as pool_path
    identities['native_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (reach_path(),pool_path(config.get('native_library')))}
    optimizer=torch.optim.AdamW(optimizer_groups(actor.net,config),lr=config['lr'],weight_decay=.001)
    progress=dict(status='Running',schema='drmc-controller-retention-training-v1',arm=config['arm'],
        updates=0,games=0,frames=0,decisions=0,paces={},checkpoints=[],
        optimizer_steps=0,consecutive_stalled_updates=0,
        target_decisions=config['target_decisions'],minimum_decisions_per_pace=config['minimum_decisions_per_pace'],
        target_updates=config['updates'],objective=config['objective'],identities=identities,
        retention_baseline=retention.baseline,retention_kl_by_pace=retention.baseline,
        anchor_rows=len(retention.records),anchor_reset_seeds=len(retention.seeds))
    if config.get('resume'):
        old=torch.load(config['resume'],map_location=device,weights_only=True)
        previous=old['training_config']
        if {k:v for k,v in previous.items() if k!='resume'}!={k:v for k,v in config.items() if k!='resume'}:
            raise ValueError('resume changed the continuation contract')
        if old['progress']['identities']!=identities:
            raise ValueError('resume changed model, opponent, anchor or motor bytes')
        progress.update(old['progress']); progress.update(status='Running')
        retention.baseline=progress['retention_baseline']
        optimizer.load_state_dict(old['optimizer']); actor.rng.set_state(old['sampling_rng'].cpu())
        restore_game_journal(output/'training-games.jsonl',old['update'])
        if not (output/'training-games.jsonl').exists():
            raise ValueError('resume requires its committed game journal')
        totals={}
        for line in (output/'training-games.jsonl').read_text().splitlines():
            row=json.loads(line); add_game_totals(totals.setdefault(row['pace'],{}),row)
        if totals!=progress['paces']:
            raise ValueError('resume journal and per-pace checkpoint counts disagree')
    elif fork:
        # Branch a variant off a running arm: identical model, optimizer, sampler,
        # retention baseline, counters and journal at the fork update; only the
        # FORK_KEYS may differ, so continued arm and variant share one history.
        old=torch.load(fork['checkpoint'],map_location=device,weights_only=True)
        if fork_contract(old['training_config'])!=fork_contract(config):
            raise ValueError('fork changed the parent run contract beyond start_mix/output/source')
        expected={k:v for k,v in identities.items() if k!='start_mix_bank'}
        if {k:v for k,v in old['progress']['identities'].items() if k!='start_mix_bank'}!=expected:
            raise ValueError('fork changed model, opponent, anchor or motor bytes')
        progress.update(old['progress']); progress.update(status='Running',checkpoints=[],identities=identities,
            fork=dict(checkpoint=fork['checkpoint'],sha256=fork['sha256'],update=old['update'],
                      frames=old['progress']['frames'],decisions=old['progress']['decisions']))
        retention.baseline=progress['retention_baseline']
        optimizer.load_state_dict(old['optimizer']); actor.rng.set_state(old['sampling_rng'].cpu())
        journal=output/'training-games.jsonl'
        shutil.copyfile(fork['journal'],journal); restore_game_journal(journal,old['update'])
        totals={}
        for line in journal.read_text().splitlines():
            row=json.loads(line); add_game_totals(totals.setdefault(row['pace'],{}),row)
        if totals!=progress['paces']:
            raise ValueError('fork journal and parent per-pace counts disagree')
        dump(output/'config.json',config)
    else:
        actor.save(output/'core-initial.pt',update=0,training_config=config)
        dump(output/'config.json',config)
    if start_mix is not None and 'start_mix_origin_frames' not in progress:
        progress['start_mix_origin_frames']=progress['frames']
    runtime=dict(config,variants={id:{'delay':4} for id in ('learner',*opponents.names)},
                 replay_games=0,mixed_core_actor=None,reactive_compute_frames=4,preparation_compute_frames=6)
    planner=ParallelPlanning(config.get('planner_workers',4))
    activity=TrainingActivity(output/'training.json',progress)
    begun=time.monotonic()
    try:
        for update in range(progress['updates']+1,config['updates']+1):
            if target_met(progress,config): break
            share=None if start_mix is None else start_mix.share(progress['frames']-progress['start_mix_origin_frames'])
            started=time.monotonic(); schedules=collection_schedule(config,update,available,opponents,start_mix,share)
            records=[]; games=[]; shards=[]; natural=Counter(); breakdown=defaultdict(float)
            progress.update(current_pace='mixed' if len(schedules)>1 else schedules[0][0]['pace'],
                            collecting_update=update,collecting_target=sum(m['games'] for m,_,_ in schedules),
                            collecting_games=0,phase='collecting',activity=None,updated_at=datetime.now(UTC).isoformat())
            dump(output/'training.json',progress)
            for match,jobs,starts in schedules:
                opponent=opponents.load(match['b']); batch=[]; offset=len(games)
                chunk=config.get('rollout_games',32)
                if chunk<2 or chunk%2: raise ValueError('rollout chunks require paired sides')
                for start in range(0,len(jobs),chunk):
                    before_games=len(games)+len(batch)
                    before_frames=sum(r['frames'] for r in games)+sum(r['frames'] for r,_,_ in batch)
                    before_requests=sum(r['a_stats'].get('decisions',0) for r in games)+sum(r['a_stats'].get('decisions',0) for r,_,_ in batch)
                    def collecting(w):
                        activity('collecting',games=before_games+w['games'],target=progress['collecting_target'],
                                 frames=before_frames+w['frames'],decision_requests=before_requests+w['decision_requests'])
                    metrics={}
                    part,elapsed=run_event_batch(runtime,match,jobs[start:start+chunk],None,planner,None,
                        policies={'learner':actor,match['b']:opponent},activity=collecting,metrics=metrics,
                        starts=None if starts is None else [k for _,k in starts[start:start+chunk]])
                    batch.extend(part); breakdown['rollout_seconds']+=elapsed
                    for k,v in metrics.items(): breakdown[k]+=v
                selected=terminal_samples(batch)
                for r in selected:
                    r.update(game_id=r['game_id']+offset,pace=match['pace'])
                rows=[dict(r,update=update,pace=match['pace'],level=match['level'],opponent=match['b'],
                           **({} if starts is None or starts[j][0] is None else {'start_row':starts[j][0]}))
                      for j,(r,_,_) in enumerate(batch)]
                natural[match['pace']]+=sum(r['reason']!='timeout' for r in rows)
                records.extend(selected); games.extend(rows); shards.append((match,selected))
                progress.update(collecting_games=len(games),collecting_frames=sum(r['frames'] for r in games),collecting_decisions=len(records))
            activity('saving_replay',decisions=len(records))
            for match,selected in shards:
                if selected:
                    write_public_replay(output/'public-replay'/f"update-{update:05d}-{match['pace']}.npz",
                        selected,games,update=update,pace=match['pace'],level=match['level'])
            tick=time.monotonic()
            revised=config['arm']=='mixed_retention'
            set_learning_rates(optimizer,config,update)
            losses=update_adapter(actor,optimizer,records,config,config['seed']+update,activity=activity,
                retention=retention if revised else None,completed_games_by_pace=dict(natural) if revised else None)
            progress['optimizer_steps']+=losses['optimizer_steps']
            progress['consecutive_stalled_updates']=(progress['consecutive_stalled_updates']+1
                if losses['optimizer_steps']==0 else 0)
            breakdown['optimizer_seconds']=time.monotonic()-tick
            # Control diagnostics use the same bank, with no gradients or veto.
            if revised or update%len(config['paces'])==0:
                progress['retention_kl_by_pace']=losses.get('retention_kl_by_pace') if revised else retention.measure()
            activity('saving_checkpoint',update=update)
            with (output/'training-games.jsonl').open('a') as journal:
                for row in games:
                    add_game_totals(progress['paces'].setdefault(row['pace'],{}),row)
                    journal.write(json.dumps(row)+'\n')
            if start_mix is not None:
                mixed=sum('start_row' in r for r in games)
                progress.update(start_mix_share=share,start_mix_games=progress.get('start_mix_games',0)+mixed,
                                start_mix_update_games=mixed)
            elapsed=time.monotonic()-started
            progress.update(updates=update,games=progress['games']+len(games),
                frames=progress['frames']+sum(r['frames'] for r in games),decisions=progress['decisions']+len(records),
                losses=losses,batch_seconds=elapsed,wall_seconds=time.monotonic()-begun,
                throughput=dict(learning_decisions_per_second=len(records)/elapsed,
                    frames_per_second=sum(r['frames'] for r in games)/elapsed,breakdown=dict(breakdown)),
                updated_at=datetime.now(UTC).isoformat())
            if sum(v.get('learning_decisions',0) for v in progress['paces'].values())!=progress['decisions']:
                raise ValueError('journal-derived learning count differs from actual PPO records')
            path=output/f'core-u{update:05d}.pt'; progress['checkpoints'].append(path.name)
            actor.save(path,update=update,optimizer=optimizer.state_dict(),sampling_rng=actor.rng.get_state(),
                       progress=progress,training_config=config)
            for name in progress['checkpoints'][:-2]: (output/name).unlink(missing_ok=True)
            progress['checkpoints']=progress['checkpoints'][-2:]
            every=config.get('checkpoint_every_frames')
            if every and progress['frames']>=every and progress['frames']//every*every>progress.get('fork',{}).get('frames',-1):
                # Stop-rule snapshots: one per crossed frame multiple (an update is far shorter).
                path=output/f"core-f{progress['frames']//every*every:011d}.pt"
                if not path.exists():
                    actor.save(path,update=update,progress=progress,training_config=config)
            for milestone in config.get('milestone_decisions',[]):
                path=output/f'core-d{milestone:09d}.pt'
                if progress['decisions']>=milestone and not path.exists():
                    actor.save(path,update=update,progress=progress,training_config=config)
            progress.update(phase='between_updates',activity=None)
            dump(output/'training.json',progress)
            print(json.dumps({k:progress[k] for k in ('updates','games','frames','decisions','current_pace','losses','throughput',
                                                      'start_mix_share','start_mix_update_games') if k in progress}),flush=True)
            if progress['consecutive_stalled_updates']>=config.get('max_stalled_updates',7):
                raise RuntimeError('seven consecutive updates accepted no optimizer steps; inspect retention and KL before spending more rollout compute')
            if losses['effective_learning_rate'] < config.get('minimum_learning_rate',0.):
                raise RuntimeError('learning rate fell below the declared useful floor; preserve the checkpoint and review retention')
            del records,games,batch,shards
        if not target_met(progress,config): raise RuntimeError('safety update cap reached before learning allocation')
        progress.update(status='Training complete',final_checkpoint='core-final.pt',updated_at=datetime.now(UTC).isoformat())
        actor.save(output/'core-final.pt',update=progress['updates'],optimizer=optimizer.state_dict(),
                   sampling_rng=actor.rng.get_state(),progress=progress,training_config=config)
        actor.save(output/'core-final-inference.pt',update=progress['updates'],progress=progress,training_config=config)
    except BaseException as error:
        progress.update(status='Failed',error=str(error),updated_at=datetime.now(UTC).isoformat()); raise
    finally:
        dump(output/'training.json',progress); planner.close()


if __name__=='__main__': main()
