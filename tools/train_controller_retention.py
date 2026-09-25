"""Matched 300M continuations: cycling PPO versus mixed-pace PPO with retention.

Both arms use natural controller games, the same per-pace game schedules and
frozen opponent population. Only the revised arm mixes a cycle before updating
and applies independent teacher retention. Outcome targets remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
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
from drmc_rl.training.showiness import apply_bonus, side_summary, validate_spec
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

    ``replay_share`` (default 0): a bank carrying a per-row ``seed`` (the source
    game's arena seed, -1 when unknown) can replay that seed for a mixed pair
    with this probability, so the pills after the preview are the source
    game's own (the row's ``pill_counter`` is the source reserve index). Only
    seeds in the run's training pool (reserve, holdout and anchor seeds
    excluded) are ever replayed; other rows always use the pair's pool seed.
    Replay decisions use a separate random stream, so a bank without seeds
    or ``replay_share`` 0 draws exactly as before.
    """
    def __init__(self, spec, available=None):
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
        self.replay_share=float(spec.get('replay_share',0.))
        if not 0<=self.replay_share<=1:
            raise ValueError('start_mix replay_share must lie in [0, 1]')
        data=np.load(spec['bank'],allow_pickle=False)
        seeds=np.asarray(data['seed'],dtype=np.int64) if 'seed' in data.files else np.full(len(self.bank),-1)
        pool=set(int(s) for s in available) if available is not None else set()
        self.replay_seed=np.where(np.isin(seeds,list(pool)),seeds,-1) if self.replay_share>0 else np.full(len(seeds),-1)
        if self.replay_share>0:
            require_training_seeds([int(s) for s in self.replay_seed if s>0],config=None,what='start_mix replay seeds')
        self.replayable=int((self.replay_seed>0).sum())
        # Optional per-tier row weights (bank ``target_tier`` 1..3), e.g. [1, 3, 6] to favour T2/T3 setups.
        self.row_p=None
        if spec.get('score_weights'):
            # [[bar, weight], ...]: a row's weight is that of the highest bar its target score reaches (0 below all).
            scores=np.asarray(data['target_score'],dtype=np.float64); w=np.zeros(len(scores))
            for bar,weight in sorted(spec['score_weights']): w[scores>=bar]=weight
            self.row_p=w/w.sum()
        elif spec.get('tier_weights'):
            tiers=np.asarray(data['target_tier'],dtype=np.int64)
            w=np.asarray(spec['tier_weights'],dtype=np.float64)[np.clip(tiers-1,0,len(spec['tier_weights'])-1)]
            self.row_p=w/w.sum()

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
        rows=[(int(rng.integers(len(self.bank))) if self.row_p is None else int(rng.choice(len(self.bank),p=self.row_p)))
              if rng.random()<share else None for _ in range(pairs)]
        if all(r is None for r in rows): return None
        replay=np.random.default_rng([config['seed'],cycle,index,0x5F])
        seeds=[int(self.replay_seed[r]) if r is not None and self.replay_seed[r]>0 and replay.random()<self.replay_share
               else None for r in rows]
        return [(r,None if r is None else self.bank.spec_kwargs(r),s) for r,s in zip(rows,seeds) for _ in (0,1)]


class StartMixes:
    """Several start banks at constant shares of the same seed pairs (e.g. showiness setups and stranded edges).

    ``specs``: ``StartMix`` specs, each with a constant ``fraction``; their sum
    must stay below 1 and the remainder is natural play. One uniform per pair
    (a stream disjoint from ``StartMix``) picks the bank, so the shares are
    exact in expectation; each bank keeps its own row weights and seed replay.
    Starts carry the bank index as a fourth element (journaled ``start_bank``).
    """
    def __init__(self, specs, available=None):
        if any('fraction' not in spec for spec in specs):
            raise ValueError('start_mixes support constant fractions only')
        self.mixes=[StartMix(spec,available) for spec in specs]
        if sum(m.share0 for m in self.mixes)>=1:
            raise ValueError('start_mixes fractions must sum below 1')
        self.sha256=hashlib.sha256(''.join(m.sha256 for m in self.mixes).encode()).hexdigest()

    def share(self, frames_since_start):
        return sum(m.share0 for m in self.mixes)

    def starts(self, config, cycle, index, match, pairs, share=None):
        live=[(k,m) for k,m in enumerate(self.mixes)
              if not (m.paces and match['pace'] not in m.paces) and match['level'] in m.levels]
        if not live: return None
        rng=np.random.default_rng([config['seed'],cycle,index,0x60])
        out,any_row=[],False
        for _ in range(pairs):
            u=rng.random(); edge=0.; pick=None
            for k,m in live:
                edge+=m.share0
                if u<edge: pick=(k,m); break
            if pick is None:
                out.extend([(None,None,None,None)]*2); continue
            k,m=pick
            row=int(rng.integers(len(m.bank))) if m.row_p is None else int(rng.choice(len(m.bank),p=m.row_p))
            seed=int(m.replay_seed[row]) if m.replay_seed[row]>0 and rng.random()<m.replay_share else None
            out.extend([(row,m.bank.spec_kwargs(row),seed,k)]*2); any_row=True
        return out if any_row else None


@contextmanager
def update_precision(config):
    """``update_tf32``: TF32 matmul/cuDNN and cudnn.benchmark for the PPO update only.

    Rollout inference stays strict FP32 (the flags are restored on exit). Batch
    shapes are not padded here: fixed-multiple padding of the afterstate conv
    batch belongs to the throughput branch (perf/ppo-throughput).
    """
    if not config.get('update_tf32'):
        yield
        return
    saved=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True
    try:
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)=saved


# Keys a resume may change: numerics of the update step only, never the objective or data.
RESUME_FREE_KEYS=('resume','update_tf32')


# Keys a fork may change relative to the run it branches from.
FORK_KEYS=('resume','fork','start_mix','output','source_commit','seed_reserve',
           'start_mixes','showiness_bonus','checkpoint_every_frames')


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
        if starts is not None:
            # A replayed pair plays its bank row's source seed on both sides.
            jobs=[(seed if s[2] is None else s[2],side,i) for (seed,side,i),s in zip(jobs,starts)]
        result.append((match,jobs,starts))
    return result


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
    actor.defer_reference=bool(config.get('deferred_reference',False))
    parent=PlainPolicy(Path(config['opponent_parent']),device,public_only=True)
    opponents=PublicOpponentPool(config['opponent_pool'],parent,config['opponent_parent'],device)
    retention=PaceRetention(actor,config['anchor_banks'],excluded_seeds=config['holdout_seeds'],
        paces=config['paces'],max_kl_increase=config.get('max_anchor_kl_increase',.03),
        coefficient=config.get('retention_coefficient',.1),batch_size=config.get('retention_batch_size',64),
        pressure_strength=config.get('retention_pressure_strength',0.),hinge=bool(config.get('retention_hinge',False)))
    require_training_seeds(retention.seeds,config=config,what='retention anchor seeds')
    available=training_seed_pool(set(config['holdout_seeds'])|retention.seeds,config=config)
    start_mix=(StartMix(config['start_mix'],available) if config.get('start_mix')
               else StartMixes(config['start_mixes'],available) if config.get('start_mixes') else None)
    bonus=validate_spec(config['showiness_bonus']) if config.get('showiness_bonus') else None
    identities=dict(opponents=opponents.identities(),anchors=retention.identities,
                    initialization=actor.parent_sha256,
                    execution_profiles={p:resolve_pace(p).to_dict() for p in config['paces']})
    if start_mix is not None:
        identities['start_mix_bank']=start_mix.sha256
    from drmc_rl.planning.native_reach import resolve_library_path as reach_path
    from drmc_rl.envs.backends.drmario_pool import resolve_library_path as pool_path
    identities['native_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (reach_path(),pool_path(config.get('native_library')))}
    optimizer=torch.optim.AdamW(actor.net.parameters(),lr=config['lr'],weight_decay=.001)
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
        if ({k:v for k,v in previous.items() if k not in RESUME_FREE_KEYS}
                !={k:v for k,v in config.items() if k not in RESUME_FREE_KEYS}):
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
            records=[]; games=[]; shards=[]; natural=Counter(); breakdown=defaultdict(float); bonus_stats=Counter()
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
                        starts=None if starts is None else [s[1] for s in starts[start:start+chunk]])
                    batch.extend(part); breakdown['rollout_seconds']+=elapsed
                    for k,v in metrics.items(): breakdown[k]+=v
                selected=terminal_samples(batch)
                if bonus is not None:
                    for k,v in apply_bonus(batch,selected,bonus).items(): bonus_stats[k]+=v
                if actor.defer_reference:
                    tick=time.monotonic(); actor.fill_reference_logits(selected)
                    breakdown['reference_seconds']+=time.monotonic()-tick
                for r in selected:
                    r.update(game_id=r['game_id']+offset,pace=match['pace'])
                rows=[dict(r,update=update,pace=match['pace'],level=match['level'],opponent=match['b'],
                           **({} if starts is None or starts[j][0] is None else {'start_row':starts[j][0]}),
                           **({'start_seed_replay':True} if starts is not None and starts[j][2] is not None else {}),
                           **({'start_bank':starts[j][3]} if starts is not None and len(starts[j])>3 and starts[j][3] is not None else {}),
                           **({'showiness':{'learner':side_summary(moves,jobs[j][1]),
                                            'opponent':side_summary(moves,1-jobs[j][1])}}
                              if config.get('journal_showiness') else {}))
                      for j,(r,moves,_) in enumerate(batch)]
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
            kl_lr=config.get('lr_kl_target')
            if kl_lr and 'adaptive_lr' not in progress: progress['adaptive_lr']=config['lr']
            with update_precision(config):
              losses=update_adapter(actor,optimizer,records,dict(config,lr=progress['adaptive_lr']) if kl_lr else config,
                config['seed']+update,activity=activity,
                retention=retention if revised else None,completed_games_by_pace=dict(natural) if revised else None)
            progress['optimizer_steps']+=losses['optimizer_steps']
            if getattr(retention,'hinge',False):
                progress['retention_hinge_active']=retention.pop_hinge_stats()
            progress['gradient']=dict(pre_clip_norm=round(losses.get('gradient_norm',0.),4),
                                      clipped_fraction=round(losses.get('gradient_clipped',0.),4))
            if kl_lr:
                # KL-target step size (needs reset_update_lr): move lr toward the target update KL,
                # at most x1.5 or x0.5 per update; halve on a KL above the alarm level.
                kl=float(losses.get('update_kl',0.)); lr=progress['adaptive_lr']
                factor=0.5 if kl>kl_lr.get('alarm',0.015) else float(np.clip((kl_lr['target']/max(kl,1e-6))**0.5,0.5,1.5))
                progress['adaptive_lr']=float(np.clip(lr*factor,kl_lr.get('min_lr',1e-6),kl_lr.get('max_lr',3e-5)))
                progress['adaptive_lr_trace']=(progress.get('adaptive_lr_trace',[])+[[update,lr,kl]])[-200:]
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
                                start_mix_update_games=mixed,
                                # Share of this update's learner decisions that came from bank games.
                                start_mix_update_decision_fraction=round(
                                    sum(r['a_stats'].get('decisions',0) for r in games if 'start_row' in r)
                                    /max(1,sum(r['a_stats'].get('decisions',0) for r in games)),4),
                                start_mix_update_game_fraction=round(mixed/max(1,len(games)),4),
                                # Per bank (start_mixes index): share of learner decisions and of games.
                                **({'start_mix_update_decision_fraction_by_bank':{str(k):round(
                                        sum(r['a_stats'].get('decisions',0) for r in games if r.get('start_bank')==k)
                                        /max(1,sum(r['a_stats'].get('decisions',0) for r in games)),4)
                                        for k in range(len(start_mix.mixes))}}
                                   if isinstance(start_mix,StartMixes) else {}),
                                start_mix_replay_games=progress.get('start_mix_replay_games',0)
                                    +sum('start_seed_replay' in r for r in games))
            if config.get('journal_showiness'):
                # Learner style this update, natural (non-bank) games only: per 100 placements.
                nat=[r['showiness']['learner'] for r in games if 'start_row' not in r and 'showiness' in r]
                pl=max(1,sum(x['placements'] for x in nat)); cl=max(1,sum(x['clears'] for x in nat))
                progress['style_update']=dict(games=len(nat),placements=pl,
                    **{f'{k}_per_100':round(100*sum(x.get(k,0) for x in nat)/pl,4) for k in ('T1_20','T1','T2','T3','horizontal')},
                    horizontal_share=round(sum(x.get('horizontal',0) for x in nat)/cl,4),
                    best=max([x['best'] for x in nat],default=0.))
            if bonus is not None:
                total=Counter(progress.get('showiness_bonus_totals',{})); total.update(bonus_stats)
                progress.update(showiness_bonus_update=dict(bonus_stats,
                                    fire_per_100_decisions=round(100*bonus_stats['events']/max(1,bonus_stats['decisions']),4)),
                                showiness_bonus_totals=dict(total))
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
            if not config.get('keep_update_checkpoints'):
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
                                                      'start_mix_share','start_mix_update_games','start_mix_update_decision_fraction','start_mix_update_decision_fraction_by_bank','showiness_bonus_update','adaptive_lr','style_update','gradient','retention_hinge_active') if k in progress}),flush=True)
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
