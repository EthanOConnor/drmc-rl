"""Fixed persistent/stateless comparison with predicted spatial construction goals."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.expressive_sequences import SCHEMA as SOURCE_SCHEMA, locked_field
from drmc_rl.human.spatial_proposer import (
    COLOR_MAP, SCHEMA, FrozenConstructionEncoder, SpatialProposer, spatial_clear_target,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.build_expressive_sequences import write_progress
from tools.fit_expressive_proposer import session_validation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(config, output, report):
    from tools.eval_policy import _build_net_from_cfg

    with np.load(config['source'], allow_pickle=False) as archive:
        data = {k:archive[k] for k in ('board','pill','preview','action','windows','sessions','metadata')}
    if json.loads(str(data['metadata']))['schema'] != SOURCE_SCHEMA:
        raise ValueError('verified construction source required')
    payload = load_checkpoint(config['checkpoint'], map_location='cpu')
    cfg = dict(payload['cfg'])
    sp = cfg.get('smdp_ppo', cfg)
    feature_device = config.get('feature_device', 'cpu')
    core, _, _ = _build_net_from_cfg(cfg, int(sp['candidate_board_channels'])+4, feature_device)
    core.load_state_dict(payload.get('ema_state_dict') or payload['state_dict'], strict=True)
    encoder = FrozenConstructionEncoder(core).to(feature_device).eval()
    # Deduplicate by actual public inputs only, never action or future payoff.
    packed = np.concatenate((data['board'].reshape(-1,128), data['pill'], data['preview']), -1)
    keys = np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, selected, data['feature_index'] = np.unique(keys, return_index=True, return_inverse=True)
    boards = data['board'][selected]
    data['planes'] = np.stack([board_bytes_to_semantic_planes(b) for b in boards])
    data['canonical_pill'] = COLOR_MAP[data['pill'][selected]]
    data['canonical_preview'] = COLOR_MAP[data['preview'][selected]]
    data['features'] = np.empty((len(selected), encoder.feature_dim), np.float32)
    batch_size = int(config.get('feature_batch', 32))
    if batch_size < 1:
        raise ValueError('positive feature batch required')
    report.update(phase='shared_features', unique_public_inputs=len(selected), feature_rows=0)
    write_progress(output, report)
    last_write = time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(selected),batch_size):
            stop = min(start+batch_size,len(selected))
            inputs = [torch.as_tensor(data[k][start:stop], device=feature_device)
                      for k in ('planes','canonical_pill','canonical_preview')]
            data['features'][start:stop] = encoder(*inputs).cpu().numpy()
            if not np.isfinite(data['features'][start:stop]).all():
                raise ValueError('nonfinite competitive features')
            report['feature_rows'] = stop
            if time.monotonic()-last_write >= 5:
                write_progress(output, report)
                last_write = time.monotonic()
    del encoder, core, payload
    if feature_device == 'mps':
        torch.mps.empty_cache()
    report.update(phase='spatial_targets', targets=0)
    write_progress(output, report)
    data['spatial_targets'] = np.empty((len(data['windows']),384),np.float32)
    target_cache = {}
    for i,(start,length,goal,_) in enumerate(data['windows']):
        end = int(start+length-1)
        key = (int(data['feature_index'][end]),int(data['action'][end]),int(goal))
        if key not in target_cache:
            result = resolve_cascade(locked_field(data['board'][end],data['pill'][end],data['action'][end]))
            target_cache[key] = spatial_clear_target(result,int(goal)).ravel()
        data['spatial_targets'][i] = target_cache[key]
        if time.monotonic()-last_write >= 5:
            report['targets'] = i+1
            write_progress(output,report)
            last_write = time.monotonic()
    report.update(targets=len(data['windows']), unique_payoffs=len(target_cache))
    np.savez_compressed(output/'prepared.npz', **{k:(v.astype(np.uint8) if k=='planes' else v)
                        for k,v in data.items() if k not in ('board','metadata')})
    report['prepared_sha256'] = sha256(output/'prepared.npz')
    write_progress(output,report)
    return data


def batch(data, ids, device, persistent):
    memory, current, goal, horizon, elapsed, owner, window, remaining = [],[],[],[],[],[],[],[]
    for slot,i in enumerate(ids):
        start,length,g,_ = map(int,data['windows'][i])
        for step in range(length):
            memory.append(start if persistent else start+step)
            current.append(start+step)
            goal.append(g)
            horizon.append(length-1 if persistent else length-step-1)
            elapsed.append(step if persistent else 0)
            remaining.append(length-step)
            owner.append(slot)
            window.append(i)
    def inputs(rows):
        features = data['feature_index'][rows]
        return tuple(torch.as_tensor(data[k][features],device=device,
                     dtype=torch.float32 if k in ('planes','features') else torch.long) for k in
                     ('planes','features','canonical_pill','canonical_preview'))
    integers = [torch.as_tensor(x,dtype=torch.long,device=device) for x in
                (goal,horizon,elapsed,owner,data['action'][current],remaining)]
    return inputs(memory),inputs(current),integers,torch.as_tensor(data['spatial_targets'][window],device=device)


def training_priors(data,training,persistent):
    """Whole-session-weighted goal/pill/preview priors; no holdout observations."""
    counts = Counter(int(data['windows'][i,3]) for i in training)
    spatial,horizon,intent = np.zeros((4,81,384)),np.zeros((4,81,6)),np.zeros((81,4))
    for i in training:
        start,length,goal,session = map(int,data['windows'][i])
        weight = len(training)/(len(counts)*counts[session]*length)
        for step in range(length):
            index = data['feature_index'][start if persistent else start+step]
            pill,preview = data['canonical_pill'][index],data['canonical_preview'][index]
            pair = int((3*pill[0]+pill[1])*9+3*preview[0]+preview[1])
            spatial[goal,pair] += weight*data['spatial_targets'][i]
            horizon[goal,pair,length-1 if persistent else length-step-1] += weight
            intent[pair,goal] += weight
    def smooth(array):
        marginal = array.sum(-2,keepdims=True)+1e-3
        marginal /= marginal.sum(-1,keepdims=True)
        return ((array+16*marginal)/(array.sum(-1,keepdims=True)+16)).astype(np.float32)
    return dict(spatial=smooth(spatial),horizon=smooth(horizon),intent=smooth(intent))


def losses(model,data,ids,device,persistent,*,free=False,priors=None):
    root_inputs,current_inputs,(goal,horizon,elapsed,owner,action,remaining),target = batch(data,ids,device,persistent)
    memory,current = model.encode(*root_inputs),model.encode(*current_inputs)
    logits,spatial,duration = model(memory,current,goal,elapsed)
    values = dict(action_nll=F.cross_entropy(logits,action,reduction='none'),
        spatial_nll=-(target*F.log_softmax(spatial,-1)).sum(-1),
        horizon_nll=F.cross_entropy(duration,horizon,reduction='none'),
        intent_nll=F.cross_entropy(model.intent(memory),goal,reduction='none'),
        action_agreement=(logits.argmax(-1)==action).float(),
        spatial_anchor_hit=target.gather(1,spatial.argmax(-1)[:,None]).squeeze(1).gt(0).float())
    if free:
        autonomous_goal = model.intent(memory).argmax(-1)
        autonomous,_,_ = model(memory,current,autonomous_goal,elapsed)
        values.update(predicted_goal_action_nll=F.cross_entropy(autonomous,action,reduction='none'),
                      predicted_goal_action_agreement=(autonomous.argmax(-1)==action).float())
    if priors is not None:
        pill,preview = root_inputs[2:4]
        pair = (3*pill[:,0]+pill[:,1])*9+3*preview[:,0]+preview[:,1]
        values.update(spatial_prior_nll=-(target*priors['spatial'][goal,pair].log()).sum(-1),
            horizon_prior_nll=-priors['horizon'][goal,pair,horizon].log(),
            intent_prior_nll=-priors['intent'][pair,goal].log())
    count = torch.bincount(owner,minlength=len(ids)).clamp_min(1)
    grouped = {k:torch.zeros(len(ids),device=device).scatter_add_(0,owner,v)/count for k,v in values.items()}
    for name,mask in [('early_setup',remaining>=3),('payoff',remaining==1)]:
        n = torch.zeros(len(ids),device=device).scatter_add_(0,owner,mask.float())
        grouped[name+'_nll'] = torch.where(n>0,torch.zeros(len(ids),device=device).scatter_add_(
            0,owner,values['action_nll']*mask)/n.clamp_min(1),torch.nan)
    return grouped,len(action)


def evaluate(model,data,ids,device,size,persistent,priors=None):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0,len(ids),size):
            chosen = ids[start:start+size]
            values,_ = losses(model,data,chosen,device,persistent,free=True,priors=priors)
            values = {k:v.cpu().numpy() for k,v in values.items()}
            records.extend(dict(session=int(data['windows'][i,3]),
                **{k:float(v[j]) for k,v in values.items()}) for j,i in enumerate(chosen))
    keys = [k for k in records[0] if k != 'session']
    sessions = sorted({r['session'] for r in records})
    by_session = {}
    for session in sessions:
        rows = [r for r in records if r['session']==session]
        by_session[str(session)] = {k:float(np.mean([r[k] for r in rows if np.isfinite(r[k])]))
                                   if any(np.isfinite(r[k]) for r in rows) else None for k in keys}
    summary = {}
    for key in keys:
        values = [r[key] for r in by_session.values() if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    return dict(summary=summary,by_session=by_session)


def fit_models(data,config,output,report):
    epochs,size = int(config.get('epochs',8)),int(config.get('batch_windows',32))
    if min(epochs,size)<1:
        raise ValueError('positive fitting exposure required')
    device = config.get('device','cpu')
    validation_session = [session_validation(str(s),int(config.get('split_seed',81029))) for s in data['sessions']]
    training = np.asarray([i for i,w in enumerate(data['windows']) if not validation_session[w[3]]])
    validation = np.asarray([i for i,w in enumerate(data['windows']) if validation_session[w[3]]])
    if not len(training) or not len(validation):
        raise ValueError('whole-session training and validation must both be nonempty')
    counts = Counter(int(data['windows'][i,3]) for i in training)
    report.update(train_sessions=len(counts),validation_sessions=len(set(data['windows'][validation,3])),
                  training_windows=len(training),validation_windows=len(validation),arms={})
    for persistent in (True,False):
        name = 'persistent' if persistent else 'stateless'
        torch.manual_seed(int(config.get('seed',19473)))
        rng = np.random.default_rng(int(config.get('seed',19473)))
        model = SpatialProposer(data['features'].shape[1],int(config.get('width',128)),
                                persistent=persistent).to(device)
        priors = {k:torch.as_tensor(v,device=device) for k,v in training_priors(data,training,persistent).items()}
        optimizer = torch.optim.AdamW(model.parameters(),lr=float(config.get('learning_rate',3e-4)))
        arm = dict(status='Running',window_presentations=0,action_presentations=0,epochs=[])
        report['arms'][name] = arm
        report.update(phase='initial_evaluation',current_arm=name)
        write_progress(output,report)
        arm['initial'] = evaluate(model,data,validation,device,size,persistent,priors)['summary']
        last_write = time.monotonic()
        for epoch in range(epochs):
            model.train()
            report.update(phase='optimizing',current_epoch=epoch+1)
            order = rng.permutation(training)
            for offset in range(0,len(order),size):
                ids = order[offset:offset+size]
                values,actions = losses(model,data,ids,device,persistent)
                weights = torch.tensor([len(training)/(len(counts)*counts[int(data['windows'][i,3])])
                                        for i in ids],device=device)
                loss = ((values['action_nll']+.2*values['spatial_nll']+
                         .1*values['horizon_nll']+.1*values['intent_nll'])*weights).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite spatial proposal loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                arm['window_presentations'] += len(ids)
                arm['action_presentations'] += actions
                if time.monotonic()-last_write>=5:
                    arm['last_loss'] = float(loss.detach())
                    write_progress(output,report)
                    last_write = time.monotonic()
            metrics = evaluate(model,data,validation,device,size,persistent,priors)
            arm['epochs'].append(dict(epoch=epoch+1,validation=metrics['summary']))
            print(json.dumps(dict(arm=name,**arm['epochs'][-1])),flush=True)
            write_progress(output,report)
        checkpoint = dict(schema=SCHEMA,feature_dim=model.feature_dim,width=model.width,
            persistent=persistent,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            source_sha256=report['source_sha256'],competitive_sha256=report['competitive_sha256'],
            feature_contract='frozen-legacy-own-bottle-mean-max-v1',diagnostic_only=True,
            quality_admission=False,selection='fixed final epoch',
            training_priors={k:v.detach().cpu() for k,v in priors.items()})
        path = output/(name+'-final.pt')
        torch.save(checkpoint,path)
        arm.update(status='Complete',checkpoint_sha256=sha256(path),final=metrics)
        write_progress(output,report)
    a,b = (report['arms'][name]['final']['by_session'] for name in ('persistent','stateless'))
    groups = sorted(a)
    rng = np.random.default_rng(91731)
    comparisons = {}
    for key in ('action_nll','predicted_goal_action_nll','early_setup_nll','payoff_nll','spatial_anchor_hit'):
        eligible = [g for g in groups if a[g][key] is not None and b[g][key] is not None]
        if not eligible:
            comparisons[key] = dict(sessions=0,status='unavailable')
            continue
        draws = rng.integers(len(eligible),size=(20000,len(eligible)))
        delta = np.asarray([a[g][key]-b[g][key] for g in eligible])
        comparisons[key] = dict(persistent_minus_stateless=float(delta.mean()),
            ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),sessions=len(eligible))
    report['paired_session_comparisons'] = comparisons
    report['training_prior_comparisons'] = {}
    draws = rng.integers(len(groups),size=(20000,len(groups)))
    for name,arm in report['arms'].items():
        rows = arm['final']['by_session']
        report['training_prior_comparisons'][name] = {}
        for key in ('spatial','horizon','intent'):
            delta = np.asarray([rows[g][key+'_nll']-rows[g][key+'_prior_nll'] for g in groups])
            report['training_prior_comparisons'][name][key] = dict(model_minus_prior=float(delta.mean()),
                ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),sessions=len(groups))
    report['comparison_scope'] = 'Descriptive recorded-prefix development validation, individual intervals. Predicted-goal scores still use actual human intermediate states, without simulating plan termination. Fresh replay confirmation and actual quality-admitted persistent play remain required.'
    return report


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(int(config.get('threads',1)))
    report = dict(schema=SCHEMA,status='Running',config=config,source_sha256=sha256(config['source']),
        competitive_sha256=sha256(config['checkpoint']),console_frames_trained=0,
        diagnostic_only=True,quality_admission=False,product_gates_passed=False)
    write_progress(output,report)
    try:
        data = prepare_data(config,output,report)
        fit_models(data,config,output,report)
        report.update(status='Complete',phase='complete',elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='Failed',error=str(error))
        raise
    finally:
        write_progress(output,report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(json.loads(parser.parse_args().config.read_text()))
