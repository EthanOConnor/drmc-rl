"""Fit a persistent proposal head from whole-session-disjoint constructions.

This learns human plan proposals only. It neither modifies the competitive
core nor uses imitation logits as calibrated regret or a style permission.
"""
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

from drmc_rl.human.expressive_proposer import ExpressiveProposer, MODEL_SCHEMA, public_planes
from drmc_rl.human.expressive_sequences import GOALS, SCHEMA
from tools.build_expressive_sequences import write_progress


def session_validation(session, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{session}'.encode()).digest()[:8], 'big') % 5 == 0


def batch(data, selected, device):
    roots, states, goals, remaining, plans, owners = [], [], [], [], [], []
    for owner, window_id in enumerate(selected):
        start, length, goal, _ = data['windows'][window_id]
        for step in range(length):
            roots.append(start)
            states.append(start+step)
            goals.append(goal)
            remaining.append(length-step)
            plans.append(goal*5+length-2)
            owners.append(owner)
    def inputs(indices):
        return (torch.as_tensor(public_planes(data['board'][indices]), device=device),
                torch.as_tensor(data['pill'][indices], dtype=torch.long, device=device),
                torch.as_tensor(data['preview'][indices], dtype=torch.long, device=device))
    tensors = [torch.as_tensor(values, dtype=torch.long, device=device)
               for values in (goals, remaining, plans, owners, data['action'][states])]
    return inputs(roots), inputs(states), tensors


def window_losses(model, data, selected, device):
    root_inputs, state_inputs, (goal, remaining, plan, owner, action) = batch(data, selected, device)
    root, current = model.encode(*root_inputs), model.encode(*state_inputs)
    logits = model(root, current, goal, remaining)
    policy_loss = F.cross_entropy(logits, action, reduction='none')
    intent_loss = F.cross_entropy(model.intent(root), plan, reduction='none')
    agreement = (logits.argmax(-1) == action).float()
    counts = torch.bincount(owner, minlength=len(selected)).clamp_min(1)
    def grouped(value):
        return torch.zeros(len(selected), device=device).scatter_add_(0, owner, value)/counts
    return grouped(policy_loss), grouped(intent_loss), grouped(agreement), len(action)


def evaluate(model, data, selected, *, device, size):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(selected), size):
            ids = selected[start:start+size]
            values = window_losses(model, data, ids, device)[:3]
            for window, p, g, a in zip(ids, *(v.cpu().tolist() for v in values), strict=True):
                records.append((int(data['windows'][window, 3]), p, g, a))
    sessions = sorted(set(r[0] for r in records))
    return {key:float(np.mean([np.mean([r[column] for r in records if r[0] == session]) for session in sessions]))
            for column, key in enumerate(('action_nll','intent_nll','action_agreement'), start=1)}


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = Path(config['source'])
    with np.load(source, allow_pickle=False) as archive:
        data = {key:archive[key] for key in ('board','pill','preview','action','windows','sessions','metadata')}
    if json.loads(str(data['metadata']))['schema'] != SCHEMA:
        raise ValueError('verified construction source required')
    seed, epochs = int(config.get('seed', 92713)), int(config.get('epochs', 8))
    size, device = int(config.get('batch_windows', 32)), config.get('device', 'cpu')
    if min(epochs, size) < 1:
        raise ValueError('positive training exposure required')
    torch.set_num_threads(int(config.get('threads', 1)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    is_validation = [session_validation(str(s), int(config.get('split_seed', 81029))) for s in data['sessions']]
    training = np.asarray([i for i, row in enumerate(data['windows']) if not is_validation[row[3]]])
    validation = np.asarray([i for i, row in enumerate(data['windows']) if is_validation[row[3]]])
    if not len(training) or not len(validation):
        raise ValueError('nonempty whole-session training and validation required')
    counts = Counter(int(data['windows'][i, 3]) for i in training)
    model = ExpressiveProposer(width=int(config.get('width', 64))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 3e-4)))
    report = dict(schema=MODEL_SCHEMA, status='Running', phase='initial_evaluation', config=config,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  train_sessions=len(counts), validation_sessions=len(set(data['windows'][validation,3])),
                  training_windows=len(training), validation_windows=len(validation),
                  window_presentations=0, action_presentations=0, console_frames_trained=0, epochs=[])
    write_progress(output, report)
    report['initial'] = evaluate(model, data, validation, device=device, size=size)
    write_progress(output, report)
    last_write = time.monotonic()
    for epoch in range(epochs):
        model.train()
        report.update(phase='optimizing', current_epoch=epoch+1)
        for start in range(0, len(training), size):
            # A single fixed permutation is chosen at each epoch below.
            if start == 0:
                order = rng.permutation(training)
            ids = order[start:start+size]
            policy, intent, _, actions = window_losses(model, data, ids, device)
            weights = torch.tensor([len(training)/(len(counts)*counts[int(data['windows'][i,3])]) for i in ids],
                                   device=device)
            loss = ((policy+.25*intent)*weights).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite proposer loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report['window_presentations'] += len(ids)
            report['action_presentations'] += actions
            if time.monotonic()-last_write >= 5:
                report.update(last_loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)
                write_progress(output, report)
                last_write = time.monotonic()
        metrics = dict(epoch=epoch+1, validation=evaluate(model, data, validation, device=device, size=size))
        report['epochs'].append(metrics)
        print(json.dumps(metrics), flush=True)
        write_progress(output, report)
    checkpoint = dict(schema=MODEL_SCHEMA, width=model.width, goals=GOALS,
                      state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                      source_sha256=report['source_sha256'], diagnostic_only=True,
                      quality_admission='unavailable; proposal order does not authorize style overrides')
    torch.save(checkpoint, output/'proposer-final.pt.next')
    (output/'proposer-final.pt.next').replace(output/'proposer-final.pt')
    report.update(status='Complete', phase='complete', elapsed_seconds=time.monotonic()-started,
                  checkpoint_sha256=hashlib.sha256((output/'proposer-final.pt').read_bytes()).hexdigest(),
                  selection='fixed final epoch; held-out sessions never select or train weights')
    write_progress(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        run(config)
    except BaseException as exc:
        output = Path(config['output'])
        if output.is_dir() and not isinstance(exc, FileExistsError):
            prior = json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
            write_progress(output, {**prior,'status':'Failed','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
