"""Fit requested-target routes; frozen root proposals and outcome actors stay fixed."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drmc_rl.human.spatial_proposer import RECURRENT_PUBLIC, SCHEMA as SPATIAL_SCHEMA
from drmc_rl.human.target_construction import SCHEMA, TargetConstruction
from tools.build_expressive_sequences import write_progress
from tools.fit_expressive_proposer import session_validation
from tools.fit_spatial_expressive import batch, prepare_data, sha256


def load_initial(study_path):
    path = Path(study_path)
    study = json.loads(path.read_text())
    if (study.get('status') != 'Complete' or study.get('schema') != SPATIAL_SCHEMA
            or study['config'].get('plan_update_schema') != RECURRENT_PUBLIC):
        raise ValueError('completed recurrent proposal study required')
    source = path.parent/'stateless-final.pt'
    if sha256(source) != study['arms']['stateless']['checkpoint_sha256']:
        raise ValueError('fixed root proposer changed')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if (payload['source_sha256'] != study['source_sha256']
            or payload['competitive_sha256'] != study['competitive_sha256']
            or payload['plan_update_schema'] != RECURRENT_PUBLIC):
        raise ValueError('proposal source contract changed')
    model = TargetConstruction(payload['feature_dim'], payload['width'])
    model.load_state_dict(payload['state_dict'], strict=True)
    return model, study, source


def requested_loss(model, data, ids, *, epoch=0):
    _, inputs, (goal, _, elapsed, owner, action, _), _ = batch(
        data, ids, 'cpu', True, replanning=True)
    current = model.encode(*inputs)
    memory = model.sequence_memory(current, owner, elapsed, len(ids))
    # Rotate uniformly through observed payoff cells. One requested colored
    # cell is fixed across the whole construction. The requested budget is
    # always six, not the observed future duration of this replay.
    anchors = []
    for index in ids:
        cells = np.flatnonzero(data['spatial_targets'][index] > 0)
        if not len(cells):
            raise ValueError('missing verified goal cells')
        anchors.append(int(cells[(int(index)+epoch) % len(cells)]))
    anchor = torch.tensor(anchors)[owner]
    logits = model.requested_actions(memory, current, goal, elapsed, anchor, 6-elapsed)
    loss = F.cross_entropy(logits, action, reduction='none')
    count = torch.bincount(owner, minlength=len(ids))
    grouped = torch.zeros(len(ids)).scatter_add_(0, owner, loss)/count
    return grouped, len(action)


def evaluate(model, data, ids, size):
    values = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), size):
            chosen = ids[start:start+size]
            losses, _ = requested_loss(model, data, chosen)
            for i, value in zip(chosen, losses.tolist(), strict=True):
                values.setdefault(str(data['sessions'][data['windows'][i, 3]]), []).append(value)
    per_session = {k:float(np.mean(v)) for k, v in values.items()}
    return dict(requested_goal_nll=float(np.mean(list(per_session.values()))),
                by_session=per_session,
                scope='Descriptive development imitation with an explicit hindsight goal request; not autonomous payoff, quality, strength or preference.')


def fit(data, model, config, output, report):
    epochs, size = int(config.get('epochs', 8)), int(config.get('batch_windows', 32))
    if epochs < 1 or size < 1:
        raise ValueError('positive fixed fitting exposure required')
    validation = [session_validation(str(s), int(config.get('split_seed', 81029))) for s in data['sessions']]
    train = np.array([i for i,w in enumerate(data['windows']) if not validation[w[3]]])
    held = np.array([i for i,w in enumerate(data['windows']) if validation[w[3]]])
    if not len(train) or not len(held):
        raise ValueError('whole-session split required')
    counts = Counter(int(data['windows'][i,3]) for i in train)
    report.update(training_windows=len(train), validation_windows=len(held),
        training_sessions=len(counts), validation_sessions=len(set(data['windows'][held,3])),
        window_presentations=0, action_presentations=0, epochs=[])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 3e-4)))
    rng = np.random.default_rng(int(config.get('seed', 94017)))
    report['initial'] = evaluate(model, data, held, size)
    last_write = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        report.update(phase='optimizing', epoch=epoch+1)
        for start in range(0, len(order), size):
            ids = order[start:start+size]
            losses, actions = requested_loss(model, data, ids, epoch=epoch)
            weights = torch.tensor([len(train)/(len(counts)*counts[int(data['windows'][i,3])]) for i in ids])
            loss = (losses*weights).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite conditional imitation')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report['window_presentations'] += len(ids)
            report['action_presentations'] += actions
            if time.monotonic()-last_write >= 5:
                report['last_loss'] = float(loss.detach())
                write_progress(output, report)
                last_write = time.monotonic()
        metric = evaluate(model, data, held, size)
        report['epochs'].append(dict(epoch=epoch+1, requested_goal_nll=metric['requested_goal_nll']))
        print(json.dumps(report['epochs'][-1]), flush=True)
        write_progress(output, report)
    payload = dict(schema=SCHEMA, feature_dim=model.feature_dim, width=model.width,
        state_dict=model.state_dict(), source_sha256=report['source_sha256'],
        competitive_sha256=report['competitive_sha256'], request_budget=6,
        root_proposer_sha256=report['root_proposer_sha256'], quality_admission=False,
        selection='fixed final epoch', request_contract='immutable-colored-cell-and-goal-v1')
    path = output/'target-final.pt'
    torch.save(payload, path)
    report.update(checkpoint_sha256=sha256(path), final=metric)
    return model


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(config.get('threads', 1)))
    started = time.monotonic()
    model, study, root_path = load_initial(config['study'])
    config = {**config, 'source':study['config']['source'], 'checkpoint':study['config']['checkpoint'],
              'prepared_from':str(Path(config['study']).resolve())}
    report = dict(schema=SCHEMA, status='Running', config=config,
        source_sha256=sha256(config['source']), competitive_sha256=sha256(config['checkpoint']),
        root_proposer=str(root_path.resolve()), root_proposer_sha256=sha256(root_path),
        console_frames_trained=0, quality_admission=False, product_gates_passed=False)
    try:
        data = prepare_data(config, output, report)
        fit(data, model, config, output, report)
        report.update(status='Complete', phase='complete')
    except BaseException as error:
        report.update(status='Failed', error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_progress(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
