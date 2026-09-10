"""Fresh, seed-held-out controller confirmation of learned motor auxiliaries.

Frozen stochastic parent play supplies positions; neither fitting nor model
selection occurs here. Exact conditional labels and seed-grouped prediction
comparisons are separate from the full-game strength tournament.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.human.motor_opportunity import MotorOpportunityLabeler, OPPORTUNITY_CONDITION
from drmc_rl.models.policy.controller_core import ControllerCorePolicy, write_public_replay
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from drmc_rl.training.motor_supervision import (
    assign_game_weights, cache_reference, evaluate_motor, load_bank, make_motor_batch,
    upgrade_motor_model, initialize_motor_priors,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.build_motor_opportunity_bank import annotate_row, selected_rows
from tools.train_pace_strategy import terminal_samples
from tools.trainer_arena_cache import MemoPlanner
from tools.trainer_event_rollout import run_event_batch
from tools.vs_head_to_head import PlainPolicy


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_seeds(seeds, training_config, bank_config, bank_records):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed <= 65535 for seed in seeds)):
        raise ValueError('confirmation needs distinct two-byte reset seeds')
    required = set(seeds)
    if (not required.issubset(training_config['holdout_seeds'])
            or not required.issubset(bank_config['holdout_seeds'])
            or required.intersection(row['game_seed'] for row in bank_records)):
        raise ValueError('confirmation seeds must be excluded from outcome training, auxiliary fitting and anchors')


def validate_confirmation_exclusions(seeds, paths):
    identities = {}
    for path in paths:
        previous = json.loads(Path(path).read_text())
        if set(seeds).intersection(previous['seeds']):
            raise ValueError('new confirmation reuses a previously inspected confirmation seed')
        identities[str(path)] = digest(path)
    return identities


def opportunity_priors(rows):
    """Fit pace/cell prevalence on training roots only, with equal game weight."""
    sums, masses = {}, defaultdict(float)
    for row in rows:
        pace = row['record']['pace']
        batch = make_motor_batch([row], device='cpu')
        valid = batch['future_mask'][0]
        if not valid.any():
            continue
        for name in ('reach', 'clear'):
            key = pace, name
            values = batch[name + '_target'][0, valid].float().mean(0).numpy()
            sums[key] = sums.get(key, np.zeros_like(values)) + row['weight'] * values
            masses[key] += row['weight']
    return {key: value / masses[key] for key, value in sums.items()}


def prior_metrics(rows, priors, pace):
    totals = defaultdict(float)
    for row in rows:
        batch = make_motor_batch([row], device='cpu')
        valid = batch['future_mask'][0]
        for name in ('reach', 'clear'):
            if (pace, name) not in priors:
                raise ValueError(f'no training-only opportunity prior for {pace}')
            target = batch[name + '_target'][0, valid].float().numpy()
            error = float(np.square(target - priors[pace, name]).mean()) if len(target) else 0.
            totals[name + '_brier'] += error * row['weight'] / len(rows)
    return dict(totals)


def summarize_seed_metrics(records, *, bootstrap_seed, draws=4000):
    """Every seed contributes once; sides, roots and parity are correlated."""
    if len({row['seed'] for row in records}) != len(records):
        raise ValueError('aggregate side/root measurements before the seed bootstrap')
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(len(records), size=(draws, len(records))) if len(records) >= 8 else None
    result = {}
    for key in records[0]['fitted']:
        base = np.array([row['initial'][key] for row in records])
        fitted = np.array([row['fitted'][key] for row in records])
        delta = fitted - base
        metrics = dict(initial=float(base.mean()), fitted=float(fitted.mean()),
                       change=float(delta.mean()), change_ci95=None)
        if indices is not None:
            metrics['change_ci95'] = np.quantile(delta[indices].mean(1), [.025, .975]).tolist()
        if key in records[0]['prior']:
            prior = np.array([row['prior'][key] for row in records])
            metrics.update(training_prior=float(prior.mean()),
                           change_from_prior=float((fitted-prior).mean()))
            if indices is not None:
                metrics['change_from_prior_ci95'] = np.quantile(
                    (fitted-prior)[indices].mean(1), [.025, .975]).tolist()
        result[key] = metrics
    return dict(independent_seeds=len(records), metrics=result,
                uncertainty='paired bootstrap of whole reset seeds; descriptive per-condition 95% intervals')


def collect_condition(config, actor, opponent, planner, condition, index, directory, report):
    replay_path = directory / 'public-replay.npz'
    games_path = directory / 'games.json'
    if replay_path.exists() and games_path.exists():
        return replay_path, json.loads(games_path.read_text())
    actor.rng.manual_seed(int(config['seed']) + index)
    jobs = [(seed, side, 2*i+side) for i, seed in enumerate(config['seeds']) for side in (0, 1)]
    match = dict(id=f'motor-confirmation-{index}', a='learner', b='parent',
                 games=len(jobs), **condition)
    delay = int(config.get('compute_frames', 4))
    rollout = dict(native_library=config.get('native_library'),
                   variants={key: dict(delay=delay) for key in ('learner', 'parent')},
                   max_game_frames=int(config.get('max_game_frames', 120000)))
    chunk = int(config.get('games_per_batch', 32))
    if chunk < 2 or chunk % 2:
        raise ValueError('confirmation batches require complete paired seeds')
    results = []
    for start in range(0, len(jobs), chunk):
        completed_frames = sum(row['frames'] for row, _, _ in results)

        def activity(work):
            report('collecting', games=len(results) + work['games'], target_games=len(jobs),
                   frames=completed_frames + work['frames'])

        part, _ = run_event_batch(rollout, match, jobs[start:start+chunk], None, planner, None,
                                 policies={'learner': actor, 'parent': opponent}, activity=activity)
        if any(row['reason'] == 'timeout' for row, _, _ in part):
            raise RuntimeError('censored source game cannot supply a natural controller confirmation')
        results.extend(part)
    records = terminal_samples(results)
    if not records:
        raise RuntimeError('no feasible source decisions for this confirmation condition')
    actor.finish_collection(records)
    games = [row for row, _, _ in results]
    write_public_replay(replay_path, records, games, update=index+1,
                        pace=condition['pace'], level=condition['level'])
    dump(games_path, games)
    return replay_path, games


def run(config):
    condition_ids = [(row['level'], row['pace']) for row in config['conditions']]
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError('confirmation conditions must be nonempty and distinct')
    torch.set_num_threads(int(config.get('threads', 1)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = config.get('device', 'cpu')
    fit = Path(config['fit_directory'])
    fit_progress = json.loads((fit / 'progress.json').read_text())
    if fit_progress['status'] != 'Complete' or not (fit / 'core-final.pt').is_file():
        raise ValueError('confirmation requires the completed immutable auxiliary fit')
    fit_config = json.loads((fit / 'config.json').read_text())
    train, validation = load_bank(config['bank'])
    bank_config = json.loads((Path(config['bank']) / 'config.json').read_text())
    source_config = json.loads(Path(config['training_config']).read_text())
    validate_seeds(config['seeds'], source_config, bank_config,
                   [row['record'] for row in train + validation])
    excluded_confirmations = validate_confirmation_exclusions(
        config['seeds'], config.get('exclude_confirmation_configs', []))
    parent_hash = digest(config['checkpoint'])
    fitted_payload = load_checkpoint(fit / 'core-final.pt', map_location=device)
    if fitted_payload['parent_sha256'] != parent_hash:
        raise ValueError('confirmation baseline differs from the auxiliary fit parent')
    identities = dict(parent_sha256=parent_hash, fitted_sha256=digest(fit / 'core-final.pt'),
                      fit_seed=fit_config['seed'])
    if excluded_confirmations:
        identities['excluded_confirmations'] = excluded_confirmations
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in (('config.json', config), ('identities.json', identities)):
        path = output / name
        if path.exists() and json.loads(path.read_text()) != expected:
            raise ValueError('confirmation resume changed its experiment or model identity')
        dump(path, expected)
    progress = dict(schema='drmc-motor-confirmation-v1', status='Running',
                    completed_conditions=0, target_conditions=len(config['conditions']))

    def report(phase, **values):
        progress.update(phase=phase, **values, updated_at=datetime.now(UTC).isoformat())
        dump(output / 'progress.json', progress)

    report('loading')
    planner = MemoPlanner(NativeReachabilityRunner())
    try:
        parent = load_checkpoint(Path(config['checkpoint']), map_location=device)
        # This reconstructs the fixed initial auxiliary heads, not a new fit.
        torch.manual_seed(fit_config['seed'])
        initial, _ = upgrade_motor_model(parent, device=device)
        if fit_config.get('head_initialization', 'legacy_random') == 'training_cell_prior':
            initialize_motor_priors(initial, train)
        fitted, _ = upgrade_motor_model(fitted_payload, device=device)
        batch_size = int(config.get('batch_size', 16))
        cache_reference(initial, validation, batch_size=batch_size, device=device)
        check = evaluate_motor(initial, validation, batch_size=batch_size, device=device)
        baseline = json.loads((fit / 'baseline.json').read_text())['validation']
        if max(abs(check[key] - baseline[key]) for key in baseline) > 1e-4:
            raise ValueError('reconstructed initial heads do not reproduce the original fitting baseline')
        priors = opportunity_priors(train)
        del train, validation, parent, fitted_payload
        actor = ControllerCorePolicy(config['checkpoint'], device=device, seed=config['seed'])
        opponent = PlainPolicy(Path(config['checkpoint']), device, public_only=True)
        reports = []
        with MotorOpportunityLabeler(planner, lib_path=config.get('native_library')) as labeler:
            for index, condition in enumerate(config['conditions']):
                label = f"{condition['level']}-{condition['pace']}"
                report('collecting', condition=label, games=0, roots=0)
                directory = output / label
                directory.mkdir(exist_ok=True)
                finished_path = directory / 'assessment.json'
                if finished_path.exists():
                    reports.append(json.loads(finished_path.read_text()))
                    progress['completed_conditions'] += 1
                    continue
                replay_path, games = collect_condition(config, actor, opponent, planner, condition,
                                                       index, directory, report)
                report('labeling', games=len(games), target_games=2*len(config['seeds']))
                with np.load(replay_path, allow_pickle=False) as payload:
                    replay = {key: payload[key] for key in payload.files}
                source_hash = digest(replay_path)
                indices = selected_rows(replay, per_game=int(config.get('roots_per_game', 3)),
                                        limit=int(config.get('roots_per_condition', 192)),
                                        seed=config['seed'] + index)
                rows = []
                (directory / 'roots').mkdir(exist_ok=True)
                for row_index in indices:
                    path = directory / 'roots' / f'{row_index:06d}.npz'
                    if path.exists():
                        with np.load(path, allow_pickle=False) as payload:
                            row = {key: payload[key] for key in payload.files if key != 'metadata'}
                            record = json.loads(str(payload['metadata']))
                        if record['source_sha256'] != source_hash:
                            raise ValueError('resumed confirmation labels refer to different source play')
                    else:
                        labels, row = annotate_row(replay, row_index, labeler)
                        record = dict(source_sha256=source_hash, source_row=int(row_index),
                                      game_seed=int(replay['game_seed'][row_index]),
                                      learner_port=int(replay['learner_port'][row_index]),
                                      split='confirmation', **condition, **labels.summary())
                        with path.with_suffix('.npz.next').open('wb') as stream:
                            np.savez_compressed(stream, **row, metadata=np.asarray(json.dumps(record)))
                        path.with_suffix('.npz.next').replace(path)
                    row['record'] = record
                    row['game_id'] = source_hash, record['game_seed'], record['learner_port']
                    rows.append(row)
                    report('labeling', roots=len(rows), target_roots=len(indices))
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row['record']['game_seed']].append(row)
                if len(grouped) < int(config.get('minimum_seeds_per_condition', 16)):
                    raise ValueError('insufficient independent confirmation seeds with feasible roots')
                cache_reference(initial, rows, batch_size=batch_size, device=device)
                by_seed = []
                for seed, group in sorted(grouped.items()):
                    assign_game_weights(group)
                    by_seed.append(dict(seed=seed, roots=len(group),
                        initial=evaluate_motor(initial, group, batch_size=batch_size, device=device),
                        fitted=evaluate_motor(fitted, group, batch_size=batch_size, device=device),
                        prior=prior_metrics(group, priors, condition['pace'])))
                    report('evaluating', evaluated_seeds=len(by_seed), target_seeds=len(grouped))
                assessment = dict(condition=condition, games=len(games), roots=len(rows),
                    source_frames=sum(game['frames'] for game in games),
                    candidates=sum(len(row['actions']) for row in rows),
                    **summarize_seed_metrics(by_seed, bootstrap_seed=config['seed'] + index),
                    seed_metrics=by_seed)
                dump(finished_path, assessment)
                reports.append(assessment)
                progress['completed_conditions'] += 1
        dump(output / 'assessment.json', dict(identities=identities, conditions=reports,
             opportunity_condition=OPPORTUNITY_CONDITION,
             source_behavior='frozen stochastic parent versus its deterministic policy, side-swapped',
             scope='held-out conditional prediction evidence; no match-strength or promotion claim'))
        progress['status'] = 'Complete'
        report('complete', evaluated_seeds=None, target_seeds=None)
    except BaseException as error:
        progress.update(status='Failed', error=str(error))
        report('failed')
        raise
    finally:
        planner.close()
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(json.loads(args.config.read_text()))
    print(json.dumps(dict(result, wall_seconds=time.perf_counter()-started), indent=2))


if __name__ == '__main__':
    main()
