"""Evaluate and stop a stranded-edge curriculum variant (A+S, C+S, B+S) forked off an afterstate arm.

Pre-registered in ``stranded-edge-v1/variant-addendum-v1.json``. For every
50M-frame snapshot ``core-fNNNNNNNNNNN.pt`` in the variant's tf3090 output:

1. Panel: the arms' fixed 896-game panel versus the champion
   (``prepare_afterstate_tournament_v1.arena_config('panel', ...)``) on the
   Mac through the distributed coordinator (``local``, 3 MPS workers).
2. Benchmark: ``tools.eval_stranded_edge`` solo at frame_perfect and normal on
   ``stranded-edge-benchmark-v1`` (one process), for the variant snapshot and,
   when it exists, the parent arm's snapshot at the same frame label.
3. Stop rule, the arms' rule: a snapshot improves when its pooled panel score
   exceeds the best so far, where "so far" starts from the parent arm's
   snapshots up to the fork. Two consecutive non-improving variant snapshots,
   or the variant ending, fire the rule: the variant's tmux session is
   interrupted and its best post-fork snapshot is written to ``selection.json``.

The Mac is shared: work starts only when no other arena coordinator or
benchmark runs and the parent arm's next snapshot is not due within 25 minutes
(its own stop rule does not wait for anyone). One ``SNAPSHOT`` line per
snapshot and one ``STOP RULE FIRED`` line go to stdout.

  python runs/review-20260909/stranded_variant_watch_v1.py A
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from assess_afterstate_tournament_v1 import assess  # noqa: E402

REMOTE_ROOT = '/home/ethan/.cache/drmc-rl/trainer-output'
DATA = Path('/Users/ethan/dev/drmario/drmc-rl-stranded-edge-data')
NATIVE = Path('/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries')
PY = '/Users/ethan/dev/drmario/drmc-rl/.venv/bin/python'
EVERY = 50_000_000
ARMS = {
    'A': dict(variant='afterstate-core-v1/ppo-v1-stranded', session='afterstate-ppo-stranded-A',
              parent='afterstate-core-v1/ppo-v1',
              parent_state=Path('/Users/ethan/dev/drmario/drmc-rl-afterstate/runs/review-20260909/afterstate-core-v1/stop-rule.json'),
              parent_data=Path('/Users/ethan/dev/drmario/drmc-rl-afterstate-data'), fork_frames=109_335_507),
    # C+S needs the arm C model code: run this watcher from the trainer/stranded-edge-armc worktree.
    'C': dict(variant='afterstate-core-full-v1/ppo-v1-stranded', session='afterstate-full-ppo-stranded-C',
              parent='afterstate-core-full-v1/ppo-v1',
              parent_state=Path('/Users/ethan/dev/drmario/drmc-rl-afterstate-full/runs/review-20260909/afterstate-core-full-v1/stop-rule.json'),
              parent_data=Path('/Users/ethan/dev/drmario/drmc-rl-afterstate-full-data'), fork_frames=None),
}
ENV = dict(PYTHONPATH=str(REPO), PATH='/usr/bin:/bin:/usr/local/bin', HOME=str(Path.home()),
           DRMARIO_REACH_LIB=str(NATIVE / 'libdrm_reach_full.dylib'), DRMARIO_POOL_LIB=str(NATIVE / 'libdrmario_pool.dylib'))


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def remote(command):
    return subprocess.run(['ssh', 'tf3090', command], capture_output=True, text=True, timeout=300).stdout


def remote_training(path):
    try:
        return json.loads(remote(f'cat {REMOTE_ROOT}/{path}/training.json'))
    except ValueError:
        return {}


def mac_busy():
    return any(subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0
               for pattern in ('trainer_arena_distributed', 'tools.eval_stranded_edge', 'tools.trainer_planning_arena'))


def parent_due_soon(arm):
    """The parent's stop rule plays its snapshot as soon as it appears; keep out of its way."""
    t = remote_training(arm['parent'])
    if t.get('status') != 'Running':
        return False
    frames = int(t.get('frames', 0))
    rate = (t.get('throughput') or {}).get('frames_per_second') or 5000
    pending = json.loads(arm['parent_state'].read_text()) if arm['parent_state'].exists() else {}
    played = {s['frames'] for s in pending.get('snapshots', [])}
    if frames // EVERY * EVERY not in played and frames >= EVERY:
        return True  # its latest snapshot is not scored yet
    return ((frames // EVERY + 1) * EVERY - frames) / rate < 25 * 60


def wait_for_mac(arm):
    while mac_busy() or parent_due_soon(arm):
        time.sleep(120)


def panel(label, checkpoint, out_dir):
    from prepare_afterstate_tournament_v1 import arena_config

    path = out_dir / f'panel-{label}.json'
    if not path.exists():
        config = arena_config('panel', checkpoint.resolve(), label, 'mps', DATA / 'panels' / f'panel-{label}')
        config['variants']['candidate']['name'] = f'Stranded-edge variant · {label}'
        path.write_text(json.dumps(config, indent=1) + '\n')
    try:
        return path, assess(path)
    except (AssertionError, FileNotFoundError):
        pass
    with (DATA / 'panels' / f'panel-{label}.log').open('a') as handle:
        subprocess.run([PY, '-m', 'tools.trainer_arena_distributed', 'local', '--config', str(path),
                        '--workers', '3', '--device', 'mps'], check=True, cwd=REPO, env=ENV, stdout=handle, stderr=handle)
    return path, assess(path)


def benchmark(label, checkpoint):
    out = DATA / 'eval' / f'bench-{label}'
    if (out / 'summary.json').exists():
        return out
    base = json.loads((DATA / 'configs' / 'champion-v1.json').read_text())
    config = dict(base, output=str(out), keep_traces=False,
                  variants={'candidate': dict(name=label, delay=4, checkpoint=str(checkpoint))},
                  schedule=[dict(mode='solo', pace='frame_perfect', candidate='candidate'),
                            dict(mode='solo', pace='normal', candidate='candidate')])
    path = DATA / 'configs' / f'bench-{label}.json'
    path.write_text(json.dumps(config, indent=1))
    with (DATA / 'eval' / f'bench-{label}.log').open('a') as handle:
        subprocess.run([PY, '-m', 'tools.eval_stranded_edge', '--config', str(path)], check=True, cwd=REPO, env=ENV,
                       stdout=handle, stderr=handle)
    return out


def compare(candidate, baseline, baseline_id=None):
    args = [PY, '-m', 'tools.eval_stranded_edge', '--compare', str(candidate), str(baseline)]
    if baseline_id:
        args += ['--baseline-id', baseline_id]
    return json.loads(subprocess.run(args, check=True, cwd=REPO, env=ENV, capture_output=True, text=True).stdout)['paired']


def panel_paired(variant_config, parent_config, draws=20000):
    """Per-seed (mean of both sides) score difference, variant minus parent, both versus the champion."""
    def per_seed(config):
        rows = [json.loads(l) for l in (Path(json.loads(Path(config).read_text())['output']) / 'games.jsonl')
                .read_text().splitlines()]
        out = {}
        for r in rows:
            out.setdefault((r['pace'], r['seed']), []).append(r['score'])
        return {k: float(np.mean(v)) for k, v in out.items()}
    a, b = per_seed(variant_config), per_seed(parent_config)
    keys = sorted(set(a) & set(b))
    diff = np.asarray([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(20260924)
    boots = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(axis=1)
    return dict(seeds=len(keys), mean=round(float(diff.mean()), 4),
                ci95=[round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])])


def main():
    key = sys.argv[1]
    arm = ARMS[key]
    out_dir = HERE / 'stranded-edge-v1' / 'afterstate-core-v1' / f'variant-{key}'  # arm C's panel wait matches this
    out_dir.mkdir(parents=True, exist_ok=True)
    (DATA / 'panels').mkdir(exist_ok=True)
    (DATA / 'checkpoints').mkdir(exist_ok=True)
    state_path = out_dir / 'stop-rule.json'
    while arm['fork_frames'] is None:  # the fork point is known once the variant has started
        fork = remote_training(arm['variant']).get('fork')
        if fork:
            arm['fork_frames'] = int(fork['frames'])
        else:
            time.sleep(600)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        parent = json.loads(arm['parent_state'].read_text())
        inherited = [dict(s, inherited=True) for s in parent['snapshots'] if s['frames'] <= arm['fork_frames']]
        state = dict(schema='drmc-stranded-variant-stop-rule-v1', arm=key, fork_frames=arm['fork_frames'],
                     rule='arms\' rule: stop after two consecutive variant snapshots whose pooled panel score does not '
                          'exceed the best so far (parent snapshots up to the fork included); select the best '
                          'post-fork variant snapshot', snapshots=inherited)
    save = lambda: state_path.write_text(json.dumps(state, indent=1) + '\n')  # noqa: E731
    save()
    remote_dir = f'{REMOTE_ROOT}/{arm["variant"]}'
    while not state.get('fired'):
        done = {s['frames'] for s in state['snapshots']}
        listing = remote(f'ls {remote_dir}').split()
        names = sorted(n for n in listing if n.startswith('core-f') and n.endswith('.pt') and int(n[6:-3]) not in done)
        status = remote_training(arm['variant']).get('status', 'unknown')
        if not names:
            if status not in ('Running', 'unknown'):
                state.update(fired=f'variant ended with status {status}')
                break
            time.sleep(300)
            continue
        name = names[0]
        frames = int(name[6:-3])
        parent_t = remote_training(arm['parent'])
        if parent_t.get('status') not in (None, 'Running'):
            parent_last = max((int(n[6:-3]) for n in remote(f"ls {REMOTE_ROOT}/{arm['parent']}").split()
                               if n.startswith('core-f') and n.endswith('.pt')), default=0)
            if frames > parent_last + EVERY:
                state.update(fired=f'equal-frames cap: parent ended at its {parent_last} snapshot')
                remote(f'tmux send-keys -t {arm["session"]} C-c')
                break
        label = f'{key}S-f{frames:011d}'
        local = DATA / 'checkpoints' / f'{key}S-{name}'
        if not local.exists():
            subprocess.run(['scp', '-q', f'tf3090:{remote_dir}/{name}', str(local)], check=True)
        wait_for_mac(arm)
        config, report = panel(label, local, out_dir)
        bench = benchmark(label, local)
        entry = dict(label=label, frames=frames, checkpoint=str(local), pooled=report['pooled_score'],
                     pooled_ci95=report['pooled_ci95'], paces={r['pace']: r['score'] for r in report['comparisons']},
                     panel_config=str(config), benchmark=str(bench))
        entry['benchmark_vs_champion'] = compare(bench, DATA / 'eval' / 'champion-v1', 'champion')['pooled']
        parent_ckpt = arm['parent_data'] / name
        parent_state = json.loads(arm['parent_state'].read_text())
        parent_entry = next((s for s in parent_state['snapshots'] if s['frames'] == frames), None)
        if parent_ckpt.exists():
            wait_for_mac(arm)
            parent_bench = benchmark(f'{key}-f{frames:011d}', parent_ckpt)
            entry['benchmark_vs_parent'] = compare(bench, parent_bench)['pooled']
        if parent_entry is not None:
            parent_config = arm['parent_state'].parent / f"panel-{parent_entry['label']}.json"
            entry['parent_pooled'] = parent_entry['pooled']
            entry['panel_vs_parent'] = panel_paired(config, parent_config)
        best = max(s['pooled'] for s in state['snapshots'])
        entry['improved'] = entry['pooled'] > best
        state['snapshots'].append(entry)
        streak = 0
        for s in reversed(state['snapshots']):
            if s.get('inherited') or s['improved']:
                break
            streak += 1
        pv = entry.get('benchmark_vs_parent', {}).get('pills_restricted')
        pp = entry.get('panel_vs_parent')
        log(f"SNAPSHOT {label}: panel pooled {entry['pooled']:.3f} [{entry['pooled_ci95'][0]:.3f}, "
            f"{entry['pooled_ci95'][1]:.3f}]" + (f" (parent {entry['parent_pooled']:.3f}; paired {pp['mean']:+.3f} "
            f"[{pp['ci95'][0]:+.3f}, {pp['ci95'][1]:+.3f}])" if pp else '') +
            f"; stranded pills vs champion {entry['benchmark_vs_champion']['pills_restricted']}" +
            (f", vs parent {pv}" if pv else '') + f"; improved={entry['improved']} streak={streak}")
        if streak >= 2:
            state.update(fired='two consecutive non-improving snapshots')
            remote(f'tmux send-keys -t {arm["session"]} C-c')
        save()
    own = [s for s in state['snapshots'] if not s.get('inherited')]
    if own:
        state['selected'] = max(own, key=lambda s: s['pooled'])
        (out_dir / 'selection.json').write_text(json.dumps(dict(reason=state['fired'], **state['selected']), indent=1) + '\n')
    save()
    log(f"STOP RULE FIRED ({state['fired']}); variant candidate {state.get('selected', {}).get('label')}")


if __name__ == '__main__':
    main()
