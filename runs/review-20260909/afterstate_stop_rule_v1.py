"""Apply the pre-registered afterstate-core stop rule to outcome-PPO snapshots.

Every 50M-frame snapshot (``core-fNNNNNNNNNNN.pt`` in the tf3090 PPO output)
plays the fixed 896-game panel against the champion on the Mac through the
distributed coordinator (``local``, 3 MPS workers). Snapshot 0 is the distilled
student (panel-distill-v1). A snapshot improves when its pooled panel score
exceeds the best so far. After two consecutive non-improving snapshots, or when
PPO finishes, the rule fires: the PPO tmux session is interrupted (the trainer
keeps its last update checkpoint), and the best-scoring snapshot is written to
``afterstate-core-v1/selection.json`` as the single tournament candidate.

State is kept in ``afterstate-core-v1/stop-rule.json``; rerunning resumes.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'afterstate-core-v1'
DATA = Path('/Users/ethan/dev/drmario/drmc-rl-afterstate-data')
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output/afterstate-core-v1/ppo-v1'
SESSION = 'afterstate-ppo-18h'
NATIVE = Path('/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries')
PY = '/Users/ethan/dev/drmario/drmc-rl/.venv/bin/python'
STATE = OUT / 'stop-rule.json'
sys.path.insert(0, str(HERE))
from assess_afterstate_tournament_v1 import assess  # noqa: E402


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def remote(command):
    return subprocess.run(['ssh', 'tf3090', command], capture_output=True, text=True, timeout=300).stdout


def save(state):
    STATE.write_text(json.dumps(state, indent=1) + '\n')


def run_panel(label, checkpoint):
    config = OUT / f'panel-{label}.json'
    if not config.exists():
        subprocess.run([PY, str(HERE / 'prepare_afterstate_tournament_v1.py'), 'panel', '--candidate', str(checkpoint),
                        '--label', label], check=True, cwd=REPO, env={'PYTHONPATH': str(REPO), 'PATH': '/usr/bin:/bin'})
    try:
        return assess(config)
    except (AssertionError, FileNotFoundError):
        pass
    env = dict(PYTHONPATH=str(REPO), PATH='/usr/bin:/bin:/usr/local/bin', HOME=str(Path.home()),
               DRMARIO_REACH_LIB=str(NATIVE / 'libdrm_reach_full.dylib'), DRMARIO_POOL_LIB=str(NATIVE / 'libdrmario_pool.dylib'))
    with (DATA / f'panel-{label}.log').open('a') as handle:
        subprocess.run([PY, '-m', 'tools.trainer_arena_distributed', 'local', '--config', str(config),
                        '--workers', '3', '--device', 'mps'], check=True, cwd=REPO, env=env, stdout=handle, stderr=handle)
    return assess(config)


def summary(label, report):
    paces = ', '.join(f"{r['pace']} {r['score']:.3f}" for r in report['comparisons'])
    return f"{label}: pooled {report['pooled_score']:.3f} [{report['pooled_ci95'][0]:.3f}, {report['pooled_ci95'][1]:.3f}]; {paces}"


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(
        schema='drmc-afterstate-stop-rule-v1', rule='stop after two consecutive snapshots whose pooled panel score '
        'does not exceed the best so far; select the best pooled snapshot', snapshots=[])
    if not state['snapshots']:
        base = assess(OUT / 'panel-distill-v1.json')
        state['snapshots'].append(dict(label='distill-v1', frames=0, checkpoint=str(DATA / 'student-final-inference.pt'),
                                       pooled=base['pooled_score'], pooled_ci95=base['pooled_ci95'],
                                       paces={r['pace']: r['score'] for r in base['comparisons']}))
        save(state)
    while not state.get('fired'):
        done = {s['label'] for s in state['snapshots']}
        listing = remote(f'ls {REMOTE}; python3 -c "import json;print(json.load(open(\'{REMOTE}/training.json\'))[\'status\'])"')
        names = sorted(n for n in listing.split() if n.startswith('core-f') and n.endswith('.pt'))
        status = listing.strip().splitlines()[-1] if listing.strip() else 'unknown'
        pending = [n for n in names if n[:-3].replace('core-', 'ppo-') not in done]
        if not pending:
            if status not in ('Running', 'unknown'):
                state.update(fired=f'PPO ended with status {status}')
                break
            time.sleep(300)
            continue
        name = pending[0]
        label = name[:-3].replace('core-', 'ppo-')
        local = DATA / name
        if not local.exists():
            subprocess.run(['scp', '-q', f'tf3090:{REMOTE}/{name}', str(local)], check=True)
        report = run_panel(label, local)
        best = max(s['pooled'] for s in state['snapshots'])
        improved = report['pooled_score'] > best
        state['snapshots'].append(dict(label=label, frames=int(name[6:-3]), checkpoint=str(local),
                                       pooled=report['pooled_score'], pooled_ci95=report['pooled_ci95'],
                                       paces={r['pace']: r['score'] for r in report['comparisons']}, improved=improved))
        streak = 0
        for s in reversed(state['snapshots'][1:]):
            if s['improved']:
                break
            streak += 1
        log(summary(label, report) + f"; improved={improved} non-improving streak={streak}")
        if streak >= 2:
            state.update(fired='two consecutive non-improving snapshots')
            remote(f'tmux send-keys -t {SESSION} C-c')
        save(state)
    choice = max(state['snapshots'], key=lambda s: s['pooled'])
    state['selected'] = choice
    save(state)
    (OUT / 'selection.json').write_text(json.dumps(dict(reason=state['fired'], **choice), indent=1) + '\n')
    log(f"STOP RULE FIRED ({state['fired']}); tournament candidate {choice['label']} pooled {choice['pooled']:.3f}")


if __name__ == '__main__':
    main()
