"""Arm C panel configs and the pre-registered stop rule for its outcome PPO.

``panel CHECKPOINT LABEL`` writes ``afterstate-core-full-v1/panel-LABEL.json``:
arm A's 896-game panel (``afterstate-core-v1/seeds.json`` group ``panel``,
``prepare_afterstate_tournament_v1.arena_config``) with the arm-C checkpoint
as candidate versus the champion.

``watch`` (default) applies arm A's stop rule to arm C: every 50M-frame
snapshot (``core-fNNNNNNNNNNN.pt`` in the tf3090 PPO output) plays that panel
on the Mac through the distributed coordinator (``local``, 3 MPS workers, this
checkout). Snapshot 0 is the initialization, which computes the champion's
function bit for bit, so its panel score is 0.500 by construction (both
side-swapped games of every seed are the same game) and it is not played. A
snapshot improves when its pooled panel score exceeds the best so far. After
two consecutive non-improving snapshots, or when PPO ends, the rule fires: the
PPO tmux session is interrupted and the best snapshot is written to
``afterstate-core-full-v1/selection.json`` (snapshot 0 selected means arm C
enters the tournament as the champion itself, i.e. no candidate).

State: ``afterstate-core-full-v1/stop-rule.json``; rerunning resumes. One
``SNAPSHOT`` line per snapshot and one ``STOP RULE FIRED`` line go to stdout.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'afterstate-core-full-v1'
DATA = Path('/Users/ethan/dev/drmario/drmc-rl-afterstate-full-data')
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output/afterstate-core-full-v1/ppo-v1'
SESSION = 'afterstate-full-ppo'
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


def write_panel(checkpoint, label):
    from prepare_afterstate_tournament_v1 import arena_config

    path = OUT / f'panel-{label}.json'
    if path.exists():
        return path
    config = arena_config('panel', Path(checkpoint).resolve(), label, 'mps', OUT / f'panel-{label}')
    config['variants']['candidate']['name'] = f'Afterstate full core (arm C) · {label}'
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    config['source'] = f'drmc-rl trainer/afterstate-core-full {commit}'
    path.write_text(json.dumps(config, indent=1) + '\n')
    return path


def run_panel(label, checkpoint):
    config = write_panel(checkpoint, label)
    try:
        return assess(config)
    except (AssertionError, FileNotFoundError):
        pass
    # The Mac is shared and memory is tight: never run alongside another arm's panel workers.
    while subprocess.run(['pgrep', '-f', 'trainer_arena_distributed.*afterstate-core-v1/'],
                         capture_output=True).returncode == 0:
        time.sleep(120)
    env = dict(PYTHONPATH=str(REPO), PATH='/usr/bin:/bin:/usr/local/bin', HOME=str(Path.home()),
               DRMARIO_REACH_LIB=str(NATIVE / 'libdrm_reach_full.dylib'), DRMARIO_POOL_LIB=str(NATIVE / 'libdrmario_pool.dylib'))
    with (DATA / f'panel-{label}.log').open('a') as handle:
        subprocess.run([PY, '-m', 'tools.trainer_arena_distributed', 'local', '--config', str(config),
                        '--workers', '3', '--device', 'mps'], check=True, cwd=REPO, env=env, stdout=handle, stderr=handle)
    return assess(config)


def summary(label, report):
    paces = ', '.join(f"{r['pace']} {r['score']:.3f}" for r in report['comparisons'])
    return f"{label}: pooled {report['pooled_score']:.3f} [{report['pooled_ci95'][0]:.3f}, {report['pooled_ci95'][1]:.3f}]; {paces}"


def watch():
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(
        schema='drmc-afterstate-stop-rule-v1', arm='C (full-size champion core + afterstate branch)',
        rule='stop after two consecutive snapshots whose pooled panel score does not exceed the best so far; '
             'select the best pooled snapshot', snapshots=[dict(
                 label='init', frames=0, checkpoint=str(DATA / 'init-arm-c.pt'), pooled=0.5, pooled_ci95=None,
                 note='not played: bitwise the champion function, so 0.500 by construction')])
    save(state)
    while not state.get('fired'):
        done = {s['label'] for s in state['snapshots']}
        listing = remote(f'ls {REMOTE}; python3 -c "import json;print(json.load(open(\'{REMOTE}/training.json\'))[\'status\'])"')
        names = sorted(n for n in listing.split() if n.startswith('core-f') and n.endswith('.pt'))
        status = listing.strip().splitlines()[-1] if listing.strip() else 'unknown'
        pending = [n for n in names if n[:-3].replace('core-', 'ppo-') not in done]
        if not pending:
            if status not in ('Running', 'unknown') and names:
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
        log('SNAPSHOT ' + summary(label, report) + f"; improved={improved} non-improving streak={streak}")
        if streak >= 2:
            state.update(fired='two consecutive non-improving snapshots')
            remote(f'tmux send-keys -t {SESSION} C-c')
        save(state)
    choice = max(state['snapshots'], key=lambda s: s['pooled'])  # ties keep the earlier snapshot
    state['selected'] = choice
    save(state)
    (OUT / 'selection.json').write_text(json.dumps(dict(reason=state['fired'], **choice), indent=1) + '\n')
    log(f"STOP RULE FIRED ({state['fired']}); arm C candidate {choice['label']} pooled {choice['pooled']:.3f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'panel':
        print(write_panel(sys.argv[2], sys.argv[3]))
    else:
        watch()


if __name__ == '__main__':
    main()
