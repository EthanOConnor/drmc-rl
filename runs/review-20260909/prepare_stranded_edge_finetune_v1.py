"""Stranded edge-virus curriculum: a retention-PPO fine-tune, or a start_mix block to fold into another run.

``finetune PARENT_CONFIG CHECKPOINT OUTPUT``: copy a finished (or running)
``trainer-controller-retention`` config from tf3090 (for example the
tournament winner's ``afterstate-core-*/ppo-v1/config.json``) and change only:

* ``checkpoint``: the winning core (tf3090 path);
* ``start_mix``: the stranded-edge training bank at ``--fraction`` of each
  collection's seed pairs (level-14 collections, all paces);
* the decision budget (``--target-decisions``, per-pace floor scaled with it),
  a fresh ``seed`` and ``output``, the current source commit;
* no ``seed_reserve: legacy``: a new output always excludes the evaluation
  reserve, including the 96 stranded-edge benchmark games.

Retention anchors, KL limits (``parent_kl``, ``max_update_kl``,
``max_anchor_kl_increase``, ``retention_coefficient``), opponent pool, objective
contract, holdouts and ``checkpoint_every_frames`` are inherited unchanged.

``fold CONFIG``: print the ``start_mix`` block to add to the next scheduled PPO
config instead of running a separate fine-tune.

The bank must already be on tf3090 at ``--bank``; its sha256 is pinned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output'
BANK = f'{REMOTE}/stranded-edge-v1/stranded-edge-train-v1.npz'
LOCAL_BANK = Path('/Users/ethan/dev/drmario/drmc-rl-stranded-edge-data/bank/stranded-edge-train-v1.npz')


def remote_json(path):
    return json.loads(subprocess.run(['ssh', 'tf3090', 'cat', path], check=True, capture_output=True, text=True).stdout)


def remote_sha(path):
    out = subprocess.run(['ssh', 'tf3090', 'sha256sum', path], check=True, capture_output=True, text=True).stdout
    return out.split()[0]


def start_mix(args):
    sha = remote_sha(args.bank)
    if LOCAL_BANK.exists() and hashlib.sha256(LOCAL_BANK.read_bytes()).hexdigest() != sha:
        raise SystemExit('tf3090 bank differs from the local stranded-edge-train-v1.npz')
    return dict(bank=args.bank, bank_sha256=sha, fraction=args.fraction, levels=[14], paces=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('finetune')
    f.add_argument('parent_config')
    f.add_argument('checkpoint')
    f.add_argument('output')
    f.add_argument('--target-decisions', type=int, default=1_000_000)
    f.add_argument('--seed', type=int, default=20260925)
    for p in (f, sub.add_parser('fold')):
        p.add_argument('--bank', default=BANK)
        p.add_argument('--fraction', type=float, default=0.15)
    args = parser.parse_args()
    mix = start_mix(args)
    if args.command == 'fold':
        print(json.dumps(dict(start_mix=mix), indent=1))
        return
    parent = remote_json(args.parent_config)
    config = {k: v for k, v in parent.items() if k not in ('resume', 'seed_reserve')}
    scale = args.target_decisions / parent['target_decisions']
    config.update(
        checkpoint=args.checkpoint,
        output=args.output,
        seed=args.seed,
        start_mix=mix,
        target_decisions=args.target_decisions,
        minimum_decisions_per_pace=int(parent['minimum_decisions_per_pace'] * scale),
        milestone_decisions=[],
        source_commit=subprocess.run(['git', '-C', str(HERE.parents[1]), 'rev-parse', '--short', 'HEAD'],
                                     capture_output=True, text=True).stdout.strip(),
    )
    path = HERE / 'stranded-edge-v1' / f'finetune-{Path(args.output).name}.json'
    path.parent.mkdir(exist_ok=True)
    assert not path.exists(), f'{path.name} already written'
    path.write_text(json.dumps(config, indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    main()
