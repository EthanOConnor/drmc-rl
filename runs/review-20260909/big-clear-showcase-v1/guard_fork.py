"""Guard-hit follow-on: snapshot the last accepted state and write the free-lineage fork config.

  python guard_fork.py RUN_DIR FREE_DIR [--accept-limit 10] [--every 12500000]

Reads RUN_DIR/retention-guard-hit.json (written by train_controller_retention with
stop_on_retention_rejection), writes RUN_DIR/core-guard-hit-uNNNNN.pt (the last accepted
update's model without optimizer state, loadable like a core-f snapshot), copies the
journal and the checkpoint into FREE_DIR/fork/, and writes FREE_DIR/finetune-free.json:
the run's config forked from that checkpoint with the retention acceptance limit lifted
(retention_accept_limit), the hinge loss, its pressure scale, max_update_kl and parent-KL unchanged.
Prints the guard-hit snapshot path.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run'); p.add_argument('free')
    p.add_argument('--accept-limit', type=float, default=10.0)
    p.add_argument('--every', type=int, default=12_500_000)
    a = p.parse_args()
    run, free = Path(a.run), Path(a.free)
    hit = json.loads((run / 'retention-guard-hit.json').read_text())
    source = Path(hit['last_accepted'])
    state = torch.load(source, map_location='cpu', weights_only=True)
    snapshot = run / f"core-guard-hit-u{int(hit['last_accepted_update']):05d}.pt"
    torch.save({k: v for k, v in state.items() if k not in ('optimizer', 'sampling_rng')}, snapshot)
    (free / 'fork').mkdir(parents=True, exist_ok=True); (free / 'tmp').mkdir(exist_ok=True)
    fork = free / 'fork' / source.name
    shutil.copyfile(source, fork)
    shutil.copyfile(run / 'training-games.jsonl', free / 'fork' / 'training-games.jsonl')
    config = dict(state['training_config'])
    config.pop('resume', None)
    config.update(output=str(free), stop_on_retention_rejection=False, retention_accept_limit=a.accept_limit,
                  checkpoint_every_frames=a.every,
                  fork=dict(checkpoint=str(fork), sha256=hashlib.sha256(fork.read_bytes()).hexdigest(),
                            journal=str(free / 'fork' / 'training-games.jsonl')))
    (free / 'finetune-free.json').write_text(json.dumps(config, indent=1) + '\n')
    print(snapshot)


if __name__ == '__main__':
    main()
