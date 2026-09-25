"""Big-clear setups: the two retention-PPO fine-tune configs (arm a standard, arm b shaped) from one parent.

Copies arm A's PPO config (``/dev/shm/afterstate-core/ppo-v1.json`` on tf3090)
and changes only:

* ``checkpoint``: the parent snapshot (pre-registered: arm A ``core-f00100000000.pt``);
* ``start_mix``: the big-clear training bank, decaying share (``share0`` 0.30,
  ``half_life_frames`` 25M, ``cutoff`` 0.02), level-14 collections, all paces,
  ``replay_share`` 0.5 (half of the mixed pairs replay the source game's pills);
* ``target_decisions`` 1,000,000 (about 100M frames; per-pace floor scaled),
  ``checkpoint_every_frames`` 25M, ``keep_update_checkpoints`` (every checkpoint
  is kept for style-lever work), ``journal_showiness`` (per-game clear tiers);
* a fresh ``seed`` shared by both arms (identical collection schedules until
  the policies diverge), ``output`` under ``big-clear-v1/``, the source commit,
  natives on the trainer-output mount;
* arm (b) only: ``showiness_bonus`` (``drmc_rl.training.showiness``).

Retention anchors, KL limits, opponent pool, objective, holdouts and every
other key are inherited unchanged. No ``seed_reserve: legacy``: new outputs
exclude the evaluation reserve.

  python runs/review-20260909/prepare_big_clear_finetune_v1.py
"""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output'
ROOT = f'{REMOTE}/big-clear-v1'
PARENT_CONFIG = '/dev/shm/afterstate-core/ppo-v1.json'
PARENT = f'{REMOTE}/afterstate-core-v1/ppo-v1/core-f00100000000.pt'
PARENT_ENTRANT = 'armA-ppo-v1-f00100000000'
BANK = f'{ROOT}/bank/big-clear-train-v1.npz'
LOCAL_BANK = Path('/Users/ethan/dev/drmario/drmc-rl-bigclear-data/bank/big-clear-train-v1.npz')
SEED = 20260926
TARGET_DECISIONS = 1_000_000
START_MIX = dict(share0=0.30, half_life_frames=25_000_000, floor=0.0, cutoff=0.02, levels=[14], paces=[],
                 replay_share=0.5)
BONUS = dict(threshold=20.0, base=0.05, per_point=0.005, event_cap=0.15, game_cap=0.30)
ARMS = {'std': {}, 'shaped': {'showiness_bonus': BONUS}}


def remote(*args):
    return subprocess.run(['ssh', 'tf3090', *args], check=True, capture_output=True, text=True).stdout


def main():
    sha = remote('sha256sum', BANK).split()[0]
    if hashlib.sha256(LOCAL_BANK.read_bytes()).hexdigest() != sha:
        raise SystemExit('tf3090 bank differs from the local big-clear-train-v1.npz')
    parent = json.loads(remote('cat', PARENT_CONFIG))
    parent_sha = remote('sha256sum', PARENT).split()[0]
    commit = subprocess.run(['git', '-C', str(HERE.parents[1]), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    scale = TARGET_DECISIONS / parent['target_decisions']
    out_dir = HERE / 'big-clear-v1'
    out_dir.mkdir(exist_ok=True)
    for arm, extra in ARMS.items():
        config = {k: v for k, v in parent.items() if k not in ('resume', 'fork', 'seed_reserve')}
        config.update(
            checkpoint=PARENT,
            parent_sha256=parent_sha,
            parent_entrant=PARENT_ENTRANT,
            output=f'{ROOT}/{arm}',
            native_library=f'{ROOT}/native/libdrmario_pool.so',
            seed=SEED,
            start_mix=dict(bank=BANK, bank_sha256=sha, **START_MIX),
            target_decisions=TARGET_DECISIONS,
            minimum_decisions_per_pace=int(parent['minimum_decisions_per_pace'] * scale),
            milestone_decisions=[],
            checkpoint_every_frames=25_000_000,
            keep_update_checkpoints=True,
            journal_showiness=True,
            source_commit=commit,
            **extra,
        )
        path = out_dir / f'finetune-{arm}.json'
        path.write_text(json.dumps(config, indent=1) + '\n')
        print(path)


if __name__ == '__main__':
    main()
