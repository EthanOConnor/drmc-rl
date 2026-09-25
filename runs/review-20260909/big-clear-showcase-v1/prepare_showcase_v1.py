"""Big-clear showcase: two shaped fine-tunes for a streamer demo (pre-registration addendum 1).

Both copy arm A's PPO config (captured in armA-ppo-v1.base.json) and change only:
checkpoint (candidate 1: the shipped champion retention-mixed-v2, fresh optimizer;
candidate 2: arm A f100M, fresh optimizer), output, a shared seed, the big-clear
start bank at a CONSTANT 0.40 of level-14 seed pairs (replay_share 0.5, rows
weighted 1/3/6 by target tier T1/T2/T3), lr 1e-5 steered to an update KL of 0.005, the
raised showiness bonus (per clear min(0.20, 0.07 + 0.007 x (score - 20)), cap
0.45 per game), 25M-frame snapshots, every checkpoint kept, per-game showiness,
fill_inference_batches and deferred_reference. Retention and KL guards unchanged.
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/home/ethan/.cache/drmc-rl/trainer-output'
ROOT = f'{REMOTE}/big-clear-showcase-v1'
BANK = f'{REMOTE}/big-clear-v1/bank/big-clear-train-v1.npz'
BANK_SHA = '96cf5e45bdb0d083eaa4f5f74c5f0bfdf4591c21d338dc39c8485f29adffce8b'
PARENTS = {
    'champ': f'{REMOTE}/review-20260909/controller-retention-mixed-v2/core-final-inference.pt',
    'armA': f'{REMOTE}/afterstate-core-v1/ppo-v1/core-f00100000000.pt',
}
ENTRANTS = {'champ': 'champion-retention-mixed-v2', 'armA': 'armA-ppo-v1-f00100000000'}
BONUS = dict(threshold=20.0, base=0.07, per_point=0.007, event_cap=0.20, game_cap=0.45)
START_MIX = dict(fraction=0.40, levels=[14], paces=[], replay_share=0.5, tier_weights=[1, 3, 6])
# Arm A's steps barely move the policy (update KL ~0.0015); start at 1e-5 and steer update KL to 0.005.
LR = 1e-5
LR_KL_TARGET = dict(target=0.005, alarm=0.015, min_lr=1e-6, max_lr=3e-5)


def main():
    base = json.loads((HERE / 'armA-ppo-v1.base.json').read_text())
    commit = (HERE / 'COMMIT').read_text().strip() if (HERE / 'COMMIT').exists() else 'uncommitted'
    for name, ckpt in PARENTS.items():
        c = {k: v for k, v in base.items() if k not in ('resume', 'fork', 'seed_reserve')}
        c.update(checkpoint=ckpt, parent_entrant=ENTRANTS[name], output=f'{ROOT}/{name}',
                 native_library=f'{ROOT}/native/libdrmario_pool.so', seed=20260925,
                 start_mix=dict(bank=BANK, bank_sha256=BANK_SHA, **START_MIX),
                 showiness_bonus=BONUS, target_decisions=3_000_000,
                 minimum_decisions_per_pace=base['minimum_decisions_per_pace'], milestone_decisions=[],
                 checkpoint_every_frames=25_000_000, keep_update_checkpoints=True, journal_showiness=True,
                 fill_inference_batches=True, deferred_reference=True, source_commit=commit,
                 lr=LR, lr_kl_target=LR_KL_TARGET)
        (HERE / f'finetune-{name}.json').write_text(json.dumps(c, indent=1) + '\n')
        print(HERE / f'finetune-{name}.json')


if __name__ == '__main__':
    main()
