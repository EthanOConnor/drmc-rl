"""Smoke test for DrMarioRetroEnv once Stable-Retro is wired.

Usage:
    python envs/retro/demo.py --mode pixel --level 0 --steps 1000 --risk-tau 0.5
"""
import argparse, time
from envs.retro.register_env import register_env_id
from gymnasium import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pixel', 'state'], default='pixel')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--risk-tau', type=float, default=0.5)
    ap.add_argument('--dump-state', action='store_true', help='Print state-plane stats (state mode)')
    args = ap.parse_args()

    register_env_id()
    env = make("DrMarioRetroEnv-v0", obs_mode=args.mode, level=args.level, risk_tau=args.risk_tau)
    obs, info = env.reset()
    t0 = time.time()
    steps, reward_sum = 0, 0.0
    for _ in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if args.dump_state and args.mode == 'state':
            core = obs['obs'] if isinstance(obs, dict) else obs
            # print non-zero counts per channel on the most recent frame in stack
            latest = core[-1]
            nz = [int(latest[c].sum()) for c in range(latest.shape[0])]
            print('nz per channel:', nz)
        reward_sum += r
        steps += 1
        if term or trunc:
            break
    dt = time.time() - t0
    fps = steps / max(dt, 1e-6)
    print(f"Ran {steps} steps, reward={reward_sum:.1f}, cleared={info.get('cleared', False)}, FPS≈{fps:.1f}")
    env.close()


if __name__ == '__main__':
    main()
