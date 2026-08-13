"""Dump raw RAM and mapped state tensors for N frames to disk."""
import argparse
import os

import gymnasium as gym
import numpy as np

from drmc_rl.envs.libretro.registration import register_env_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--outdir', type=str, default='debug_dumps')
    ap.add_argument('--mode', choices=['state', 'pixel'], default='state')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    register_env_id()
    env = gym.make('DrMarioLibretroEnv-v0', obs_mode=args.mode, level=0)
    obs, info = env.reset()
    for i in range(args.frames):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        core = obs if isinstance(obs, np.ndarray) else obs.get('obs', None)
        if core is not None:
            np.save(os.path.join(args.outdir, f'state_{i:04d}.npy'), core[-1] if core.ndim == 4 else core)
        if term or trunc:
            break
    env.close()
    print(f"Dumped up to {i+1} frames into {args.outdir}")


if __name__ == '__main__':
    main()

