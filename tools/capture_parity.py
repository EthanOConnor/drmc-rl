#!/usr/bin/env python3
"""Capture parity fixtures (RAM + planes) over seeds for later simulator tests.

Usage:
  python tools/capture_parity.py --episodes 3 --steps 256 --out re/out/parity --mode state \
      [--frame-offset 0]

Notes:
- Requires Stable-Retro environment available for RAM capture; will fallback to state-only dumps.
"""
import argparse, os, json
import numpy as np
from pathlib import Path
from envs.retro.register_env import register_env_id
import gymnasium as gym


def save_np(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--out', type=str, default='re/out/parity')
    ap.add_argument('--mode', choices=['state', 'pixel'], default='state')
    ap.add_argument('--frame-offset', type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    register_env_id()

    env = gym.make('DrMarioRetroEnv-v0', obs_mode=args.mode, level=0)

    for ep in range(args.episodes):
        obs, info = env.reset(options={'frame_offset': args.frame_offset})
        # Try to detect raw RAM access via the underlying retro env
        retro = getattr(env.unwrapped, '_retro', None)
        using_retro = getattr(env.unwrapped, '_using_retro', False) and retro is not None
        for t in range(args.steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            core = obs if isinstance(obs, np.ndarray) else obs.get('obs', None)
            if core is not None:
                save_np(outdir / f'ep{ep:02d}_t{t:04d}_planes.npy', core[-1] if core.ndim == 4 else core)
            if using_retro:
                try:
                    ram = retro.get_ram()
                    (outdir / 'ram').mkdir(exist_ok=True)
                    (outdir / 'meta').mkdir(exist_ok=True)
                    (outdir / 'ram' / f'ep{ep:02d}_t{t:04d}.bin').write_bytes(bytes(ram))
                except Exception:
                    pass
            if term or trunc:
                break
    env.close()
    print(f"Captured up to {args.episodes} episodes x {args.steps} steps into {outdir}")


if __name__ == '__main__':
    main()

