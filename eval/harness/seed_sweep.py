"""Seed sweep evaluator.

Loads seeds from envs/retro/seeds/registry.json and runs N episodes per seed,
computing E[T], Var[T], success %, and CVaR_α. Writes Parquet and CSV.

This script assumes your Gym env id is 'DrMarioRetroEnv-v0' or that you have
a local `make_env()` helper.
"""

import argparse, json, os
from typing import Any, List
import numpy as np
import pandas as pd


def cvar(x: np.ndarray, alpha: float = 0.25, minimize: bool = True) -> float:
    if minimize:
        q = np.quantile(x, alpha)
        return x[x <= q].mean() if (x <= q).any() else float(q)
    else:
        q = np.quantile(x, 1 - alpha)
        return x[x >= q].mean() if (x >= q).any() else float(q)


def run_eval(env_ctor, seeds, episodes_per_seed: int = 100, alpha_list=(0.05, 0.25)):
    rows = []
    for seed in seeds:
        Ts: List[int] = []
        succ = 0
        env = env_ctor(seed=seed)
        for _ in range(episodes_per_seed):
            obs, info = env.reset(seed=seed)
            done = False
            t = 0
            while not done:
                action = env.action_space.sample()  # replace with policy
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                t += 1
            Ts.append(info.get('t', t))
            succ += 1 if (info.get('viruses_remaining', 1) == 0 and terminated) else 0
        T = np.array(Ts, dtype=np.float32)
        row = {
            'seed': seed,
            'episodes': episodes_per_seed,
            'E_T': float(T.mean()),
            'Var_T': float(T.var()),
            'success_rate': succ / episodes_per_seed,
        }
        for a in alpha_list:
            row[f'CVaR_{int(a * 100)}'] = float(cvar(T, alpha=a, minimize=True))
        rows.append(row)
        env.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=str, default='envs/retro/seeds/registry.json')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--out', type=str, default='eval/seed_metrics.parquet')
    args = ap.parse_args()

    with open(args.registry, 'r') as f:
        reg = json.load(f)
    # Support either a dict of ids or a list of seed dicts
    if isinstance(reg, dict) and 'seeds' in reg:
        seeds = [s['id'] for s in reg['seeds']]
    elif isinstance(reg, dict):
        seeds = list(reg.keys())
    else:
        seeds = [e['id'] for e in reg]

    try:
        from eval.harness.seed_sweep_env_ctor import make_env as env_ctor
    except Exception as e:
        raise RuntimeError('Please provide eval.harness.seed_sweep_env_ctor.make_env') from e

    df = run_eval(env_ctor, seeds, episodes_per_seed=args.episodes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_csv(args.out.replace('.parquet', '.csv'), index=False)
    print(f'Wrote {args.out} and CSV twin.')


if __name__ == '__main__':
    main()
