from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from envs.retro import DrMarioRetroEnv


def cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return np.nan
    cutoff = np.quantile(values, alpha)
    return values[values <= cutoff].mean()


@dataclass
class EvalResult:
    mean: float
    var: float
    cvar5: float
    cvar25: float
    success_rate: float


def run_seed(env_kwargs: Dict, episodes: int = 100) -> EvalResult:
    env = DrMarioRetroEnv(**env_kwargs)
    Ts: List[int] = []
    successes = 0
    for _ in range(episodes):
        obs, info = env.reset()
        t = 0
        done = False
        trunc = False
        while not (done or trunc):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            t += 1
        Ts.append(t)
        if done and info.get("viruses_remaining", 1) == 0:
            successes += 1
    T = np.array(Ts, dtype=np.float32)
    return EvalResult(
        mean=float(T.mean()),
        var=float(T.var()),
        cvar5=float(cvar(T, 0.05)),
        cvar25=float(cvar(T, 0.25)),
        success_rate=float(successes / episodes),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--obs-mode", choices=["pixel", "state"], default="state")
    args = ap.parse_args()

    res = run_seed({"obs_mode": args.obs_mode}, episodes=args.episodes)
    print("E[T]:", res.mean)
    print("Var[T]:", res.var)
    print("CVaR5%:", res.cvar5)
    print("CVaR25%:", res.cvar25)
    print("Success%:", 100.0 * res.success_rate)


if __name__ == "__main__":
    main()
