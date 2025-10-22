#!/usr/bin/env python3
"""Automated smoke test for the placement planner/translator stack."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym

from envs.retro.register_env import register_env_id, register_placement_env_id
from envs.retro.placement_wrapper import DrMarioPlacementEnv


@dataclass
class PlacementStats:
    episodes: int = 0
    placements: int = 0
    rewards: List[float] = field(default_factory=list)
    plan_latencies_ms: List[float] = field(default_factory=list)
    options: List[int] = field(default_factory=list)
    replan_triggers: int = 0
    replan_failures: int = 0

    def record_step(self, info: Dict[str, float], reward: float, latency_ms: float, options: int) -> None:
        self.placements += 1
        self.rewards.append(float(reward))
        if not math.isnan(latency_ms):
            self.plan_latencies_ms.append(latency_ms)
        self.options.append(int(options))
        if info.get("placements/replan_triggered"):
            self.replan_triggers += 1
        if info.get("placements/replan_fail"):
            self.replan_failures += int(info.get("placements/replan_fail", 0))

    def summary(self) -> Dict[str, float]:
        def safe_mean(values: List[float]) -> float:
            return float(statistics.mean(values)) if values else float("nan")

        def safe_std(values: List[float]) -> float:
            return float(statistics.stdev(values)) if len(values) > 1 else 0.0

        return {
            "episodes": float(self.episodes),
            "placements": float(self.placements),
            "avg_reward": safe_mean(self.rewards),
            "avg_plan_latency_ms": safe_mean(self.plan_latencies_ms),
            "plan_latency_std_ms": safe_std(self.plan_latencies_ms),
            "avg_options": safe_mean(self.options),
            "replan_triggers": float(self.replan_triggers),
            "replan_failures": float(self.replan_failures),
        }


def choose_action(mask: np.ndarray, greedy: bool) -> int:
    available = np.flatnonzero(mask)
    if available.size == 0:
        return 0
    if greedy:
        return int(available[0])
    return int(np.random.choice(available))


def run_episode(
    env: DrMarioPlacementEnv,
    stats: PlacementStats,
    max_steps: int,
    greedy: bool,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    translator = env._translator  # type: ignore[attr-defined]
    done = False
    step = 0
    episode_reward = 0.0
    while not done and step < max_steps:
        trans_info = translator.info()
        diag = translator.diagnostics()
        mask = trans_info["placements/feasible_mask"]
        action = choose_action(mask, greedy)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = bool(terminated or truncated)
        step += 1
        episode_reward += float(reward)
        stats.record_step(step_info, reward, float(diag.get("plan_latency_ms", float("nan"))), int(trans_info["placements/options"]))
        obs = next_obs
        info = step_info
    stats.episodes += 1


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Placement planner diagnostic runner")
    parser.add_argument("--env-id", default="DrMarioRetroEnv-v0", help="Base Gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run")
    parser.add_argument("--max-steps", type=int, default=120, help="Placement decisions per episode")
    parser.add_argument("--greedy", action="store_true", help="Always take the first feasible action")
    parser.add_argument("--seed", type=int, default=None, help="Seed passed to env.reset")
    args = parser.parse_args(argv)

    register_env_id(args.env_id)
    register_placement_env_id()

    base_env = gym.make(args.env_id, obs_mode="state")
    env = DrMarioPlacementEnv(base_env)

    stats = PlacementStats()
    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        run_episode(env, stats, args.max_steps, args.greedy, seed=ep_seed)

    summary = stats.summary()
    print("Placement diagnostics summary")
    for key, value in summary.items():
        if math.isnan(value):
            print(f"  {key}: n/a")
        else:
            print(f"  {key}: {value:.3f}")

    env.close()


if __name__ == "__main__":
    main()
