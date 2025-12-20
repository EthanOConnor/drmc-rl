#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

import numpy as np

from training.envs.dr_mario_vec import VecEnvConfig, make_vec_env


def _parse_num_envs(value: str) -> List[int]:
    if not value:
        return [1, 2, 4, 8, 16]
    items = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return items if items else [1, 2, 4, 8, 16]


def _extract_tau(info: Dict[str, Any]) -> int:
    if not isinstance(info, dict):
        return 1
    tau = info.get("placements/tau", 1)
    if isinstance(tau, np.ndarray):
        try:
            tau = tau.item()
        except Exception:
            return 1
    try:
        return max(1, int(tau))
    except Exception:
        return 1


def _sample_action(info: Dict[str, Any], rng: np.random.Generator, action_space: Any) -> int:
    mask = None
    for key in ("placements/feasible_mask", "placements/legal_mask", "mask"):
        mask = info.get(key)
        if mask is not None:
            break
    if mask is not None:
        try:
            m = np.asarray(mask).reshape(-1)
            idxs = np.flatnonzero(m)
            if idxs.size > 0:
                return int(rng.choice(idxs))
        except Exception:
            pass
    if action_space is not None and hasattr(action_space, "sample"):
        try:
            return int(action_space.sample())
        except Exception:
            pass
    return int(rng.integers(0, 1))


def _run_bench(
    cfg: VecEnvConfig, *, duration_sec: float, warmup_steps: int, seed: int
) -> Dict[str, float]:
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=seed)
        if isinstance(infos, (list, tuple)):
            infos_list = list(infos)
        else:
            infos_list = [infos for _ in range(cfg.num_envs)]

        action_space = getattr(env, "single_action_space", None)
        rng = np.random.default_rng(seed)

        # Warmup steps
        for _ in range(max(0, warmup_steps)):
            actions = np.array(
                [_sample_action(infos_list[i], rng, action_space) for i in range(cfg.num_envs)],
                dtype=np.int64,
            )
            _, _, _, _, infos = env.step(actions)
            infos_list = list(infos) if isinstance(infos, (list, tuple)) else infos_list

        frames_total = 0
        decisions_total = 0
        t0 = time.perf_counter()
        while True:
            actions = np.array(
                [_sample_action(infos_list[i], rng, action_space) for i in range(cfg.num_envs)],
                dtype=np.int64,
            )
            _, _, _, _, infos = env.step(actions)
            infos_list = list(infos) if isinstance(infos, (list, tuple)) else infos_list

            tau_sum = 0
            for i in range(cfg.num_envs):
                info_i = infos_list[i] if i < len(infos_list) else {}
                tau_sum += _extract_tau(info_i)
            frames_total += int(tau_sum)
            decisions_total += int(cfg.num_envs)

            if time.perf_counter() - t0 >= duration_sec:
                break

        elapsed = max(1e-6, float(time.perf_counter() - t0))
        fps_total = float(frames_total) / elapsed
        dps_total = float(decisions_total) / elapsed
        fps_per_env = fps_total / float(max(1, cfg.num_envs))
        dps_per_env = dps_total / float(max(1, cfg.num_envs))
        return {
            "fps_total": fps_total,
            "fps_per_env": fps_per_env,
            "dps_total": dps_total,
            "dps_per_env": dps_per_env,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-env scaling for Dr. Mario")
    parser.add_argument("--env-id", default="DrMarioPlacementEnv-v0")
    parser.add_argument("--backend", default="cpp-engine")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--state-repr", default="bitplane_bottle")
    parser.add_argument("--num-envs", default="1,2,4,8,16")
    parser.add_argument("--vectorization", default="both", choices=["sync", "async", "both"])
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--emit-raw-ram", action="store_true", help="Include raw_ram in info payloads"
    )
    args = parser.parse_args()

    num_envs_list = _parse_num_envs(args.num_envs)
    vectorizations = ["sync", "async"] if args.vectorization == "both" else [args.vectorization]

    results = []
    for vec in vectorizations:
        for n in num_envs_list:
            cfg = VecEnvConfig(
                id=args.env_id,
                obs_mode=args.obs_mode,
                num_envs=int(n),
                frame_stack=1,
                render=False,
                randomize_rng=True,
                backend=args.backend,
                state_repr=args.state_repr,
                vectorization=vec,
                emit_raw_ram=bool(args.emit_raw_ram),
            )
            metrics = _run_bench(
                cfg,
                duration_sec=float(args.duration_sec),
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            )
            results.append(
                {
                    "vectorization": vec,
                    "num_envs": n,
                    **metrics,
                }
            )

    # Compute speedup/efficiency per vectorization.
    baselines = {}
    for row in results:
        if row["num_envs"] == 1:
            baselines[row["vectorization"]] = row["fps_total"]

    header = (
        "vectorization",
        "num_envs",
        "fps_total",
        "fps/env",
        "speedup",
        "efficiency",
        "dps_total",
    )
    print(" ".join(f"{h:>12}" for h in header))
    for row in results:
        base = baselines.get(row["vectorization"])
        speedup = (row["fps_total"] / base) if base else 0.0
        efficiency = (speedup / float(row["num_envs"])) if base else 0.0
        print(
            f"{row['vectorization']:>12}"
            f"{int(row['num_envs']):>12}"
            f"{row['fps_total']:>12.1f}"
            f"{row['fps_per_env']:>12.1f}"
            f"{speedup:>12.2f}"
            f"{efficiency:>12.2f}"
            f"{row['dps_total']:>12.1f}"
        )


if __name__ == "__main__":
    main()
