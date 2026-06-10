#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _infos_to_list(
    infos: Any, num_envs: int, previous: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    if isinstance(infos, (list, tuple)):
        out = [dict(i) if isinstance(i, dict) else {} for i in infos]
        if len(out) < int(num_envs):
            out.extend({} for _ in range(int(num_envs) - len(out)))
        return out[: int(num_envs)]
    if isinstance(infos, dict):
        return [infos for _ in range(int(num_envs))]
    if previous is not None:
        return previous
    return [{} for _ in range(int(num_envs))]


def _sample_action(
    info: Dict[str, Any],
    rng: np.random.Generator,
    action_space: Any,
    *,
    action_mode: str,
) -> int:
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
                if str(action_mode) == "first":
                    return int(idxs[0])
                return int(rng.choice(idxs))
        except Exception:
            pass
    if action_space is not None and hasattr(action_space, "sample"):
        try:
            return int(action_space.sample())
        except Exception:
            pass
    return int(rng.integers(0, 1))


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (float(percentile) / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _run_bench(
    cfg: VecEnvConfig,
    *,
    duration_sec: float,
    warmup_steps: int,
    seed: int,
    action_mode: str = "first",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    t_create0 = time.perf_counter()
    env = make_vec_env(cfg)
    env_create_sec = float(time.perf_counter() - t_create0)
    try:
        t_reset0 = time.perf_counter()
        obs, infos = env.reset(seed=seed)
        reset_sec = float(time.perf_counter() - t_reset0)
        _ = obs
        infos_list = _infos_to_list(infos, cfg.num_envs)

        action_space = getattr(env, "single_action_space", None)
        rng = np.random.default_rng(seed)

        t_warmup0 = time.perf_counter()
        for _ in range(max(0, warmup_steps)):
            actions = np.array(
                [
                    _sample_action(
                        infos_list[i], rng, action_space, action_mode=str(action_mode)
                    )
                    for i in range(cfg.num_envs)
                ],
                dtype=np.int64,
            )
            _, _, _, _, infos = env.step(actions)
            infos_list = _infos_to_list(infos, cfg.num_envs, infos_list)
        warmup_sec = float(time.perf_counter() - t_warmup0)

        frames_total = 0
        decisions_total = 0
        batches_total = 0
        action_select_sec = 0.0
        env_step_sec = 0.0
        info_extract_sec = 0.0
        batch_wall_ms: List[float] = []
        env_step_ms: List[float] = []
        t0 = time.perf_counter()
        while True:
            batch_t0 = time.perf_counter()
            t_action0 = time.perf_counter()
            actions = np.array(
                [
                    _sample_action(
                        infos_list[i], rng, action_space, action_mode=str(action_mode)
                    )
                    for i in range(cfg.num_envs)
                ],
                dtype=np.int64,
            )
            action_select_sec += float(time.perf_counter() - t_action0)

            t_step0 = time.perf_counter()
            _, _, _, _, infos = env.step(actions)
            step_elapsed = float(time.perf_counter() - t_step0)
            env_step_sec += step_elapsed
            env_step_ms.append(step_elapsed * 1000.0)

            t_info0 = time.perf_counter()
            infos_list = _infos_to_list(infos, cfg.num_envs, infos_list)
            tau_sum = 0
            for i in range(cfg.num_envs):
                info_i = infos_list[i] if i < len(infos_list) else {}
                tau_sum += _extract_tau(info_i)
            info_extract_sec += float(time.perf_counter() - t_info0)

            frames_total += int(tau_sum)
            decisions_total += int(cfg.num_envs)
            batches_total += 1
            batch_wall_ms.append(float(time.perf_counter() - batch_t0) * 1000.0)

            elapsed_so_far = time.perf_counter() - t0
            if (
                max_batches is not None
                and int(max_batches) > 0
                and batches_total >= int(max_batches)
            ):
                break
            if elapsed_so_far >= float(duration_sec) and batches_total > 0:
                break

        elapsed = max(1e-6, float(time.perf_counter() - t0))
        fps_total = float(frames_total) / elapsed
        dps_total = float(decisions_total) / elapsed
        fps_per_env = fps_total / float(max(1, cfg.num_envs))
        dps_per_env = dps_total / float(max(1, cfg.num_envs))
        return {
            "env_create_sec": env_create_sec,
            "reset_sec": reset_sec,
            "warmup_sec": warmup_sec,
            "elapsed_sec": elapsed,
            "batches_total": float(batches_total),
            "frames_total": float(frames_total),
            "decisions_total": float(decisions_total),
            "avg_tau": float(frames_total) / float(max(1, decisions_total)),
            "fps_total": fps_total,
            "fps_per_env": fps_per_env,
            "dps_total": dps_total,
            "dps_per_env": dps_per_env,
            "action_select_sec_total": float(action_select_sec),
            "env_step_sec_total": float(env_step_sec),
            "info_extract_sec_total": float(info_extract_sec),
            "action_select_ms_per_batch": (action_select_sec / max(1, batches_total)) * 1000.0,
            "env_step_ms_mean": statistics.fmean(env_step_ms) if env_step_ms else 0.0,
            "env_step_ms_p50": _percentile(env_step_ms, 50.0),
            "env_step_ms_p95": _percentile(env_step_ms, 95.0),
            "batch_wall_ms_mean": statistics.fmean(batch_wall_ms) if batch_wall_ms else 0.0,
            "batch_wall_ms_p50": _percentile(batch_wall_ms, 50.0),
            "batch_wall_ms_p95": _percentile(batch_wall_ms, 95.0),
            "info_extract_ms_per_batch": (info_extract_sec / max(1, batches_total)) * 1000.0,
            "harness_overhead_frac": (
                (action_select_sec + info_extract_sec) / elapsed if elapsed > 0 else 0.0
            ),
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _group_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(row["vectorization"]), int(row["num_envs"]), str(row["action_mode"]))


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda k: (k[0], k[2], k[1])):
        group = grouped[key]
        metric_names = [
            "fps_total",
            "fps_per_env",
            "dps_total",
            "dps_per_env",
            "avg_tau",
            "env_create_sec",
            "reset_sec",
            "warmup_sec",
            "elapsed_sec",
            "batches_total",
            "frames_total",
            "decisions_total",
            "action_select_ms_per_batch",
            "env_step_ms_mean",
            "env_step_ms_p50",
            "env_step_ms_p95",
            "batch_wall_ms_mean",
            "batch_wall_ms_p50",
            "batch_wall_ms_p95",
            "info_extract_ms_per_batch",
            "harness_overhead_frac",
        ]
        summary: Dict[str, Any] = {
            "vectorization": key[0],
            "num_envs": key[1],
            "action_mode": key[2],
            "repeats": len(group),
        }
        for name in metric_names:
            values = [float(row.get(name, 0.0)) for row in group]
            summary[f"{name}_mean"] = statistics.fmean(values) if values else 0.0
            summary[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(summary)

    baselines: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for row in out:
        if int(row["num_envs"]) == 1:
            baselines[(str(row["vectorization"]), str(row["action_mode"]))] = (
                1,
                float(row["fps_total_mean"]),
            )
    for row in out:
        key = (str(row["vectorization"]), str(row["action_mode"]))
        if key in baselines:
            continue
        same = [r for r in out if (str(r["vectorization"]), str(r["action_mode"])) == key]
        base_row = min(same, key=lambda r: int(r["num_envs"]))
        baselines[key] = (int(base_row["num_envs"]), float(base_row["fps_total_mean"]))
    for row in out:
        base = baselines.get((str(row["vectorization"]), str(row["action_mode"])))
        if base and base[1]:
            base_n, base_fps = base
            speedup = float(row["fps_total_mean"]) / float(base_fps)
            efficiency = speedup / (float(max(1, int(row["num_envs"]))) / float(max(1, base_n)))
        else:
            speedup = 0.0
            efficiency = 0.0
        row["speedup"] = speedup
        row["efficiency"] = efficiency
    return out


def _write_json(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
) -> None:
    payload = {
        "schema": "drmc-rl.bench_multienv.v2",
        "args": vars(args),
        "rows": rows,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows_list for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-env scaling for Dr. Mario")
    parser.add_argument("--env-id", default="DrMarioPlacementEnv-v0")
    parser.add_argument("--backend", default="cpp-pool")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--state-repr", default="bitplane_bottle_conn_mask")
    parser.add_argument("--num-envs", default="1,2,4,8,16")
    parser.add_argument("--vectorization", default="sync", choices=["sync", "async", "both"])
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap on measured env.step batches per row (0 means duration-based).",
    )
    parser.add_argument(
        "--action-mode",
        choices=["first", "random"],
        default="first",
        help="'first' minimizes action-selection overhead; 'random' samples a feasible action.",
    )
    parser.add_argument(
        "--emit-raw-ram", action="store_true", help="Include raw_ram in info payloads"
    )
    parser.add_argument("--json-out", type=str, default=None, help="Write detailed rows + summary")
    parser.add_argument("--csv-out", type=str, default=None, help="Write detailed rows as CSV")
    args = parser.parse_args()

    num_envs_list = _parse_num_envs(args.num_envs)
    vectorizations = ["sync", "async"] if args.vectorization == "both" else [args.vectorization]

    results = []
    for vec in vectorizations:
        for n in num_envs_list:
            for repeat in range(max(1, int(args.repeats))):
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
                    seed=int(args.seed) + repeat,
                    action_mode=str(args.action_mode),
                    max_batches=int(args.max_batches) if int(args.max_batches) > 0 else None,
                )
                results.append(
                    {
                        "vectorization": vec,
                        "num_envs": int(n),
                        "repeat": int(repeat),
                        "seed": int(args.seed) + repeat,
                        "action_mode": str(args.action_mode),
                        **metrics,
                    }
                )

    summary = _summarize(results)

    if args.json_out:
        _write_json(Path(args.json_out), args=args, rows=results, summary=summary)
    if args.csv_out:
        _write_csv(Path(args.csv_out), results)

    header = [
        "vectorization",
        "num_envs",
        "repeats",
        "action",
        "fps_mean",
        "fps_std",
        "fps/env",
        "speedup",
        "eff",
        "dps_mean",
        "avg_tau",
        "step_ms",
        "p95_ms",
        "overhead%",
    ]
    print(" ".join(f"{h:>12}" for h in header))
    for row in summary:
        print(
            f"{row['vectorization']:>12}"
            f"{int(row['num_envs']):>12}"
            f"{int(row['repeats']):>12}"
            f"{row['action_mode']:>12}"
            f"{row['fps_total_mean']:>12.1f}"
            f"{row['fps_total_std']:>12.1f}"
            f"{row['fps_per_env_mean']:>12.1f}"
            f"{row['speedup']:>12.2f}"
            f"{row['efficiency']:>12.2f}"
            f"{row['dps_total_mean']:>12.1f}"
            f"{row['avg_tau_mean']:>12.1f}"
            f"{row['env_step_ms_mean_mean']:>12.3f}"
            f"{row['env_step_ms_p95_mean']:>12.3f}"
            f"{100.0 * row['harness_overhead_frac_mean']:>12.1f}"
        )


if __name__ == "__main__":
    main()
