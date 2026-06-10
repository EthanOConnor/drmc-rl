#!/usr/bin/env python3
"""Benchmark placement policy network components.

This is intentionally separate from `tools.bench_multienv`: the env benchmark
measures simulator/planner throughput, while this tool measures policy forward
cost, candidate-packing overhead, parameter counts, and decisions/sec.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from models.policy.candidate_packing import pack_feasible_candidates
from models.policy.candidate_policy import CandidatePlacementPolicyNet
from models.policy.placement_heads import PlacementPolicyNet

GRID_H = 16
GRID_W = 8
MACRO_ACTIONS = 4 * GRID_H * GRID_W


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


def _count_params(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _sync_device(device: Any) -> None:
    if torch is None:
        return
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif str(device).startswith("mps") and getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
    except Exception:
        pass


def _preview_colors(info: Dict[str, Any]) -> np.ndarray:
    preview = info.get("preview_pill")
    if isinstance(preview, dict):
        return np.asarray(
            [int(preview.get("first_color", 0)), int(preview.get("second_color", 0))],
            dtype=np.int64,
        )
    arr = np.asarray(preview if preview is not None else [0, 0], dtype=np.int64).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    return np.zeros((2,), dtype=np.int64)


def _collect_cpp_pool_batch(
    batch_size: int, seed: int, *, state_repr: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    from training.envs.dr_mario_vec import VecEnvConfig, make_vec_env

    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=int(batch_size),
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-pool",
        state_repr=str(state_repr),
        vectorization="sync",
        emit_raw_ram=False,
    )
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=int(seed))
        return np.asarray(obs, dtype=np.float32), [dict(i) for i in infos]
    finally:
        if hasattr(env, "close"):
            env.close()


def _valid_macro_actions() -> np.ndarray:
    actions: List[int] = []
    for orient in range(4):
        dr = [0, 1, 0, -1][orient]
        dc = [1, 0, -1, 0][orient]
        for row in range(GRID_H):
            for col in range(GRID_W):
                row2 = row + dr
                col2 = col + dc
                if 0 <= row2 < GRID_H and 0 <= col2 < GRID_W:
                    actions.append(int(orient * GRID_H * GRID_W + row * GRID_W + col))
    return np.asarray(actions, dtype=np.int32)


def _synthetic_batch(
    batch_size: int, seed: int, feasible_count: int, channels: int = 12
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    obs = rng.random((int(batch_size), int(channels), GRID_H, GRID_W), dtype=np.float32)
    valid_actions = _valid_macro_actions()
    infos: List[Dict[str, Any]] = []
    for _ in range(int(batch_size)):
        mask = np.zeros((MACRO_ACTIONS,), dtype=np.uint8)
        count = int(max(1, min(valid_actions.size, int(feasible_count))))
        idx = rng.choice(valid_actions, size=count, replace=False)
        mask[idx] = 1
        costs = np.full((MACRO_ACTIONS,), 0xFFFF, dtype=np.uint16)
        costs[idx] = rng.integers(12, 160, size=count, dtype=np.uint16)
        pill = rng.integers(0, 3, size=(2,), dtype=np.int64)
        preview = rng.integers(0, 3, size=(2,), dtype=np.int64)
        infos.append(
            {
                "placements/feasible_mask": mask.reshape(4, GRID_H, GRID_W),
                "placements/cost_to_lock": costs.reshape(4, GRID_H, GRID_W),
                "next_pill_colors": pill,
                "preview_pill": {
                    "first_color": int(preview[0]),
                    "second_color": int(preview[1]),
                    "rotation": 0,
                },
            }
        )
    return obs, infos


def _batch_tensors(
    obs: np.ndarray,
    infos: List[Dict[str, Any]],
    *,
    device: Any,
    candidate_max: int,
) -> Dict[str, Any]:
    masks = np.stack(
        [np.asarray(info["placements/feasible_mask"], dtype=np.uint8) for info in infos], axis=0
    )
    costs = np.stack(
        [np.asarray(info["placements/cost_to_lock"]) for info in infos],
        axis=0,
    )
    pill = np.stack(
        [np.asarray(info.get("next_pill_colors", [0, 0]), dtype=np.int64)[:2] for info in infos],
        axis=0,
    )
    preview = np.stack([_preview_colors(info) for info in infos], axis=0)

    packed_t0 = time.perf_counter()
    packed = [
        pack_feasible_candidates(masks[i], costs[i], max_candidates=int(candidate_max))
        for i in range(len(infos))
    ]
    pack_sec = float(time.perf_counter() - packed_t0)
    feasible_counts = np.count_nonzero(masks.reshape(len(infos), -1), axis=1)

    cand_actions = np.stack([p.actions for p in packed], axis=0)
    cand_mask = np.stack([p.mask for p in packed], axis=0)
    cand_cost = np.stack([p.cost for p in packed], axis=0)

    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(masks.astype(np.bool_), dtype=torch.bool, device=device),
        "pill": torch.as_tensor(pill, dtype=torch.long, device=device),
        "preview": torch.as_tensor(preview, dtype=torch.long, device=device),
        "cand_actions": torch.as_tensor(cand_actions, dtype=torch.long, device=device),
        "cand_mask": torch.as_tensor(cand_mask, dtype=torch.bool, device=device),
        "cand_cost": torch.as_tensor(cand_cost, dtype=torch.float32, device=device),
        "pack_sec": pack_sec,
        "feasible_count_mean": float(np.mean(feasible_counts)) if feasible_counts.size else 0.0,
        "candidate_count_mean": float(np.mean([p.count for p in packed])) if packed else 0.0,
        "candidate_truncation_frac": float(np.mean(feasible_counts > int(candidate_max)))
        if packed
        else 0.0,
    }


def _time_forward(fn: Any, *, warmup: int, repeats: int, device: Any) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(int(max(0, warmup))):
            fn()
        _sync_device(device)

        samples: List[float] = []
        for _ in range(int(max(1, repeats))):
            t0 = time.perf_counter()
            fn()
            _sync_device(device)
            samples.append(float(time.perf_counter() - t0) * 1000.0)

    return {
        "forward_ms_mean": statistics.fmean(samples),
        "forward_ms_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "forward_ms_p50": _percentile(samples, 50.0),
        "forward_ms_p95": _percentile(samples, 95.0),
    }


def _bench_heatmap(
    tensors: Dict[str, Any],
    *,
    device: Any,
    encoder_blocks: int,
    repeats: int,
    warmup: int,
) -> Dict[str, float]:
    net = PlacementPolicyNet(
        in_channels=int(tensors["obs"].shape[1]),
        head_type="dense",
        pill_embed_dim=128,
        pill_embed_type="ordered_pair",
        encoder_blocks=int(encoder_blocks),
        aux_dim=0,
    ).to(device)
    net.eval()

    def _forward() -> Tuple[Any, Any]:
        return net(tensors["obs"], tensors["pill"], tensors["preview"], tensors["mask"])

    metrics = _time_forward(_forward, warmup=warmup, repeats=repeats, device=device)
    metrics["params"] = float(_count_params(net))
    return metrics


def _bench_candidate(
    tensors: Dict[str, Any],
    *,
    device: Any,
    board_encoder: str,
    candidate_max: int,
    patch_kernel: int,
    repeats: int,
    warmup: int,
    board_channels: int = 8,
) -> Dict[str, float]:
    net = CandidatePlacementPolicyNet(
        in_channels=int(tensors["obs"].shape[1]),
        board_channels=int(board_channels),
        board_encoder=str(board_encoder),
        encoder_blocks=0,
        d_model=128,
        pill_embed_dim=128,
        pill_embed_type="ordered_pair",
        aux_dim=0,
        pos_embed_dim=32,
        cost_embed_dim=32,
        cand_hidden_dim=256,
        transformer_layers=4,
        transformer_heads=4,
        transformer_ff_mult=4,
        patch_kernel=int(patch_kernel),
    ).to(device)
    net.eval()

    def _forward() -> Tuple[Any, Any]:
        return net(
            tensors["obs"],
            tensors["pill"],
            tensors["preview"],
            tensors["cand_actions"],
            tensors["cand_cost"],
            tensors["cand_mask"],
        )

    metrics = _time_forward(_forward, warmup=warmup, repeats=repeats, device=device)
    metrics["params"] = float(_count_params(net))
    metrics["pack_ms_total"] = float(tensors["pack_sec"]) * 1000.0
    metrics["pack_ms_per_decision"] = (
        float(tensors["pack_sec"]) * 1000.0 / max(1, int(tensors["obs"].shape[0]))
    )
    metrics["candidate_count_mean"] = float(tensors["candidate_count_mean"])
    metrics["feasible_count_mean"] = float(tensors["feasible_count_mean"])
    metrics["candidate_truncation_frac"] = float(tensors["candidate_truncation_frac"])
    metrics["candidate_max"] = float(candidate_max)
    return metrics


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Dr. Mario placement policy networks")
    parser.add_argument("--source", choices=["cpp-pool", "synthetic"], default="cpp-pool")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=128)
    parser.add_argument("--candidate-board-channels", type=int, default=8)
    parser.add_argument("--patch-kernel", type=int, default=9)
    parser.add_argument("--state-repr", type=str, default="bitplane_bottle_conn_mask")
    parser.add_argument("--synthetic-feasible", type=int, default=52)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)

    if torch is None:
        raise RuntimeError("PyTorch is required for tools.bench_policy")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    device = torch.device(str(args.device))
    if args.source == "cpp-pool":
        obs, infos = _collect_cpp_pool_batch(
            int(args.batch_size), int(args.seed), state_repr=str(args.state_repr)
        )
    else:
        repr_norm = str(args.state_repr).strip().lower().replace("-", "_")
        if repr_norm == "bitplane_bottle_conn_mask":
            channels = 12
        elif repr_norm in {"bitplane_bottle_conn", "bitplane_bottle_mask"}:
            channels = 8
        else:
            channels = 4
        obs, infos = _synthetic_batch(
            int(args.batch_size), int(args.seed), int(args.synthetic_feasible), channels=channels
        )
    tensors = _batch_tensors(obs, infos, device=device, candidate_max=int(args.candidate_max))

    rows: List[Dict[str, Any]] = []
    heatmap_variants = [("heatmap_dense_b0", 0), ("heatmap_dense_b32", 32)]
    for name, blocks in heatmap_variants:
        metrics = _bench_heatmap(
            tensors,
            device=device,
            encoder_blocks=int(blocks),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        rows.append({"policy": name, "batch_size": int(args.batch_size), **metrics})

    for name, encoder in [("candidate_cnn", "cnn"), ("candidate_col_tx", "col_transformer")]:
        metrics = _bench_candidate(
            tensors,
            device=device,
            board_encoder=encoder,
            board_channels=int(args.candidate_board_channels),
            candidate_max=int(args.candidate_max),
            patch_kernel=int(args.patch_kernel),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        rows.append({"policy": name, "batch_size": int(args.batch_size), **metrics})

    for row in rows:
        row["decisions_per_sec"] = 1000.0 * float(args.batch_size) / max(
            1e-9, float(row["forward_ms_mean"])
        )

    if args.json_out:
        payload = {"schema": "drmc-rl.bench_policy.v1", "args": vars(args), "rows": rows}
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = [
        "policy",
        "batch",
        "params",
        "fwd_ms",
        "p95_ms",
        "dec/s",
        "pack_ms",
        "feas_n",
        "cand_n",
        "trunc%",
    ]
    print(" ".join(f"{h:>16}" for h in header))
    for row in rows:
        print(
            f"{row['policy']:>16}"
            f"{int(row['batch_size']):>16}"
            f"{int(row['params']):>16}"
            f"{row['forward_ms_mean']:>16.3f}"
            f"{row['forward_ms_p95']:>16.3f}"
            f"{row['decisions_per_sec']:>16.1f}"
            f"{row.get('pack_ms_per_decision', 0.0):>16.4f}"
            f"{row.get('feasible_count_mean', 0.0):>16.1f}"
            f"{row.get('candidate_count_mean', 0.0):>16.1f}"
            f"{100.0 * row.get('candidate_truncation_frac', 0.0):>16.1f}"
        )


if __name__ == "__main__":
    main()
