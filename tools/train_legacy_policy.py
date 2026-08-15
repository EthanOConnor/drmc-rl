"""Train full-corpus versions of the original rating-banded human policies.

This is the maintained historical baseline: own-board state, candidate geometry,
and chosen-placement imitation only.  It deliberately omits opponent state,
continuous rating conditioning, timing, outcomes, and afterstates so its arena
results isolate the value of scaling the original Human Legacy idea.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.training.utils.checkpoint_io import save_checkpoint
from tools.eval_policy import _build_net_from_cfg
from tools.train_human_policy import (
    DEFAULT_DATASET,
    KMAX,
    _prefetched,
    _sample_weights,
    _training_mask,
    fields_to_planes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "human_policy" / "legacy_full"
BANDS = ("lt1600", "1600to2000", "gt2000")


@dataclass(slots=True)
class BandStatistics:
    rows: int
    player_counts: dict[int, int]
    rating_edges: np.ndarray
    rating_counts: np.ndarray
    validation: dict[str, np.ndarray]


def band_masks(ratings: np.ndarray) -> dict[str, np.ndarray]:
    ratings = np.asarray(ratings)
    return {
        "lt1600": ratings < 1600.0,
        "1600to2000": (ratings >= 1600.0) & (ratings < 2000.0),
        "gt2000": ratings >= 2000.0,
    }


def legacy_policy_config(capacity: str = "small") -> dict[str, Any]:
    sizes = {
        "small": (96, 2, 2, 192, 8),
        "medium": (128, 3, 3, 256, 8),
        "large": (192, 4, 4, 384, 16),
    }
    if capacity not in sizes:
        raise ValueError(f"unknown capacity {capacity!r}")
    d_model, blocks, layers, hidden, board_channels = sizes[capacity]
    return {
        "smdp_ppo": {
            "policy_type": "candidate",
            "aux_spec": "none",
            "pill_embed_dim": d_model,
            "pill_embed_type": "ordered_pair",
            "encoder_blocks": blocks,
            "candidate_max_candidates": KMAX,
            "candidate_d_model": d_model,
            "candidate_pos_embed_dim": 32,
            "candidate_cost_embed_dim": 32,
            "candidate_hidden_dim": hidden,
            "candidate_board_encoder": "cnn",
            "candidate_board_channels": board_channels,
            "candidate_transformer_layers": layers,
            "candidate_transformer_heads": 4,
            "candidate_transformer_ff_mult": 4,
            "candidate_patch_kernel": 9,
        }
    }


def legacy_batch_inputs(arrays: dict[str, np.ndarray], rows: np.ndarray):
    from drmc_rl.game.specs.ram_to_state import COLOR_VALUE_TO_INDEX

    rows = np.asarray(rows, dtype=np.int64)
    batch = len(rows)
    actions = arrays["candidate_actions"][rows].astype(np.int32)
    counts = arrays["candidate_count"][rows].astype(np.int64)
    mask = np.arange(KMAX)[None] < counts[:, None]
    costs = arrays["candidate_costs"][rows].astype(np.float32)
    raw_pills = arrays["pill"][rows].astype(np.int64)
    raw_previews = arrays["preview"][rows].astype(np.int64)
    color_map = np.zeros(4, dtype=np.int64)
    for raw, canonical in COLOR_VALUE_TO_INDEX.items():
        color_map[int(raw) & 3] = int(canonical)
    observations = np.zeros((batch, 12, 16, 8), dtype=np.float32)
    observations[:, :8] = fields_to_planes(arrays["field"][rows])
    batch_rows, slots = np.nonzero(mask)
    valid_actions = actions[batch_rows, slots]
    observations[:, 8:12].reshape(batch, 512)[batch_rows, valid_actions] = 1.0
    same_color = raw_pills[:, 0] == raw_pills[:, 1]
    observations[same_color, 6:8] = 0.0
    return (
        observations,
        color_map[raw_pills],
        color_map[raw_previews],
        actions,
        costs,
        mask,
        arrays["chosen_slot"][rows].astype(np.int64),
    )


def _row_arrays(arrays: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    source_rows = len(arrays["rating"])
    return {
        key: value[rows]
        for key, value in arrays.items()
        if np.asarray(value).ndim > 0 and len(value) == source_rows
    }


def _statistics(
    paths: list[Path], *, seed: int, validation_rows_per_shard: int
) -> dict[str, BandStatistics]:
    rng = np.random.default_rng(int(seed))
    rows = {band: 0 for band in BANDS}
    players: dict[str, dict[int, int]] = {band: {} for band in BANDS}
    edges = {band: np.linspace(0.0, 4000.0, 21) for band in BANDS}
    counts = {band: np.zeros(20, dtype=np.int64) for band in BANDS}
    validation: dict[str, list[dict[str, np.ndarray]]] = {band: [] for band in BANDS}
    for _path, arrays in _prefetched(paths):
        train = _training_mask(arrays)
        held_out = ~train
        for band, rating_mask in band_masks(arrays["rating"]).items():
            band_train = train & rating_mask
            band_ratings = arrays["rating"][band_train]
            rows[band] += len(band_ratings)
            counts[band] += np.histogram(band_ratings, bins=edges[band])[0]
            if "player_key" in arrays:
                keys, key_counts = np.unique(arrays["player_key"][band_train], return_counts=True)
                for key, key_count in zip(keys, key_counts):
                    player = int(key)
                    players[band][player] = players[band].get(player, 0) + int(key_count)
            available = np.flatnonzero(held_out & rating_mask)
            if len(available):
                chosen = rng.choice(
                    available,
                    size=min(len(available), int(validation_rows_per_shard)),
                    replace=False,
                )
                validation[band].append(_row_arrays(arrays, chosen))
    result = {}
    for band in BANDS:
        if not rows[band] or not validation[band]:
            raise ValueError(f"band {band} has no training or validation rows")
        keys = validation[band][0]
        result[band] = BandStatistics(
            rows=rows[band],
            player_counts=players[band],
            rating_edges=edges[band],
            rating_counts=counts[band],
            validation={key: np.concatenate([part[key] for part in validation[band]]) for key in keys},
        )
    return result


def _tensor_inputs(arrays, rows, device):
    import torch

    return tuple(torch.from_numpy(value).to(device) for value in legacy_batch_inputs(arrays, rows))


def _evaluate(net, arrays, *, device: str, batch_size: int, autocast) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    total = correct = top3 = 0
    nll = 0.0
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["rating"]), int(batch_size)):
            rows = np.arange(start, min(start + int(batch_size), len(arrays["rating"])))
            obs, pill, preview, actions, costs, mask, chosen = _tensor_inputs(
                arrays, rows, device
            )
            with autocast():
                logits, _value = net(obs, pill, preview, actions, costs, mask, aux=None)
            logits = logits.float().masked_fill(~mask, -1e9)
            nll += float(F.cross_entropy(logits, chosen, reduction="sum"))
            rank = (logits > logits.gather(1, chosen[:, None])).sum(dim=1)
            correct += int((rank == 0).sum())
            top3 += int((rank < 3).sum())
            total += len(rows)
    return {
        "rows": float(total),
        "nll": nll / max(total, 1),
        "top1": correct / max(total, 1),
        "top3": top3 / max(total, 1),
    }


def train(args) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    paths = sorted(Path(args.dataset).glob("*.npz"))
    if args.max_shards is not None:
        paths = paths[: int(args.max_shards)]
    if not paths:
        raise ValueError(f"no dataset shards under {args.dataset}")
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    statistics = _statistics(
        paths, seed=int(args.seed), validation_rows_per_shard=int(args.validation_rows_per_shard)
    )
    cfg = legacy_policy_config(args.capacity)
    nets = {}
    optimizers = {}
    schedulers = {}
    for band in BANDS:
        net, aux_dim, _candidate_max = _build_net_from_cfg(cfg, 12, args.device)
        if aux_dim:
            raise ValueError("legacy policy unexpectedly requires auxiliary features")
        net.train()
        nets[band] = net
        optimizer_args: dict[str, Any] = {"lr": float(args.lr), "weight_decay": 1e-4}
        if str(args.device).startswith("cuda"):
            optimizer_args["fused"] = True
        optimizers[band] = torch.optim.AdamW(net.parameters(), **optimizer_args)
        schedulers[band] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[band], T_max=max(int(args.epochs), 1), eta_min=float(args.lr) * 0.05
        )
    use_bf16 = str(args.device).startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        if use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    best = {band: float("inf") for band in BANDS}
    steps = {band: 0 for band in BANDS}
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    lineage = args.output / "lineage"
    lineage.mkdir(parents=True, exist_ok=True)
    epoch_metrics: dict[str, Any] = {}
    decisions = 0
    updates = 0
    for epoch in range(1, int(args.epochs) + 1):
        for band in BANDS:
            nets[band].train()
        ordered_paths = [paths[i] for i in rng.permutation(len(paths))]
        for _path, arrays in _prefetched(ordered_paths):
            train = _training_mask(arrays)
            for band, rating_mask in band_masks(arrays["rating"]).items():
                indices = np.flatnonzero(train & rating_mask)
                rng.shuffle(indices)
                stat = statistics[band]
                weights = _sample_weights(
                    arrays["rating"][indices],
                    player_keys=(arrays["player_key"][indices] if "player_key" in arrays else None),
                    player_counts=stat.player_counts,
                    rating_edges=stat.rating_edges,
                    rating_counts=stat.rating_counts,
                )
                for start in range(0, len(indices), int(args.batch_size)):
                    batch_rows = indices[start : start + int(args.batch_size)]
                    if len(batch_rows) < 2:
                        continue
                    obs, pill, preview, actions, costs, mask, chosen = _tensor_inputs(
                        arrays, batch_rows, args.device
                    )
                    weight = torch.from_numpy(weights[start : start + len(batch_rows)]).to(args.device)
                    with autocast():
                        logits, _value = nets[band](
                            obs, pill, preview, actions, costs, mask, aux=None
                        )
                        losses = F.cross_entropy(logits, chosen, reduction="none")
                        loss = (losses * weight).sum() / weight.sum().clamp_min(1e-8)
                    optimizers[band].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(nets[band].parameters(), 5.0)
                    optimizers[band].step()
                    steps[band] += 1
                    decisions += len(batch_rows)
                    updates += 1
                    if int(args.log_every) and updates % int(args.log_every) == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            f"epoch={epoch} updates={updates} decisions={decisions:,} "
                            f"decisions/s={decisions / elapsed:,.0f} band={band} "
                            f"loss={float(loss.detach()):.4f}",
                            flush=True,
                        )
        for band in BANDS:
            metrics = _evaluate(
                nets[band],
                statistics[band].validation,
                device=args.device,
                batch_size=int(args.batch_size),
                autocast=autocast,
            )
            epoch_metrics[band] = metrics
            payload = {
                "state_dict": {key: value.cpu() for key, value in nets[band].state_dict().items()},
                "cfg": cfg,
                "step": steps[band],
                "bc_meta": {
                    "schema": "drmc-human-legacy-full-v1",
                    "band": band,
                    "training_rows": statistics[band].rows,
                    "players": len(statistics[band].player_counts),
                    "parameters": sum(value.numel() for value in nets[band].parameters()),
                    "dataset": str(args.dataset),
                    "capacity": args.capacity,
                    "epoch": epoch,
                    "metrics": metrics,
                    "balance": "sqrt-inverse-player-and-rating-frequency",
                    "trained_at": time.time(),
                },
            }
            save_checkpoint(payload, lineage / f"bc_full_{band}_epoch{epoch:02d}.pt.gz")
            if metrics["nll"] < best[band]:
                best[band] = metrics["nll"]
                save_checkpoint(payload, args.output / f"bc_full_{band}.pt.gz")
            schedulers[band].step()
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            json.dumps(
                {"epoch": epoch, "elapsed_seconds": elapsed, "steps": steps, "metrics": epoch_metrics}
            ),
            flush=True,
        )
    result = {"output": str(args.output), "best_nll": best, "metrics": epoch_metrics}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capacity", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--validation-rows-per-shard", type=int, default=1024)
    parser.add_argument("--max-shards", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
