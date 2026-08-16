"""Train the V3 afterstate policy and rating-to-regret calibration.

This is deliberately not rating-conditioned value learning. Competitive
quality is learned from outcomes and exact tactical consequences. A separate
human-style head learns the observed action. After training, the human choice's
regret against all legal alternatives is measured and monotonically calibrated
against WHR-C.
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

from drmc_rl.human.afterstate_model import (
    HUMAN_AFTERSTATE_SCHEMA,
    afterstate_policy_config,
    build_afterstate_policy,
)
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.afterstate_sim import decode_sparse_deltas
from drmc_rl.human.model import POLICY_CONDITION_DIM, build_timing_model
from drmc_rl.human.strength import RegretCalibration, quality_opportunity
from drmc_rl.training.utils.checkpoint_io import save_checkpoint
from tools.train_human_policy import _sample_weights, condition_features, timing_features


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "human_vs" / "human_policy_v2"
DEFAULT_AFTERSTATES = REPO_ROOT / "data" / "human_vs" / "human_afterstates_v3"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "human_policy" / "human_afterstate_v3.pt.gz"


@dataclass(slots=True)
class Shard:
    arrays: dict[str, np.ndarray]
    delta_offsets: np.ndarray
    delta_cells: np.ndarray
    delta_values: np.ndarray
    targets: dict[str, np.ndarray]


@dataclass(slots=True)
class TrainingStatistics:
    condition: HumanSkillCondition
    rows: int
    player_counts: dict[int, int]
    rating_edges: np.ndarray
    rating_counts: np.ndarray


def _use_bf16(torch_module, *, device: str, precision: str) -> bool:
    cuda_device = str(device).startswith("cuda")
    native_bf16 = (
        cuda_device
        and torch_module.cuda.is_bf16_supported()
        and torch_module.cuda.get_device_capability(device)[0] >= 8
    )
    if precision == "bf16" and not native_bf16:
        raise ValueError(f"native bf16 is unavailable on {device}")
    return precision == "bf16" or (precision == "auto" and native_bf16)


def _load_shard(source: Path, afterstates: Path) -> Shard:
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    delta_offsets = np.load(afterstates / f"{source.stem}.delta_offsets.npy", mmap_mode="r")
    delta_cells = np.memmap(
        afterstates / f"{source.stem}.delta_cells.bin", mode="r", dtype=np.uint8
    )
    delta_values = np.memmap(
        afterstates / f"{source.stem}.delta_values.bin", mode="r", dtype=np.uint8
    )
    with np.load(afterstates / f"{source.stem}.targets.npz", allow_pickle=False) as data:
        targets = {key: data[key] for key in data.files}
    candidates = int(targets["candidate_offsets"][-1])
    if len(delta_offsets) != candidates + 1:
        raise ValueError(f"afterstate offset mismatch for {source.name}")
    if int(delta_offsets[-1]) != len(delta_cells) or len(delta_cells) != len(delta_values):
        raise ValueError(f"afterstate delta mismatch for {source.name}")
    annotated_rows = len(targets["candidate_offsets"]) - 1
    source_rows = len(arrays["rating"])
    if annotated_rows > source_rows:
        raise ValueError(f"afterstate row mismatch for {source.name}")
    if annotated_rows < source_rows:
        arrays = {
            key: value[:annotated_rows] if value.ndim and len(value) == source_rows else value
            for key, value in arrays.items()
        }
    return Shard(arrays, delta_offsets, delta_cells, delta_values, targets)


def _canonical_colors(raw: np.ndarray) -> np.ndarray:
    # Raw NES 0=Y, 1=R, 2=B; model canonical 0=R, 1=Y, 2=B.
    return np.asarray((1, 0, 2, 2), dtype=np.int64)[np.asarray(raw, dtype=np.int64) & 3]


def make_batch(
    shard: Shard, index: np.ndarray, condition: HumanSkillCondition
) -> dict[str, np.ndarray]:
    arrays, targets = shard.arrays, shard.targets
    rows = np.asarray(index, dtype=np.int64)
    batch = len(rows)
    counts = arrays["candidate_count"][rows].astype(np.int64)
    width = int(counts.max(initial=1))
    mask = np.arange(width)[None] < counts[:, None]
    afterstate = np.full((batch, width, 128), 0xFF, dtype=np.uint8)
    terminal = np.zeros((batch, width), dtype=np.uint8)
    viruses = np.zeros((batch, width), dtype=np.float32)
    nonviruses = np.zeros((batch, width), dtype=np.float32)
    events = np.zeros((batch, width), dtype=np.float32)
    offsets = targets["candidate_offsets"]
    for out_row, source_row in enumerate(rows):
        start, stop = int(offsets[source_row]), int(offsets[source_row + 1])
        count = stop - start
        afterstate[out_row, :count] = decode_sparse_deltas(
            arrays["field"][source_row],
            start,
            stop,
            shard.delta_offsets,
            shard.delta_cells,
            shard.delta_values,
        )
        terminal[out_row, :count] = targets["terminal_reason"][start:stop]
        viruses[out_row, :count] = targets["viruses_cleared"][start:stop]
        nonviruses[out_row, :count] = targets["nonviruses_cleared"][start:stop]
        events[out_row, :count] = targets["clear_events"][start:stop]
        if targets["invalid"][start:stop].any():
            mask[out_row, :count] &= ~targets["invalid"][start:stop]
    timing_x, timing_y = timing_features(arrays, rows, condition)
    return {
        "afterstate": afterstate,
        "root": arrays["field"][rows].astype(np.uint8),
        "opponent": arrays["opponent_field"][rows].astype(np.uint8),
        "pill": _canonical_colors(arrays["pill"][rows]),
        "preview": _canonical_colors(arrays["preview"][rows]),
        "actions": arrays["candidate_actions"][rows, :width].astype(np.int64),
        "costs": arrays["candidate_costs"][rows, :width].astype(np.float32),
        "mask": mask,
        "condition": condition_features(arrays, rows, condition),
        "chosen": arrays["chosen_slot"][rows].astype(np.int64),
        "won": arrays["won"][rows].astype(np.float32),
        "rating": arrays["rating"][rows].astype(np.float32),
        "terminal": terminal,
        "viruses": viruses,
        "nonviruses": nonviruses,
        "events": events,
        "timing_x": timing_x,
        "timing_y": timing_y,
    }


def _tensor_batch(batch: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    import torch

    return {key: torch.from_numpy(value).to(device) for key, value in batch.items()}


def _weighted_mean(values, weights):
    """Mean one loss per decision, with stable batch-local normalization."""

    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _masked_row_mean(values, mask):
    """Reduce candidate losses per decision before reducing across decisions."""

    valid = mask.to(values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _losses(model, timing, batch, *, autocast):
    import torch
    import torch.nn.functional as F

    with autocast():
        output = model(
            batch["afterstate"],
            batch["root"],
            batch["opponent"],
            batch["pill"],
            batch["preview"],
            batch["actions"],
            batch["costs"],
            batch["mask"],
            batch["condition"],
        )
        row = torch.arange(len(batch["chosen"]), device=batch["chosen"].device)
        chosen = batch["chosen"]
        weights = batch.get("row_weight")
        if weights is None:
            weights = torch.ones_like(batch["won"])
        style = _weighted_mean(
            F.cross_entropy(output["human_logits"], chosen, reduction="none"), weights
        )
        outcome = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output["outcome_logit"][row, chosen], batch["won"], reduction="none"
            ),
            weights,
        )
        quality_outcome = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output["competitive_score"][row, chosen], batch["won"], reduction="none"
            ),
            weights,
        )
        # Bootstrap a coherent rating-independent policy from the observed
        # corpus action. Outcome/search fine-tuning can improve this head later;
        # without this term, only one candidate per row receives a long-horizon
        # label and the remaining ranking is nearly arbitrary.
        quality_policy = _weighted_mean(
            F.cross_entropy(output["competitive_score"], chosen, reduction="none"), weights
        )
        valid = batch["mask"]
        clear_target = (batch["terminal"] == 1).to(output["clear_logit"].dtype)
        topout_target = (batch["terminal"] == 2).to(output["topout_logit"].dtype)

        def balanced_bce(logits, target, valid):
            positives = target[valid].sum()
            negatives = valid.sum() - positives
            if float(positives) > 0:
                positive_weight = (negatives / positives).clamp(1.0, 100.0)
            else:
                positive_weight = torch.ones((), dtype=logits.dtype, device=logits.device)
            per_candidate = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight, reduction="none"
            )
            return _weighted_mean(_masked_row_mean(per_candidate, valid), weights)

        clear = balanced_bce(output["clear_logit"], clear_target, valid)
        topout = balanced_bce(output["topout_logit"], topout_target, valid)
        virus = _weighted_mean(
            _masked_row_mean(
                F.smooth_l1_loss(
                    output["virus_delta"], torch.log1p(batch["viruses"]), reduction="none"
                ),
                valid,
            ),
            weights,
        )
        attack_target = torch.log1p(batch["nonviruses"] + 2.0 * batch["events"])
        attack = _weighted_mean(
            _masked_row_mean(
                F.smooth_l1_loss(output["attack"], attack_target, reduction="none"), valid
            ),
            weights,
        )

        # Immediate native consequences bootstrap within-position ordering;
        # centering prevents this auxiliary target from defining absolute value.
        tactical = (
            2.5 * batch["viruses"]
            + 0.25 * batch["nonviruses"]
            + batch["events"]
            + 8.0 * clear_target
            - 8.0 * topout_target
        )
        quality = output["competitive_score"]
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        quality_centered = quality - (quality.masked_fill(~valid, 0.0).sum(1, keepdim=True) / count)
        tactical_centered = tactical - (
            tactical.masked_fill(~valid, 0.0).sum(1, keepdim=True) / count
        )
        ordering = _weighted_mean(
            _masked_row_mean(
                F.smooth_l1_loss(
                    quality_centered, tactical_centered / 4.0, reduction="none"
                ),
                valid,
            ),
            weights,
        )

        timing_output = timing(batch["timing_x"])
        mean = timing_output[:, 0]
        log_std = timing_output[:, 1].clamp(-3.0, 2.0)
        timing_nll = _weighted_mean(
            0.5 * ((batch["timing_y"] - mean) / log_std.exp()).square() + log_std,
            weights,
        )
        total = (
            style
            + 0.5 * outcome
            + 0.25 * quality_outcome
            + 0.50 * quality_policy
            + 0.12 * ordering
            + 0.08 * clear
            + 0.08 * topout
            + 0.08 * virus
            + 0.05 * attack
            + 0.12 * timing_nll
        )
    return (
        total,
        {
            "style": style.detach(),
            "outcome": outcome.detach(),
            "quality_outcome": quality_outcome.detach(),
            "quality_policy": quality_policy.detach(),
            "ordering": ordering.detach(),
            "clear": clear.detach(),
            "topout": topout.detach(),
            "virus": virus.detach(),
            "attack": attack.detach(),
            "timing": timing_nll.detach(),
        },
        output,
    )


def _source_paths(dataset: Path, afterstates: Path, max_shards: int | None) -> list[Path]:
    paths = [
        path
        for path in sorted(dataset.glob("*.npz"))
        if (afterstates / f"{path.stem}.delta_offsets.npy").is_file()
        and (afterstates / f"{path.stem}.delta_cells.bin").is_file()
        and (afterstates / f"{path.stem}.delta_values.bin").is_file()
        and (afterstates / f"{path.stem}.targets.npz").is_file()
    ]
    if max_shards is not None:
        paths = paths[: int(max_shards)]
    if not paths:
        raise ValueError("no source shards with matching afterstate annotations")
    return paths


def _fit_training_statistics(paths: list[Path]) -> TrainingStatistics:
    """Collect global balance statistics without retaining every shard."""

    count = 0
    rating_sum = 0.0
    rating_square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    player_counts: dict[int, int] = {}
    rating_edges = np.linspace(0.0, 4000.0, 21)
    rating_counts = np.zeros(20, dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            train = (data["split"] == 0) & (data["player_fold"] != 0) & (data["time_split"] == 0)
            ratings = data["rating"][train].astype(np.float64)
            count += len(ratings)
            rating_sum += float(ratings.sum())
            rating_square_sum += float(np.square(ratings).sum())
            rating_counts += np.histogram(ratings, bins=rating_edges)[0]
            if len(ratings):
                minimum = min(minimum, float(ratings.min()))
                maximum = max(maximum, float(ratings.max()))
            if "player_key" in data:
                keys, counts = np.unique(data["player_key"][train], return_counts=True)
                for key, player_count in zip(keys, counts):
                    player_counts[int(key)] = player_counts.get(int(key), 0) + int(player_count)
    if count == 0:
        raise ValueError("afterstate dataset contains no training rows")
    mean = rating_sum / count
    variance = max(rating_square_sum / count - mean * mean, 1.0)
    return TrainingStatistics(
        condition=HumanSkillCondition(
            mean=float(mean),
            scale=float(np.sqrt(variance)),
            minimum=float(minimum),
            maximum=float(maximum),
        ),
        rows=count,
        player_counts=player_counts,
        rating_edges=rating_edges,
        rating_counts=rating_counts,
    )


def _evaluate(
    model,
    timing,
    paths: list[Path],
    *,
    afterstates: Path,
    condition: HumanSkillCondition,
    device: str,
    batch_size: int,
    max_rows: int,
    autocast,
) -> tuple[dict[str, float], RegretCalibration]:
    import torch

    model.eval()
    timing.eval()
    total_rows = 0
    objective_sum = top1_sum = quality_top1_sum = brier_sum = 0.0
    ratings: list[np.ndarray] = []
    regrets: list[np.ndarray] = []
    opportunities: list[np.ndarray] = []
    rows_per_shard = max(1, int(np.ceil(max_rows / len(paths))))
    with torch.inference_mode():
        for path in paths:
            if total_rows >= max_rows:
                break
            shard = _load_shard(path, afterstates)
            arrays = shard.arrays
            validation = (
                (arrays["split"] != 0) | (arrays["player_fold"] == 0) | (arrays["time_split"] != 0)
            )
            available = np.flatnonzero(validation)
            take = min(rows_per_shard, len(available), max_rows - total_rows)
            if take < len(available):
                positions = np.linspace(0, len(available) - 1, take, dtype=np.int64)
                rows = available[positions]
            else:
                rows = available
            for start in range(0, len(rows), batch_size):
                index = rows[start : start + batch_size]
                numpy_batch = make_batch(shard, index, condition)
                batch = _tensor_batch(numpy_batch, device)
                loss, _parts, output = _losses(model, timing, batch, autocast=autocast)
                row = torch.arange(len(index), device=batch["chosen"].device)
                chosen = batch["chosen"]
                quality = output["competitive_score"].float()
                regret = quality.max(dim=1).values - quality[row, chosen]
                prediction = output["human_logits"].argmax(dim=1)
                quality_prediction = output["competitive_score"].argmax(dim=1)
                probability = torch.sigmoid(output["outcome_logit"][row, chosen].float())
                n = len(index)
                objective_sum += float(loss) * n
                top1_sum += float((prediction == chosen).sum())
                quality_top1_sum += float((quality_prediction == chosen).sum())
                brier_sum += float(((probability - batch["won"]) ** 2).sum())
                ratings.append(numpy_batch["rating"])
                regrets.append(regret.cpu().numpy())
                opportunities.append(
                    quality_opportunity(
                        quality.cpu().numpy(), numpy_batch["mask"]
                    )
                )
                total_rows += n
    if total_rows < 20:
        raise ValueError("not enough held-out rows for V3 evaluation")
    rating_array = np.concatenate(ratings)
    regret_array = np.concatenate(regrets)
    opportunity_array = np.concatenate(opportunities)
    bins = min(12, max(2, total_rows // 100))
    calibration = RegretCalibration.fit(
        rating_array,
        regret_array,
        opportunity_array,
        rating_bins=bins,
    )
    low_cut, high_cut = np.quantile(rating_array, (0.2, 0.8))
    low_regret = regret_array[rating_array <= low_cut]
    high_regret = regret_array[rating_array >= high_cut]
    metrics = {
        "validation_objective": objective_sum / total_rows,
        "validation_top1": top1_sum / total_rows,
        "validation_quality_top1": quality_top1_sum / total_rows,
        "validation_outcome_brier": brier_sum / total_rows,
        "validation_mean_regret": float(regret_array.mean()),
        "validation_regret_q90": float(np.quantile(regret_array, 0.9)),
        "validation_low_rating_regret_q90": float(np.quantile(low_regret, 0.9)),
        "validation_high_rating_regret_q90": float(np.quantile(high_regret, 0.9)),
        "validation_regret_tail_gap": float(
            np.quantile(low_regret, 0.9) - np.quantile(high_regret, 0.9)
        ),
        "validation_rows": float(total_rows),
    }
    return metrics, calibration


def _checkpoint_payload(
    model,
    timing,
    *,
    cfg: dict[str, Any],
    condition: HumanSkillCondition,
    calibration: RegretCalibration,
    args,
    parameter_count: int,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    statistics: TrainingStatistics,
) -> dict[str, Any]:
    return {
        "schema": HUMAN_AFTERSTATE_SCHEMA,
        "cfg": cfg,
        "state_dict": model.state_dict(),
        "timing_state_dict": timing.state_dict(),
        "human_meta": {
            "skill_condition": condition.to_dict(),
            "regret_calibration": calibration.to_dict(),
            "source_dataset": str(args.dataset),
            "afterstate_dataset": str(args.afterstates),
            "parameters": parameter_count,
            "optimizer_steps": step,
            "epoch": epoch,
            "metrics": metrics,
            "training_balance": {
                "scheme": "sqrt-inverse-player-and-rating-frequency",
                "training_rows": statistics.rows,
                "players": len(statistics.player_counts),
                "rating_bins": len(statistics.rating_counts),
            },
            "trained_at": time.time(),
        },
    }


def train(args) -> dict[str, Any]:
    import torch

    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    paths = _source_paths(args.dataset, args.afterstates, args.max_shards)
    statistics = _fit_training_statistics(paths)
    condition = statistics.condition
    cfg = afterstate_policy_config(capacity=args.capacity)
    model = build_afterstate_policy(cfg, condition_dim=POLICY_CONDITION_DIM, device=args.device)
    timing = build_timing_model(device=args.device)
    if args.init_checkpoint is not None:
        from drmc_rl.training.utils.checkpoint_io import load_checkpoint

        initial = load_checkpoint(args.init_checkpoint, map_location="cpu")
        if initial.get("schema") != HUMAN_AFTERSTATE_SCHEMA:
            raise ValueError(f"not a {HUMAN_AFTERSTATE_SCHEMA} checkpoint: {args.init_checkpoint}")
        model.load_state_dict(initial["state_dict"])
        timing.load_state_dict(initial["timing_state_dict"])
    parameters = list(model.parameters()) + list(timing.parameters())
    optimizer_args: dict[str, Any] = {"lr": args.lr, "weight_decay": 1e-4}
    if str(args.device).startswith("cuda"):
        optimizer_args["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1), eta_min=float(args.lr) * 0.05
    )
    use_bf16 = _use_bf16(torch, device=args.device, precision=args.precision)

    def autocast():
        if use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"schema={HUMAN_AFTERSTATE_SCHEMA} parameters={parameter_count:,} "
        f"shards={len(paths)} precision={'bf16' if use_bf16 else 'fp32'}",
        flush=True,
    )
    step = 0
    best_objective = float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_calibration: RegretCalibration | None = None
    started = time.perf_counter()
    for epoch in range(int(args.epochs)):
        model.train()
        timing.train()
        order = rng.permutation(len(paths))
        for shard_number in order:
            path = paths[int(shard_number)]
            shard = _load_shard(path, args.afterstates)
            train_rows = np.flatnonzero(
                (shard.arrays["split"] == 0)
                & (shard.arrays["player_fold"] != 0)
                & (shard.arrays["time_split"] == 0)
            )
            if args.max_rows_per_shard is not None:
                train_rows = train_rows[: int(args.max_rows_per_shard)]
            rng.shuffle(train_rows)
            weights = _sample_weights(
                shard.arrays["rating"][train_rows],
                player_keys=(
                    shard.arrays["player_key"][train_rows]
                    if "player_key" in shard.arrays
                    else None
                ),
                player_counts=statistics.player_counts,
                rating_edges=statistics.rating_edges,
                rating_counts=statistics.rating_counts,
            )
            for start in range(0, len(train_rows), int(args.batch_size)):
                index = train_rows[start : start + int(args.batch_size)]
                if len(index) < 2:
                    continue
                numpy_batch = make_batch(shard, index, condition)
                numpy_batch["row_weight"] = weights[start : start + len(index)]
                batch = _tensor_batch(numpy_batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts, _output = _losses(model, timing, batch, autocast=autocast)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 2.0)
                optimizer.step()
                step += 1
                if step % int(args.log_every) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    values = " ".join(f"{key}={float(value):.4f}" for key, value in parts.items())
                    print(
                        f"epoch={epoch + 1} step={step} decisions/s="
                        f"{step * int(args.batch_size) / elapsed:,.0f} "
                        f"loss={float(loss.detach()):.4f} {values}",
                        flush=True,
                    )
        scheduler.step()
        metrics, calibration = _evaluate(
            model,
            timing,
            paths,
            afterstates=args.afterstates,
            condition=condition,
            device=args.device,
            batch_size=int(args.batch_size),
            max_rows=int(args.calibration_rows),
            autocast=autocast,
        )
        payload = _checkpoint_payload(
            model,
            timing,
            cfg=cfg,
            condition=condition,
            calibration=calibration,
            args=args,
            parameter_count=parameter_count,
            step=step,
            epoch=epoch + 1,
            metrics=metrics,
            statistics=statistics,
        )
        lineage = (
            args.lineage_dir
            if args.lineage_dir is not None
            else args.output.parent / "v3_lineage"
        )
        lineage.mkdir(parents=True, exist_ok=True)
        epoch_path = lineage / f"human_afterstate_v3_epoch{epoch + 1:02d}.pt.gz"
        save_checkpoint(payload, epoch_path)
        if metrics["validation_objective"] < best_objective:
            best_objective = metrics["validation_objective"]
            best_epoch = epoch + 1
            best_metrics = metrics
            best_calibration = calibration
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(payload, args.output)
        print(
            json.dumps(
                {"epoch": epoch + 1, "checkpoint": str(epoch_path), "metrics": metrics},
                indent=2,
            ),
            flush=True,
        )
    assert best_calibration is not None
    result = {
        "checkpoint": str(args.output),
        "parameters": parameter_count,
        "optimizer_steps": step,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "regret_calibration": best_calibration.to_dict(),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--afterstates", type=Path, default=DEFAULT_AFTERSTATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--lineage-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
        help="auto uses bf16 only on GPUs with native bf16 tensor cores",
    )
    parser.add_argument("--capacity", choices=("small", "base", "large"), default="base")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--calibration-rows", type=int, default=100_000)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-rows-per-shard", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
