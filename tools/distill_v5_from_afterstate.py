"""Distill a V3 exact-afterstate teacher into the rollout-fast V5 policy.

V3 receives resolved candidate afterstates.  V5 receives the corresponding
root/opponent bottles and candidate geometry used by PPO.  Candidate order is
the corpus contract, so the teacher distribution transfers without action
remapping.  The competitive ``v1_vs`` auxiliary tail is zero during offline
pretraining; live G4 teaching and PPO subsequently train that context.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from drmc_rl.human.afterstate_model import HUMAN_AFTERSTATE_SCHEMA, build_afterstate_policy
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.training.distillation import masked_distillation_loss
from drmc_rl.training.utils.checkpoint_io import load_checkpoint, save_checkpoint
from tools.eval_policy import _build_net_from_cfg
from tools.train_afterstate_policy import (
    _fit_training_statistics,
    _load_shard,
    _sample_weights,
    _source_paths,
    _tensor_batch,
    make_batch,
)
from tools.train_human_policy import batch_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "drmc_rl" / "training" / "configs" / "vs_g5_maximal_tf3090.yaml"


def _student_batch(shard, rows: np.ndarray, condition, *, aux_dim: int, device: str):
    obs, pills, previews, actions, costs, mask, chosen, _human_condition = batch_inputs(
        shard.arrays, rows, condition
    )
    return {
        "obs": torch.from_numpy(obs).to(device),
        "pill": torch.from_numpy(pills).to(device),
        "preview": torch.from_numpy(previews).to(device),
        "actions": torch.from_numpy(actions).to(device),
        "costs": torch.from_numpy(costs).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "chosen": torch.from_numpy(chosen).to(device),
        "aux": torch.zeros((len(rows), aux_dim), dtype=torch.float32, device=device),
    }


def _load_teacher(path: Path, device: str):
    payload = load_checkpoint(path, map_location="cpu")
    if payload.get("schema") != HUMAN_AFTERSTATE_SCHEMA:
        raise ValueError(f"not a {HUMAN_AFTERSTATE_SCHEMA} checkpoint: {path}")
    meta = payload.get("human_meta") or {}
    condition_meta = meta.get("skill_condition")
    if not isinstance(condition_meta, dict):
        raise ValueError("V3 checkpoint is missing skill-condition metadata")
    cfg = payload.get("cfg") or {}
    condition_dim = int(cfg.get("condition_dim", 40))
    model = build_afterstate_policy(cfg, condition_dim=condition_dim, device=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, HumanSkillCondition.from_dict(condition_meta)


def _student_forward(student, batch):
    return student(
        batch["obs"],
        batch["pill"],
        batch["preview"],
        batch["actions"],
        batch["costs"],
        batch["mask"],
        aux=batch["aux"],
    )


def _outcome_value_loss(
    value: torch.Tensor,
    won: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute probability-form BCE in FP32, outside any active autocast region."""

    with torch.autocast(device_type=value.device.type, enabled=False):
        probability = ((value.squeeze(-1).float() + 1.0) * 0.5).clamp(
            1e-5, 1.0 - 1e-5
        )
        return F.binary_cross_entropy(
            probability,
            won.float(),
            weight=row_weight.float(),
        )


def _evaluate(
    student,
    teacher,
    paths: list[Path],
    *,
    afterstates: Path,
    condition,
    aux_dim: int,
    device: str,
    batch_size: int,
    rows_per_shard: int,
    temperature: float,
    autocast,
) -> dict[str, float]:
    student.eval()
    totals = {"rows": 0.0, "loss": 0.0, "competitive_agreement": 0.0, "human_agreement": 0.0}
    with torch.inference_mode():
        for path in paths:
            shard = _load_shard(path, afterstates)
            held_out = np.flatnonzero(
                (shard.arrays["split"] != 0)
                | (shard.arrays["player_fold"] == 0)
                | (shard.arrays["time_split"] != 0)
            )[: int(rows_per_shard)]
            for start in range(0, len(held_out), int(batch_size)):
                rows = held_out[start : start + int(batch_size)]
                if len(rows) == 0:
                    continue
                teacher_np = make_batch(shard, rows, condition)
                teacher_batch = _tensor_batch(teacher_np, device)
                student_batch = _student_batch(
                    shard, rows, condition, aux_dim=aux_dim, device=device
                )
                width = int(teacher_batch["mask"].shape[1])
                with autocast():
                    teacher_out = teacher(
                        teacher_batch["afterstate"],
                        teacher_batch["root"],
                        teacher_batch["opponent"],
                        teacher_batch["pill"],
                        teacher_batch["preview"],
                        teacher_batch["actions"],
                        teacher_batch["costs"],
                        teacher_batch["mask"],
                        teacher_batch["condition"],
                    )
                    student_logits, _value = _student_forward(student, student_batch)
                    student_logits = student_logits[:, :width]
                    loss = masked_distillation_loss(
                        student_logits,
                        teacher_out["competitive_score"],
                        teacher_batch["mask"],
                        temperature=temperature,
                    )
                n = len(rows)
                totals["rows"] += n
                totals["loss"] += float(loss) * n
                totals["competitive_agreement"] += float(
                    (student_logits.argmax(1) == teacher_out["competitive_score"].argmax(1)).sum()
                )
                totals["human_agreement"] += float(
                    (student_logits.argmax(1) == teacher_out["human_logits"].argmax(1)).sum()
                )
    rows = max(totals.pop("rows"), 1.0)
    return {key: value / rows for key, value in totals.items()} | {"rows": rows}


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    cfg = yaml.safe_load(args.student_config.read_text(encoding="utf-8")) or {}
    student, aux_dim, candidate_max = _build_net_from_cfg(cfg, 20, args.device)
    if str((cfg.get("smdp_ppo") or {}).get("candidate_architecture", "g4")) != "g5":
        raise ValueError("student config must select candidate_architecture=g5")
    if candidate_max != 128:
        raise ValueError(f"corpus contract requires 128 candidate slots, got {candidate_max}")
    teacher, teacher_condition = _load_teacher(args.teacher, args.device)
    paths = _source_paths(args.dataset, args.afterstates, args.max_shards)
    statistics = _fit_training_statistics(paths)

    optimizer_args: dict[str, Any] = {"lr": args.lr, "weight_decay": args.weight_decay}
    if str(args.device).startswith("cuda"):
        optimizer_args["fused"] = True
    optimizer = torch.optim.AdamW(student.parameters(), **optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1), eta_min=float(args.lr) * 0.05
    )
    use_bf16 = str(args.device).startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        if use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lineage = args.lineage_dir or args.output.parent / "v5_v3_distill_lineage"
    lineage.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    started = time.perf_counter()
    best_metrics: dict[str, float] = {}
    for epoch in range(int(args.epochs)):
        student.train()
        for shard_index in rng.permutation(len(paths)):
            shard = _load_shard(paths[int(shard_index)], args.afterstates)
            rows = np.flatnonzero(
                (shard.arrays["split"] == 0)
                & (shard.arrays["player_fold"] != 0)
                & (shard.arrays["time_split"] == 0)
            )
            rng.shuffle(rows)
            if args.max_rows_per_shard is not None:
                rows = rows[: int(args.max_rows_per_shard)]
            weights = _sample_weights(
                shard.arrays["rating"][rows],
                player_keys=(
                    shard.arrays["player_key"][rows] if "player_key" in shard.arrays else None
                ),
                player_counts=statistics.player_counts,
                rating_edges=statistics.rating_edges,
                rating_counts=statistics.rating_counts,
            )
            for start in range(0, len(rows), int(args.batch_size)):
                batch_rows = rows[start : start + int(args.batch_size)]
                if len(batch_rows) < 2:
                    continue
                teacher_np = make_batch(shard, batch_rows, teacher_condition)
                teacher_batch = _tensor_batch(teacher_np, args.device)
                student_batch = _student_batch(
                    shard, batch_rows, teacher_condition, aux_dim=aux_dim, device=args.device
                )
                row_weight = torch.from_numpy(weights[start : start + len(batch_rows)]).to(
                    args.device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.inference_mode(), autocast():
                    teacher_out = teacher(
                        teacher_batch["afterstate"],
                        teacher_batch["root"],
                        teacher_batch["opponent"],
                        teacher_batch["pill"],
                        teacher_batch["preview"],
                        teacher_batch["actions"],
                        teacher_batch["costs"],
                        teacher_batch["mask"],
                        teacher_batch["condition"],
                    )
                width = int(teacher_batch["mask"].shape[1])
                with autocast():
                    student_logits, value = _student_forward(student, student_batch)
                    student_logits = student_logits[:, :width]
                    policy_loss = masked_distillation_loss(
                        student_logits,
                        teacher_out["competitive_score"],
                        teacher_batch["mask"],
                        temperature=args.temperature,
                        row_weight=row_weight,
                    )
                value_loss = _outcome_value_loss(value, teacher_batch["won"], row_weight)
                loss = policy_loss + float(args.value_coef) * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
                optimizer.step()
                step += 1
                if step % int(args.log_every) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"epoch={epoch + 1} step={step} decisions/s="
                        f"{step * int(args.batch_size) / elapsed:,.0f} loss={float(loss):.4f} "
                        f"policy={float(policy_loss):.4f} value={float(value_loss):.4f}",
                        flush=True,
                    )
        scheduler.step()
        metrics = _evaluate(
            student,
            teacher,
            paths,
            afterstates=args.afterstates,
            condition=teacher_condition,
            aux_dim=aux_dim,
            device=args.device,
            batch_size=args.batch_size,
            rows_per_shard=args.validation_rows_per_shard,
            temperature=args.temperature,
            autocast=autocast,
        )
        payload = {
            "schema": "drmc-v5-v3-distill-v1",
            "cfg": cfg,
            "state_dict": student.state_dict(),
            "ema_state_dict": student.state_dict(),
            "step": 0,
            "decision_step": 0,
            "distillation": {
                "teacher": str(args.teacher),
                "teacher_schema": HUMAN_AFTERSTATE_SCHEMA,
                "dataset": str(args.dataset),
                "afterstates": str(args.afterstates),
                "epoch": epoch + 1,
                "optimizer_steps": step,
                "metrics": metrics,
                "temperature": float(args.temperature),
                "aux_context": "zero-v1-vs",
            },
        }
        epoch_path = lineage / f"v5_v3_distill_epoch{epoch + 1:02d}.pt.gz"
        save_checkpoint(payload, epoch_path)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics
            save_checkpoint(payload, args.output)
        print(
            json.dumps({"epoch": epoch + 1, "checkpoint": str(epoch_path), "metrics": metrics}),
            flush=True,
        )
    result = {"checkpoint": str(args.output), "optimizer_steps": step, "metrics": best_metrics}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--afterstates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-dir", type=Path)
    parser.add_argument("--student-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-rows-per-shard", type=int, default=2048)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-rows-per-shard", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
