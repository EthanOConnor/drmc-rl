"""Train one continuous Fightcade-strength human opponent.

The supported input is an immutable :class:`drmc_rl.data.HumanCorpus`
release. Replay facts and the release's current WHR-C trajectories are joined
at extraction time; no live database or sibling-repository imports cross this
boundary.

The checkpoint contains two deliberately separate models:

* a candidate placement policy conditioned on continuous WHR-C;
* a timing model for human slack beyond the exact planner-minimal script.

Examples (tf3090):

    uv run python -m tools.train_human_policy extract --planner cuda --sample-modulus 32
    uv run python -m tools.train_human_policy train --device cuda --epochs 8
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.data.human_corpus import DEFAULT_ROOT, HumanCorpus
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.model import (
    HUMAN_POLICY_SCHEMA,
    build_human_policy,
    build_timing_model,
    canonicalize_same_color_action,
    human_policy_config,
)
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.planning.fast_reach import compute_speed_threshold
from tools.annotate_replay_events import (
    GRID_H,
    GRID_W,
    POSE_TO_ACTION,
    occupancy_cols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "human_vs" / "human_policy_v1.npz"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "human_policy" / "human_policy_v1.pt.gz"
KMAX = 128

_COLUMNS = (
    "decision_id",
    "day",
    "player",
    "player_fold",
    "random_split",
    "field",
    "opp_field",
    "opp_state_age_frames",
    "pill_left",
    "pill_right",
    "preview_left",
    "preview_right",
    "speed",
    "speed_ups",
    "frame_counter",
    "held_at_spawn",
    "horizontal_velocity",
    "speed_counter",
    "lock_x",
    "lock_y_top",
    "lock_rotation",
    "tau_frames",
)


def _stable_keep(decision_id: str, modulus: int, seed: int) -> bool:
    digest = hashlib.blake2b(
        f"{int(seed)}:{decision_id}".encode(), digest_size=8, person=b"drmc-hum"
    ).digest()
    return int.from_bytes(digest, "little") % int(modulus) == 0


def _controller_state(held: int) -> tuple[int, int]:
    hold_dir = 1 if held & 0x02 else 2 if held & 0x01 else 0
    rot_hold = 1 if held & 0x80 else 2 if held & 0x40 else 0
    return hold_dir, rot_hold


def _empty_arrays() -> dict[str, list[Any]]:
    return {
        key: []
        for key in (
            "field",
            "opponent_field",
            "opponent_state_age_frames",
            "pill",
            "preview",
            "candidate_actions",
            "candidate_costs",
            "candidate_count",
            "chosen_slot",
            "rating",
            "rating_sd",
            "tau_frames",
            "chosen_cost",
            "speed",
            "speed_ups",
            "player_fold",
            "split",
            "time_split",
        )
    }


def _append_batch(
    output: dict[str, list[Any]],
    rows: list[dict[str, Any]],
    costs: np.ndarray,
    corpus: HumanCorpus,
) -> int:
    kept = 0
    for i, row in enumerate(rows):
        rating, rating_sd = corpus.rating_at(str(row["player"]), int(row["day"]))
        if rating is None:
            continue
        action_cost = np.full(512, 0xFFFF, dtype=np.uint16)
        for pose in np.flatnonzero(costs[i] != 0xFFFF):
            action = int(POSE_TO_ACTION[pose])
            if action >= 0:
                action_cost[action] = costs[i, pose]

        x = int(row["lock_x"])
        y = int(row["lock_y_top"])
        rotation = int(row["lock_rotation"]) & 3
        if not (0 <= x < GRID_W and 0 <= y < GRID_H):
            continue
        pose = rotation * 128 + y * GRID_W + x
        chosen = int(POSE_TO_ACTION[pose])
        if chosen < 0 or action_cost[chosen] == 0xFFFF:
            continue

        pill = (int(row["pill_left"]) & 3, int(row["pill_right"]) & 3)
        if pill[0] == pill[1]:
            action_cost[256:] = 0xFFFF
            chosen = canonicalize_same_color_action(chosen)
            if action_cost[chosen] == 0xFFFF:
                continue
        packed = pack_feasible_candidates(
            (action_cost != 0xFFFF).reshape(4, GRID_H, GRID_W),
            action_cost.reshape(4, GRID_H, GRID_W),
            max_candidates=KMAX,
            sort_by_cost=True,
        )
        slot = np.flatnonzero(packed.actions[: packed.count] == chosen)
        if slot.size == 0:
            continue

        output["field"].append(np.frombuffer(row["field"], dtype=np.uint8).copy())
        output["opponent_field"].append(np.frombuffer(row["opp_field"], dtype=np.uint8).copy())
        output["opponent_state_age_frames"].append(int(row["opp_state_age_frames"]))
        output["pill"].append(pill)
        output["preview"].append(
            (int(row["preview_left"]) & 3, int(row["preview_right"]) & 3)
        )
        output["candidate_actions"].append(packed.actions.astype(np.int16))
        output["candidate_costs"].append(packed.cost.astype(np.uint16))
        output["candidate_count"].append(int(packed.count))
        output["chosen_slot"].append(int(slot[0]))
        output["rating"].append(float(rating))
        output["rating_sd"].append(float(rating_sd or 0.0))
        output["tau_frames"].append(int(row["tau_frames"]))
        output["chosen_cost"].append(int(action_cost[chosen]))
        output["speed"].append(int(row["speed"]))
        output["speed_ups"].append(int(row["speed_ups"]))
        output["player_fold"].append(int(row["player_fold"]))
        output["split"].append({"train": 0, "validation": 1, "test": 2}[row["random_split"]])
        output["time_split"].append(
            {"train": 0, "validation": 1, "test": 2}[corpus.time_split(int(row["day"]))]
        )
        kept += 1
    return kept


def extract_dataset(
    corpus: HumanCorpus,
    *,
    planner_backend: str,
    sample_modulus: int,
    seed: int,
    max_rows: int | None,
    months: list[str] | None = None,
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    from tools.annotate_replay_events import make_batch_planner

    planner = make_batch_planner(planner_backend)
    output = _empty_arrays()
    scanned = sampled = kept = 0
    next_log = 10_000
    started = time.time()
    for batch in corpus.batches(
        "decisions", columns=list(_COLUMNS), months=months, batch_size=batch_size
    ):
        source_rows = batch.to_pylist()
        scanned += len(source_rows)
        rows = [
            row
            for row in source_rows
            if _stable_keep(str(row["decision_id"]), sample_modulus, seed)
        ]
        if max_rows is not None:
            rows = rows[: max(0, int(max_rows) - sampled)]
        if not rows:
            if max_rows is not None and sampled >= int(max_rows):
                break
            continue
        sampled += len(rows)

        n = len(rows)
        cols = np.zeros((n, 8), dtype=np.uint16)
        parity = np.zeros(n, dtype=np.uint8)
        threshold = np.zeros(n, dtype=np.uint8)
        speed_counter = np.zeros(n, dtype=np.uint8)
        horizontal_velocity = np.zeros(n, dtype=np.uint8)
        hold_dir = np.zeros(n, dtype=np.uint8)
        rot_hold = np.zeros(n, dtype=np.uint8)
        for i, row in enumerate(rows):
            field = np.frombuffer(row["field"], dtype=np.uint8).reshape(GRID_H, GRID_W)
            cols[i] = occupancy_cols(field)
            parity[i] = int(row["frame_counter"]) & 1
            threshold[i] = compute_speed_threshold(int(row["speed"]), int(row["speed_ups"]))
            speed_counter[i] = int(row["speed_counter"]) & 0xFF
            horizontal_velocity[i] = int(row["horizontal_velocity"]) & 0x0F
            hold_dir[i], rot_hold[i] = _controller_state(int(row["held_at_spawn"]))
        costs = planner(
            cols,
            parity,
            threshold,
            sc=speed_counter,
            hv=horizontal_velocity,
            hd=hold_dir,
            rh=rot_hold,
        )
        kept += _append_batch(output, rows, costs, corpus)
        if sampled >= next_log:
            print(
                f"scanned={scanned:,} sampled={sampled:,} kept={kept:,} "
                f"rate={sampled / max(time.time() - started, 1e-6):,.0f}/s",
                flush=True,
            )
            next_log += 10_000
        if max_rows is not None and sampled >= int(max_rows):
            break

    arrays = {key: np.asarray(value) for key, value in output.items()}
    arrays["corpus_release_id"] = np.asarray(corpus.release_id)
    arrays["sample_modulus"] = np.asarray(int(sample_modulus))
    arrays["seed"] = np.asarray(int(seed))
    return arrays


def batch_inputs(
    arrays: dict[str, np.ndarray], idx: np.ndarray, condition: HumanSkillCondition
):
    from drmc_rl.game.specs.ram_to_state import COLOR_VALUE_TO_INDEX

    count = len(idx)
    obs = np.zeros((count, 20, GRID_H, GRID_W), dtype=np.float32)
    actions = arrays["candidate_actions"][idx].astype(np.int32)
    candidate_count = arrays["candidate_count"][idx].astype(np.int64)
    mask = np.arange(KMAX)[None, :] < candidate_count[:, None]
    costs = arrays["candidate_costs"][idx].astype(np.float32)
    raw_pills = arrays["pill"][idx].astype(np.int64)
    raw_previews = arrays["preview"][idx].astype(np.int64)
    color_map = np.zeros(4, dtype=np.int64)
    for raw, canonical in COLOR_VALUE_TO_INDEX.items():
        color_map[int(raw) & 3] = int(canonical)
    pills = color_map[raw_pills]
    previews = color_map[raw_previews]
    obs[:, :8] = fields_to_planes(arrays["field"][idx])
    obs[:, 8:16] = fields_to_planes(arrays["opponent_field"][idx])
    rows, slots = np.nonzero(mask)
    valid_actions = actions[rows, slots]
    obs[:, 16:20].reshape(count, 512)[rows, valid_actions] = 1.0
    same_color = raw_pills[:, 0] == raw_pills[:, 1]
    obs[same_color, 6:8] = 0.0
    condition_batch = condition_features(arrays, idx, condition)
    return (
        obs,
        pills,
        previews,
        actions,
        costs,
        mask,
        arrays["chosen_slot"][idx].astype(np.int64),
        condition_batch,
    )


def fields_to_planes(fields: np.ndarray) -> np.ndarray:
    """Vectorized corpus field bytes -> canonical eight semantic planes."""

    field = np.asarray(fields, dtype=np.uint8).reshape(-1, GRID_H, GRID_W)
    high = field & 0xF0
    color = field & 0x03
    empty = field == 0xFF
    zero = field == 0x00
    clearing = (high == 0xB0) | ((high == 0xF0) & ~empty)
    valid = ~(empty | zero | clearing)
    planes = np.zeros((len(field), 8, GRID_H, GRID_W), dtype=np.float32)
    planes[:, 0] = (color == 1) & valid
    planes[:, 1] = (color == 0) & valid
    planes[:, 2] = (color == 2) & valid
    planes[:, 3] = high == 0xD0
    for code, channel in ((0x50, 4), (0x40, 5), (0x70, 6), (0x60, 7)):
        planes[:, channel] = high == code
    return planes


def condition_features(arrays: dict[str, np.ndarray], idx: np.ndarray, condition=None) -> np.ndarray:
    if condition is None:
        raise ValueError("condition is required")
    skill = condition.encode(arrays["rating"][idx])
    age = np.minimum(
        np.maximum(arrays["opponent_state_age_frames"][idx].astype(np.float32), 0.0), 240.0
    ) / 240.0
    return np.column_stack((skill, age)).astype(np.float32)


def timing_features(
    arrays: dict[str, np.ndarray], idx: np.ndarray, condition: HumanSkillCondition
) -> tuple[np.ndarray, np.ndarray]:
    boards = fields_to_planes(arrays["field"][idx])
    occupied = boards[:, :3].sum(axis=1) > 0
    any_by_row = occupied.any(axis=2)
    has_tiles = any_by_row.any(axis=1)
    first_row = np.argmax(any_by_row, axis=1)
    heights = np.where(has_tiles, GRID_H - first_row, 0).astype(np.float32)
    skill = condition.encode(arrays["rating"][idx])
    chosen_cost = arrays["chosen_cost"][idx].astype(np.float32)
    features = np.column_stack(
        (
            skill,
            chosen_cost / 120.0,
            occupied.mean(axis=(1, 2)),
            heights / GRID_H,
            arrays["speed"][idx].astype(np.float32) / 2.0,
            np.minimum(arrays["speed_ups"][idx].astype(np.float32), 20.0) / 20.0,
            np.minimum(arrays["candidate_count"][idx].astype(np.float32), 128.0) / 128.0,
        )
    ).astype(np.float32)
    slack = np.maximum(arrays["tau_frames"][idx].astype(np.float32) - chosen_cost, 0.0)
    return features, np.log1p(slack).astype(np.float32)


def _sample_weights(ratings: np.ndarray, bins: int = 20) -> np.ndarray:
    counts, edges = np.histogram(ratings, bins=bins)
    bucket = np.clip(np.searchsorted(edges, ratings, side="right") - 1, 0, bins - 1)
    weights = 1.0 / np.sqrt(np.maximum(counts[bucket], 1))
    return (weights / weights.mean()).astype(np.float32)


def train(
    arrays: dict[str, np.ndarray],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    capacity: str,
) -> tuple[Any, Any, dict[str, Any], HumanSkillCondition]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    use_bfloat16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        if use_bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    validation = (
        (arrays["split"] != 0)
        | (arrays["player_fold"] == 0)
        | (arrays["time_split"] != 0)
    )
    train_idx = np.flatnonzero(~validation)
    val_idx = np.flatnonzero(validation)
    condition = HumanSkillCondition.fit(arrays["rating"][train_idx])
    cfg = human_policy_config(capacity=capacity, candidate_max=KMAX)
    policy = build_human_policy(cfg, device=device)
    timing = build_timing_model(device=device)
    policy.train()
    timing.train()
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(timing.parameters()), lr=float(lr), weight_decay=1e-4
    )
    # Train only on old, randomly assigned training replays from non-held-out
    # players. The three independent holdouts diagnose replay memorization,
    # identity generalization, and temporal drift separately.
    weights = np.ones(len(arrays["rating"]), dtype=np.float32)
    weights[train_idx] = _sample_weights(arrays["rating"][train_idx])
    rng = np.random.default_rng(int(seed))
    losses: list[float] = []
    for epoch in range(int(epochs)):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                arrays, idx, condition
            )
            tf, timing_target = timing_features(arrays, idx, condition)
            mask_t = torch.from_numpy(mask).to(device)
            with autocast():
                logits, _ = policy(
                    torch.from_numpy(obs).to(device),
                    torch.from_numpy(pills).to(device),
                    torch.from_numpy(previews).to(device),
                    torch.from_numpy(actions).to(device),
                    torch.from_numpy(costs).to(device),
                    mask_t,
                    aux=torch.from_numpy(aux).to(device),
                )
                ce = F.cross_entropy(
                    logits.masked_fill(~mask_t, -1e9),
                    torch.from_numpy(slots).to(device),
                    reduction="none",
                )
                timing_out = timing(torch.from_numpy(tf).to(device))
                mean = timing_out[:, 0]
                log_std = timing_out[:, 1].clamp(-3.0, 2.0)
                target = torch.from_numpy(timing_target).to(device)
                nll = 0.5 * ((target - mean) / log_std.exp()).square() + log_std
                weight_t = torch.from_numpy(weights[idx]).to(device)
                loss = (ce * weight_t).mean() + 0.15 * (nll * weight_t).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(timing.parameters()), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} loss={np.mean(losses[-max(1, len(order)//batch_size):]):.4f}")

    policy.eval()
    timing.eval()
    metrics: dict[str, Any] = {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "loss_final": float(np.mean(losses[-100:])),
        "bfloat16": bool(use_bfloat16),
    }
    with torch.inference_mode():
        evaluation_sets = (
            ("train", train_idx[:20000]),
            (
                "replay_holdout",
                np.flatnonzero(
                    (arrays["split"] != 0)
                    & (arrays["player_fold"] != 0)
                    & (arrays["time_split"] == 0)
                )[:50000],
            ),
            (
                "player_holdout",
                np.flatnonzero(
                    (arrays["player_fold"] == 0)
                    & (arrays["split"] == 0)
                    & (arrays["time_split"] == 0)
                )[:50000],
            ),
            (
                "future_holdout",
                np.flatnonzero(
                    (arrays["time_split"] != 0)
                    & (arrays["split"] == 0)
                    & (arrays["player_fold"] != 0)
                )[:50000],
            ),
        )
        for name, eval_idx in evaluation_sets:
            correct = total = 0
            nll_sum = 0.0
            mean_rating_nll_sum = 0.0
            timing_abs_log_sum = 0.0
            timing_abs_frames_sum = 0.0
            for start in range(0, len(eval_idx), 512):
                idx = eval_idx[start : start + 512]
                obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                    arrays, idx, condition
                )
                with autocast():
                    logits, _ = policy(
                        torch.from_numpy(obs).to(device),
                        torch.from_numpy(pills).to(device),
                        torch.from_numpy(previews).to(device),
                        torch.from_numpy(actions).to(device),
                        torch.from_numpy(costs).to(device),
                        torch.from_numpy(mask).to(device),
                        aux=torch.from_numpy(aux).to(device),
                    )
                log_probs = logits.masked_fill(~torch.from_numpy(mask).to(device), -1e9).log_softmax(-1)
                targets = torch.from_numpy(slots).to(device)
                correct += int((log_probs.argmax(-1) == targets).sum().item())
                nll_sum += float((-log_probs[torch.arange(len(idx), device=device), targets]).sum().item())
                mean_aux = aux.copy()
                mean_aux[:, :2] = condition.encode(condition.mean)
                with autocast():
                    mean_logits, _ = policy(
                        torch.from_numpy(obs).to(device),
                        torch.from_numpy(pills).to(device),
                        torch.from_numpy(previews).to(device),
                        torch.from_numpy(actions).to(device),
                        torch.from_numpy(costs).to(device),
                        torch.from_numpy(mask).to(device),
                        aux=torch.from_numpy(mean_aux).to(device),
                    )
                mean_log_probs = mean_logits.masked_fill(
                    ~torch.from_numpy(mask).to(device), -1e9
                ).log_softmax(-1)
                mean_rating_nll_sum += float(
                    (-mean_log_probs[torch.arange(len(idx), device=device), targets]).sum().item()
                )
                timing_input, timing_target = timing_features(arrays, idx, condition)
                with autocast():
                    timing_output = timing(torch.from_numpy(timing_input).to(device))
                predicted_log = timing_output[:, 0].float().cpu().numpy()
                timing_abs_log_sum += float(np.abs(predicted_log - timing_target).sum())
                predicted_frames = np.maximum(np.expm1(predicted_log), 0.0)
                target_frames = np.expm1(timing_target)
                timing_abs_frames_sum += float(np.abs(predicted_frames - target_frames).sum())
                total += len(idx)
            metrics[f"{name}_top1"] = correct / max(total, 1)
            metrics[f"{name}_nll"] = nll_sum / max(total, 1)
            metrics[f"{name}_mean_rating_nll"] = mean_rating_nll_sum / max(total, 1)
            metrics[f"{name}_rating_gain_nll"] = (
                mean_rating_nll_sum - nll_sum
            ) / max(total, 1)
            metrics[f"{name}_timing_log_mae"] = timing_abs_log_sum / max(total, 1)
            metrics[f"{name}_timing_frames_mae"] = timing_abs_frames_sum / max(total, 1)
    return policy, timing, {"cfg": cfg, "metrics": metrics}, condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--corpus-root", default=str(DEFAULT_ROOT))
    extract.add_argument("--release", default="latest")
    extract.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    extract.add_argument("--planner", choices=("cpu", "cuda"), default="cpu")
    extract.add_argument("--sample-modulus", type=int, default=64)
    extract.add_argument("--max-rows", type=int)
    extract.add_argument(
        "--months",
        help="optional comma-separated YYYY-MM shards (use a spread across eras for pilots)",
    )
    extract.add_argument("--seed", type=int, default=0)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--capacity", choices=("small", "medium"), default="medium")
    args = parser.parse_args()

    if args.command == "extract":
        corpus = HumanCorpus(args.corpus_root, release=args.release)
        arrays = extract_dataset(
            corpus,
            planner_backend=args.planner,
            sample_modulus=max(1, args.sample_modulus),
            seed=args.seed,
            max_rows=args.max_rows,
            months=None if not args.months else [value.strip() for value in args.months.split(",")],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **arrays)
        print(f"wrote {args.out} rows={len(arrays['rating']):,} release={corpus.release_id}")
        return

    import torch
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    torch.set_num_threads(2)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    policy, timing, result, condition = train(
        arrays,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        capacity=args.capacity,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        {
            "schema": HUMAN_POLICY_SCHEMA,
            "state_dict": {key: value.cpu() for key, value in policy.state_dict().items()},
            "timing_state_dict": {key: value.cpu() for key, value in timing.state_dict().items()},
            "cfg": result["cfg"],
            "human_meta": {
                "skill_condition": condition.to_dict(),
                "corpus_release_id": str(arrays["corpus_release_id"]),
                "dataset": str(args.dataset),
                "metrics": result["metrics"],
                "trained_at": time.time(),
            },
        },
        args.out,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
