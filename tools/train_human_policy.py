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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.data.human_corpus import DEFAULT_ROOT, HumanCorpus
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.model import (
    HUMAN_POLICY_SCHEMA,
    HISTORY_FEATURE_DIM,
    HISTORY_STEPS,
    build_human_policy,
    build_timing_model,
    canonicalize_same_color_action,
    history_features,
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
DEFAULT_DATASET = REPO_ROOT / "data" / "human_vs" / "human_policy_v2"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "human_policy" / "human_policy_v2.pt.gz"
KMAX = 128

_COLUMNS = (
    "decision_id",
    "game_id",
    "day",
    "player",
    "opponent",
    "player_slot",
    "player_fold",
    "random_split",
    "won",
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


def _attach_temporal_context(
    rows: list[dict[str, Any]],
    histories: dict[tuple[str, int], list[dict[str, float | int]]],
    decision_counts: dict[tuple[str, int], int],
    *,
    retained_ids: set[str] | None = None,
) -> None:
    """Attach leak-free prior-decision features before sampling current rows."""

    for row in rows:
        key = (str(row["game_id"]), int(row["player_slot"]))
        recent = histories.setdefault(key, [])
        count = decision_counts.get(key, 0)
        retained = retained_ids is None or str(row["decision_id"]) in retained_ids
        if retained:
            row["_history"] = history_features(recent)
            row["_game_phase"] = min(count, 100) / 100.0
        x = int(row["lock_x"])
        y = int(row["lock_y_top"])
        rotation = int(row["lock_rotation"]) & 3
        action = -1
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            action = int(POSE_TO_ACTION[rotation * 128 + y * GRID_W + x])
            if action >= 0 and int(row["pill_left"]) == int(row["pill_right"]):
                action = canonicalize_same_color_action(action)
        if action >= 0:
            recent.insert(0, {"action": action, "tau_frames": int(row["tau_frames"])})
            del recent[HISTORY_STEPS:]
        decision_counts[key] = count + 1


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
            "opponent_rating",
            "opponent_rating_sd",
            "game_phase",
            "history",
            "won",
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
        opponent_rating, opponent_rating_sd = corpus.rating_at(
            str(row["opponent"]), int(row["day"])
        )
        if opponent_rating is None:
            opponent_rating = rating
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
        output["opponent_rating"].append(float(opponent_rating))
        output["opponent_rating_sd"].append(float(opponent_rating_sd or 0.0))
        output["game_phase"].append(float(row["_game_phase"]))
        output["history"].append(row["_history"])
        output["won"].append(bool(row["won"]))
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
    histories: dict[tuple[str, int], list[dict[str, float | int]]] = {}
    decision_counts: dict[tuple[str, int], int] = {}
    scanned = sampled = kept = 0
    next_log = 10_000
    started = time.time()
    for batch in corpus.batches(
        "decisions", columns=list(_COLUMNS), months=months, batch_size=batch_size
    ):
        source_rows = batch.to_pylist()
        scanned += len(source_rows)
        retained_ids = {
            str(row["decision_id"])
            for row in source_rows
            if _stable_keep(str(row["decision_id"]), sample_modulus, seed)
        }
        _attach_temporal_context(
            source_rows, histories, decision_counts, retained_ids=retained_ids
        )
        rows = [row for row in source_rows if str(row["decision_id"]) in retained_ids]
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


def corpus_months(corpus: HumanCorpus) -> list[str]:
    """Return authoritative YYYY-MM behavior partitions in release order."""

    months = []
    for entry in corpus.files("decisions"):
        parts = Path(entry.path).parts
        year = next(part.split("=", 1)[1] for part in parts if part.startswith("year="))
        month = next(part.split("=", 1)[1] for part in parts if part.startswith("month="))
        months.append(f"{year}-{month}")
    return sorted(set(months))


def extract_shards(
    corpus: HumanCorpus,
    output_dir: Path,
    *,
    planner_backend: str,
    sample_modulus: int,
    seed: int,
    months: list[str] | None = None,
    max_rows: int | None = None,
) -> list[Path]:
    """Extract independently loadable month shards for bounded-memory training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    remaining = None if max_rows is None else int(max_rows)
    for month in months or corpus_months(corpus):
        if remaining is not None and remaining <= 0:
            break
        arrays = extract_dataset(
            corpus,
            planner_backend=planner_backend,
            sample_modulus=sample_modulus,
            seed=seed,
            max_rows=remaining,
            months=[month],
        )
        path = output_dir / f"{month}.npz"
        np.savez_compressed(path, **arrays)
        rows = len(arrays["rating"])
        print(f"wrote {path} rows={rows:,}", flush=True)
        paths.append(path)
        if remaining is not None:
            remaining -= rows
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": HUMAN_POLICY_SCHEMA,
                "corpus_release_id": corpus.release_id,
                "sample_modulus": int(sample_modulus),
                "seed": int(seed),
                "shards": [path.name for path in paths],
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _prefetched(paths: list[Path]):
    """Load one compressed shard ahead while the current shard trains."""

    if not paths:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="human-shard") as executor:
        future = executor.submit(_load_npz, paths[0])
        for i, path in enumerate(paths):
            arrays = future.result()
            if i + 1 < len(paths):
                future = executor.submit(_load_npz, paths[i + 1])
            yield path, arrays


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
    opponent_skill = condition.encode(arrays["opponent_rating"][idx])[:, 0]
    age = np.minimum(
        np.maximum(arrays["opponent_state_age_frames"][idx].astype(np.float32), 0.0), 240.0
    ) / 240.0
    base = np.column_stack(
        (
            skill,
            np.minimum(np.maximum(arrays["rating_sd"][idx], 0.0), 500.0) / 500.0,
            opponent_skill,
            np.clip(
                (arrays["rating"][idx] - arrays["opponent_rating"][idx]) / condition.scale,
                -4.0,
                4.0,
            )
            / 4.0,
            np.minimum(np.maximum(arrays["opponent_rating_sd"][idx], 0.0), 500.0)
            / 500.0,
            age,
            arrays["game_phase"][idx],
        )
    )
    history = arrays["history"][idx].reshape(len(idx), HISTORY_STEPS * HISTORY_FEATURE_DIM)
    return np.concatenate((base, history), axis=1).astype(np.float32)


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
    opponent_skill = condition.encode(arrays["opponent_rating"][idx])[:, 0]
    previous_tau = np.expm1(arrays["history"][idx, 7] * 6.0)
    chosen_cost = arrays["chosen_cost"][idx].astype(np.float32)
    features = np.column_stack(
        (
            skill,
            np.minimum(np.maximum(arrays["rating_sd"][idx], 0.0), 500.0) / 500.0,
            np.clip(opponent_skill, -4.0, 4.0) / 4.0,
            arrays["game_phase"][idx],
            np.minimum(previous_tau, 300.0) / 300.0,
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
    if len(ratings) == 0:
        return np.empty(0, dtype=np.float32)
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
    patience: int = 3,
    value_weight: float = 0.25,
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
    parameters = list(policy.parameters()) + list(timing.parameters())
    optimizer_kwargs: dict[str, Any] = {"lr": float(lr), "weight_decay": 1e-4}
    if str(device).startswith("cuda"):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.05
    )
    # Train only on old, randomly assigned training replays from non-held-out
    # players. The three independent holdouts diagnose replay memorization,
    # identity generalization, and temporal drift separately.
    weights = np.ones(len(arrays["rating"]), dtype=np.float32)
    weights[train_idx] = _sample_weights(arrays["rating"][train_idx])
    rng = np.random.default_rng(int(seed))
    losses: list[float] = []
    selection_idx = rng.choice(val_idx, size=min(len(val_idx), 50_000), replace=False)
    best_objective = float("inf")
    best_epoch = 0
    best_policy: dict[str, Any] | None = None
    best_timing: dict[str, Any] | None = None
    epochs_without_improvement = 0

    def validation_objective() -> float:
        policy.eval()
        timing.eval()
        total = 0
        objective = 0.0
        with torch.inference_mode():
            for start in range(0, len(selection_idx), 512):
                idx = selection_idx[start : start + 512]
                obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                    arrays, idx, condition
                )
                tf, timing_target = timing_features(arrays, idx, condition)
                mask_t = torch.from_numpy(mask).to(device)
                with autocast():
                    logits, values = policy(
                        torch.from_numpy(obs).to(device),
                        torch.from_numpy(pills).to(device),
                        torch.from_numpy(previews).to(device),
                        torch.from_numpy(actions).to(device),
                        torch.from_numpy(costs).to(device),
                        mask_t,
                        aux=torch.from_numpy(aux).to(device),
                    )
                    behavior = F.cross_entropy(
                        logits.masked_fill(~mask_t, -1e9),
                        torch.from_numpy(slots).to(device),
                    )
                    outcome = F.binary_cross_entropy_with_logits(
                        values.reshape(-1),
                        torch.from_numpy(arrays["won"][idx].astype(np.float32)).to(device),
                    )
                    timing_out = timing(torch.from_numpy(tf).to(device))
                    mean = timing_out[:, 0]
                    log_std = timing_out[:, 1].clamp(-3.0, 2.0)
                    target = torch.from_numpy(timing_target).to(device)
                    timing_nll = (
                        0.5 * ((target - mean) / log_std.exp()).square() + log_std
                    ).mean()
                    batch_objective = behavior + float(value_weight) * outcome + 0.15 * timing_nll
                objective += float(batch_objective) * len(idx)
                total += len(idx)
        policy.train()
        timing.train()
        return objective / max(total, 1)

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
                logits, values = policy(
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
                outcome = F.binary_cross_entropy_with_logits(
                    values.reshape(-1),
                    torch.from_numpy(arrays["won"][idx].astype(np.float32)).to(device),
                    reduction="none",
                )
                weight_t = torch.from_numpy(weights[idx]).to(device)
                loss = (
                    (ce * weight_t).mean()
                    + 0.15 * (nll * weight_t).mean()
                    + float(value_weight) * (outcome * weight_t).mean()
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val_objective = validation_objective()
        print(
            f"epoch={epoch + 1} loss={np.mean(losses[-max(1, len(order)//batch_size):]):.4f} "
            f"validation={val_objective:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if val_objective < best_objective - 1e-4:
            best_objective = val_objective
            best_epoch = epoch + 1
            best_policy = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            best_timing = {key: value.detach().cpu().clone() for key, value in timing.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch + 1 >= 2 and epochs_without_improvement >= int(patience):
                print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch}")
                break

    if best_policy is not None and best_timing is not None:
        policy.load_state_dict(best_policy)
        timing.load_state_dict(best_timing)

    policy.eval()
    timing.eval()
    metrics: dict[str, Any] = {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "loss_final": float(np.mean(losses[-100:])),
        "bfloat16": bool(use_bfloat16),
        "best_epoch": int(best_epoch),
        "best_validation_objective": float(best_objective),
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
            outcome_bce_sum = 0.0
            outcome_brier_sum = 0.0
            outcome_correct = 0
            for start in range(0, len(eval_idx), 512):
                idx = eval_idx[start : start + 512]
                obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                    arrays, idx, condition
                )
                with autocast():
                    logits, values = policy(
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
                outcome_targets = arrays["won"][idx].astype(np.float32)
                outcome_probs = torch.sigmoid(values.reshape(-1).float()).cpu().numpy()
                outcome_bce_sum += float(
                    -(
                        outcome_targets * np.log(np.maximum(outcome_probs, 1e-7))
                        + (1.0 - outcome_targets)
                        * np.log(np.maximum(1.0 - outcome_probs, 1e-7))
                    ).sum()
                )
                outcome_brier_sum += float(((outcome_probs - outcome_targets) ** 2).sum())
                outcome_correct += int(((outcome_probs >= 0.5) == outcome_targets).sum())
                total += len(idx)
            metrics[f"{name}_top1"] = correct / max(total, 1)
            metrics[f"{name}_nll"] = nll_sum / max(total, 1)
            metrics[f"{name}_mean_rating_nll"] = mean_rating_nll_sum / max(total, 1)
            metrics[f"{name}_rating_gain_nll"] = (
                mean_rating_nll_sum - nll_sum
            ) / max(total, 1)
            metrics[f"{name}_timing_log_mae"] = timing_abs_log_sum / max(total, 1)
            metrics[f"{name}_timing_frames_mae"] = timing_abs_frames_sum / max(total, 1)
            metrics[f"{name}_outcome_bce"] = outcome_bce_sum / max(total, 1)
            metrics[f"{name}_outcome_brier"] = outcome_brier_sum / max(total, 1)
            metrics[f"{name}_outcome_accuracy"] = outcome_correct / max(total, 1)
    return policy, timing, {"cfg": cfg, "metrics": metrics}, condition


def _training_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (arrays["split"] == 0)
        & (arrays["player_fold"] != 0)
        & (arrays["time_split"] == 0)
    )


def _concat_rows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no validation rows found in dataset shards")
    row_keys = [
        key
        for key, value in parts[0].items()
        if np.asarray(value).ndim > 0 and len(value) == len(parts[0]["rating"])
    ]
    return {key: np.concatenate([part[key] for part in parts]) for key in row_keys}


def _shard_statistics(
    paths: list[Path], *, seed: int, validation_rows_per_shard: int = 2048
) -> tuple[HumanSkillCondition, dict[str, np.ndarray], int]:
    """Streaming skill moments plus a deterministic cross-shard validation set."""

    count = 0
    rating_sum = rating_square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    validation_parts = []
    rng = np.random.default_rng(int(seed))
    for _path, arrays in _prefetched(paths):
        train_mask = _training_mask(arrays)
        ratings = arrays["rating"][train_mask].astype(np.float64)
        count += len(ratings)
        rating_sum += float(ratings.sum())
        rating_square_sum += float(np.square(ratings).sum())
        if len(ratings):
            minimum = min(minimum, float(ratings.min()))
            maximum = max(maximum, float(ratings.max()))
        groups = (
            (1, (arrays["split"] != 0) & (arrays["player_fold"] != 0) & (arrays["time_split"] == 0)),
            (2, (arrays["player_fold"] == 0) & (arrays["split"] == 0) & (arrays["time_split"] == 0)),
            (3, (arrays["time_split"] != 0) & (arrays["split"] == 0) & (arrays["player_fold"] != 0)),
        )
        per_group = max(1, int(validation_rows_per_shard) // len(groups))
        for group, group_mask in groups:
            validation = np.flatnonzero(group_mask)
            if not len(validation):
                continue
            chosen = rng.choice(
                validation,
                size=min(len(validation), per_group),
                replace=False,
            )
            part = {
                key: value[chosen]
                for key, value in arrays.items()
                if np.asarray(value).ndim > 0 and len(value) == len(arrays["rating"])
            }
            part["evaluation_group"] = np.full(len(chosen), group, dtype=np.uint8)
            validation_parts.append(part)
    if count == 0:
        raise ValueError("dataset shards contain no training rows")
    mean = rating_sum / count
    variance = max(rating_square_sum / count - mean * mean, 1.0)
    condition = HumanSkillCondition(
        mean=float(mean),
        scale=float(np.sqrt(variance)),
        minimum=float(minimum),
        maximum=float(maximum),
    )
    return condition, _concat_rows(validation_parts), count


def _evaluate_compact(
    policy,
    timing,
    arrays: dict[str, np.ndarray],
    condition: HumanSkillCondition,
    *,
    device: str,
    batch_size: int = 512,
) -> dict[str, float]:
    """Bounded validation used for epoch selection in streaming training."""

    import torch
    import torch.nn.functional as F

    policy.eval()
    timing.eval()
    totals = {
        "rows": 0.0,
        "behavior_nll": 0.0,
        "top1": 0.0,
        "outcome_bce": 0.0,
        "outcome_brier": 0.0,
        "timing_nll": 0.0,
        "timing_frames_mae": 0.0,
    }
    use_bfloat16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bfloat16
        else contextlib.nullcontext()
    )
    with torch.inference_mode():
        for start in range(0, len(arrays["rating"]), int(batch_size)):
            idx = np.arange(start, min(start + int(batch_size), len(arrays["rating"])))
            obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                arrays, idx, condition
            )
            tf, timing_target = timing_features(arrays, idx, condition)
            mask_t = torch.from_numpy(mask).to(device)
            targets = torch.from_numpy(slots).to(device)
            with autocast():
                logits, values = policy(
                    torch.from_numpy(obs).to(device),
                    torch.from_numpy(pills).to(device),
                    torch.from_numpy(previews).to(device),
                    torch.from_numpy(actions).to(device),
                    torch.from_numpy(costs).to(device),
                    mask_t,
                    aux=torch.from_numpy(aux).to(device),
                )
                timing_out = timing(torch.from_numpy(tf).to(device))
            log_probs = logits.masked_fill(~mask_t, -1e9).float().log_softmax(-1)
            outcome_targets = torch.from_numpy(arrays["won"][idx].astype(np.float32)).to(device)
            outcome_probs = torch.sigmoid(values.reshape(-1).float())
            log_std = timing_out[:, 1].float().clamp(-3.0, 2.0)
            timing_targets = torch.from_numpy(timing_target).to(device)
            timing_nll = (
                0.5 * ((timing_targets - timing_out[:, 0].float()) / log_std.exp()).square()
                + log_std
            )
            totals["rows"] += len(idx)
            totals["behavior_nll"] += float(-log_probs[torch.arange(len(idx), device=device), targets].sum())
            totals["top1"] += float((log_probs.argmax(-1) == targets).sum())
            totals["outcome_bce"] += float(
                F.binary_cross_entropy(outcome_probs, outcome_targets, reduction="sum")
            )
            totals["outcome_brier"] += float(torch.square(outcome_probs - outcome_targets).sum())
            totals["timing_nll"] += float(timing_nll.sum())
            predicted_frames = np.maximum(np.expm1(timing_out[:, 0].float().cpu().numpy()), 0.0)
            target_frames = np.expm1(timing_target)
            totals["timing_frames_mae"] += float(np.abs(predicted_frames - target_frames).sum())
    rows = max(totals.pop("rows"), 1.0)
    return {key: value / rows for key, value in totals.items()} | {"rows": rows}


def train_sharded(
    paths: list[Path],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    capacity: str,
    patience: int = 3,
    value_weight: float = 0.25,
) -> tuple[Any, Any, dict[str, Any], HumanSkillCondition]:
    """Train over every row while keeping at most one month shard resident."""

    import torch
    import torch.nn.functional as F

    condition, validation, train_rows = _shard_statistics(paths, seed=seed)
    cfg = human_policy_config(capacity=capacity, candidate_max=KMAX)
    policy = build_human_policy(cfg, device=device)
    timing = build_timing_model(device=device)
    parameters = list(policy.parameters()) + list(timing.parameters())
    optimizer_kwargs: dict[str, Any] = {"lr": float(lr), "weight_decay": 1e-4}
    if str(device).startswith("cuda"):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.05
    )
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    use_bfloat16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bfloat16
        else contextlib.nullcontext()
    )
    best_objective = float("inf")
    best_epoch = 0
    best_policy = best_timing = None
    stale_epochs = 0
    steps = 0
    for epoch in range(int(epochs)):
        policy.train()
        timing.train()
        epoch_loss = 0.0
        epoch_rows = 0
        ordered_paths = [paths[i] for i in rng.permutation(len(paths))]
        for path, arrays in _prefetched(ordered_paths):
            train_idx = np.flatnonzero(_training_mask(arrays))
            rng.shuffle(train_idx)
            weights = _sample_weights(arrays["rating"][train_idx])
            for start in range(0, len(train_idx), int(batch_size)):
                idx = train_idx[start : start + int(batch_size)]
                local = np.arange(start, start + len(idx))
                obs, pills, previews, actions, costs, mask, slots, aux = batch_inputs(
                    arrays, idx, condition
                )
                tf, timing_target = timing_features(arrays, idx, condition)
                mask_t = torch.from_numpy(mask).to(device)
                with autocast():
                    logits, values = policy(
                        torch.from_numpy(obs).to(device),
                        torch.from_numpy(pills).to(device),
                        torch.from_numpy(previews).to(device),
                        torch.from_numpy(actions).to(device),
                        torch.from_numpy(costs).to(device),
                        mask_t,
                        aux=torch.from_numpy(aux).to(device),
                    )
                    weight_t = torch.from_numpy(weights[local]).to(device)
                    behavior = F.cross_entropy(
                        logits.masked_fill(~mask_t, -1e9),
                        torch.from_numpy(slots).to(device),
                        reduction="none",
                    )
                    timing_out = timing(torch.from_numpy(tf).to(device))
                    log_std = timing_out[:, 1].clamp(-3.0, 2.0)
                    timing_target_t = torch.from_numpy(timing_target).to(device)
                    timing_nll = (
                        0.5
                        * ((timing_target_t - timing_out[:, 0]) / log_std.exp()).square()
                        + log_std
                    )
                    outcome = F.binary_cross_entropy_with_logits(
                        values.reshape(-1),
                        torch.from_numpy(arrays["won"][idx].astype(np.float32)).to(device),
                        reduction="none",
                    )
                    loss = (
                        (behavior * weight_t).mean()
                        + 0.15 * (timing_nll * weight_t).mean()
                        + float(value_weight) * (outcome * weight_t).mean()
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
                epoch_rows += len(idx)
                steps += 1
            print(f"epoch={epoch + 1} shard={path.name} rows={epoch_rows:,}", flush=True)
        scheduler.step()
        val = _evaluate_compact(policy, timing, validation, condition, device=device)
        objective = val["behavior_nll"] + float(value_weight) * val["outcome_bce"] + 0.15 * val["timing_nll"]
        print(
            f"epoch={epoch + 1} loss={epoch_loss / max(epoch_rows, 1):.4f} "
            f"validation={objective:.4f} top1={val['top1']:.4f} brier={val['outcome_brier']:.4f}",
            flush=True,
        )
        if objective < best_objective - 1e-4:
            best_objective = objective
            best_epoch = epoch + 1
            best_policy = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            best_timing = {key: value.detach().cpu().clone() for key, value in timing.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break
    if best_policy is not None and best_timing is not None:
        policy.load_state_dict(best_policy)
        timing.load_state_dict(best_timing)
    metrics = _evaluate_compact(policy, timing, validation, condition, device=device)
    historyless = dict(validation)
    historyless["history"] = np.zeros_like(validation["history"])
    historyless_metrics = _evaluate_compact(
        policy, timing, historyless, condition, device=device
    )
    skillless = dict(validation)
    skillless["rating"] = np.full_like(validation["rating"], condition.mean)
    skillless["opponent_rating"] = np.full_like(
        validation["opponent_rating"], condition.mean
    )
    skillless["rating_sd"] = np.zeros_like(validation["rating_sd"])
    skillless["opponent_rating_sd"] = np.zeros_like(validation["opponent_rating_sd"])
    skillless_metrics = _evaluate_compact(policy, timing, skillless, condition, device=device)
    metrics.update(
        {
            "history_gain_behavior_nll": historyless_metrics["behavior_nll"]
            - metrics["behavior_nll"],
            "history_gain_outcome_brier": historyless_metrics["outcome_brier"]
            - metrics["outcome_brier"],
            "skill_gain_behavior_nll": skillless_metrics["behavior_nll"]
            - metrics["behavior_nll"],
            "skill_gain_outcome_brier": skillless_metrics["outcome_brier"]
            - metrics["outcome_brier"],
        }
    )
    group_names = {1: "replay_holdout", 2: "player_holdout", 3: "future_holdout"}
    for group, name in group_names.items():
        selected = np.flatnonzero(validation["evaluation_group"] == group)
        if not len(selected):
            continue
        part = {
            key: value[selected]
            for key, value in validation.items()
            if np.asarray(value).ndim > 0 and len(value) == len(validation["rating"])
        }
        group_metrics = _evaluate_compact(policy, timing, part, condition, device=device)
        metrics.update({f"{name}_{key}": value for key, value in group_metrics.items()})
    metrics.update(
        {
            "train_rows": int(train_rows),
            "validation_rows": int(len(validation["rating"])),
            "best_epoch": int(best_epoch),
            "best_validation_objective": float(best_objective),
            "optimizer_steps": int(steps),
            "shards": int(len(paths)),
            "bfloat16": bool(use_bfloat16),
        }
    )
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
    train_parser.add_argument("--capacity", choices=("small", "medium", "large"), default="medium")
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--value-weight", type=float, default=0.25)
    args = parser.parse_args()

    if args.command == "extract":
        corpus = HumanCorpus(args.corpus_root, release=args.release)
        paths = extract_shards(
            corpus,
            args.out,
            planner_backend=args.planner,
            sample_modulus=max(1, args.sample_modulus),
            seed=args.seed,
            max_rows=args.max_rows,
            months=None if not args.months else [value.strip() for value in args.months.split(",")],
        )
        print(f"wrote {len(paths)} shards under {args.out} release={corpus.release_id}")
        return

    import torch
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    torch.set_num_threads(2)
    if args.dataset.is_dir():
        paths = sorted(args.dataset.glob("*.npz"))
        if not paths:
            raise SystemExit(f"no .npz shards under {args.dataset}")
        policy, timing, result, condition = train_sharded(
            paths,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            capacity=args.capacity,
            patience=args.patience,
            value_weight=args.value_weight,
        )
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        corpus_release_id = str(manifest["corpus_release_id"])
    else:
        arrays = _load_npz(args.dataset)
        policy, timing, result, condition = train(
            arrays,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            capacity=args.capacity,
            patience=args.patience,
            value_weight=args.value_weight,
        )
        corpus_release_id = str(arrays["corpus_release_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        {
            "schema": HUMAN_POLICY_SCHEMA,
            "state_dict": {key: value.cpu() for key, value in policy.state_dict().items()},
            "timing_state_dict": {key: value.cpu() for key, value in timing.state_dict().items()},
            "cfg": result["cfg"],
            "human_meta": {
                "skill_condition": condition.to_dict(),
                "corpus_release_id": corpus_release_id,
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
