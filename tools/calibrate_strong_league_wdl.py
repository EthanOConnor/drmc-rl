"""Fit aggregate and member-specific Strong League links to grouped W/D/L."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from drmc_rl.eval.wdl_calibration import calibration_report
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.search.strong_league import FrozenStrongLeagueMixture
from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_infer(mixture: FrozenStrongLeagueMixture, env, obs):
    batch = env.policy_batch("v1_vs")
    size = len(obs)
    actions = np.full((size, 128), -1, dtype=np.int32)
    masks = np.zeros((size, 128), dtype=bool)
    costs = np.zeros((size, 128), dtype=np.float32)
    for index in range(size):
        packed = pack_feasible_candidates(
            batch.feasible_mask[index],
            batch.cost_to_lock[index],
            max_candidates=128,
            sort_by_cost=True,
        )
        actions[index], masks[index], costs[index] = (
            packed.actions,
            packed.mask,
            packed.cost,
        )
    probabilities = np.zeros((size, 128), dtype=np.float64)
    member_values = np.zeros((size, len(mixture.members)), dtype=np.float64)
    for member_index, (weight, member) in enumerate(
        zip(mixture.weights, mixture.members, strict=True)
    ):
        with torch.inference_mode():
            logits, value = member.net(
                torch.from_numpy(obs[:, :16].astype(np.float32)).to(member.device),
                torch.from_numpy(batch.pill_colors.astype(np.int64)).to(member.device),
                torch.from_numpy(batch.preview_pill_colors.astype(np.int64)).to(member.device),
                torch.from_numpy(actions).to(member.device),
                torch.from_numpy(costs).to(member.device),
                torch.from_numpy(masks).to(member.device),
                aux=torch.from_numpy(batch.aux).to(member.device),
            )
        logits_np = logits.float().cpu().numpy()
        logits_np[~masks] = -np.inf
        valid_rows = masks.any(axis=1)
        maxima = np.zeros((size, 1), dtype=np.float32)
        maxima[valid_rows] = np.max(logits_np[valid_rows], axis=1, keepdims=True)
        member_probability = np.zeros_like(logits_np, dtype=np.float64)
        member_probability[valid_rows] = np.exp(
            np.clip(logits_np[valid_rows] - maxima[valid_rows], -60.0, 0.0)
        )
        member_probability[~masks] = 0.0
        member_probability /= np.maximum(member_probability.sum(axis=1, keepdims=True), 1e-12)
        probabilities += float(weight) * member_probability
        member_values[:, member_index] = value.reshape(-1).float().cpu().numpy()
    aggregate_values = member_values @ mixture.weights
    slots = probabilities.argmax(axis=1)
    chosen = actions[np.arange(size), slots]
    chosen[~masks.any(axis=1)] = -1
    return chosen.astype(np.int32), aggregate_values, member_values


def collect(
    mixture: FrozenStrongLeagueMixture,
    *,
    games_per_stratum: int,
    pairs: int,
    seed: int,
    lib_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    scores: list[float] = []
    member_scores: list[np.ndarray] = []
    outcomes: list[int] = []
    groups: list[int] = []
    strata: list[dict[str, int]] = []
    next_group = 0
    for level in (5, 10, 15, 20):
        for speed in (0, 1, 2):
            env = DrMarioVsPoolVecEnv(
                num_pairs=min(pairs, games_per_stratum),
                state_repr="bitplane_bottle_conn_mask_vs",
                level=level,
                speed_setting=speed,
                randomize_rng=True,
                garbage_reward_coef=0.0,
                match_horizon_pills=256,
                direct_policy_batch=True,
                lib_path=lib_path,
            )
            obs, infos = env.reset(seed=seed + level * 101 + speed * 1009)
            pending: list[list[float]] = [[] for _ in range(env.num_sides)]
            pending_members: list[list[np.ndarray]] = [
                [] for _ in range(env.num_sides)
            ]
            decisions = np.zeros(env.num_sides, dtype=np.int64)
            completed = 0
            attempted = 0
            truncated_games = 0
            try:
                while completed < games_per_stratum:
                    actions, values, values_by_member = _batch_infer(mixture, env, obs)
                    active = env.policy_batch("none").feasible_mask.reshape(
                        env.num_sides, -1
                    ).any(axis=1)
                    for side in range(env.num_sides):
                        # Fixed logarithmic-ish cadence avoids excessive within-game rows.
                        if active[side] and (decisions[side] < 8 or decisions[side] % 8 == 0):
                            pending[side].append(float(values[side]))
                            pending_members[side].append(values_by_member[side].copy())
                        decisions[side] += int(active[side])
                    obs, _reward, terminated, truncated, infos = env.step(actions)
                    for pair in range(env.num_pairs):
                        first = 2 * pair
                        if not bool(terminated[first] or truncated[first]):
                            continue
                        attempted += 1
                        if completed >= games_per_stratum:
                            for side in (first, first + 1):
                                pending[side].clear()
                                pending_members[side].clear()
                            continue
                        if bool(truncated[first]) and not bool(terminated[first]):
                            for side in (first, first + 1):
                                pending[side].clear()
                                pending_members[side].clear()
                            decisions[first : first + 2] = 0
                            truncated_games += 1
                            if attempted > max(100, 20 * games_per_stratum):
                                raise RuntimeError(
                                    "too many horizon truncations to collect natural W/D/L outcomes"
                                )
                            continue
                        for side in (first, first + 1):
                            outcome = infos[side].get("drm", {}).get("vs_outcome")
                            if outcome not in {"win", "draw", "loss"}:
                                raise RuntimeError(
                                    "natural terminal game is missing an authoritative VS outcome"
                                )
                            if len(pending[side]) != len(pending_members[side]):
                                raise RuntimeError("aggregate/member calibration rows diverged")
                            target = {"win": 0, "draw": 1, "loss": 2}[outcome]
                            scores.extend(pending[side])
                            member_scores.extend(pending_members[side])
                            outcomes.extend([target] * len(pending[side]))
                            groups.extend([next_group] * len(pending[side]))
                            pending[side].clear()
                            pending_members[side].clear()
                            decisions[side] = 0
                        next_group += 1
                        completed += 1
                strata.append(
                    {
                        "level": level,
                        "speed": speed,
                        "games": games_per_stratum,
                        "attempts": attempted,
                        "horizon_truncations_excluded": truncated_games,
                    }
                )
            finally:
                env.close()
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(member_scores, dtype=np.float32),
        np.asarray(outcomes, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        strata,
    )


def _heldout(report: dict[str, object]) -> dict[str, object]:
    crossfit = report["crossfit"]  # type: ignore[index]
    return {
        "validation_rows": report["rows"],
        "validation_games": report["games"],
        "baseline": crossfit["baseline"],  # type: ignore[index]
        "calibrated": crossfit["calibrated"],  # type: ignore[index]
        "paired_game_bootstrap": crossfit["paired_game_bootstrap"],  # type: ignore[index]
        "outcome_games": report["outcome_games"],
        "natural_draw_games": report["natural_draw_games"],
        "draw_identifiable": report["draw_identifiable"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-stratum", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-lib")
    args = parser.parse_args()
    if args.games_per_stratum < 1 or args.folds < 2 or args.bootstrap_samples < 1:
        parser.error("game, fold, and bootstrap counts must be positive")

    # Collection uses member policies/values only; fitted links replace this
    # temporary identity link before any counterfactual release.
    from drmc_rl.search.strong_league import DavidsonCalibration, MixtureMember

    payload = json.loads(args.mixture_manifest.read_text())
    base = args.mixture_manifest.parent
    members = []
    for item in payload["members"]:
        path = Path(item["checkpoint"])
        members.append(
            MixtureMember(
                str(item["id"]),
                path if path.is_absolute() else base / path,
                str(item["sha256"]),
                float(item["weight"]),
            )
        )
    mixture = FrozenStrongLeagueMixture(
        members,
        DavidsonCalibration(1.0, 0.0, -3.0, "collection-only"),
        device=args.device,
    )
    scores, member_scores, outcomes, groups, strata = collect(
        mixture,
        games_per_stratum=args.games_per_stratum,
        pairs=args.pairs,
        seed=args.seed,
        lib_path=args.native_lib,
    )
    report = calibration_report(
        scores,
        outcomes,
        groups,
        seed=args.seed,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    member_calibrations = {
        member.id: calibration_report(
            member_scores[:, index],
            outcomes,
            groups,
            seed=args.seed + 1000 + index,
            folds=args.folds,
            bootstrap_samples=args.bootstrap_samples,
        )
        for index, member in enumerate(members)
    }
    result = {
        "schema": "drmc-strong-league-wdl-calibration-v2",
        "mixture_manifest_sha256": _sha256(args.mixture_manifest),
        "parameters": report["parameters"],
        "heldout_metrics": _heldout(report),
        "grouped_calibration": report,
        "member_calibrations": member_calibrations,
        "member_parameters": {
            member_id: value["parameters"]
            for member_id, value in member_calibrations.items()
        },
        "collection": {
            "seed": args.seed,
            "rows": int(len(scores)),
            "games": int(sum(item["games"] for item in strata)),
            "strata": strata,
            "sampling": "first-8-and-every-8th-decision-per-side",
            "weighting": "equal-total-weight-per-game",
            "member_ids": [member.id for member in members],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
