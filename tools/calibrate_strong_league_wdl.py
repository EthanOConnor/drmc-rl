"""Fit a frozen Strong League value mixture to held-out native W/D/L outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

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
    values = np.zeros(size, dtype=np.float64)
    for weight, member in zip(mixture.weights, mixture.members, strict=True):
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
        values += float(weight) * value.reshape(-1).float().cpu().numpy()
    slots = probabilities.argmax(axis=1)
    chosen = actions[np.arange(size), slots]
    chosen[~masks.any(axis=1)] = -1
    return chosen.astype(np.int32), values


def collect(
    mixture: FrozenStrongLeagueMixture,
    *,
    games_per_stratum: int,
    pairs: int,
    seed: int,
    lib_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    scores: list[float] = []
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
            decisions = np.zeros(env.num_sides, dtype=np.int64)
            completed = 0
            attempted = 0
            truncated_games = 0
            try:
                while completed < games_per_stratum:
                    actions, values = _batch_infer(mixture, env, obs)
                    active = env.policy_batch("none").feasible_mask.reshape(
                        env.num_sides, -1
                    ).any(axis=1)
                    for side in range(env.num_sides):
                        # Fixed logarithmic-ish cadence avoids overweighting long endgames.
                        if active[side] and (
                            decisions[side] < 8 or decisions[side] % 8 == 0
                        ):
                            pending[side].append(float(values[side]))
                        decisions[side] += int(active[side])
                    obs, _reward, terminated, truncated, infos = env.step(actions)
                    for pair in range(env.num_pairs):
                        first = 2 * pair
                        if not bool(terminated[first] or truncated[first]):
                            continue
                        attempted += 1
                        if completed >= games_per_stratum:
                            pending[first].clear()
                            pending[first + 1].clear()
                            continue
                        if bool(truncated[first]) and not bool(terminated[first]):
                            # A search-horizon cutoff is not a competitive draw.
                            # Discard all samples from that game and wait for a
                            # natural terminal result from the autoreset pair.
                            pending[first].clear()
                            pending[first + 1].clear()
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
                            target = {"win": 0, "draw": 1, "loss": 2}.get(outcome)
                            if target is not None:
                                scores.extend(pending[side])
                                outcomes.extend([target] * len(pending[side]))
                                groups.extend([next_group] * len(pending[side]))
                            pending[side].clear()
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
        np.asarray(outcomes, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        strata,
    )


def _probabilities(scores: np.ndarray, slope: float, bias: float, draw_logit: float):
    strength = slope * scores.astype(np.float64) + bias
    logits = np.stack((strength, np.full_like(strength, draw_logit), -strength), axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def _metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    one_hot = np.eye(3, dtype=np.float64)[target]
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(np.maximum(probability[np.arange(len(target)), target], 1e-12))))
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == target
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= low) & (confidence < low + 0.1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"brier": brier, "log_loss": log_loss, "ece_10": ece}


def fit(scores: np.ndarray, outcomes: np.ndarray, groups: np.ndarray, split_seed: int):
    group_digest = {
        int(group): int.from_bytes(
            hashlib.sha256(f"{split_seed}:{int(group)}".encode()).digest()[:8], "big"
        )
        for group in np.unique(groups)
    }
    digest = np.asarray([group_digest[int(group)] for group in groups], dtype=np.uint64)
    validation = digest % 5 == 0
    train = ~validation
    x = torch.from_numpy(scores[train]).double()
    y = torch.from_numpy(outcomes[train]).long()
    raw_slope = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    draw = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [raw_slope, bias, draw], max_iter=200, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        slope = torch.nn.functional.softplus(raw_slope) + 1e-6
        strength = slope * x + bias
        logits = torch.stack((strength, draw.expand_as(strength), -strength), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    parameters = {
        "slope": float(torch.nn.functional.softplus(raw_slope).detach() + 1e-6),
        "bias": float(bias.detach()),
        "draw_logit": float(draw.detach()),
    }
    calibrated = _probabilities(scores[validation], **parameters)
    baseline = _probabilities(scores[validation], 1.0, 0.0, -3.0)
    return parameters, {
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "train_games": int(len(np.unique(groups[train]))),
        "validation_games": int(len(np.unique(groups[validation]))),
        "baseline": _metrics(baseline, outcomes[validation]),
        "calibrated": _metrics(calibrated, outcomes[validation]),
        "outcomes": {
            "win": int((outcomes[validation] == 0).sum()),
            "draw": int((outcomes[validation] == 1).sum()),
            "loss": int((outcomes[validation] == 2).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-stratum", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-lib")
    args = parser.parse_args()
    # Temporary identity link: collection uses only member policies/values;
    # the fitted artifact replaces this object before search release.
    from drmc_rl.search.strong_league import DavidsonCalibration

    payload = json.loads(args.mixture_manifest.read_text())
    base = args.mixture_manifest.parent
    from drmc_rl.search.strong_league import MixtureMember

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
    scores, outcomes, groups, strata = collect(
        mixture,
        games_per_stratum=args.games_per_stratum,
        pairs=args.pairs,
        seed=args.seed,
        lib_path=args.native_lib,
    )
    parameters, metrics = fit(scores, outcomes, groups, args.seed)
    result = {
        "schema": "drmc-strong-league-wdl-calibration-v1",
        "mixture_manifest_sha256": _sha256(args.mixture_manifest),
        "parameters": parameters,
        "heldout_metrics": metrics,
        "collection": {
            "seed": args.seed,
            "rows": int(len(scores)),
            "games": int(sum(item["games"] for item in strata)),
            "strata": strata,
            "sampling": "first-8-and-every-8th-decision-per-side",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
