"""Calibrate requested human rating/search weight by native head-to-head play.

The ladder pits each requested rating against a fixed anchor using the same
checkpoint and alternates which side owns the probe.  This measures realized
playing strength rather than assuming WHR-conditioned imitation likelihood is
already a monotone win-rate dial.

Example:
    python -m tools.calibrate_human_strength \
      --checkpoint runs/human_policy/human_policy_v2.pt.gz \
      --ratings 1000,1300,1600,1900,2200 --anchor 1600 --matches 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.human.runtime import HumanPolicyRuntime
from drmc_rl.human.search import HumanValueSearch, blend_human_and_search, competitive_scores
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

_RAW_TO_CANONICAL = np.asarray([1, 0, 2], dtype=np.int64)


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def elo_from_win_rate(win_rate: float, anchor: float) -> float:
    """Standard logistic-Elo equivalent, clipped away from infinite tails."""

    p = float(np.clip(win_rate, 0.01, 0.99))
    return float(anchor + 400.0 * math.log10(p / (1.0 - p)))


def relative_elo(matchups: list[dict[str, Any]], labels: list[str]) -> dict[str, float]:
    """Fit centered Bradley-Terry ratings; the zero point is intentionally arbitrary."""

    count = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    scores = np.full(count, 0.5, dtype=np.float64)
    games = np.ones((count, count), dtype=np.float64)
    np.fill_diagonal(games, 0.0)
    for row in matchups:
        left = index[str(row["left"])]
        right = index[str(row["right"])]
        scores[left] += float(row["wins"]) + 0.5 * float(row["draws"])
        scores[right] += float(row["losses"]) + 0.5 * float(row["draws"])
        games[left, right] += float(row["matches"])
        games[right, left] += float(row["matches"])
    ability = np.ones(count, dtype=np.float64)
    for _ in range(10_000):
        denominator = np.zeros(count, dtype=np.float64)
        for i in range(count):
            for j in range(count):
                if i != j and games[i, j] > 0:
                    denominator[i] += games[i, j] / (ability[i] + ability[j])
        updated = scores / np.maximum(denominator, 1e-12)
        updated /= np.exp(np.log(np.maximum(updated, 1e-12)).mean())
        if np.max(np.abs(np.log(updated / ability))) < 1e-10:
            ability = updated
            break
        ability = updated
    scale = 400.0 / math.log(10.0)
    ratings = scale * np.log(ability)
    ratings -= ratings.mean()
    return {label: float(ratings[i]) for i, label in enumerate(labels)}


def parse_contestants(value: str) -> list[dict[str, float | str]]:
    """Parse ``label:rating:search_weight`` tournament contestants."""

    contestants: list[dict[str, float | str]] = []
    for item in value.split(","):
        label, rating, weight = item.split(":", 2)
        contestants.append(
            {"label": label, "rating": float(rating), "search_weight": float(weight)}
        )
    if len(contestants) < 2 or len({row["label"] for row in contestants}) != len(contestants):
        raise ValueError("contestants require at least two unique labels")
    return contestants


class LadderPolicy:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        search_weight: float,
        search_deadline_ms: float,
        search_beam: int,
        search_num_sim_envs: int,
        seed: int,
    ) -> None:
        self.runtime = HumanPolicyRuntime(checkpoint, device=device, seed=seed)
        self.search_weight = max(float(search_weight), 0.0)
        self.search_deadline_ms = float(search_deadline_ms)
        self.search = (
            HumanValueSearch(
                self.runtime,
                device=device,
                beam=search_beam,
                seed=seed,
                num_sim_envs=search_num_sim_envs,
                gpu_planner=str(device).startswith("cuda"),
            )
            if self.search_weight > 0
            else None
        )

    def close(self) -> None:
        if self.search is not None:
            self.search.close()

    def act(
        self,
        obs: np.ndarray,
        infos: list[dict[str, Any]],
        ratings: np.ndarray,
        opponent_ratings: np.ndarray,
        search_weights: np.ndarray,
        histories: list[list[dict[str, float | int]]],
        decisions: np.ndarray,
        *,
        speed: int,
        level: int,
    ) -> np.ndarray:
        actions_out = np.full(len(obs), -1, dtype=np.int32)
        for side, info in enumerate(infos):
            if not bool(info.get("placements/needs_action", False)):
                actions_out[side] = -2
                continue
            packed = pack_feasible_candidates(
                np.asarray(info["placements/feasible_mask"], dtype=bool),
                np.asarray(info["placements/cost_to_lock"]),
                max_candidates=128,
                sort_by_cost=True,
            )
            if packed.count == 0:
                continue
            pill = np.asarray(info["next_pill_colors"], dtype=np.int64)
            preview_raw = info["preview_pill"]
            preview = _RAW_TO_CANONICAL[
                [int(preview_raw["first_color"]), int(preview_raw["second_color"])]
            ]
            logits, _value, resolved, _clamped = self.runtime.score(
                board_planes=obs[side, :8],
                opponent_board_planes=obs[side, 8:16],
                opponent_state_age_frames=0,
                opponent_rating=float(opponent_ratings[side]),
                game_phase=min(float(decisions[side]) / 100.0, 1.0),
                recent_decisions=histories[side],
                pill=pill,
                preview=preview,
                candidate_actions=packed.actions,
                candidate_costs=packed.cost,
                candidate_mask=packed.mask,
                rating=float(ratings[side]),
            )
            valid_actions = packed.actions[packed.mask]
            valid_logits = logits[packed.mask]
            scores = valid_logits
            side_search_weight = float(search_weights[side])
            if self.search is not None and side_search_weight > 0.0:
                sinfo = self.search.analyze(
                    board_planes=obs[side, :8],
                    opponent_board_planes=obs[side, 8:16],
                    pill=pill,
                    preview=preview,
                    feasible_mask512=np.asarray(info["placements/feasible_mask"], dtype=bool),
                    cost_to_lock512=np.asarray(info["placements/cost_to_lock"]),
                    speed=speed,
                    speed_ups=min(max(level - 20, 0) + int(decisions[side]) // 10, 0x31),
                    level=level,
                    rating=resolved,
                    opponent_rating=float(opponent_ratings[side]),
                    game_phase=min(float(decisions[side]) / 100.0, 1.0),
                    recent_decisions=histories[side],
                    deadline_ms=self.search_deadline_ms,
                )
                scores = blend_human_and_search(
                    valid_logits,
                    competitive_scores(valid_actions, sinfo),
                    weight=side_search_weight,
                )
            actions_out[side] = int(valid_actions[int(np.argmax(scores))])
        return actions_out


def run_probe(
    checkpoint: Path,
    *,
    rating: float,
    anchor: float,
    matches: int,
    pairs: int,
    level: int,
    speed: int,
    device: str,
    search_weight: float,
    search_deadline_ms: float,
    seed: int,
    search_beam: int = 8,
    search_num_sim_envs: int = 64,
    anchor_search_weight: float = 0.0,
) -> dict[str, Any]:
    if pairs < 2:
        raise ValueError("pairs must be >= 2 so probe side can be balanced")
    env = DrMarioVsPoolVecEnv(
        num_pairs=pairs,
        state_repr="bitplane_bottle_conn_mask_vs",
        level=level,
        speed_setting=speed,
        randomize_rng=True,
    )
    policy = LadderPolicy(
        checkpoint,
        device=device,
        search_weight=max(search_weight, anchor_search_weight),
        search_deadline_ms=search_deadline_ms,
        search_beam=search_beam,
        search_num_sim_envs=search_num_sim_envs,
        seed=seed,
    )
    obs, infos = env.reset(seed=seed)
    sides = env.num_sides
    probe_side = np.asarray([(pair % 2) for pair in range(pairs)], dtype=np.int64)
    ratings = np.full(sides, anchor, dtype=np.float32)
    for pair, side in enumerate(probe_side):
        ratings[2 * pair + int(side)] = rating
    opponent_ratings = ratings[np.arange(sides) ^ 1]
    search_weights = np.full(sides, anchor_search_weight, dtype=np.float32)
    for pair, side in enumerate(probe_side):
        search_weights[2 * pair + int(side)] = search_weight
    histories: list[list[dict[str, float | int]]] = [[] for _ in range(sides)]
    decisions = np.zeros(sides, dtype=np.int64)
    wins = losses = draws = 0
    started = time.perf_counter()
    try:
        while wins + losses + draws < matches:
            actions = policy.act(
                obs,
                infos,
                ratings,
                opponent_ratings,
                search_weights,
                histories,
                decisions,
                speed=speed,
                level=level,
            )
            acted = np.asarray(
                [bool(info.get("placements/needs_action", False)) for info in infos]
            ) & (actions >= 0)
            obs, _reward, term, trunc, infos = env.step(actions)
            for side in np.flatnonzero(acted):
                histories[side].insert(
                    0,
                    {
                        "action": int(actions[side]),
                        "tau_frames": int(infos[side].get("placements/tau", 0)),
                    },
                )
                del histories[side][4:]
                decisions[side] += 1
            for pair in range(pairs):
                left = 2 * pair
                if not bool(term[left] or trunc[left]):
                    continue
                probe = left + int(probe_side[pair])
                outcome = str(infos[probe].get("vs/outcome", ""))
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                else:
                    draws += 1
                histories[left].clear()
                histories[left + 1].clear()
                decisions[left : left + 2] = 0
                if wins + losses + draws >= matches:
                    break
    finally:
        policy.close()
        env.close()
    decisive = wins + losses
    rate = (wins + 0.5 * draws) / max(wins + losses + draws, 1)
    lo, hi = wilson_interval(wins, decisive)
    return {
        "requested_rating": float(rating),
        "anchor_rating": float(anchor),
        "search_weight": float(search_weight),
        "anchor_search_weight": float(anchor_search_weight),
        "search_beam": int(search_beam),
        "search_num_sim_envs": int(search_num_sim_envs),
        "matches": int(wins + losses + draws),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "score_rate": float(rate),
        "decisive_win_rate": float(wins / max(decisive, 1)),
        "decisive_win_rate_ci95": [float(lo), float(hi)],
        "effective_rating": elo_from_win_rate(rate, anchor),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ratings", default="1000,1300,1600,1900,2200")
    parser.add_argument(
        "--contestants",
        help="Round robin as label:rating:search_weight comma-separated entries",
    )
    parser.add_argument("--anchor", type=float, default=1600.0)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--level", type=int, default=20)
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--search-weight", type=float, default=0.0)
    parser.add_argument("--search-deadline-ms", type=float, default=100.0)
    parser.add_argument("--search-beam", type=int, default=8)
    parser.add_argument("--search-num-sim-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.contestants:
        contestants = parse_contestants(args.contestants)
        rows = []
        for left in range(len(contestants)):
            for right in range(left + 1, len(contestants)):
                lhs, rhs = contestants[left], contestants[right]
                row = run_probe(
                    args.checkpoint,
                    rating=float(lhs["rating"]),
                    anchor=float(rhs["rating"]),
                    matches=args.matches,
                    pairs=args.pairs,
                    level=args.level,
                    speed=args.speed,
                    device=args.device,
                    search_weight=float(lhs["search_weight"]),
                    anchor_search_weight=float(rhs["search_weight"]),
                    search_deadline_ms=args.search_deadline_ms,
                    search_beam=args.search_beam,
                    search_num_sim_envs=args.search_num_sim_envs,
                    seed=args.seed + len(rows),
                )
                row.update(left=str(lhs["label"]), right=str(rhs["label"]))
                rows.append(row)
        labels = [str(row["label"]) for row in contestants]
        payload = {
            "schema": "drmc-human-strength-tournament-v1",
            "contestants": contestants,
            "relative_elo": relative_elo(rows, labels),
            "matchups": rows,
        }
    else:
        ratings = [float(value) for value in args.ratings.split(",")]
        rows = [
            run_probe(
                args.checkpoint,
                rating=rating,
                anchor=args.anchor,
                matches=args.matches,
                pairs=args.pairs,
                level=args.level,
                speed=args.speed,
                device=args.device,
                search_weight=args.search_weight,
                search_deadline_ms=args.search_deadline_ms,
                search_beam=args.search_beam,
                search_num_sim_envs=args.search_num_sim_envs,
                seed=args.seed + index,
            )
            for index, rating in enumerate(ratings)
        ]
        payload = {"schema": "drmc-human-strength-calibration-v1", "probes": rows}
    output = json.dumps(payload, indent=2)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")


if __name__ == "__main__":
    main()
