"""Inference runtime for the V3 afterstate and calibrated-regret policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.human.afterstate_model import (
    HUMAN_AFTERSTATE_SCHEMA,
    build_afterstate_policy,
)
from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.model import (
    build_timing_model,
    policy_condition_features,
    timing_feature_vector,
)
from drmc_rl.human.search import semantic_planes_to_nes_board
from drmc_rl.human.strength import RegretCalibration, RegretStrengthController
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


_CANONICAL_TO_NES = np.asarray((1, 0, 2), dtype=np.uint8)


class AfterstatePolicyRuntime:
    """Exact one-placement inference with separate quality and human style."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu", seed: int = 0):
        import torch

        self.path = Path(checkpoint)
        payload = load_checkpoint(self.path, map_location="cpu")
        if payload.get("schema") != HUMAN_AFTERSTATE_SCHEMA:
            raise ValueError(f"not a {HUMAN_AFTERSTATE_SCHEMA} checkpoint: {self.path}")
        self.cfg = payload["cfg"]
        self.meta = payload["human_meta"]
        self.condition = HumanSkillCondition.from_dict(self.meta["skill_condition"])
        self.calibration = RegretCalibration.from_dict(self.meta["regret_calibration"])
        self.controller = RegretStrengthController(self.calibration, seed=seed)
        self.device = torch.device(device)
        self.policy = build_afterstate_policy(
            self.cfg,
            condition_dim=int(self.cfg.get("condition_dim", 40)),
            device=device,
        )
        self.policy.load_state_dict(payload["state_dict"])
        self.policy.eval()
        self.timing = build_timing_model(device=device)
        self.timing.load_state_dict(payload["timing_state_dict"])
        self.timing.eval()
        self.simulator = NativeAfterstateSimulator(num_envs=128)
        self.rng = np.random.default_rng(int(seed))

    def close(self) -> None:
        self.simulator.close()

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": HUMAN_AFTERSTATE_SCHEMA,
            "checkpoint": self.path.name,
            "source_dataset": self.meta.get("source_dataset"),
            "rating_range": [self.condition.minimum, self.condition.maximum],
            "parameters": self.meta.get("parameters"),
            "competitive_quality": "rating-independent exact afterstate value",
            "strength_control": "monotone corpus-calibrated action regret",
        }

    def score(
        self,
        *,
        board_planes: np.ndarray,
        opponent_board_planes: np.ndarray,
        opponent_state_age_frames: int,
        rating_sd: float = 0.0,
        opponent_rating: float | None = None,
        opponent_rating_sd: float = 0.0,
        game_phase: float = 0.0,
        recent_decisions=(),
        pill: np.ndarray,
        preview: np.ndarray,
        candidate_actions: np.ndarray,
        candidate_costs: np.ndarray,
        candidate_mask: np.ndarray,
        rating: float,
        speed: int,
        speed_ups: int,
    ) -> dict[str, Any]:
        return self.score_batch(
            [
                {
                    "board_planes": board_planes,
                    "opponent_board_planes": opponent_board_planes,
                    "opponent_state_age_frames": opponent_state_age_frames,
                    "rating_sd": rating_sd,
                    "opponent_rating": opponent_rating,
                    "opponent_rating_sd": opponent_rating_sd,
                    "game_phase": game_phase,
                    "recent_decisions": recent_decisions,
                    "pill": pill,
                    "preview": preview,
                    "candidate_actions": candidate_actions,
                    "candidate_costs": candidate_costs,
                    "candidate_mask": candidate_mask,
                    "rating": rating,
                    "speed": speed,
                    "speed_ups": speed_ups,
                }
            ]
        )[0]

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score independent decisions with one native simulation and network pass."""
        import torch

        if not requests:
            return []
        batch = len(requests)
        widths = [len(np.asarray(request["candidate_actions"]).reshape(-1)) for request in requests]
        width = max(widths)
        actions = np.full((batch, width), -1, dtype=np.int64)
        costs = np.zeros((batch, width), dtype=np.float32)
        mask = np.zeros((batch, width), dtype=np.bool_)
        roots = np.empty((batch, 128), dtype=np.uint8)
        opponents = np.empty((batch, 128), dtype=np.uint8)
        pills = np.empty((batch, 2), dtype=np.int64)
        previews = np.empty((batch, 2), dtype=np.int64)
        counts = np.empty(batch, dtype=np.int64)
        conditions = []
        resolved_ratings = []
        clamped_ratings = []
        for row, request in enumerate(requests):
            row_width = widths[row]
            row_actions = np.asarray(request["candidate_actions"], dtype=np.int64).reshape(-1)
            row_costs = np.asarray(request["candidate_costs"], dtype=np.float32).reshape(-1)
            row_mask = np.asarray(request["candidate_mask"], dtype=np.bool_).reshape(-1)
            if len(row_costs) != row_width or len(row_mask) != row_width:
                raise ValueError("V3 candidate arrays must have matching widths")
            count = int(row_mask.sum())
            if count <= 0 or not row_mask[:count].all() or row_mask[count:].any():
                raise ValueError("V3 candidates must be packed contiguously")
            actions[row, :row_width] = row_actions
            costs[row, :row_width] = row_costs
            mask[row, :row_width] = row_mask
            counts[row] = count
            roots[row] = semantic_planes_to_nes_board(request["board_planes"])
            opponents[row] = semantic_planes_to_nes_board(request["opponent_board_planes"])
            pills[row] = np.asarray(request["pill"], dtype=np.int64)
            previews[row] = np.asarray(request["preview"], dtype=np.int64)
            resolved, clamped = self.condition.resolve(float(request["rating"]))
            resolved_ratings.append(resolved)
            clamped_ratings.append(clamped)
            conditions.append(
                policy_condition_features(
                    self.condition,
                    rating=resolved,
                    rating_sd=float(request.get("rating_sd", 0.0)),
                    opponent_rating=request.get("opponent_rating"),
                    opponent_rating_sd=float(request.get("opponent_rating_sd", 0.0)),
                    opponent_state_age_frames=int(request["opponent_state_age_frames"]),
                    game_phase=float(request.get("game_phase", 0.0)),
                    recent_decisions=request.get("recent_decisions", ()),
                )
            )
        simulated = self.simulator.simulate_packed(
            fields=roots,
            pills=_CANONICAL_TO_NES[pills],
            previews=_CANONICAL_TO_NES[previews],
            candidate_actions=actions,
            candidate_costs=costs.astype(np.uint16),
            candidate_count=counts,
            speed=np.asarray([request["speed"] for request in requests]),
            speed_ups=np.asarray([request["speed_ups"] for request in requests]),
        )
        afterstates = np.full((batch, width, 128), 0xFF, dtype=np.uint8)
        offset = 0
        for row, count in enumerate(counts):
            afterstates[row, :count] = simulated.fields[offset : offset + count]
            offset += int(count)
        with torch.inference_mode():
            output = self.policy(
                torch.from_numpy(afterstates).to(self.device),
                torch.from_numpy(roots).to(self.device),
                torch.from_numpy(opponents).to(self.device),
                torch.from_numpy(pills).to(self.device),
                torch.from_numpy(previews).to(self.device),
                torch.from_numpy(actions).to(self.device),
                torch.from_numpy(costs).to(self.device),
                torch.from_numpy(mask).to(self.device),
                torch.from_numpy(np.stack(conditions)).to(self.device),
            )
        arrays = {key: value.float().cpu().numpy() for key, value in output.items()}
        results = []
        for row, row_width in enumerate(widths):
            result = {key: value[row, :row_width] for key, value in arrays.items()}
            result.update(
                resolved_rating=resolved_ratings[row], rating_clamped=clamped_ratings[row]
            )
            results.append(result)
        return results

    def choose_strength(
        self,
        competitive_scores: np.ndarray,
        human_logits: np.ndarray,
        candidate_mask: np.ndarray,
        *,
        rating: float,
        temperature: float = 1.0,
    ) -> tuple[int, dict[str, float | int]]:
        style = np.asarray(human_logits, dtype=np.float64)
        if temperature > 0:
            style = style / float(temperature)
        return self.controller.choose(
            competitive_scores,
            style,
            candidate_mask,
            rating=rating,
            deterministic=temperature <= 0,
        )

    def choose_style(
        self,
        human_logits: np.ndarray,
        candidate_mask: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> int:
        """Sample pure corpus imitation without claiming a strength transform."""

        valid = np.flatnonzero(np.asarray(candidate_mask, dtype=np.bool_))
        if valid.size == 0:
            raise ValueError("cannot choose without a valid candidate")
        scores = np.asarray(human_logits, dtype=np.float64)[valid]
        if temperature <= 0:
            return int(valid[int(np.argmax(scores))])
        scaled = (scores - scores.max()) / float(temperature)
        probability = np.exp(scaled)
        probability /= probability.sum()
        return int(self.rng.choice(valid, p=probability))

    @staticmethod
    def choose_quality(competitive_scores: np.ndarray, candidate_mask: np.ndarray) -> int:
        """Choose the rating-independent competitive head's best candidate."""

        valid = np.flatnonzero(np.asarray(candidate_mask, dtype=np.bool_))
        if valid.size == 0:
            raise ValueError("cannot choose without a valid candidate")
        scores = np.asarray(competitive_scores, dtype=np.float64)[valid]
        return int(valid[int(np.argmax(scores))])

    def timing_prediction(
        self,
        *,
        board_planes: np.ndarray,
        rating: float,
        rating_sd: float = 0.0,
        opponent_rating: float | None = None,
        game_phase: float = 0.0,
        previous_tau_frames: float = 0.0,
        chosen_cost: float,
        speed: int,
        speed_ups: int,
        candidate_count: int,
    ) -> dict[str, float]:
        import torch

        resolved, _ = self.condition.resolve(rating)
        features = timing_feature_vector(
            self.condition.encode(resolved),
            rating_sd=rating_sd,
            opponent_skill_z=float(
                self.condition.encode(resolved if opponent_rating is None else opponent_rating)[0]
            ),
            game_phase=game_phase,
            previous_tau_frames=previous_tau_frames,
            chosen_cost=chosen_cost,
            board_planes=board_planes,
            speed=speed,
            speed_ups=speed_ups,
            candidate_count=candidate_count,
        )
        with torch.inference_mode():
            output = self.timing(torch.from_numpy(features[None]).to(self.device))[0]
        mean = float(output[0].cpu())
        std = float(np.exp(float(output[1].clamp(-3.0, 2.0).cpu())))
        return {
            "log_slack_mean": mean,
            "log_slack_std": std,
            "slack_frames_median": float(max(np.expm1(mean), 0.0)),
        }

    def sample_slack_frames(self, timing: dict[str, float], *, scale: float = 1.0) -> int:
        if scale <= 0:
            return 0
        value = self.rng.normal(timing["log_slack_mean"], timing["log_slack_std"])
        return int(np.clip(np.rint(max(np.expm1(value), 0.0) * float(scale)), 0, 300))


__all__ = ["AfterstatePolicyRuntime"]
