"""Checkpoint-backed inference for the continuous human model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.model import (
    HUMAN_POLICY_SCHEMA,
    build_human_policy,
    build_timing_model,
    timing_feature_vector,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


class HumanPolicyRuntime:
    def __init__(self, checkpoint: str | Path, *, device: str = "cpu", seed: int = 0):
        import torch

        self.path = Path(checkpoint)
        payload = load_checkpoint(self.path, map_location="cpu")
        if payload.get("schema") != HUMAN_POLICY_SCHEMA:
            raise ValueError(f"not a {HUMAN_POLICY_SCHEMA} checkpoint: {self.path}")
        self.cfg = payload["cfg"]
        self.meta = payload["human_meta"]
        self.condition = HumanSkillCondition.from_dict(self.meta["skill_condition"])
        self.device = torch.device(device)
        self.policy = build_human_policy(self.cfg, device=device)
        self.policy.load_state_dict(payload["state_dict"])
        self.policy.eval()
        self.timing = build_timing_model(device=device)
        self.timing.load_state_dict(payload["timing_state_dict"])
        self.timing.eval()
        self.rng = np.random.default_rng(int(seed))

    @property
    def identity(self) -> dict[str, Any]:
        metrics = self.meta.get("metrics", {})
        return {
            "schema": HUMAN_POLICY_SCHEMA,
            "checkpoint": self.path.name,
            "corpus_release_id": self.meta.get("corpus_release_id"),
            "rating_range": [self.condition.minimum, self.condition.maximum],
            "replay_holdout_top1": metrics.get("replay_holdout_top1"),
            "player_holdout_top1": metrics.get("player_holdout_top1"),
            "future_holdout_top1": metrics.get("future_holdout_top1"),
        }

    def score(
        self,
        *,
        board_planes: np.ndarray,
        opponent_board_planes: np.ndarray,
        opponent_state_age_frames: int,
        pill: np.ndarray,
        preview: np.ndarray,
        candidate_actions: np.ndarray,
        candidate_costs: np.ndarray,
        candidate_mask: np.ndarray,
        rating: float,
    ) -> tuple[np.ndarray, float, bool]:
        import torch

        planes = np.asarray(board_planes, dtype=np.float32)
        if planes.shape != (8, 16, 8):
            raise ValueError(f"board_planes must have shape (8,16,8), got {planes.shape}")
        opponent = np.asarray(opponent_board_planes, dtype=np.float32)
        if opponent.shape != (8, 16, 8):
            raise ValueError(
                f"opponent_board_planes must have shape (8,16,8), got {opponent.shape}"
            )
        actions = np.asarray(candidate_actions, dtype=np.int32)
        costs = np.asarray(candidate_costs, dtype=np.float32)
        mask = np.asarray(candidate_mask, dtype=np.bool_)
        if actions.ndim != 1 or costs.shape != actions.shape or mask.shape != actions.shape:
            raise ValueError("candidate arrays must be equal-length vectors")
        resolved, clamped = self.condition.resolve(rating)
        skill = self.condition.encode(np.asarray([resolved], dtype=np.float32))
        age = min(max(int(opponent_state_age_frames), 0), 240) / 240.0
        aux = np.concatenate((skill, np.asarray([[age]], dtype=np.float32)), axis=1)
        feasible = np.zeros((4, 16, 8), dtype=np.float32)
        valid_actions = actions[mask]
        feasible.reshape(-1)[valid_actions] = 1.0
        obs = np.concatenate((planes, opponent, feasible), axis=0)[None]
        pill_arr = np.asarray(pill, dtype=np.int64).reshape(1, 2)
        preview_arr = np.asarray(preview, dtype=np.int64).reshape(1, 2)
        if pill_arr[0, 0] == pill_arr[0, 1]:
            obs[:, 6:8] = 0.0
        with torch.inference_mode():
            logits, _ = self.policy(
                torch.from_numpy(obs).to(self.device),
                torch.from_numpy(pill_arr).to(self.device),
                torch.from_numpy(preview_arr).to(self.device),
                torch.from_numpy(actions[None]).to(self.device),
                torch.from_numpy(costs[None]).to(self.device),
                torch.from_numpy(mask[None]).to(self.device),
                aux=torch.from_numpy(aux).to(self.device),
            )
        return logits[0].float().cpu().numpy(), resolved, clamped

    def choose(self, logits: np.ndarray, mask: np.ndarray, *, temperature: float = 1.0) -> int:
        valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
        if valid.size == 0:
            raise ValueError("cannot choose without a feasible candidate")
        scores = np.asarray(logits, dtype=np.float64)[valid]
        if temperature <= 0:
            return int(valid[int(np.argmax(scores))])
        scaled = (scores - scores.max()) / float(temperature)
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(self.rng.choice(valid, p=probs))

    def timing_prediction(
        self,
        *,
        board_planes: np.ndarray,
        rating: float,
        chosen_cost: float,
        speed: int,
        speed_ups: int,
        candidate_count: int,
    ) -> dict[str, float]:
        import torch

        resolved, _ = self.condition.resolve(rating)
        features = timing_feature_vector(
            self.condition.encode(resolved),
            chosen_cost=chosen_cost,
            board_planes=board_planes,
            speed=speed,
            speed_ups=speed_ups,
            candidate_count=candidate_count,
        )
        with torch.inference_mode():
            output = self.timing(torch.from_numpy(features[None]).to(self.device))[0]
        mean = float(output[0].cpu())
        log_std = float(output[1].clamp(-3.0, 2.0).cpu())
        return {
            "log_slack_mean": mean,
            "log_slack_std": float(np.exp(log_std)),
            "slack_frames_median": float(max(np.expm1(mean), 0.0)),
        }
