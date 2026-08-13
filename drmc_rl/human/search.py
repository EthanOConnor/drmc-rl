"""Value-guided native search for human sparring and coaching."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from drmc_rl.human.model import POLICY_CONDITION_DIM, policy_condition_features
from drmc_rl.models.policy.search_policy import ObsContext, SearchPolicy

_CANONICAL_TO_NES = np.asarray([1, 0, 2], dtype=np.uint8)


def semantic_planes_to_nes_board(planes: np.ndarray) -> np.ndarray:
    """Invert the shared eight semantic planes into native checkpoint bytes."""

    value = np.asarray(planes, dtype=np.float32)
    if value.shape != (8, 16, 8):
        raise ValueError(f"semantic board must be [8,16,8], got {value.shape}")
    colors = value[:3].argmax(axis=0)
    occupied = value[:3].sum(axis=0) > 0.5
    board = np.full((16, 8), 0xFF, dtype=np.uint8)
    low = _CANONICAL_TO_NES[colors]
    high = np.full((16, 8), 0x80, dtype=np.uint8)
    high[value[3] > 0.5] = 0xD0
    for channel, tile_type in ((4, 0x50), (5, 0x40), (6, 0x70), (7, 0x60)):
        high[(value[channel] > 0.5) & ~(value[3] > 0.5)] = tile_type
    board[occupied] = (high | low)[occupied]
    return board.reshape(-1)


@dataclass(slots=True)
class _ContextAux:
    """Marker consumed by SearchPolicy's fixed semantic-context aux path."""

    aux_dim: int = POLICY_CONDITION_DIM
    context_only: bool = True


class HumanValueSearch:
    """Depth-2 search retaining the human prior and outcome-trained value."""

    def __init__(
        self,
        runtime,
        *,
        device: str = "cpu",
        beam: int = 8,
        deadline_ms: float = 100.0,
        num_sim_envs: int = 64,
        seed: int = 0,
        gpu_planner: bool = False,
    ) -> None:
        self.runtime = runtime
        self.search = SearchPolicy.from_net(
            copy.deepcopy(runtime.policy),
            aux_shim=_ContextAux(),
            candidate_max=128,
            beam=beam,
            deadline_ms=deadline_ms,
            device=device,
            num_sim_envs=num_sim_envs,
            win_value=8.0,
            loss_value=-8.0,
            reward_mode="vs",
            gamma=1.0,
            opponent_model="none",
            seed=seed,
            warmup=False,
            gpu_planner=gpu_planner,
        )

    def close(self) -> None:
        self.search.close()

    def analyze(
        self,
        *,
        board_planes: np.ndarray,
        opponent_board_planes: np.ndarray,
        pill: np.ndarray,
        preview: np.ndarray,
        feasible_mask512: np.ndarray,
        cost_to_lock512: np.ndarray,
        speed: int,
        speed_ups: int,
        level: int,
        rating: float,
        rating_sd: float = 0.0,
        opponent_rating: float | None = None,
        opponent_rating_sd: float = 0.0,
        opponent_state_age_frames: int = 0,
        game_phase: float = 0.0,
        recent_decisions=(),
        deadline_ms: float | None = None,
    ) -> dict[str, Any]:
        resolved, _ = self.runtime.condition.resolve(rating)
        aux = policy_condition_features(
            self.runtime.condition,
            rating=resolved,
            rating_sd=rating_sd,
            opponent_rating=opponent_rating,
            opponent_rating_sd=opponent_rating_sd,
            opponent_state_age_frames=opponent_state_age_frames,
            game_phase=game_phase,
            recent_decisions=recent_decisions,
        )
        raw_pill = _CANONICAL_TO_NES[np.asarray(pill, dtype=np.int64)]
        raw_preview = _CANONICAL_TO_NES[np.asarray(preview, dtype=np.int64)]
        action, info = self.search.decide(
            semantic_planes_to_nes_board(board_planes),
            raw_pill,
            raw_preview,
            np.asarray(feasible_mask512, dtype=np.bool_),
            np.asarray(cost_to_lock512),
            int(speed),
            int(speed_ups),
            int(level),
            deadline_ms=deadline_ms,
            obs_context=ObsContext(
                opp_planes=np.asarray(opponent_board_planes, dtype=np.float32),
                aux_tail=aux,
            ),
        )
        return {"action": int(action), **info}


def competitive_scores(
    candidate_actions: np.ndarray,
    search_info: dict[str, Any],
) -> np.ndarray:
    """Map partial beam Q values onto every candidate using root value completion."""

    actions = np.asarray(candidate_actions, dtype=np.int64)
    baseline = float(search_info.get("value_root") or 0.0)
    scores = np.full(len(actions), baseline, dtype=np.float64)
    searched_actions = search_info.get("ply1_actions")
    q_values = search_info.get("q_values")
    if searched_actions is not None and q_values is not None:
        positions = {int(action): i for i, action in enumerate(actions)}
        for action, q in zip(searched_actions, q_values):
            if int(action) in positions and np.isfinite(q):
                scores[positions[int(action)]] = float(q)
    return scores


def blend_human_and_search(
    human_logits: np.ndarray,
    competitive: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend human plausibility with bounded within-position value advantage."""

    human = np.asarray(human_logits, dtype=np.float64)
    values = np.asarray(competitive, dtype=np.float64)
    centered = values - np.median(values)
    scale = max(float(np.median(np.abs(centered))), 0.25)
    advantage = np.clip(centered / scale, -4.0, 4.0)
    return human + max(float(weight), 0.0) * advantage


__all__ = [
    "HumanValueSearch",
    "blend_human_and_search",
    "competitive_scores",
    "semantic_planes_to_nes_board",
]
