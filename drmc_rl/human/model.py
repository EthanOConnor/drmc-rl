"""Continuous human placement and execution-timing models."""

from __future__ import annotations

from typing import Any, Mapping


HUMAN_POLICY_SCHEMA = "drmc-human-policy-v2"
SKILL_FEATURE_DIM = 2
HISTORY_STEPS = 4
HISTORY_FEATURE_DIM = 8
# own z/z², own uncertainty, opponent z, rating gap, opponent uncertainty,
# opponent-snapshot age, game phase, then four recent placement summaries.
POLICY_BASE_CONDITION_DIM = 8
POLICY_CONDITION_DIM = POLICY_BASE_CONDITION_DIM + HISTORY_STEPS * HISTORY_FEATURE_DIM
TIMING_FEATURE_DIM = 12


def canonicalize_same_color_action(action: int) -> int:
    """Collapse monochrome o2/o3 placements onto equivalent o0/o1 actions."""

    orientation, row_column = divmod(int(action), 128)
    row, column = divmod(row_column, 8)
    if orientation == 2:
        return row * 8 + column - 1
    if orientation == 3:
        return 128 + (row - 1) * 8 + column
    return int(action)


def timing_feature_vector(
    skill_features,
    *,
    rating_sd: float = 0.0,
    opponent_skill_z: float = 0.0,
    game_phase: float = 0.0,
    previous_tau_frames: float = 0.0,
    chosen_cost: float,
    board_planes,
    speed: int,
    speed_ups: int,
    candidate_count: int,
):
    """Build the timing-model feature vector from semantic state."""

    import numpy as np

    planes = np.asarray(board_planes, dtype=np.float32)
    occupied = planes[:3].sum(axis=0) > 0
    rows = np.flatnonzero(occupied.any(axis=1))
    height = 0.0 if rows.size == 0 else float(16 - rows.min())
    skill = np.asarray(skill_features, dtype=np.float32).reshape(2)
    return np.asarray(
        [
            skill[0],
            skill[1],
            min(max(float(rating_sd), 0.0), 500.0) / 500.0,
            float(np.clip(opponent_skill_z, -4.0, 4.0)) / 4.0,
            float(np.clip(game_phase, 0.0, 1.0)),
            min(max(float(previous_tau_frames), 0.0), 300.0) / 300.0,
            float(chosen_cost) / 120.0,
            float(occupied.mean()),
            height / 16.0,
            float(speed) / 2.0,
            min(float(speed_ups), 20.0) / 20.0,
            min(float(candidate_count), 128.0) / 128.0,
        ],
        dtype=np.float32,
    )


def history_features(recent_decisions) -> "Any":
    """Encode up to four preceding placements, newest first.

    Each row is ``present, x, y, rotation-one-hot, log(tau)``.  The compact
    fixed-size form is cheap to store in corpus shards and keeps the host
    protocol semantic: callers send placements, never opaque hidden state.
    """

    import numpy as np

    out = np.zeros((HISTORY_STEPS, HISTORY_FEATURE_DIM), dtype=np.float32)
    for i, item in enumerate(list(recent_decisions or ())[:HISTORY_STEPS]):
        action = int(item.get("action", -1))
        if action < 0:
            continue
        orientation, cell = divmod(action, 128)
        row, column = divmod(cell, 8)
        out[i, 0] = 1.0
        out[i, 1] = np.clip(column / 7.0, 0.0, 1.0)
        out[i, 2] = np.clip(row / 15.0, 0.0, 1.0)
        out[i, 3 + (orientation & 3)] = 1.0
        out[i, 7] = np.clip(np.log1p(max(float(item.get("tau_frames", 0.0)), 0.0)) / 6.0, 0.0, 1.0)
    return out.reshape(-1)


def policy_condition_features(
    condition,
    *,
    rating: float,
    rating_sd: float = 0.0,
    opponent_rating: float | None = None,
    opponent_rating_sd: float = 0.0,
    opponent_state_age_frames: int = 0,
    game_phase: float = 0.0,
    recent_decisions=(),
):
    """Build the shared semantic conditioning vector for training/inference."""

    import numpy as np

    opponent = float(rating if opponent_rating is None else opponent_rating)
    own_skill = condition.encode(float(rating))
    opponent_z = float(condition.encode(opponent)[0])
    base = np.asarray(
        [
            own_skill[0],
            own_skill[1],
            np.clip(float(rating_sd), 0.0, 500.0) / 500.0,
            opponent_z,
            np.clip((float(rating) - opponent) / condition.scale, -4.0, 4.0) / 4.0,
            np.clip(float(opponent_rating_sd), 0.0, 500.0) / 500.0,
            np.clip(float(opponent_state_age_frames), 0.0, 240.0) / 240.0,
            np.clip(float(game_phase), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate((base, history_features(recent_decisions))).astype(np.float32)


def human_policy_config(*, capacity: str = "medium", candidate_max: int = 128) -> dict[str, Any]:
    sizes = {
        "small": (96, 2, 2, 192),
        "medium": (128, 3, 3, 256),
        "large": (192, 4, 4, 384),
    }
    if capacity not in sizes:
        raise ValueError(f"unknown capacity {capacity!r}")
    d_model, blocks, layers, hidden = sizes[capacity]
    return {
        "schema": HUMAN_POLICY_SCHEMA,
        "human_condition_dim": POLICY_CONDITION_DIM,
        "in_channels": 20,
        "smdp_ppo": {
            "policy_type": "candidate",
            "aux_spec": "none",
            "pill_embed_dim": d_model,
            "pill_embed_type": "ordered_pair",
            "encoder_blocks": blocks,
            "candidate_max_candidates": int(candidate_max),
            "candidate_d_model": d_model,
            "candidate_pos_embed_dim": 32,
            "candidate_cost_embed_dim": 32,
            "candidate_hidden_dim": hidden,
            "candidate_board_encoder": "cnn",
            "candidate_board_channels": 8,
            "candidate_transformer_layers": layers,
            "candidate_transformer_heads": 4,
            "candidate_transformer_ff_mult": 4,
            "candidate_patch_kernel": 9,
        },
    }


def build_human_policy(cfg: Mapping[str, Any], *, device: str = "cpu"):
    from drmc_rl.models.policy.candidate_policy import CandidatePlacementPolicyNet

    sp = cfg.get("smdp_ppo", cfg)

    def g(key: str, default: Any) -> Any:
        return sp.get(key, default)

    return CandidatePlacementPolicyNet(
        in_channels=int(cfg.get("in_channels", 20)),
        board_channels=int(g("candidate_board_channels", 8)),
        board_encoder=str(g("candidate_board_encoder", "cnn")),
        encoder_blocks=int(g("encoder_blocks", 0)),
        d_model=int(g("candidate_d_model", 128)),
        pill_embed_dim=int(g("pill_embed_dim", 128)),
        pill_embed_type=str(g("pill_embed_type", "ordered_pair")),
        num_colors=3,
        aux_dim=int(cfg.get("human_condition_dim", SKILL_FEATURE_DIM)),
        pos_embed_dim=int(g("candidate_pos_embed_dim", 32)),
        cost_embed_dim=int(g("candidate_cost_embed_dim", 32)),
        cand_hidden_dim=int(g("candidate_hidden_dim", 256)),
        transformer_layers=int(g("candidate_transformer_layers", 3)),
        cross_layers=int(g("candidate_cross_layers", 0)),
        transformer_heads=int(g("candidate_transformer_heads", 4)),
        transformer_ff_mult=int(g("candidate_transformer_ff_mult", 4)),
        patch_kernel=int(g("candidate_patch_kernel", 9)),
    ).to(device)


def build_timing_model(*, device: str = "cpu"):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(TIMING_FEATURE_DIM, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 2),
    ).to(device)
