"""Continuous human placement and execution-timing models."""

from __future__ import annotations

from typing import Any, Mapping


HUMAN_POLICY_SCHEMA = "drmc-human-policy-v1"
SKILL_FEATURE_DIM = 2
POLICY_CONDITION_DIM = 3  # skill z, skill z², opponent-snapshot age
TIMING_FEATURE_DIM = 8


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
            float(chosen_cost) / 120.0,
            float(occupied.mean()),
            height / 16.0,
            float(speed) / 2.0,
            min(float(speed_ups), 20.0) / 20.0,
            min(float(candidate_count), 128.0) / 128.0,
        ],
        dtype=np.float32,
    )


def human_policy_config(*, capacity: str = "medium", candidate_max: int = 128) -> dict[str, Any]:
    sizes = {
        "small": (96, 2, 2, 192),
        "medium": (128, 3, 3, 256),
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
