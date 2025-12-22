from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import envs.specs.ram_to_state as ram_specs
from models.policy.candidate_packing import pack_feasible_candidates
from models.policy.candidate_policy import CandidatePlacementPolicyNet
from models.policy.placement_dist import MaskedPlacementDist
from training.envs.dr_mario_vec import VecEnvConfig, make_vec_env


ENGINE_PATH = Path("game_engine/drmario_engine")


def _extract_cost(info: dict) -> np.ndarray:
    value = info.get("placements/cost_to_lock")
    if value is None:
        value = info.get("placements/costs")
    if value is None:
        return np.full((4, 16, 8), np.inf, dtype=np.float32)
    arr = np.asarray(value)
    if arr.shape != (4, 16, 8):
        return np.full((4, 16, 8), np.inf, dtype=np.float32)
    if arr.dtype == np.uint16:
        out = arr.astype(np.float32)
        out[out >= np.float32(0xFFFE)] = np.inf
        return out
    return arr.astype(np.float32, copy=False)


def _preview_colors(info: dict) -> np.ndarray:
    preview = info.get("preview_pill")
    if not isinstance(preview, dict):
        return np.array([0, 0], dtype=np.int64)
    left_raw = int(preview.get("first_color", 0)) & 0x03
    right_raw = int(preview.get("second_color", 0)) & 0x03
    left = int(ram_specs.COLOR_VALUE_TO_INDEX.get(left_raw, 0))
    right = int(ram_specs.COLOR_VALUE_TO_INDEX.get(right_raw, 0))
    return np.array([left, right], dtype=np.int64)


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_candidate_policy_runs_on_cpp_engine_backend() -> None:
    torch = pytest.importorskip("torch")

    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=1,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-engine",
        state_repr="bitplane_bottle_mask",
        vectorization="sync",
        emit_raw_ram=False,
        speed_setting=2,
    )
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=0)
        info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
        assert isinstance(info0, dict)

        net = CandidatePlacementPolicyNet(
            in_channels=4,
            board_channels=4,
            board_encoder="cnn",
            encoder_blocks=0,
            d_model=64,
            pill_embed_dim=64,
            aux_dim=0,
            transformer_layers=2,
            transformer_heads=2,
            transformer_ff_mult=2,
            patch_kernel=3,
        )
        net.eval()

        # Run a few macro decisions end-to-end.
        for _ in range(4):
            info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            mask = np.asarray(info0.get("placements/feasible_mask"), dtype=bool)
            assert mask.shape == (4, 16, 8)
            cost = _extract_cost(info0)

            packed = pack_feasible_candidates(mask, cost, max_candidates=512, sort_by_cost=True)
            cand_actions = torch.from_numpy(packed.actions[None, :].astype(np.int64))
            cand_mask = torch.from_numpy(packed.mask[None, :].astype(bool))
            cand_cost = torch.from_numpy(packed.cost[None, :].astype(np.float32))

            board = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            colors = torch.from_numpy(np.asarray(info0.get("next_pill_colors", [0, 0]), dtype=np.int64)[None, :])
            preview = torch.from_numpy(_preview_colors(info0)[None, :])

            with torch.no_grad():
                logits, _value = net(board, colors, preview, cand_actions, cand_cost, cand_mask)

            dist = MaskedPlacementDist(logits, cand_mask)
            slot = dist.mode()
            action = int(cand_actions[0, int(slot.item())].item())

            obs, _rewards, _terminated, _truncated, infos = env.step(np.array([action], dtype=np.int64))
    finally:
        env.close()

