from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.human.coach import analyze_choice
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.planning.native_reach import is_library_present


def test_skill_condition_is_continuous_and_clamped() -> None:
    condition = HumanSkillCondition.fit(np.array([1000.0, 1500.0, 2000.0]))
    encoded = condition.encode(np.array([1499.0, 1500.0, 1501.0]))
    assert encoded.shape == (3, 2)
    assert np.linalg.norm(encoded[2] - encoded[1]) < 0.01
    assert condition.resolve(500.0) == (1000.0, True)
    assert condition.resolve(1750.0) == (1750.0, False)


def test_coach_keeps_human_norm_and_competitive_quality_separate() -> None:
    result = analyze_choice(
        [10, 20, 30],
        [3.0, 2.0, 1.0],
        chosen_action=20,
        competitive_scores=[0.1, 0.0, 1.0],
    )
    assert result["alternatives"][0]["action"] == 10
    assert result["alternatives"][0]["human_rank"] == 1
    assert result["chosen"]["human_rank"] == 2
    assert result["chosen"]["competitive_rank"] == 3
    assert result["interpretation"]["human_probability"] != result["interpretation"]["competitive_score"]


def test_training_batch_contains_opponent_context_and_continuous_condition() -> None:
    from tools.train_human_policy import KMAX, batch_inputs

    n = 3
    candidates = np.full((n, KMAX), -1, dtype=np.int16)
    candidates[:, :2] = (120, 121)
    costs = np.zeros((n, KMAX), dtype=np.uint16)
    costs[:, :2] = (30, 31)
    own = np.full((n, 128), 0xFF, dtype=np.uint8)
    opponent = np.full((n, 128), 0xFF, dtype=np.uint8)
    own[:, 120] = 0xD0
    opponent[:, 127] = 0xD2
    arrays = {
        "field": own,
        "opponent_field": opponent,
        "opponent_state_age_frames": np.asarray([0, 120, 999]),
        "pill": np.asarray([[1, 0]] * n),
        "preview": np.asarray([[2, 1]] * n),
        "candidate_actions": candidates,
        "candidate_costs": costs,
        "candidate_count": np.asarray([2] * n),
        "chosen_slot": np.asarray([0, 1, 0]),
        "rating": np.asarray([1000.0, 1500.0, 2000.0]),
    }
    condition = HumanSkillCondition.fit(arrays["rating"])
    obs, _pill, _preview, _actions, _costs, mask, slots, aux = batch_inputs(
        arrays, np.arange(n), condition
    )
    assert obs.shape == (n, 20, 16, 8)
    assert obs[:, 3, 15, 0].all()  # own virus plane
    assert obs[:, 8 + 3, 15, 7].all()  # opponent virus plane
    assert mask[:, :2].all()
    assert slots.tolist() == [0, 1, 0]
    assert aux.shape == (n, 3)
    assert aux[:, 2].tolist() == [0.0, 0.5, 1.0]


def test_tiny_end_to_end_training_smoke() -> None:
    from tools.train_human_policy import KMAX, train

    n = 16
    candidates = np.full((n, KMAX), -1, dtype=np.int16)
    candidates[:, :2] = (120, 121)
    costs = np.zeros((n, KMAX), dtype=np.uint16)
    costs[:, :2] = (30, 31)
    fields = np.full((n, 128), 0xFF, dtype=np.uint8)
    fields[:, 120] = 0xD0
    arrays = {
        "field": fields,
        "opponent_field": fields.copy(),
        "opponent_state_age_frames": np.arange(n),
        "pill": np.asarray([[1, 0]] * n),
        "preview": np.asarray([[2, 1]] * n),
        "candidate_actions": candidates,
        "candidate_costs": costs,
        "candidate_count": np.asarray([2] * n),
        "chosen_slot": np.arange(n) % 2,
        "rating": np.linspace(1000, 2000, n, dtype=np.float32),
        "tau_frames": np.asarray([45] * n),
        "chosen_cost": np.asarray([30] * n),
        "speed": np.asarray([2] * n),
        "speed_ups": np.asarray([0] * n),
        "split": np.asarray([0] * 12 + [1] * 4),
        "time_split": np.asarray([0] * n),
        "player_fold": np.asarray([1] * 10 + [0] * 2 + [1] * 4),
    }
    policy, timing, result, condition = train(
        arrays,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=3,
        capacity="small",
    )
    assert policy.training is False
    assert timing.training is False
    assert result["metrics"]["train_rows"] == 10
    assert np.isfinite(result["metrics"]["replay_holdout_nll"])
    assert condition.minimum == 1000.0


def _checkpoint(path) -> None:
    from drmc_rl.human.conditioning import HumanSkillCondition
    from drmc_rl.human.model import (
        HUMAN_POLICY_SCHEMA,
        build_human_policy,
        build_timing_model,
        human_policy_config,
    )
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    cfg = human_policy_config(capacity="small")
    policy = build_human_policy(cfg)
    timing = build_timing_model()
    save_checkpoint(
        {
            "schema": HUMAN_POLICY_SCHEMA,
            "cfg": cfg,
            "state_dict": policy.state_dict(),
            "timing_state_dict": timing.state_dict(),
            "human_meta": {
                "skill_condition": HumanSkillCondition.fit(
                    np.asarray([1000.0, 1500.0, 2000.0])
                ).to_dict(),
                "corpus_release_id": "fixture",
                "metrics": {"replay_holdout_top1": 0.25},
            },
        },
        path,
    )


@pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")
def test_backend_contract_is_semantic_monotonic_and_stale_safe(tmp_path) -> None:
    from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA

    checkpoint = tmp_path / "human.pt.gz"
    _checkpoint(checkpoint)
    backend = HumanBackend(str(checkpoint), seed=7)

    hello = backend.handle({"schema": PROTOCOL_SCHEMA, "type": "hello"})
    assert hello["type"] == "capabilities"
    assert hello["capabilities"]["model"]["corpus_release_id"] == "fixture"
    assert "board_planes" in hello["capabilities"]["state"]

    planes = np.zeros((8, 16, 8), dtype=np.float32)
    planes[0, 15, 0] = 1.0
    planes[3, 15, 0] = 1.0
    request = {
        "schema": PROTOCOL_SCHEMA,
        "type": "decide",
        "request_id": 1,
        "frame_id": 100,
        "deadline_ms": 10_000,
        "target_rating": 1500,
        "temperature": 0,
        "state": {
            "board_planes": planes.tolist(),
            "opponent_board_planes": np.zeros((8, 16, 8), dtype=np.float32).tolist(),
            "opponent_state_age_frames": 12,
            "pill": [0, 1],
            "preview": [2, 0],
            "speed": 2,
            "speed_ups": 0,
            "falling": {
                "x": 3,
                "y": 0,
                "rotation": 0,
                "speed_counter": 0,
                "horizontal_velocity": 0,
                "hold_dir": 0,
                "rotation_hold": 0,
                "frame_parity": 0,
            },
        },
    }
    result = backend.handle(request)
    assert result["type"] == "result"
    assert result["result"]["resolved_rating"] == 1500
    assert result["result"]["controller_frames"]
    assert result["result"]["placement"]["action"] in result["result"]["candidate_actions"]

    stale = backend.handle({**request, "request_id": 2, "frame_id": 99})
    assert stale == {
        **stale,
        "type": "stale",
        "latest_frame_id": 100,
    }
    duplicate = backend.handle({**request, "request_id": 2, "frame_id": 101})
    assert duplicate["type"] == "error"
    assert "monotonically" in duplicate["error"]["message"]
    coach = backend.handle(
        {
            **request,
            "type": "coach",
            "request_id": 3,
            "frame_id": 101,
            "chosen_action": result["result"]["placement"]["action"],
        }
    )
    assert coach["type"] == "result"
    assert coach["result"]["coach"]["chosen"]["feasible"] is True
    assert coach["result"]["coach"]["alternatives"]
