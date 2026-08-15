from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.human.coach import analyze_choice
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.search import blend_human_and_search, semantic_planes_to_nes_board
from drmc_rl.planning.native_reach import is_library_present


def test_skill_condition_is_continuous_and_clamped() -> None:
    condition = HumanSkillCondition.fit(np.array([1000.0, 1500.0, 2000.0]))
    encoded = condition.encode(np.array([1499.0, 1500.0, 1501.0]))
    assert encoded.shape == (3, 2)
    assert np.linalg.norm(encoded[2] - encoded[1]) < 0.01
    assert condition.resolve(500.0) == (1000.0, True)
    assert condition.resolve(1750.0) == (1750.0, False)


def test_human_v2_trunk_encodes_both_boards() -> None:
    from drmc_rl.human.model import build_human_policy, human_policy_config

    net = build_human_policy(human_policy_config(capacity="small"))
    assert net.in_channels == 20
    assert net.board_channels == 16


def test_strength_calibration_math() -> None:
    from tools.calibrate_human_strength import (
        elo_from_win_rate,
        parse_contestants,
        relative_elo,
        wilson_interval,
    )

    assert elo_from_win_rate(0.5, 1600) == 1600
    assert elo_from_win_rate(0.75, 1600) > 1750
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    contestants = parse_contestants("low:1000:0,high:1600:0.5")
    assert contestants[1]["search_weight"] == 0.5
    ratings = relative_elo(
        [{"left": "high", "right": "low", "wins": 75, "losses": 25, "draws": 0, "matches": 100}],
        ["low", "high"],
    )
    assert ratings["high"] > ratings["low"]
    assert abs(sum(ratings.values())) < 1e-9


def test_adaptive_search_budget_preserves_headroom_and_recovers_from_misses() -> None:
    from drmc_rl.human.backend import AdaptiveSearchBudget

    budget = AdaptiveSearchBudget()
    assert budget.resolve(100.0) == pytest.approx(66.0)
    assert budget.resolve(100.0, 40.0) == 40.0
    before = budget.utilization
    budget.observe(deadline_exceeded=True)
    assert budget.utilization < before
    reduced = budget.resolve(100.0)
    budget.observe(deadline_exceeded=False)
    assert budget.resolve(100.0) > reduced


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
    assert (
        result["interpretation"]["human_probability"]
        != result["interpretation"]["competitive_score"]
    )


def test_semantic_board_round_trip_input_and_search_blend() -> None:
    planes = np.zeros((8, 16, 8), dtype=np.float32)
    planes[0, 15, 0] = 1.0
    planes[3, 15, 0] = 1.0
    planes[2, 14, 1] = 1.0
    planes[6, 14, 1] = 1.0
    board = semantic_planes_to_nes_board(planes).reshape(16, 8)
    assert board[15, 0] == 0xD1  # canonical red -> raw red virus
    assert board[14, 1] == 0x72  # canonical blue, connected left
    assert board[0, 0] == 0xFF

    human = np.asarray([2.0, 1.0, 0.0])
    value = np.asarray([0.0, 3.0, -1.0])
    np.testing.assert_array_equal(blend_human_and_search(human, value, weight=0), human)
    assert np.argmax(blend_human_and_search(human, value, weight=2.0)) == 1


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
        "rating_sd": np.asarray([50.0] * n),
        "opponent_rating": np.asarray([1200.0, 1500.0, 1800.0]),
        "opponent_rating_sd": np.asarray([60.0] * n),
        "game_phase": np.asarray([0.1, 0.2, 0.3]),
        "history": np.zeros((n, 32), dtype=np.float32),
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
    assert aux.shape == (n, 40)
    assert aux[:, 6].tolist() == [0.0, 0.5, 1.0]


def test_temporal_context_uses_prior_decisions_in_same_round() -> None:
    from tools.train_human_policy import _attach_temporal_context

    common = {
        "decision_id": "one",
        "game_id": "round-1",
        "lock_x": 3,
        "lock_y_top": 10,
        "lock_rotation": 0,
        "pill_left": 0,
        "pill_right": 1,
        "tau_frames": 60,
    }
    rows = [
        {**common, "player_slot": 1},
        {**common, "decision_id": "two", "player_slot": 2},
        {**common, "decision_id": "three", "player_slot": 1, "tau_frames": 70},
    ]
    _attach_temporal_context(rows, {}, {})
    assert rows[0]["_history"].sum() == 0
    assert rows[1]["_history"].sum() == 0
    assert rows[2]["_history"][0] == 1.0
    assert rows[2]["_game_phase"] == 0.01

    sampled = [dict(row) for row in rows]
    for row in sampled:
        row.pop("_history", None)
        row.pop("_game_phase", None)
    _attach_temporal_context(sampled, {}, {}, retained_ids={"three"})
    assert "_history" not in sampled[0]
    assert "_history" not in sampled[1]
    assert sampled[2]["_history"][0] == 1.0


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
        "rating_sd": np.full(n, 50.0, dtype=np.float32),
        "opponent_rating": np.linspace(1100, 1900, n, dtype=np.float32),
        "opponent_rating_sd": np.full(n, 60.0, dtype=np.float32),
        "game_phase": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "history": np.zeros((n, 32), dtype=np.float32),
        "won": np.arange(n) % 2 == 0,
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


def test_sharded_training_streams_and_selects_checkpoint(tmp_path) -> None:
    from tools.train_human_policy import KMAX, train_sharded

    n = 24
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
        "rating_sd": np.full(n, 50.0, dtype=np.float32),
        "opponent_rating": np.linspace(1100, 1900, n, dtype=np.float32),
        "opponent_rating_sd": np.full(n, 60.0, dtype=np.float32),
        "game_phase": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "history": np.zeros((n, 32), dtype=np.float32),
        "won": np.arange(n) % 2 == 0,
        "tau_frames": np.asarray([45] * n),
        "chosen_cost": np.asarray([30] * n),
        "speed": np.asarray([2] * n),
        "speed_ups": np.asarray([0] * n),
        "split": np.asarray([0] * 18 + [1] * 6),
        "time_split": np.asarray([0] * n),
        "player_fold": np.asarray([1] * 16 + [0] * 2 + [1] * 6),
    }
    paths = []
    for shard, selected in enumerate((np.arange(0, 12), np.arange(12, 24))):
        path = tmp_path / f"shard-{shard}.npz"
        np.savez_compressed(path, **{key: value[selected] for key, value in arrays.items()})
        paths.append(path)
    policy, timing, result, condition = train_sharded(
        paths,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=3,
        capacity="small",
    )
    assert policy.training is False
    assert timing.training is False
    assert result["metrics"]["shards"] == 2
    assert result["metrics"]["best_epoch"] == 1
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


def _afterstate_checkpoint(path) -> None:
    from drmc_rl.human.afterstate_model import (
        HUMAN_AFTERSTATE_SCHEMA,
        afterstate_policy_config,
        build_afterstate_policy,
    )
    from drmc_rl.human.conditioning import HumanSkillCondition
    from drmc_rl.human.model import POLICY_CONDITION_DIM, build_timing_model
    from drmc_rl.human.strength import RegretCalibration
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    condition = HumanSkillCondition.fit(np.asarray([800.0, 1500.0, 2400.0]))
    ratings = np.repeat(np.asarray([800.0, 1500.0, 2400.0]), 80)
    calibration = RegretCalibration.fit(
        ratings,
        np.repeat(np.asarray([1.0, 0.5, 0.1]), 80),
        np.tile(np.linspace(0.1, 1.0, 80), 3),
        rating_bins=3,
        opportunity_bins=2,
    )
    cfg = afterstate_policy_config(capacity="small")
    model = build_afterstate_policy(cfg, condition_dim=POLICY_CONDITION_DIM)
    timing = build_timing_model()
    save_checkpoint(
        {
            "schema": HUMAN_AFTERSTATE_SCHEMA,
            "cfg": cfg,
            "state_dict": model.state_dict(),
            "timing_state_dict": timing.state_dict(),
            "human_meta": {
                "skill_condition": condition.to_dict(),
                "regret_calibration": calibration.to_dict(),
                "source_dataset": "fixture",
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
        },
        path,
    )


def test_opponent_pool_loads_fixed_rating_human_checkpoint(tmp_path) -> None:
    from drmc_rl.training.envs.vs_opponents import OpponentPool

    checkpoint = tmp_path / "human.pt.gz"
    _checkpoint(checkpoint)
    pool = OpponentPool(tmp_path / "pool", device="cpu")
    pool.seed_humans([{"checkpoint": str(checkpoint), "rating": 1750, "rating_sd": 80}])
    entry = pool.entries[0]
    pool.ensure_loaded(entry)
    assert entry.kind == "human"
    assert entry.rating == 1750
    assert entry.aux_spec == "human_v2"
    assert entry.aux_dim == 40
    assert entry.net.in_channels == 20

    reloaded = OpponentPool(tmp_path / "pool", device="cpu")
    assert reloaded.entries[0].kind == "human"
    assert reloaded.entries[0].rating == 1750


@pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")
def test_opponent_pool_loads_explicit_afterstate_teacher(tmp_path) -> None:
    from drmc_rl.training.envs.vs_opponents import OpponentPool

    checkpoint = tmp_path / "afterstate.pt.gz"
    _afterstate_checkpoint(checkpoint)
    pool = OpponentPool(tmp_path / "pool", device="cpu")
    pool.seed_afterstates(
        [
            {
                "checkpoint": str(checkpoint),
                "id": "v3-quality",
                "selection": "quality",
                "rating": 1900,
            }
        ]
    )
    entry = pool.entries[0]
    pool.ensure_loaded(entry)
    try:
        assert entry.kind == "afterstate"
        assert entry.selection == "quality"
        assert entry.rating == 1900
        assert entry.runtime is not None
        assert entry.net is entry.runtime
    finally:
        pool.close()
    assert entry.runtime is None
    assert entry.net is None


@pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")
def test_vs_pool_can_step_against_fixed_rating_human(tmp_path) -> None:
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv
    from drmc_rl.training.envs.vs_opponents import OpponentPool

    checkpoint = tmp_path / "human.pt.gz"
    _checkpoint(checkpoint)
    pool = OpponentPool(tmp_path / "pool", device="cpu")
    pool.seed_humans([{"checkpoint": str(checkpoint), "rating": 1750}])
    env = DrMarioVsPoolVecEnv(
        num_pairs=1,
        state_repr="bitplane_bottle_conn_mask_vs",
        opponent_pool_cfg={"enabled": True, "pool": pool},
    )
    try:
        obs, infos = env.reset(seed=3)
        assert obs.shape == (1, 20, 16, 8)
        mask = np.asarray(infos[0]["placements/feasible_mask"], dtype=bool).reshape(-1)
        action = int(np.flatnonzero(mask)[0]) if mask.any() else -1
        next_obs, _reward, _term, _trunc, _infos = env.step([action])
        assert next_obs.shape == obs.shape
        assert env._human_recent_actions[1, 0] >= 0
    finally:
        env.close()


@pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")
def test_vs_pool_batches_exact_afterstate_opponents(tmp_path) -> None:
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv
    from drmc_rl.training.envs.vs_opponents import OpponentPool

    checkpoint = tmp_path / "afterstate.pt.gz"
    _afterstate_checkpoint(checkpoint)
    pool = OpponentPool(tmp_path / "pool", device="cpu")
    pool.seed_afterstates(
        [{"checkpoint": checkpoint, "selection": "quality", "rating": 1900}]
    )
    env = DrMarioVsPoolVecEnv(
        num_pairs=2,
        state_repr="bitplane_bottle_conn_mask_vs",
        opponent_pool_cfg={"enabled": True, "pool": pool},
    )
    try:
        obs, infos = env.reset(seed=5)
        actions = []
        for info in infos:
            mask = np.asarray(info["placements/feasible_mask"], dtype=bool).reshape(-1)
            actions.append(int(np.flatnonzero(mask)[0]) if mask.any() else -1)
        next_obs, _reward, _term, _trunc, _infos = env.step(actions)
        assert next_obs.shape == obs.shape
        assert np.all(env._human_recent_actions[1::2, 0] >= 0)
    finally:
        env.close()


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
    assert coach["result"]["search"] is not None
    assert "competitive_rank" in coach["result"]["coach"]["chosen"]
    backend.close()


@pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")
def test_backend_v3_uses_exact_afterstate_quality_and_regret_control(tmp_path) -> None:
    from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA

    checkpoint = tmp_path / "human-v3.pt.gz"
    _afterstate_checkpoint(checkpoint)
    backend = HumanBackend(str(checkpoint), seed=3)
    try:
        hello = backend.handle({"schema": PROTOCOL_SCHEMA, "type": "hello"})
        assert hello["capabilities"]["model"]["schema"] == "drmc-human-afterstate-v3"
        assert hello["capabilities"]["search"]["exact_afterstate"] is True
        planes = np.zeros((8, 16, 8), dtype=np.float32)
        planes[0, 15, 0] = 1.0
        planes[3, 15, 0] = 1.0
        response = backend.handle(
            {
                "schema": PROTOCOL_SCHEMA,
                "type": "decide",
                "request_id": 1,
                "frame_id": 10,
                "deadline_ms": 10_000,
                "target_rating": 1600,
                "temperature": 0,
                "state": {
                    "board_planes": planes.tolist(),
                    "opponent_board_planes": np.zeros_like(planes).tolist(),
                    "pill": [0, 1],
                    "preview": [2, 0],
                    "speed": 2,
                    "speed_ups": 0,
                    "falling": {"x": 3, "y": 0, "rotation": 0, "frame_parity": 0},
                },
            }
        )
        assert response["type"] == "result"
        result = response["result"]
        assert result["search"]["stage"] == "exact-afterstate"
        assert len(result["competitive_scores"]) == result["candidate_count"]
        assert result["strength"]["chosen_regret"] >= 0
    finally:
        backend.close()


def test_corpus_weights_reduce_prolific_player_dominance() -> None:
    from tools.train_human_policy import _sample_weights, _stable_player_key

    ratings = np.full(4, 1800.0, dtype=np.float32)
    keys = np.asarray([1, 1, 1, 2], dtype=np.uint64)
    weights = _sample_weights(
        ratings,
        player_keys=keys,
        player_counts={1: 100, 2: 1},
    )
    assert weights[3] > weights[0]
    assert float(weights.mean()) == pytest.approx(1.0)
    assert _stable_player_key("player-a") == _stable_player_key("player-a")
    assert _stable_player_key("player-a") != _stable_player_key("player-b")
