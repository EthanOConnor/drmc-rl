import copy
import json
import os

import numpy as np
import pytest
import torch

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import BY_ID
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.human.anticipation import (
    NextTurnGeometryPreparer, NextTurnPreparer, execution_for_action,
)
from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA, plan_candidates
from drmc_rl.human.controller_context import live_controller_state
from drmc_rl.planning.native_reach import NativeReachabilityRunner


def live_state(pool):
    wire = pool.public_state(1).to_dict()
    for side in wire["sides"]:
        del side["board_b64"]
    wire.update(schema="public-controller-history-v1", compute_frames=4)
    return pool.semantic(1) | {"public_live_context": wire}


@pytest.mark.parametrize("pace_id", ["sloth", "top_humans", "frame_perfect"])
def test_geometry_matches_full_fresh_frontier_for_every_preview_and_parity(pace_id):
    planner = NativeReachabilityRunner()
    preparer = NextTurnGeometryPreparer(planner)
    pace = BY_ID[pace_id]
    delay = max(4, pace.reaction_frames)
    try:
        with FrameVsPool(lib_path=os.environ.get("DRMARIO_POOL_LIB")) as pool:
            pool.reset([17291])
            while not pool.states[1].falling:
                pool.step()
            state = live_controller_state(live_state(pool))
            # Same-color next pills must retain all rotations for context actors.
            state["preview"] = [1, 1]
            state["pill_counter_total"], state["speed_ups"] = 9, 0
            candidate = plan_candidates(planner, state, delay, pace)
            action = int(candidate[-2].actions[0])
            move = execution_for_action(candidate, action, pace, delay=delay)
            prepared = preparer.prepare(state, move, pace, delay)
            assert prepared is not None and prepared.state["speed_ups"] == 1
            assert "public_pair_state" not in prepared.state
            assert "public_live_context" not in prepared.state
            assert "opponent_board_planes" not in prepared.state
            for parity in (0, 1):
                for left in range(3):
                    for right in range(3):
                        observed = {**prepared.state, "preview": [left, right],
                                    "opponent_board_planes": state["board_planes"],
                                    "falling": {**prepared.state["falling"], "frame_parity": parity}}
                        cached, reason = prepared.select(observed, pace, delay)
                        assert reason == "hit"
                        fresh = plan_candidates(planner, observed, delay, pace)
                        for index in (0, 1, 2, 3, -1):
                            np.testing.assert_array_equal(cached[index], fresh[index])
                        np.testing.assert_array_equal(cached[-2].actions, fresh[-2].actions)
                        np.testing.assert_array_equal(cached[-2].mask, fresh[-2].mask)
                        assert cached[6] == fresh[6]
                        # Every legal action retains an identical independently
                        # validated witness, including the carried DAS/parity.
                        if left == right == 0:
                            for action in fresh[-2].actions[fresh[-2].mask]:
                                assert execution_for_action(cached, action, pace, delay=delay) == \
                                    execution_for_action(fresh, action, pace, delay=delay)
            assert prepared.select(observed, pace, delay - 1)[1] == "execution_profile"
            assert prepared.select(observed, BY_ID["normal"], delay)[1] == "execution_profile"
            changed = copy.deepcopy(observed)
            changed["board_planes"][0, 0, 0] = 1
            assert prepared.select(changed, pace, delay)[1] == "own_state"
            for field in ("x", "y", "rotation", "speed_counter", "horizontal_velocity",
                          "hold_dir", "rotation_hold"):
                changed = copy.deepcopy(observed)
                changed["falling"][field] += 1
                assert prepared.select(changed, pace, delay)[1] == "microstate"
            changed = {**observed, "public_context_schema": None}
            assert prepared.select(changed, pace, delay)[1] == "observation_contract"
            if pace_id == "sloth":
                # A faster gravity step leaves no control after the reaction
                # delay on this bottle. Preparation must not relax that delay.
                # The current 45-frame Sloth reaction needs a faster gravity
                # fixture than the old 60-frame preset did. This root remains
                # playable, but action 42 leaves no next-turn control window.
                state["speed_ups"] = 5
                candidate = plan_candidates(planner, state, delay, pace)
                assert 42 in candidate[-2].actions[candidate[-2].mask]
                move = execution_for_action(candidate, 42, pace, delay=delay)
                assert preparer.prepare(state, move, pace, delay) is None
    finally:
        preparer.close()
        planner.close()


@pytest.fixture
def context_backend(tmp_path):
    from test_human_backend import _afterstate_checkpoint
    from tools.eval_policy import _build_net_from_cfg

    torch.manual_seed(8341)
    human = tmp_path / "human.pt.gz"
    _afterstate_checkpoint(human)
    cfg = dict(candidate_architecture="g5", candidate_board_channels=16,
               candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8,
               candidate_hidden_dim=24, candidate_cross_layers=1,
               candidate_interaction_layers=1, candidate_transformer_heads=2,
               candidate_patch_kernel=3, aux_spec=PUBLIC_CONTEXT_SCHEMA,
               env={"public_observations": True})
    net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    core = tmp_path / "core.pt"
    torch.save({"cfg": cfg, "state_dict": net.state_dict()}, core)
    backend = HumanBackend(str(human), competitive_checkpoint=str(core), seed=3)
    yield backend
    backend.close()


def test_live_geometry_consumes_actual_history_and_reuses_no_scores(context_backend, monkeypatch):
    backend = context_backend
    pace, delay = BY_ID["top_humans"], 6
    assert backend.capabilities()["geometry_preparation"]["available"]
    assert not backend.capabilities()["anticipation"]["available"]
    with pytest.raises(ValueError, match="fresh late context"):
        NextTurnPreparer(backend.competitive, backend.planner)
    with FrameVsPool(lib_path=os.environ.get("DRMARIO_POOL_LIB")) as pool:
        pool.reset([17291])
        while not pool.states[1].falling:
            pool.step()
        state = live_state(pool)
        candidate = backend._candidates(live_controller_state(state), delay, pace)
        move = execution_for_action(candidate, int(candidate[-2].actions[0]), pace, delay=delay)
        request = {"schema": PROTOCOL_SCHEMA, "type": "prepare_geometry", "request_id": 0,
                   "frame_id": 1, "state": state, "committed": move,
                   "pace": pace.id, "execution_delay_frames": delay}
        def reject_score(*args, **kwargs):
            raise AssertionError("geometry preparation must not infer future policy inputs")
        with monkeypatch.context() as scoped:
            scoped.setattr(backend.competitive, "score", reject_score)
            preparation = backend.handle(request)
        assert preparation["type"] == "geometry_prepared", preparation
        token = preparation["geometry_token"]
        assert token and len(preparation["candidate_counts"]) == 2
        json.dumps(preparation)  # No native buffers, history or speculative states.
        assert "state" not in preparation and "prepared" not in preparation
        pool.step(count=delay)
        for bits in move["controller_frames"]:
            pool.step([bits, bits])
        while not pool.states[1].falling and not pool.states[1].terminal:
            pool.step()
        assert not pool.states[1].terminal
        actual = live_state(pool)
        assert actual["public_live_context"]["frame_id"] > state["public_live_context"]["frame_id"]
        assert not np.array_equal(actual["opponent_board_planes"], state["opponent_board_planes"])
        decision = {**request, "type": "decide", "target_rating": 1600, "temperature": 0,
                    "strength_control": "quality", "state": actual, "frame_id": 1000}
        calls = []
        original_score = backend.competitive.score
        def capture(obs, infos):
            calls.append((obs.copy(), copy.deepcopy(infos)))
            return original_score(obs, infos)
        monkeypatch.setattr(backend.competitive, "score", capture)
        fresh = backend._infer(decision, remaining_ms=10000)
        def reject_plan(*args, **kwargs):
            raise AssertionError("matching prepared geometry should not replan")
        with monkeypatch.context() as scoped:
            scoped.setattr(backend, "_candidates", reject_plan)
            cached = backend._infer({**decision, "geometry_token": token}, remaining_ms=10000)
        assert cached["geometry_preparation"]["status"] == "hit"
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0][0], calls[1][0])
        for key in calls[0][1][0]:
            a, b = calls[0][1][0][key], calls[1][1][0][key]
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b
        for key in ("placement", "controller_frames", "controller_states", "execution",
                    "competitive_scores", "candidate_actions", "candidate_count"):
            assert cached[key] == fresh[key]
        repeated = backend._infer({**decision, "geometry_token": token}, remaining_ms=10000)
        assert repeated["geometry_preparation"]["status"] == "unavailable"
        assert repeated["placement"] == fresh["placement"]
        missing = {**decision, "request_id": 1, "state": {**actual}, "geometry_token": "old-process"}
        missing["state"].pop("public_live_context")
        rejected = backend.handle(missing)
        assert rejected["type"] == "error" and "live public history" in str(rejected)
        # Two players may prepare concurrently; a third evicts only the oldest.
        tokens = [backend.handle(request)["geometry_token"] for _ in range(3)]
        assert list(backend.prepared_geometry) == tokens[1:]


@pytest.mark.parametrize("orientation", [2, 3])
def test_live_context_executes_the_chosen_same_color_orientation(context_backend, monkeypatch, orientation):
    backend = context_backend
    pace, delay = BY_ID["top_humans"], 6
    with FrameVsPool(lib_path=os.environ.get("DRMARIO_POOL_LIB")) as pool:
        for seed in range(20):
            pool.reset([seed])
            while not pool.states[1].falling:
                pool.step()
            state = live_state(pool)
            if state["pill"][0] == state["pill"][1]:
                break
        assert state["pill"][0] == state["pill"][1]
        candidate = backend._candidates(live_controller_state(state), delay, pace)
        choices = candidate[-2].actions[candidate[-2].mask]
        action = int(next(a for a in choices if a // 128 == orientation))
        expected = execution_for_action(candidate, action, pace, delay=delay, frame_id=100)
        def choose_exact_pose(obs, infos):
            legal = np.flatnonzero(infos[0]["placements/feasible_mask"].reshape(512))
            return legal[None], np.ones((1,len(legal)),bool), (legal==action)[None].astype(float)
        monkeypatch.setattr(backend.competitive,"score",choose_exact_pose)
        result = backend._infer({"type":"decide","strength_control":"quality","target_rating":1600,
            "temperature":0,"pace":pace.id,"execution_delay_frames":delay,"frame_id":100,
            "state":state}, remaining_ms=10000)
        assert result["placement"] == expected["placement"]
        assert result["controller_frames"] == expected["controller_frames"]
        assert result["controller_states"] == expected["controller_states"]
        assert result["execution"] == expected["execution"]
