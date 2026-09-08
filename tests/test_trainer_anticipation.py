import os

import numpy as np

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import NextTurnPreparer, execution_for_action, own_board_only, public_policy_inputs, select_prepared
from drmc_rl.human.backend import plan_candidates
from drmc_rl.planning.native_reach import NativeReachabilityRunner


class CostPolicy:
    def score(self, _obs, infos):
        costs = np.stack([np.asarray(i["placements/cost_to_lock"]).reshape(-1) for i in infos])
        actions = np.broadcast_to(np.arange(512), costs.shape)
        return actions, costs != 65535, -costs.astype(float)


def test_preparation_protocol_serializes_public_planes_and_expired_decisions_skip_inference():
    import json
    from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA

    backend = HumanBackend.__new__(HumanBackend)
    backend.planner = NativeReachabilityRunner()
    backend.competitive = CostPolicy()
    backend.preparer = NextTurnPreparer(backend.competitive, backend.planner,
        lib_path=os.environ.get("DRMC_FRAME_LIBRARY"))
    backend.requests = backend.errors = 0
    backend.latencies_ms = []
    backend.last_request_id = 17
    backend.latest_frame_id = -1
    backend.cancelled = set()
    try:
        with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
            pool.reset([61183])
            pool.step(count=2)
            state = pool.states[0].semantic(pool.states[1])
            pace = resolve_pace("frame_perfect")
            candidate = plan_candidates(backend.planner, state, 0, pace)
            move = execution_for_action(candidate, int(candidate[-2].actions[0]), pace)
            response = backend.handle({"schema":PROTOCOL_SCHEMA, "type":"prepare_next", "request_id":17,
                "frame_id":2, "state":state, "committed":move, "pace":"frame_perfect"})
            assert response["type"] == "prepared" and response["prepared"] is not None
            assert backend.last_request_id == 17
            encoded = json.loads(json.dumps(response))
            assert isinstance(encoded["prepared"]["state"]["board_planes"][0][0][0], int)
            assert len(encoded["prepared"]["branches"]) == 2
            response = backend.handle({"schema":PROTOCOL_SCHEMA, "type":"decide", "request_id":18,
                "frame_id":3, "deadline_ms":0})
            assert response["type"] == "deadline_exceeded" and backend.errors == 0
    finally:
        backend.preparer.close()
        backend.planner.close()


def test_opponent_ablation_changes_only_opponent_actor_inputs():
    own = np.zeros((8,16,8), dtype=np.float32)
    own[0,15,0] = 1
    first = {"board_planes": own, "opponent_board_planes": np.ones_like(own),
             "pill": [0,1], "preview": [1,2], "opponent_pill": [1,2]}
    second = {**first, "opponent_board_planes": np.zeros_like(own), "opponent_pill": [2,2]}
    a, b = own_board_only(first), own_board_only(second)
    assert a["board_planes"] is own and b["board_planes"] is own
    assert first["opponent_board_planes"].any()
    np.testing.assert_array_equal(a["opponent_board_planes"], b["opponent_board_planes"])
    costs = np.full(512,65535,np.uint16)
    costs[0] = 20
    encoded, _ = public_policy_inputs(a["board_planes"], a["opponent_board_planes"],
        a["pill"], a["opponent_pill"], costs, [a["preview"]])
    assert not encoded[:,8:16].any()
    np.testing.assert_array_equal(encoded[0,:8], own)


def test_predicted_bottle_and_spawn_match_controller_continuation():
    lib = os.environ.get("DRMC_FRAME_LIBRARY")
    planner = NativeReachabilityRunner()
    prepare = NextTurnPreparer(CostPolicy(), planner, lib_path=lib)
    pace = resolve_pace("frame_perfect")
    rng = np.random.default_rng(829)
    try:
        with FrameVsPool(lib_path=lib) as pool:
            pool.reset([19291])
            while not pool.states[0].falling:
                pool.step()
            for _ in range(10):
                state = pool.states[0].semantic(pool.states[1])
                candidate = plan_candidates(planner, state, 0, pace)
                packed = candidate[-2]
                # Exercise different orientations and DAS carry, not only Down.
                action = int(rng.choice(packed.actions[packed.mask]))
                move = execution_for_action(candidate, action, pace)
                prediction = prepare.predict(state, move)
                for bits in move["controller_frames"]:
                    pool.step([bits, bits])
                while not pool.states[0].falling and not pool.states[0].terminal:
                    pool.step()
                if pool.states[0].terminal:
                    break
                actual = pool.states[0].semantic(pool.states[1])
                assert prediction is not None
                for key in ("board_planes", "pill", "speed_ups"):
                    np.testing.assert_array_equal(prediction[key], actual[key])
                for key in ("x", "y", "rotation", "speed_counter", "horizontal_velocity", "hold_dir", "rotation_hold"):
                    assert prediction["falling"][key] == actual["falling"][key]
    finally:
        prepare.close()
        planner.close()


def test_all_previews_and_parities_are_conditional_and_cache_rejects_change():
    lib = os.environ.get("DRMC_FRAME_LIBRARY")
    planner = NativeReachabilityRunner()
    preparer = NextTurnPreparer(CostPolicy(), planner, lib_path=lib)
    pace = resolve_pace("frame_perfect")
    try:
        with FrameVsPool(lib_path=lib) as pool:
            pool.reset([61183])
            pool.step(count=2)
            state = pool.states[0].semantic(pool.states[1])
            candidate = plan_candidates(planner, state, 0, pace)
            move = execution_for_action(candidate, int(candidate[-2].actions[0]), pace)
            prepared = preparer.prepare(state, move, pace)
            assert prepared is not None
            for parity in (0, 1):
                for left in range(3):
                    for right in range(3):
                        observed = {**prepared["state"], "preview": [left, right],
                            "falling": {**prepared["state"]["falling"], "frame_parity": parity}}
                        chosen, reason = select_prepared(prepared, observed)
                        assert chosen and reason == "hit"
                        assert chosen["execution"]["falling"]["frame_parity"] == parity
            observed["speed_ups"] += 1
            assert select_prepared(prepared, observed) == (None, "own_state")
            observed["speed_ups"] -= 1
            observed["opponent_pill"] = [(observed["opponent_pill"][0]+1)%3, 0]
            assert select_prepared(prepared, observed) == (None, "opponent")
    finally:
        preparer.close()
        planner.close()
