from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.execution.pace import PACES
from drmc_rl.human.movement import MODEL_PATH, load_model, movement_for_pace
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame
from drmc_rl.planning.native_reach import is_library_present

HUMAN_PACES = ("sloth", "relaxed", "normal", "fast", "top_humans", "super_human")
native = pytest.mark.skipif(not is_library_present(), reason="native planner library is not built")


def test_model_covers_named_paces_and_keeps_frame_perfect_exact():
    model = load_model()
    assert model["schema"] == "drmc-human-movement-model-v1"
    assert tuple(model["profiles"]) == HUMAN_PACES
    assert movement_for_pace("frame_perfect") is None
    for pace in PACES[:-1]:
        profile = movement_for_pace(pace.id).to_dict()
        assert profile["id"] == pace.id and profile["label"] == pace.label
        assert "rating" not in profile["label"].lower()
    # Compact enough for the browser package and a per-decision lookup.
    assert MODEL_PATH.stat().st_size < 100_000


def test_style_and_decisions_are_deterministic_and_vary_by_game():
    mv = movement_for_pace("normal")
    assert np.array_equal(mv.style(11), mv.style(11))
    assert not np.array_equal(mv.style(11), mv.style(12))
    first = mv.decide(11, 5, threshold=13)
    assert first == mv.decide(11, 5, threshold=13)
    reactions = {mv.decide(seed, 5, threshold=13).reaction_frames for seed in range(40)}
    assert len(reactions) > 3


def test_sampled_reaction_orders_the_paces():
    medians = []
    for pace in HUMAN_PACES:
        mv = movement_for_pace(pace)
        medians.append(np.median([mv.decide(seed, key, threshold=13).reaction_frames
                                  for seed in range(30) for key in range(20)]))
    assert medians == sorted(medians, reverse=True)
    assert medians[0] > medians[1] and medians[-2] > medians[-1]


def _situation(rng):
    cols = np.zeros(8, np.uint16)
    height = int(rng.choice([rng.integers(0, 9), rng.integers(9, 15)]))
    for c in range(8):
        for y in range(16 - int(np.clip(height + rng.integers(-4, 3), 0, 15)), 16):
            if rng.random() < 0.8:
                cols[c] |= 1 << y
    cols[3] &= ~np.uint16(3)
    cols[4] &= ~np.uint16(3)
    threshold = compute_speed_threshold(int(rng.integers(0, 3)), int(rng.integers(0, 50)))
    spawn = FrameState(x=3, y=0, rot=0, speed_counter=int(rng.integers(0, threshold + 1)),
                       hor_velocity=int(rng.integers(0, 16)), hold_dir=HoldDir(int(rng.integers(0, 3))),
                       rot_hold=Rotation(int(rng.integers(0, 3))), frame_parity=int(rng.integers(0, 2)))
    return cols, threshold, spawn


@native
@pytest.mark.parametrize("pace", HUMAN_PACES)
def test_generated_scripts_lock_exactly_and_deterministically(pace):
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    mv = movement_for_pace(pace)
    runner = NativeReachabilityRunner()
    rng = np.random.default_rng([7, len(pace)])
    routes, checked = set(), 0
    try:
        while checked < 60:
            cols, threshold, spawn = _situation(rng)
            decision = mv.decide(int(rng.integers(1 << 30)), checked, threshold=threshold)
            delay, start = max(decision.reaction_frames, 4), spawn
            for _ in range(delay):
                start = simulate_frame(cols, start, 0, speed_threshold=threshold)
            if start.locked:
                continue
            reach = runner.bfs_full(cols, start, speed_threshold=threshold, **mv.planning.planner_args(0))
            poses = np.flatnonzero(reach.costs_u16 != 0xFFFF)
            if not len(poses):
                continue
            pose = int(rng.choice(poses))
            target = (pose & 7, (pose >> 3) & 15, (pose >> 7) & 3)
            witness = reach.script_for_pose(*target).copy()
            script, info = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                                       witness=witness, execution_delay=delay)
            again, _ = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                                   witness=witness, execution_delay=delay)
            assert np.array_equal(script, again)
            state = start
            for index, action in enumerate(script, 1):
                state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
                if state.locked:
                    break
            assert state.locked and index == len(script) and (state.x, state.y, state.rot) == target
            mv.floor.validate(cols, start, script, speed_threshold=threshold, execution_delay=delay)
            assert info["validated"] and not info["unrestricted_fallback"]
            routes.add(info["route"])
            checked += 1
    finally:
        runner.close()
    assert "human" in routes


@native
def test_human_movement_is_slower_than_the_planner_route_on_open_boards():
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    runner = NativeReachabilityRunner()
    cols = np.zeros(8, np.uint16)
    for c in range(8):
        cols[c] = 0xFF00  # eight settled rows
    start = FrameState(x=3, y=0, rot=0, speed_counter=0, hor_velocity=0, hold_dir=HoldDir.NEUTRAL, frame_parity=0)
    threshold = compute_speed_threshold(1, 0)
    lengths = {}
    try:
        for pace in ("relaxed", "top_humans"):
            mv = movement_for_pace(pace)
            reach = runner.bfs_full(cols, start, speed_threshold=threshold, **mv.planning.planner_args(0))
            witness = reach.script_for_pose(0, 7, 1).copy()
            lengths[pace] = np.median([
                len(mv.generate(mv.decide(seed, 0, threshold=threshold), cols, start, (0, 7, 1),
                                speed_threshold=threshold, witness=witness, execution_delay=0)[0])
                for seed in range(40)])
            assert lengths[pace] >= len(witness)
    finally:
        runner.close()
    assert lengths["relaxed"] > lengths["top_humans"]


def test_arena_keys_human_movement_separately():
    from tools.trainer_planning_arena import bind_execution_profiles

    def config(movement):
        return {"variants": {"a": {"movement": movement}, "b": {}},
                "schedule": [{"id": "x", "a": "a", "b": "b", "pace": "normal"}]}
    exact, human = config("exact"), config("human")
    bind_execution_profiles(exact)
    bind_execution_profiles(human)
    assert exact["schedule"][0]["execution_key"] != human["schedule"][0]["execution_key"]
    assert human["schedule"][0]["movement_profiles"]["a"]["movement"] == "human"
    with pytest.raises(ValueError):
        bind_execution_profiles(config("robot"))


@native
@pytest.mark.parametrize("pace", [p.id for p in PACES])
def test_backend_human_movement_option(tmp_path, pace):
    from drmc_rl.human.backend import ACTION_TO_POSE, HumanBackend, PROTOCOL_SCHEMA, _columns, _board_planes
    from tests.test_human_backend import _afterstate_checkpoint

    checkpoint = tmp_path / "human-v3.pt.gz"
    _afterstate_checkpoint(checkpoint)
    backend = HumanBackend(str(checkpoint), seed=3)
    planes = np.zeros((8, 16, 8), dtype=np.float32)
    planes[0, 15, 0] = planes[3, 15, 0] = 1.0
    planes[1, 12:16, 5] = 1.0
    request = {
        "schema": PROTOCOL_SCHEMA, "type": "decide", "request_id": 1, "frame_id": 10,
        "deadline_ms": 10_000, "target_rating": 1600, "temperature": 0, "strength_control": "regret",
        "pace": pace, "movement": "human", "movement_seed": 99, "execution_delay_frames": 4,
        "state": {"board_planes": planes.tolist(), "opponent_board_planes": planes.tolist(),
                  "pill": [0, 1], "opponent_pill": [2, 2], "preview": [2, 0], "speed": 2, "speed_ups": 0,
                  "pill_counter_total": 7, "falling": {"x": 3, "y": 0, "rotation": 0, "frame_parity": 0}},
    }
    try:
        caps = backend.handle({"schema": PROTOCOL_SCHEMA, "type": "hello"})["capabilities"]
        assert caps["movement"]["modes"] == ["exact", "human"]
        assert "frame_perfect" not in caps["movement"]["human_paces"]
        response = backend.handle(request)
        assert response["type"] == "result", response
        result = response["result"]
        timing = result["timing"]
        assert timing["execution_profile"]["id"] == pace
        assert timing["movement"]["validated"] and timing["movement"]["unrestricted_fallback"] is False
        expected = "constrained-frame-search-v1" if pace == "frame_perfect" else "human-movement-v1"
        assert timing["movement"]["algorithm"] == expected
        # The host's start frame is unchanged; sampled reaction beyond it is neutral input.
        assert result["execution"]["start_frame"] == 14
        assert result["controller_states"][0] == result["execution"]["falling"]
        assert len(result["controller_states"]) == len(result["controller_frames"])
        if pace != "frame_perfect":
            pad = timing["movement"]["reaction_pad_frames"]
            assert result["controller_frames"][:pad] == [0] * pad
            assert timing["movement"]["first_input_delay_frames"] == 4 + pad
        # Replay the controller bytes to the reported placement.
        from drmc_rl.human.backend import falling_frame
        state = falling_frame({"falling": result["execution"]["falling"]})
        cols = _columns(_board_planes(planes))
        mask_to_action = {0: 0, 2: 6, 1: 12, 4: 3, 0x80: 1, 0x40: 2, 0x82: 7, 0x42: 8, 0x81: 13, 0x41: 14,
                          0x84: 4, 0x44: 5}
        for mask in result["controller_frames"]:
            state = simulate_frame(cols, state, mask_to_action[mask], speed_threshold=compute_speed_threshold(2, 0))
        placement = result["placement"]
        assert state.locked and (state.x, state.y, state.rot) == (placement["x"], placement["y_top"], placement["rotation"])
        assert int(ACTION_TO_POSE[placement["action"]]) >= 0
        again = backend.handle({**request, "request_id": 2})["result"]
        assert again["controller_frames"] == result["controller_frames"]
    finally:
        backend.close()


@native
@pytest.mark.parametrize("ablation", [
    {"steering": "planner", "descent": "prompt"}, {"steering": "planner"}, {"descent": "prompt"},
    {"reaction": "pace"}, {"corrections": False, "pauses": False},
])
def test_ablated_movement_still_locks_exactly(ablation):
    from drmc_rl.human.movement import MovementAblation
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    mv, switches = movement_for_pace("fast"), MovementAblation.from_dict(ablation)
    runner = NativeReachabilityRunner()
    rng = np.random.default_rng(23)
    checked, routes = 0, set()
    try:
        while checked < 30:
            cols, threshold, start = _situation(rng)
            decision = mv.decide(int(rng.integers(1 << 30)), checked, threshold=threshold)
            reach = runner.bfs_full(cols, start, speed_threshold=threshold, **mv.planning.planner_args(0))
            poses = np.flatnonzero(reach.costs_u16 != 0xFFFF)
            if not len(poses):
                continue
            pose = int(rng.choice(poses))
            target = (pose & 7, (pose >> 3) & 15, (pose >> 7) & 3)
            witness = reach.script_for_pose(*target).copy()
            script, info = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                                       witness=witness, execution_delay=4, ablation=switches)
            state = start
            for action in script:
                state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
            assert state.locked and (state.x, state.y, state.rot) == target
            assert info["ablation"] == ablation
            if switches.reaction == "pace":
                assert info.get("hesitation_frames", 0) == 0
            if switches.steering == "planner" and switches.descent == "prompt":
                assert len(script) <= len(witness) + 1
            routes.add(info["route"])
            checked += 1
    finally:
        runner.close()
    assert routes & {"human", "planner_steering"}


def test_arena_keys_movement_ablations_and_context():
    from tools.trainer_planning_arena import bind_execution_profiles

    def config(**extra):
        return {"variants": {"a": {"movement": "human", **extra}, "b": {}},
                "schedule": [{"id": "x", "a": "a", "b": "b", "pace": "fast"}]}
    keys = set()
    for extra in ({}, {"movement_ablation": {"descent": "prompt"}}, {"context_pace": "normal"}):
        c = config(**extra)
        bind_execution_profiles(c)
        keys.add(c["schedule"][0]["execution_key"])
    assert len(keys) == 3
    with pytest.raises(ValueError):
        bind_execution_profiles({"variants": {"a": {"context_pace": "normal"}, "b": {}},
                                 "schedule": [{"id": "x", "a": "a", "b": "b", "pace": "fast"}]})
