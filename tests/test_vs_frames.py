"""Controller arena checks against the independent Python frame stepper."""
import os

import numpy as np
import pytest

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.backend import ACTION_TO_BUTTONS, ACTION_TO_POSE, _columns, plan_candidates
from drmc_rl.planning.fast_reach import compute_speed_threshold, simulate_frame
from drmc_rl.planning.native_reach import NativeReachabilityRunner


@pytest.mark.parametrize("level,delay,parity", [(14,0,0), (14,3,1), (14,8,0), (20,3,0), (20,8,1)])
def test_controller_script_matches_independent_physics(level, delay, parity):
    with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
        pool.reset([17291], level=level)
        while not all(s.falling for s in pool.states):
            pool.step()
        if pool.states[0].frame_parity != parity:
            pool.step()
        planner = NativeReachabilityRunner()
        try:
            state = pool.states[0].semantic(pool.states[1])
            own, _, _, _, speed, ups, frame, reach, packed, _ = plan_candidates(
                planner, state, delay, resolve_pace("frame_perfect"))
            action = int(packed.actions[packed.count // 2])
            pose = int(ACTION_TO_POSE[action])
            x, y, rot = pose % 8, (pose // 8) % 16, pose // 128
            script = reach.script_for_pose(x, y, rot)
            pool.step(count=delay)
            for buttons in script:
                observed = pool.states[0]
                assert (observed.x, observed.y_top, observed.rotation,
                        observed.speed_counter, observed.horizontal_velocity,
                        observed.frame_parity) == (frame.x, frame.y, frame.rot,
                        frame.speed_counter, frame.hor_velocity, frame.frame_parity)
                frame = simulate_frame(_columns(own), frame, int(buttons),
                                       speed_threshold=compute_speed_threshold(speed, ups))
                pool.step([ACTION_TO_BUTTONS[int(buttons)]] * 2)
            assert frame.locked and not pool.states[0].falling
            assert (pool.states[0].x, pool.states[0].y_top, pool.states[0].rotation) == (x,y,rot)
            assert bytes(pool.states[0].board) == bytes(pool.states[1].board)
        finally:
            planner.close()


def test_frame_reset_is_seeded_and_terminal_stops_both_sides():
    with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
        pool.reset([40219])
        first = bytes(pool.states[0].board)
        pool.reset([40219])
        assert bytes(pool.states[0].board) == first
        pool.reset([19073])
        assert bytes(pool.states[0].board) != first
        for _ in range(20):
            pool.step(count=6000)
            if pool.states[0].terminal:
                break
        assert pool.states[0].terminal and pool.states[1].terminal
        assert pool.states[0].outcome == pool.states[1].outcome == 3
        assert pool.states[0].frame == pool.states[1].frame
        final = bytes(pool.states)
        pool.step(count=10)
        assert bytes(pool.states) == final
