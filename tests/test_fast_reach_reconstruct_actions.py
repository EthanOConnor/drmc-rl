from __future__ import annotations

import numpy as np

from envs.retro.fast_reach import (
    FrameState,
    HoldDir,
    ReachabilityConfig,
    build_reachability,
    reconstruct_actions,
    simulate_frame,
)


def test_reconstruct_actions_reaches_locked_terminal_pose() -> None:
    # Empty board: dropping straight down should lock at the bottom.
    cols = np.zeros((8,), dtype=np.uint16)
    spawn = FrameState(
        x=3,
        y=3,
        rot=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        locked=False,
    )

    # Use a typical early-level speed threshold; soft-drop dominates anyway.
    speed_threshold = 0x45
    result = build_reachability(cols, spawn, speed_threshold=speed_threshold, config=ReachabilityConfig(max_frames=256))

    node_idx = result.terminal_nodes.get((3, 15, 0))
    assert node_idx is not None

    actions = reconstruct_actions(result, int(node_idx))
    assert actions, "Expected a non-empty action sequence to reach a locked state"

    st = spawn
    for a in actions:
        st = simulate_frame(cols, st, int(a), speed_threshold=speed_threshold)

    assert st.locked is True
    assert (st.x, st.y, st.rot) == (3, 15, 0)

