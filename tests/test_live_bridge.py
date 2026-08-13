"""Local tests for the Fightcade live-play bridge (no Fightcade required).

Simulates the Lua side in Python: synthesizes state.jsonl lines from a
DrMarioPoolVecEnv reset (board bytes + spawn state), runs the server over the
file IPC, and asserts that plans arrive with valid frame-indexed button
scripts. Also unit-tests the 18-action -> NES-button mapping against the
table in vendor/drmario_native/DrMarioPool.cpp (buttons_mask_from_reach_action).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present as pool_lib_present

# Transcribed from DrMarioPool::buttons_mask_from_reach_action:
#   idx = hold_dir*6 + hold_down*3 + rot
#   hold_dir: 0 neutral, 1 left (0x02), 2 right (0x01); down 0x04;
#   rot: 0 none, 1 A (0x80), 2 B (0x40).
EXPECTED_BUTTONS = {
    0: 0x00, 1: 0x80, 2: 0x40,
    3: 0x04, 4: 0x84, 5: 0x44,
    6: 0x02, 7: 0x82, 8: 0x42,
    9: 0x06, 10: 0x86, 11: 0x46,
    12: 0x01, 13: 0x81, 14: 0x41,
    15: 0x05, 16: 0x85, 17: 0x45,
}


def test_action_to_buttons_matches_pool_table() -> None:
    from tools.live_agent_server import ACTION_TO_BUTTONS

    assert len(ACTION_TO_BUTTONS) == 18
    for idx, expected in EXPECTED_BUTTONS.items():
        assert ACTION_TO_BUTTONS[idx] == expected, f"action {idx}"


@pytest.mark.skipif(
    not pool_lib_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
def test_live_bridge_plan_roundtrip(tmp_path) -> None:
    from drmc_rl.planning.fast_reach import (
        FrameState,
        HoldDir,
        compute_speed_threshold,
        simulate_frame,
    )
    from tools.live_agent_server import (
        ACTION_TO_BUTTONS,
        LiveAgentServer,
        cols_from_field,
        field_from_hex,
    )
    from drmc_rl.training.envs.drmario_pool_vec import DrMarioPoolVecEnv, _canonical_to_raw_color

    # Take one real decision-time snapshot from the pool backend (light: 1 env).
    env = DrMarioPoolVecEnv(
        num_envs=1,
        state_repr="bitplane_bottle_conn_mask",
        level=4,
        speed_setting=2,
        emit_board=True,
    )
    try:
        _obs, infos = env.reset(seed=123)
        info = infos[0]
        board = np.asarray(info["board"], dtype=np.uint8).reshape(-1)
        pill = np.asarray(info["next_pill_colors"], dtype=np.int64)
        prev = info["preview_pill"]
    finally:
        env.close()

    margin = 6
    line = {
        "f": 100, "nesf": 100, "mode": 4, "side": 1, "pc": 1, "na": 0,
        "pill": [_canonical_to_raw_color(int(pill[0])), _canonical_to_raw_color(int(pill[1]))],
        "prev": [int(prev["first_color"]), int(prev["second_color"])],
        "x": 3, "y": 15, "rot": 0, "sc": 0, "hv": 0,
        "spd": 2, "spdups": 0, "level": 4, "atk_us": 0, "atk_opp": 0,
        "field": board.tobytes().hex(),
    }
    (tmp_path / "state.jsonl").write_text(json.dumps(line) + "\n")

    server = LiveAgentServer(tmp_path, side=1, policy="greedy-cost", margin=margin)
    server.run(once=True, from_start=True)

    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["plan_id"] == 1
    assert plan["spawn"] == 1
    assert plan["start_frame"] == line["f"] + margin  # respects the latency margin
    buttons = plan["buttons"]
    assert len(buttons) > 0
    assert all(b in EXPECTED_BUTTONS.values() for b in buttons)

    decision = server.last_decision
    assert decision is not None
    assert decision["script_len"] == len(buttons)
    assert decision["cost"] == len(buttons)
    assert buttons == [ACTION_TO_BUTTONS[int(a)] for a in decision["script"]]

    # Independent re-simulation: neutral over the margin, then the script must
    # lock exactly at the decided pose (validates feasibility of every action).
    cols = cols_from_field(field_from_hex(line["field"]))
    thr = compute_speed_threshold(line["spd"], line["spdups"])
    st = FrameState(
        x=line["x"], y=15 - line["y"], rot=line["rot"],
        speed_counter=line["sc"], hor_velocity=line["hv"],
        hold_dir=HoldDir.NEUTRAL, frame_parity=line["nesf"] & 1,
    )
    for _ in range(margin):
        st = simulate_frame(cols, st, 0, speed_threshold=thr)
        assert not st.locked
    for a in decision["script"]:
        st = simulate_frame(cols, st, int(a), speed_threshold=thr)
    assert st.locked
    assert (st.x, st.y, st.rot & 3) == tuple(decision["pose"])

    # Same spawn, state line at/after start_frame => the plan missed its window
    # (or desynced): the server must re-plan from the fresh micro-state.
    line2 = dict(line)
    line2["f"] = 120
    line2["nesf"] = 120
    plan2 = server.process_line(line2)
    assert plan2 is not None
    assert plan2["plan_id"] == 2
    assert plan2["start_frame"] == 126
    assert plan2["spawn"] == 1

    # New spawn => new decision.
    line3 = dict(line)
    line3["f"] = 400
    line3["nesf"] = 400
    line3["pc"] = 2
    plan3 = server.process_line(line3)
    assert plan3 is not None
    assert plan3["plan_id"] == 3
    assert plan3["spawn"] == 2
