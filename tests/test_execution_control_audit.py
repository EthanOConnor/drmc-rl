import numpy as np
import pytest

from drmc_rl.planning.native_reach import NativeReachabilityRunner, is_library_present
from tools.audit_execution_control import action, audit_row, buttons, columns, initial_state, metrics, replay, summary, weights


def test_controller_alphabet_round_trip():
    assert [action(buttons(i)) for i in range(18)] == list(range(18))


def test_equal_player_and_game_weights():
    rows = [{"player_id": "a", "game_id": "1"}] * 9 + [
        {"player_id": "a", "game_id": "2"},
        {"player_id": "b", "game_id": "3"},
    ]
    w = weights(rows)
    assert w[:9].sum() == pytest.approx(w[9])
    assert w[:10].sum() == pytest.approx(w[10])


def test_carried_input_does_not_imply_fresh_reaction():
    m = metrics([4, 4, 4, 0, 2], 4)
    assert m["reaction_frames"] == 0
    assert m["first_change_frames"] == 3
    assert m["carried_input"] == 1


def test_summary_handles_undefined_edge_intervals():
    rows = [{"player_id": "a", "game_id": "1", "difference": {}},
            {"player_id": "b", "game_id": "2", "difference": {"interval": 2}}]
    assert summary(rows, "difference")["metrics"]["interval"]["mean"] == 2


@pytest.mark.skipif(not is_library_present(), reason="native planner unavailable")
def test_neutral_lead_loses_escape_from_high_board():
    board = np.full((16, 8), 255, dtype=np.uint8)
    board[1, 3] = 0xD0
    row = {"player_id": "a", "game_id": "1", "cohort": "1600/speed2", "partition": "test",
           "field_hex": board.tobytes().hex(), "initial_buttons": 0,
           "initial_speed_counter": 12, "initial_horizontal_velocity": 0,
           "initial_frame_parity": 0, "speed": 2, "speed_ups": 0, "high_board": True, "rating": 1800.0}
    script = [1, 0, 1, 0, 1] + [4] * 40
    lock, tau = replay(columns(row), initial_state(row), script, 13)
    assert lock.locked and (lock.x, lock.y, lock.rot) == (6, 15, 0)
    row.update(script=script[:tau], lock_pose=[6, 15, 0])
    row["human"] = metrics(row["script"], 0)
    result = audit_row(row, NativeReachabilityRunner(max_frames=128), [0, 4, 8])
    assert result["lead_0"]["target_available"] == 1
    assert result["lead_4"]["target_available"] == 0
    assert result["lead_8"]["target_available"] == 0
