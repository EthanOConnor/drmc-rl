import struct

import pytest

from tools.extract_execution_corpus import execution_row


def row():
    return {"decision_id": "d", "game_id": "g", "source_blob_sha256": "abc",
            "player": "fixture", "day": 1, "random_split": "validation", "player_fold": 2,
            "held_at_spawn": 4, "held_before_spawn": 0, "horizontal_velocity": 0,
            "field": bytes([0xFF] * 128), "speed_counter": 0, "frame_counter": 1,
            "input_frames": 32, "input_rle_u16_u8": struct.pack("<HBHB", 31, 4, 1, 0),
            "tau_frames": 31, "spawn_frame": 100, "lock_frame": 131,
            "lock_x": 3, "lock_y_top": 15, "lock_rotation": 0,
            "speed": 0, "speed_ups": 0, "lock_repaired": False}


def test_execution_sample_preserves_the_verified_fbneo_movement_window():
    result = execution_row(row(), rating=1600, rating_sd=30)
    assert result["script"] == [4] * 31
    assert result["initial_buttons"] == 0
    assert result["initial_horizontal_velocity"] == 0
    assert result["lock_pose"] == [3, 15, 0]
    assert result["recorded_lock_verified"] is True
    assert result["initial_frame_parity"] == 0
    assert result["split"] == "validation"
    assert "player" not in result


@pytest.mark.parametrize("change", [
    {"held_at_spawn": 0}, {"input_frames": 5}, {"tau_frames": 4}, {"lock_repaired": True},
    {"held_before_spawn": None}, {"held_before_spawn": 3},
    {"lock_x": 4}, {"frame_counter": 0},
])
def test_execution_sample_rejects_broken_recording_boundaries(change):
    with pytest.raises(ValueError):
        execution_row({**row(), **change}, rating=1600, rating_sd=30)
