import struct

from tools.audit_execution_replay import replay_row


def recording(initial=0, buttons=4, rotation=0):
    return dict(field=bytes([0xFF]*128), held_at_spawn=initial, horizontal_velocity=0,
        speed_counter=0, frame_counter=0, speed=0, speed_ups=0,
        lock_x=3, lock_y_top=15, lock_rotation=rotation, lock_repaired=False,
        input_frames=32, tau_frames=31,
        input_rle_u16_u8=struct.pack("<HBHB", 1, initial, 31, buttons))


def test_recorded_window_replays_to_the_exact_lock_frame_and_pose():
    assert replay_row(recording())["status"] == "match"
    assert replay_row(recording(buttons=0x84, rotation=3))["status"] == "match"


def test_spawn_held_rotation_is_not_a_new_press():
    assert replay_row(recording(initial=0x80, buttons=0x84))["status"] == "match"


def test_wrong_frame_parity_is_visible_as_a_timing_mismatch():
    result = replay_row({**recording(), "frame_counter": 1})
    assert result["status"] == "mismatch"
    assert not result["time_match"]


def test_unsupported_controller_chords_are_explicitly_excluded():
    result = replay_row(recording(buttons=0xC4))
    assert result == {"status": "excluded", "reason": "both_rotation_buttons"}


def test_input_visible_before_movement_requires_explicit_recording_alignment():
    row = recording(initial=4)
    row["input_rle_u16_u8"] = struct.pack("<HBHB", 31, 4, 1, 0)
    row["frame_counter"] = 1
    assert replay_row(row)["status"] == "mismatch"
    assert replay_row(row, parity_xor=1, input_delay_frames=1)["status"] == "match"


def test_prior_held_byte_distinguishes_a_fresh_rotation_visible_at_spawn():
    row = recording(initial=0x84, rotation=3)
    row.update(frame_counter=1, held_before_spawn=0,
               input_rle_u16_u8=struct.pack("<HB", 32, 0x84))
    assert replay_row(row, parity_xor=1, input_delay_frames=1)["status"] == "match"
    row["held_before_spawn"] = 0x80
    assert not replay_row(row, parity_xor=1, input_delay_frames=1)["pose_match"]
