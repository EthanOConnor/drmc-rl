from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BottleSpec:
    base_addr: int
    width: int
    height: int
    stride: int
    encoding: str


BOTTLE = BottleSpec(
    base_addr=0x0400,
    width=8,
    height=16,
    stride=8,
    encoding="hi:type, lo:color",
)

# Current falling pill metadata for player one (player two mirrors with +0x80 offset).
FALLING_PILL: Dict[str, int] = {
    "row_addr": 0x0306,
    "col_addr": 0x0305,
    "orient_addr": 0x0325,
    "size_addr": 0x0326,
    "left_color_addr": 0x0301,
    "right_color_addr": 0x0302,
}

PREVIEW_PILL: Dict[str, int] = {
    "left_color_addr": 0x031A,
    "right_color_addr": 0x031B,
    "rotation_addr": 0x0322,
    "size_addr": 0x0323,
}

GRAVITY_LOCK: Dict[str, int] = {
    "gravity_counter_addr": 0x0312,
    "lock_counter_addr": 0x0307,
    "speed_index_addr": 0x0320,
    "speed_setting_addr": 0x030B,
}

TIMERS: Dict[str, int] = {
    "frame_counter_addr": 0x0043,
    "wait_frames_addr": 0x0051,
    "music_frames_since_last_beat_addr": 0x067F,
}

LEVEL: Dict[str, int] = {"addr": 0x0316}

GAME_STATUS: Dict[str, int] = {
    "mode_addr": 0x0046,
    "mode_in_game": 0x0004,
    "stage_clear_flag_addr": 0x0055,
    "stage_clear_value": 0x0001,
    "ending_state_addr": 0x0053,
    "ending_non_value": 0x000A,
    "player_count_addr": 0x0727,
    "pill_counter_addr": 0x0310,
}

PLAYER_BASE_OFFSETS: Dict[str, int] = {"p1": 0x0000, "p2": 0x0080}
