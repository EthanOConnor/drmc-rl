from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from . import retail_wram as ram


@dataclass(frozen=True)
class BottleSpec:
    base_addr: int
    width: int
    height: int
    stride: int
    encoding: str


BOTTLE = BottleSpec(
    base_addr=ram.P1_FIELD,
    width=8,
    height=16,
    stride=8,
    encoding="hi:type, lo:color",
)

# Current falling pill metadata for player one (player two mirrors with +0x80 offset).
FALLING_PILL: Dict[str, int] = {
    "row_addr": ram.p1(ram.FALLING_PILL_ROW),
    "col_addr": ram.p1(ram.FALLING_PILL_COLUMN),
    "orient_addr": ram.p1(ram.FALLING_PILL_ROTATION),
    "size_addr": ram.p1(ram.FALLING_PILL_SIZE),
    "left_color_addr": ram.p1(ram.FALLING_PILL_COLOR_LEFT),
    "right_color_addr": ram.p1(ram.FALLING_PILL_COLOR_RIGHT),
}

PREVIEW_PILL: Dict[str, int] = {
    "left_color_addr": ram.p1(ram.NEXT_PILL_COLOR_LEFT),
    "right_color_addr": ram.p1(ram.NEXT_PILL_COLOR_RIGHT),
    "rotation_addr": ram.p1(ram.NEXT_PILL_ROTATION),
    "size_addr": ram.p1(ram.NEXT_PILL_SIZE),
}

GRAVITY_LOCK: Dict[str, int] = {
    "gravity_counter_addr": ram.p1(ram.SPEED_COUNTER),
    "lock_counter_addr": ram.p1(ram.LOCK_COUNTER),
    "speed_index_addr": ram.p1(ram.SPEED_INDEX),
    "speed_setting_addr": ram.p1(ram.SPEED_SETTING),
}

TIMERS: Dict[str, int] = {
    "frame_counter_addr": ram.FRAME_COUNTER,
    "wait_frames_addr": ram.WAIT_FRAMES,
    "music_frames_since_last_beat_addr": ram.MUSIC_FRAMES_SINCE_LAST_BEAT,
}

LEVEL: Dict[str, int] = {"addr": ram.p1(ram.LEVEL)}

GAME_STATUS: Dict[str, int] = {
    "mode_addr": ram.GAME_MODE,
    "mode_in_game": 0x0004,
    "stage_clear_flag_addr": ram.WHO_WON,
    "stage_clear_value": 0x0001,
    "ending_state_addr": ram.FINAL_CUTSCENE_STEP,
    "ending_non_value": 0x0000,
    "player_count_addr": ram.PLAYER_COUNT,
    "pill_counter_addr": ram.p1(ram.PILLS_COUNTER_DECIMAL),
}

PLAYER_BASE_OFFSETS: Dict[str, int] = {"p1": 0x0000, "p2": 0x0080}
