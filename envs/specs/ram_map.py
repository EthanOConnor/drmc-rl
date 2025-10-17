from dataclasses import dataclass
@dataclass(frozen=True)
class BottleSpec:
    base_addr: int; width: int; height: int; stride: int; encoding: str
BOTTLE = BottleSpec(base_addr=1024, width=8, height=16, stride=8, encoding="hi:type, lo:color")
FALLING_PILL = {'row_addr': 774, 'col_addr': 773, 'orient_addr': 805, 'size_addr': 806, 'left_color_addr': 769, 'right_color_addr': 770}
PREVIEW_PILL = {'left_color_addr': 794, 'right_color_addr': 795, 'rotation_addr': 802, 'size_addr': 803}
GRAVITY_LOCK = {'gravity_counter_addr': 786, 'lock_counter_addr': 775, 'speed_index_addr': 800, 'speed_setting_addr': 779}
TIMERS = {'frame_counter_addr': 67, 'wait_frames_addr': 81, 'music_frames_since_last_beat_addr': 1663}
LEVEL = {'addr': 790}
