"""Versioned trainer motor limits, applied inside exact reachability.

These are product presets, not certified human-percentile envelopes. The
Top Humans name describes the requested setting; corpus calibration remains
separate. Every slower profile is a strict subset of the faster operation set.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from drmc_rl.planning.fast_reach import FrameState, simulate_frame


@dataclass(frozen=True)
class Pace:
    id: str
    label: str
    reaction_frames: int
    edge_interval: int
    motion_interval: int
    max_buttons: int

    def planner_args(self, execution_delay: int = 0) -> dict[str, int]:
        return dict(reaction_frames=max(0, self.reaction_frames - execution_delay),
                    edge_interval=self.edge_interval, motion_interval=self.motion_interval,
                    max_buttons=self.max_buttons)

    def to_dict(self) -> dict:
        return {"schema": "drmc-motor-pace-v1", **asdict(self), "human_calibrated": False}

    def validate(self, columns: np.ndarray, spawn: FrameState, script,
                 *, speed_threshold: int, execution_delay: int = 0) -> dict[str, int]:
        """Independent per-frame physics and motor audit; never repairs a script."""
        previous = spawn.hold_dir.value * 6 + spawn.rot_hold.value
        last_edge = last_motion = -10000
        edge_gaps, motion_gaps = [], []
        reaction = max(0, self.reaction_frames - execution_delay)
        state = spawn
        for index, value in enumerate(script):
            action = int(value)
            if state.locked or not 0 <= action < 18:
                raise ValueError("invalid paced script length or action")
            buttons = int(action // 6 != 0) + int(action % 6 >= 3) + int(action % 3 != 0)
            if buttons > self.max_buttons or (index < reaction and action != 0):
                raise ValueError("paced script violates reaction or button overlap")
            if action != previous:
                if index - last_edge < self.edge_interval:
                    raise ValueError("paced script violates button-change interval")
                if last_edge >= 0:
                    edge_gaps.append(index - last_edge)
                last_edge = index
            next_state = simulate_frame(columns, state, action, speed_threshold=speed_threshold)
            if (next_state.x, next_state.rot) != (state.x, state.rot):
                if index - last_motion < self.motion_interval:
                    raise ValueError("paced script violates steering interval")
                if last_motion >= 0:
                    motion_gaps.append(index - last_motion)
                last_motion = index
            state, previous = next_state, action
        if not state.locked:
            raise ValueError("paced script does not lock")
        return {"min_edge_interval": min(edge_gaps, default=0),
                "min_motion_interval": min(motion_gaps, default=0),
                "x": state.x, "y": state.y, "rotation": state.rot}


PACES = (
    Pace("sloth", "Sloth", 60, 12, 24, 1),
    Pace("relaxed", "Relaxed", 36, 7, 14, 1),
    Pace("normal", "Normal", 22, 4, 8, 1),
    Pace("fast", "Fast", 12, 3, 5, 2),
    Pace("top_humans", "Top Humans", 6, 2, 3, 2),
    Pace("super_human", "Super Human", 2, 1, 2, 3),
    Pace("frame_perfect", "Frame Perfect", 0, 0, 0, 3),
)
BY_ID = {pace.id: pace for pace in PACES}


def resolve_pace(name: str | None = None, timing_scale: float = 1.0) -> Pace:
    if name is not None:
        if name not in BY_ID:
            raise ValueError(f"unknown execution pace {name!r}")
        return BY_ID[name]
    scale = float(timing_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("timing_scale must be finite and non-negative")
    # Migration for clients that predate named mechanical profiles.
    return BY_ID["frame_perfect" if scale < 0.25 else "fast" if scale < 0.75
                 else "normal" if scale < 1.25 else "relaxed" if scale < 1.75 else "sloth"]
