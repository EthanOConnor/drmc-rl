from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from envs.retro.drmario_env import Action
from envs.specs.ram_to_state import ram_to_state


class IntentAction(enum.IntEnum):
    NOOP = 0
    LEFT1 = 1
    RIGHT1 = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    ROTATE_180 = 5
    DOWN_UNTIL_LOCK = 6
    DOWN1 = 7
    RELEASE_ALL = 8

    @staticmethod
    def size() -> int:
        return len(IntentAction)


@dataclass
class PillState:
    spawn_id: int
    col_raw: int
    row_raw: int
    orient: int
    drop_counter: int
    falling_coords: List[Tuple[int, int]]
    static_mask: np.ndarray  # shape (16,8) bool
    orientation_plane: float
    falling: bool

    @property
    def board_rows(self) -> List[int]:
        return [r for r, _ in self.falling_coords]

    @property
    def board_cols(self) -> List[int]:
        return [c for _, c in self.falling_coords]


def _decode_pill_state(env) -> PillState:
    """Extract pill state using RAM and ram_to_state helper."""
    unwrapped = env.unwrapped
    ram_arr = unwrapped._read_ram_array(refresh=True)
    if ram_arr is None:
        ram_arr = np.zeros(0x800, dtype=np.uint8)
    ram_bytes = ram_arr.tobytes()
    offsets = unwrapped._ram_offsets

    # Raw coordinates
    col_addr = offsets["falling_pill"]["col_addr"]
    row_addr = offsets["falling_pill"]["row_addr"]
    orient_addr = offsets["falling_pill"]["orient_addr"]
    spawn_addr = offsets["pill_counter"]["addr"]
    drop_addr = offsets["gravity_lock"]["gravity_counter_addr"]

    col_raw = int(ram_arr[int(col_addr, 16)]) if col_addr else 0
    row_raw = int(ram_arr[int(row_addr, 16)]) if row_addr else 0
    orient = int(ram_arr[int(orient_addr, 16)]) & 0x03 if orient_addr else 0
    spawn_id = int(ram_arr[int(spawn_addr, 16)]) if spawn_addr else 0
    drop_counter = int(ram_arr[int(drop_addr, 16)]) if drop_addr else 0

    state = ram_to_state(ram_bytes, offsets)
    falling_mask = (state[6:9].sum(axis=0) > 0.5)
    static_mask = (state[0:6].sum(axis=0) > 0.5)

    coords = [(int(r), int(c)) for r, c in zip(*np.where(falling_mask))]
    falling = len(coords) > 0

    return PillState(
        spawn_id=spawn_id,
        col_raw=int(col_raw),
        row_raw=int(row_raw),
        orient=int(orient),
        drop_counter=int(drop_counter),
        falling_coords=coords,
        static_mask=static_mask,
        orientation_plane=float(state[9, 0, 0]),
        falling=falling,
    )


def _board_hit(static_mask: np.ndarray, coords: List[Tuple[int, int]], direction: str) -> bool:
    if not coords:
        return False
    H, W = static_mask.shape
    if direction == "left":
        for r, c in coords:
            if c <= 0:
                return True
            if static_mask[r, c - 1]:
                return True
    elif direction == "right":
        for r, c in coords:
            if c >= W - 1:
                return True
            if static_mask[r, c + 1]:
                return True
    return False


@dataclass
class IntentStatus:
    active: int = -1
    done: bool = False
    failed: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)


class IntentTranslator:
    """Translate sticky intent actions into discrete controller commands."""

    TAP_FRAMES = 2
    TAP_COOLDOWN = 1
    EDGE_GUARD_FRAMES = 1
    LOCK_STILL_FRAMES = 3

    def __init__(self, safe_writes: bool = False):
        self.safe_writes = safe_writes
        self.reset()

    def reset(self) -> None:
        self._active: Optional[Tuple[IntentAction, Dict[str, Any]]] = None
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False
        self._tap_a = 0
        self._tap_b = 0
        self._cool_a = 0
        self._cool_b = 0
        self._edge_guard_left = 0
        self._edge_guard_right = 0
        self._still_frames = 0
        self._prev_coords: Optional[List[Tuple[int, int]]] = None
        self._prev_spawn: Optional[int] = None
        self._stats = {
            "timeouts_move": 0,
            "timeouts_rotate": 0,
            "timeouts_drop": 0,
        }

    @property
    def hold_left(self) -> bool:
        return self._hold_left

    @property
    def hold_right(self) -> bool:
        return self._hold_right

    @property
    def hold_down(self) -> bool:
        return self._hold_down

    def on_reset(self, state: PillState) -> None:
        self.reset()
        self._prev_spawn = state.spawn_id
        self._prev_coords = list(state.falling_coords)

    def set_intent(self, intent: IntentAction) -> None:
        if intent == IntentAction.NOOP:
            if self._active and self._active[0] != IntentAction.NOOP:
                return
            self._active = None
            return
        if intent == IntentAction.RELEASE_ALL:
            self._hold_left = self._hold_right = self._hold_down = False
            self._active = None
            return
        if self._active and self._active[0] == intent:
            return
        self._active = (IntentAction(intent), {})

    def _gravity_timeouts(self, state: PillState) -> Tuple[int, int, int]:
        g = max(1, int(state.drop_counter))
        move_to = int(np.clip(2 * g, 8, 40))
        rot_to = int(np.clip(2 * g, 6, 30))
        drop_to = int(np.clip(10 * g, 20, 200))
        return move_to, rot_to, drop_to

    def _press_a(self) -> None:
        if self._cool_a == 0:
            self._tap_a = self.TAP_FRAMES
            self._cool_a = self.TAP_FRAMES + self.TAP_COOLDOWN

    def _press_b(self) -> None:
        if self._cool_b == 0:
            self._tap_b = self.TAP_FRAMES
            self._cool_b = self.TAP_FRAMES + self.TAP_COOLDOWN

    def _update_spawn(self, state: PillState) -> bool:
        spawn_changed = False
        if self._prev_spawn is None or state.spawn_id != self._prev_spawn:
            self._prev_spawn = state.spawn_id
            spawn_changed = True
            self._active = None
            self._hold_left = self._hold_right = self._hold_down = False
            self._edge_guard_left = self._edge_guard_right = 0
        return spawn_changed

    def _update_stillness(self, state: PillState) -> None:
        if not state.falling:
            self._still_frames += 1
            self._prev_coords = None
            return
        current = sorted(state.falling_coords)
        if self._prev_coords is not None and current == sorted(self._prev_coords):
            self._still_frames += 1
        else:
            self._still_frames = 0
        self._prev_coords = current

    def step(self, state: PillState) -> Tuple[int, bool, bool, bool, IntentStatus]:
        spawn_changed = self._update_spawn(state)
        self._update_stillness(state)

        move_to, rot_to, drop_to = self._gravity_timeouts(state)

        # decrement timers
        self._tap_a = max(0, self._tap_a - 1)
        self._tap_b = max(0, self._tap_b - 1)
        self._cool_a = max(0, self._cool_a - 1)
        self._cool_b = max(0, self._cool_b - 1)
        self._edge_guard_left = max(0, self._edge_guard_left - 1)
        self._edge_guard_right = max(0, self._edge_guard_right - 1)

        status = IntentStatus()
        status.payload["spawn_changed"] = spawn_changed

        if not state.falling and self._active:
            # Pill locked; clear intent automatically.
            status.failed = False
            status.done = True
            status.active = int(self._active[0])
            self._active = None
            self._hold_down = False

        if self._active is not None:
            intent, payload = self._active
            status.active = int(intent)

            if intent == IntentAction.LEFT1 or intent == IntentAction.RIGHT1:
                if payload.get("start") is None or spawn_changed:
                    payload["start"] = list(state.falling_coords)
                    payload["timer"] = 0
                    payload["stable"] = 0
                dir_delta = -1 if intent == IntentAction.LEFT1 else 1
                expected = {(r, c + dir_delta) for r, c in payload["start"]}
                coords = set(state.falling_coords)
                if coords and coords == expected:
                    payload["stable"] += 1
                    if payload["stable"] >= 1:
                        if intent == IntentAction.LEFT1:
                            self._hold_left = False
                        else:
                            self._hold_right = False
                        self._active = None
                        status.done = True
                else:
                    payload["stable"] = 0
                    if intent == IntentAction.LEFT1:
                        if self._edge_guard_left == 0:
                            self._hold_left = True
                            self._hold_right = False
                    else:
                        if self._edge_guard_right == 0:
                            self._hold_right = True
                            self._hold_left = False
                if intent == IntentAction.LEFT1 and _board_hit(state.static_mask, state.falling_coords, "left"):
                    self._edge_guard_left = self.EDGE_GUARD_FRAMES
                    self._hold_left = False
                    self._active = None
                    status.failed = True
                if intent == IntentAction.RIGHT1 and _board_hit(state.static_mask, state.falling_coords, "right"):
                    self._edge_guard_right = self.EDGE_GUARD_FRAMES
                    self._hold_right = False
                    self._active = None
                    status.failed = True
                payload["timer"] += 1
                if payload["timer"] > move_to:
                    self._hold_left = False
                    self._hold_right = False
                    self._active = None
                    status.failed = True
                    self._stats["timeouts_move"] += 1

            elif intent == IntentAction.ROTATE_CW or intent == IntentAction.ROTATE_CCW or intent == IntentAction.ROTATE_180:
                if payload.get("start_orient") is None or spawn_changed:
                    payload["start_orient"] = state.orient
                    payload["timer"] = 0
                    payload["target"] = (state.orient + (1 if intent == IntentAction.ROTATE_CW else 3 if intent == IntentAction.ROTATE_CCW else 2)) % 4
                    payload["nudge_attempted"] = False
                if state.orient == payload["target"]:
                    self._active = None
                    status.done = True
                else:
                    if intent in (IntentAction.ROTATE_CW, IntentAction.ROTATE_180):
                        self._press_a()
                    if intent in (IntentAction.ROTATE_CCW, IntentAction.ROTATE_180):
                        self._press_b()
                    payload["timer"] += 1
                    if payload["timer"] > rot_to:
                        self._active = None
                        status.failed = True
                        self._stats["timeouts_rotate"] += 1

            elif intent == IntentAction.DOWN_UNTIL_LOCK:
                self._hold_down = True
                payload["timer"] = payload.get("timer", 0) + 1
                if not state.falling or self._still_frames >= self.LOCK_STILL_FRAMES:
                    self._hold_down = False
                    self._active = None
                    status.done = True
                elif payload["timer"] > drop_to:
                    self._hold_down = False
                    self._active = None
                    status.failed = True
                    self._stats["timeouts_drop"] += 1

            elif intent == IntentAction.DOWN1:
                if payload.get("start_rows") is None or spawn_changed:
                    payload["start_rows"] = [r for r, _ in state.falling_coords]
                    payload["timer"] = 0
                self._hold_down = True
                rows = [r for r, _ in state.falling_coords]
                if not rows:
                    self._hold_down = False
                    self._active = None
                    status.done = True
                elif rows and payload["start_rows"] and min(rows) > min(payload["start_rows"]):
                    self._hold_down = False
                    self._active = None
                    status.done = True
                payload["timer"] += 1
                if payload["timer"] > move_to:
                    self._hold_down = False
                    self._active = None
                    status.failed = True
                    self._stats["timeouts_move"] += 1

            else:
                self._active = None

        # Compose action index
        action_idx = Action.NOOP
        if self._tap_a > 0 and self._tap_b > 0:
            action_idx = Action.BOTH_ROT
        elif self._tap_a > 0:
            action_idx = Action.ROTATE_A
        elif self._tap_b > 0:
            action_idx = Action.ROTATE_B
        elif self._hold_down and not (self._hold_left or self._hold_right):
            # ensure down is registered even if already latched
            action_idx = Action.DOWN_HOLD if self._hold_down else Action.NOOP
        else:
            action_idx = Action.NOOP

        return (
            int(action_idx),
            bool(self._hold_left and not self._hold_right),
            bool(self._hold_right and not self._hold_left),
            bool(self._hold_down),
            status,
        )


class DrMarioIntentEnv(gym.Wrapper):
    """Wrapper that exposes a sticky intent action space."""

    def __init__(self, env: gym.Env, *, safe_writes: bool = False):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(IntentAction.size())
        self._translator = IntentTranslator(safe_writes=safe_writes)
        self._last_status: Dict[str, Any] = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pill_state = _decode_pill_state(self.env)
        self._translator.on_reset(pill_state)
        self._last_status = {"intent_active": -1, "intent_done": False, "intent_failed": False}
        if info is None:
            info = {}
        info.update(self._intent_info(pill_state, self._last_status))
        return obs, info

    def step(self, action: int):
        intent = IntentAction(int(action))
        self._translator.set_intent(intent)
        pill_state = _decode_pill_state(self.env)
        action_idx, hold_left, hold_right, hold_down, status = self._translator.step(pill_state)

        base = self.env.unwrapped
        base._hold_left = bool(hold_left)
        base._hold_right = bool(hold_right)
        base._hold_down = bool(hold_down)

        obs, reward, terminated, truncated, info = self.env.step(action_idx)

        next_state = _decode_pill_state(self.env)
        info.update(self._intent_info(next_state, {
            "intent_active": status.active,
            "intent_done": status.done,
            "intent_failed": status.failed,
        }))
        self._last_status = {
            "intent_active": status.active,
            "intent_done": status.done,
            "intent_failed": status.failed,
        }
        return obs, reward, terminated, truncated, info

    def _intent_info(self, pill_state: PillState, status: Dict[str, Any]) -> Dict[str, Any]:
        coords = pill_state.falling_coords
        info = {
            "intent/active": status.get("intent_active", -1),
            "intent/done": status.get("intent_done", False),
            "intent/failed": status.get("intent_failed", False),
            "pill/col_raw": pill_state.col_raw,
            "pill/row_raw": pill_state.row_raw,
            "pill/orient": pill_state.orient,
            "pill/spawn_id": pill_state.spawn_id,
            "pill/drop_counter": pill_state.drop_counter,
            "pill/falling": pill_state.falling,
        }
        if coords:
            info["pill/cells"] = tuple(coords)
        return info


__all__ = ["IntentAction", "DrMarioIntentEnv"]
