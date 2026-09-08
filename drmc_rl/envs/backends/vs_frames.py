"""Real controller-frame VS access, with an explicitly public observation ABI."""

from __future__ import annotations

import ctypes as C

import numpy as np

from drmc_rl.envs.backends.drmario_pool import _load_cdll, resolve_library_path
from drmc_rl.envs.backends.drmario_vs_pool import (
    _DrmVsPoolConfig, _DrmVsResetSpec, build_vs_reset_spec,
)
from drmc_rl.game.observation import board_bytes_to_semantic_planes


class FrameState(C.Structure):
    _fields_ = [
        ("frame", C.c_uint64), ("garbage_sent_total", C.c_uint32),
        ("pill_counter_total", C.c_uint16), ("board", C.c_uint8 * 128),
        ("pill", C.c_uint8 * 2), ("preview", C.c_uint8 * 2),
        *[(name, C.c_uint8) for name in (
            "mode", "phase", "subphase", "spawn_id", "level", "speed", "speed_ups",
            "x", "y_top", "rotation", "speed_counter", "horizontal_velocity",
            "held_buttons", "frame_parity", "terminal", "outcome", "event_type",
        )],
    ]

    @property
    def falling(self):
        return self.mode == 4 and self.phase == 0 and not self.terminal

    def copy(self):
        return FrameState.from_buffer_copy(self)

    def semantic(self, opponent):
        raw_to_canon = (1, 0, 2)
        held = self.held_buttons
        return {
            "board_planes": board_bytes_to_semantic_planes(bytes(self.board)),
            "opponent_board_planes": board_bytes_to_semantic_planes(bytes(opponent.board)),
            "pill": [raw_to_canon[c & 3] for c in self.pill],
            "preview": [raw_to_canon[c & 3] for c in self.preview],
            "opponent_pill": [raw_to_canon[c & 3] for c in opponent.pill],
            "level": self.level, "speed": self.speed, "speed_ups": self.speed_ups,
            "pill_counter_total": self.pill_counter_total,
            "falling": {
                "x": self.x, "y": self.y_top, "rotation": self.rotation,
                "speed_counter": self.speed_counter,
                "horizontal_velocity": self.horizontal_velocity,
                "frame_parity": self.frame_parity,
                "hold_dir": 1 if held & 2 else 2 if held & 1 else 0,
                "rotation_hold": 1 if held & 128 else 2 if held & 64 else 0,
            },
        }


class FrameVsPool:
    def __init__(self, num_pairs=1, *, lib_path=None):
        self.num_pairs = int(num_pairs)
        if self.num_pairs < 1:
            raise ValueError("num_pairs must be positive")
        self.lib = _load_cdll(resolve_library_path(lib_path))
        cfg = _DrmVsPoolConfig(2, C.sizeof(_DrmVsPoolConfig), self.num_pairs, 2048, 6000, 1)
        self.lib.drm_vspool_create.argtypes = [C.POINTER(_DrmVsPoolConfig)]
        self.lib.drm_vspool_create.restype = C.c_void_p
        self.lib.drm_vspool_destroy.argtypes = [C.c_void_p]
        self.lib.drm_vspool_destroy.restype = None
        self.lib.drm_vspool_frame_reset.argtypes = [C.c_void_p, C.POINTER(C.c_uint8),
            C.POINTER(_DrmVsResetSpec), C.POINTER(FrameState), C.c_size_t]
        self.lib.drm_vspool_frame_step.argtypes = [C.c_void_p, C.POINTER(C.c_uint8),
            C.c_uint32, C.POINTER(FrameState), C.c_size_t]
        self.handle = self.lib.drm_vspool_create(C.byref(cfg))
        if not self.handle:
            raise RuntimeError("frame VS pool creation failed")
        self.states = (FrameState * (2 * self.num_pairs))()
        self.buttons = (C.c_uint8 * (2 * self.num_pairs))()

    def reset(self, seeds, *, level=14, speed=2, mask=None):
        if len(seeds) != self.num_pairs:
            raise ValueError("one seed per pair required")
        specs = (_DrmVsResetSpec * self.num_pairs)(*[
            build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
                                rng_override=True, rng_state=(int(seed) & 255, (int(seed) >> 8) & 255))
            for seed in seeds])
        cmask = None if mask is None else (C.c_uint8 * self.num_pairs)(*mask)
        self._check(self.lib.drm_vspool_frame_reset(self.handle, cmask, specs,
                                                  self.states, C.sizeof(FrameState)))
        return self.states

    def step(self, buttons=None, count=1):
        if buttons is not None:
            if len(buttons) != len(self.buttons) or any(not 0 <= int(b) <= 255 for b in buttons):
                raise ValueError("one NES controller byte per side required")
            self.buttons[:] = buttons
        else:
            self.buttons[:] = [0] * len(self.buttons)
        self._check(self.lib.drm_vspool_frame_step(self.handle, self.buttons, count,
                                                 self.states, C.sizeof(FrameState)))
        return self.states

    @staticmethod
    def _check(rc):
        if rc:
            raise RuntimeError(f"controller-frame VS call failed: {rc}")

    def close(self):
        if self.handle:
            self.lib.drm_vspool_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ControllerFrame(C.Structure):
    _fields_ = [(name, C.c_uint8) for name in (
        "buttons", "x", "y_top", "rotation", "speed_counter", "horizontal_velocity",
        "hold_dir", "rotation_hold", "frame_parity")]


class FrameScript(C.Structure):
    _fields_ = [("start_frame", C.c_uint64), ("frames", C.POINTER(ControllerFrame)),
               ("length", C.c_uint32), ("pill_counter_total", C.c_uint16),
               ("spawn_id", C.c_uint8), ("accepted", C.c_uint8)]


class FrameAdvance(C.Structure):
    _fields_ = [("validated_input_frames", C.c_uint32), ("locks", C.c_uint32),
               ("unplanned_locks", C.c_uint32), ("needs_action", C.c_uint8)]


class EventVsPool(FrameVsPool):
    """Park independent pairs at decisions, executing every input frame in C++."""
    def __init__(self, num_pairs=1, *, lib_path=None):
        super().__init__(num_pairs, lib_path=lib_path)
        self.advance_fn = self.lib.drm_vspool_frame_advance
        self.advance_fn.argtypes = [C.c_void_p, C.POINTER(FrameScript), C.c_uint64,
            C.POINTER(FrameState), C.c_size_t, C.POINTER(FrameAdvance)]
        self.advance_fn.restype = C.c_int
        self.scripts = (FrameScript * (2*self.num_pairs))()
        self.progress = (FrameAdvance * (2*self.num_pairs))()
        self.script_storage = [None] * (2*self.num_pairs)

    def reset(self, seeds, *, level=14, speed=2, mask=None):
        states = super().reset(seeds,level=level,speed=speed,mask=mask)
        for pair in range(self.num_pairs):
            if mask is None or mask[pair]:
                for side in (2*pair,2*pair+1):
                    self.scripts[side] = FrameScript()
                    self.script_storage[side] = None
        return states

    def install(self, side, move=None, *, delay=0):
        state = self.states[side]
        script = self.scripts[side]
        script.start_frame = state.frame + delay
        script.spawn_id, script.pill_counter_total, script.accepted = state.spawn_id, state.pill_counter_total, 1
        frames = [] if move is None else move["controller_states"]
        if move is not None and len(move["controller_frames"]) != len(frames):
            raise ValueError("one expected microstate per controller frame required")
        storage = (ControllerFrame * len(frames))(*[
            ControllerFrame(buttons, f["x"], f["y"], f["rotation"], f["speed_counter"],
                f["horizontal_velocity"], f["hold_dir"], f["rotation_hold"], f["frame_parity"])
            for buttons, f in zip([] if move is None else move["controller_frames"], frames)])
        self.script_storage[side] = storage
        script.frames, script.length = storage, len(frames)

    def advance(self, frame_limit):
        self._check(self.advance_fn(self.handle, self.scripts, frame_limit,
                                   self.states, C.sizeof(FrameState), self.progress))
        return self.progress
