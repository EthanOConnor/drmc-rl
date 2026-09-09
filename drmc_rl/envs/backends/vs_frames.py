"""Real controller-frame VS access, with an explicitly public observation ABI."""

from __future__ import annotations

import ctypes as C

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


class PublicFrameEvent(C.Structure):
    _fields_ = [
        ("frame", C.c_uint64), ("salt_frames", C.c_uint16),
        *[(name, C.c_uint8) for name in (
            "kind", "side", "tiles_cleared", "viruses_cleared", "lines_cleared",
            "x", "y_top", "rotation", "garbage_size")],
        ("cols", C.c_uint8 * 4), ("colors", C.c_uint8 * 4), ("outcome", C.c_uint8),
    ]

    def public(self):
        from drmc_rl.game.pair_state import PairEvent, PairEventKind

        kinds = (None, PairEventKind.SPAWN, PairEventKind.LOCK, PairEventKind.CLEAR,
                 PairEventKind.VOLLEY, PairEventKind.TOP_OUT, PairEventKind.STAGE_CLEAR,
                 PairEventKind.TERMINAL)
        if not 1 <= self.kind < len(kinds) or self.side not in (0, 1):
            raise ValueError("invalid native public event")
        payload = {}
        if self.kind in (1, 2):
            payload.update(column=int(self.x), row_top=C.c_int8(self.y_top).value,
                           rotation=int(self.rotation))
        elif self.kind == 3:
            payload.update(tiles_cleared=int(self.tiles_cleared),
                           viruses_cleared=int(self.viruses_cleared),
                           lines_cleared=int(self.lines_cleared))
        elif self.kind == 4:
            payload.update(garbage_size=int(self.garbage_size),
                           columns=list(self.cols[:self.garbage_size]),
                           colors=[(1, 0, 2)[c] for c in self.colors[:self.garbage_size]],
                           salt_frames=int(self.salt_frames), sender=1 - int(self.side))
        elif self.kind == 7:
            payload["outcome"] = {1: 1, 2: -1, 3: 0}[self.outcome]
        return PairEvent(kinds[self.kind], int(self.frame), int(self.side), payload)


class FrameHistory(C.Structure):
    _fields_ = [("spawn_frame", C.c_uint64 * 2), ("viruses_remaining", C.c_uint8 * 2),
               ("count", C.c_uint8), ("events", PublicFrameEvent * 32)]


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
        self._history = None

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
        self._history = None
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
        self._history = None
        return self.states

    def public_state(self, side):
        """A causal full-pair view with events captured inside every native tick."""
        from drmc_rl.game.pair_state import (
            DecisionBoundary, FallingPillView, PublicPairState, VisibleSideState,
        )

        if not 0 <= side < len(self.states):
            raise ValueError("invalid controller side")
        if self._history is None:
            if not hasattr(self.lib, "drm_vspool_frame_history"):
                raise RuntimeError("public-context actors require the native frame-history ABI")
            fn = self.lib.drm_vspool_frame_history
            fn.argtypes = [C.c_void_p, C.POINTER(FrameHistory), C.c_size_t]
            fn.restype = C.c_int
            history = (FrameHistory * self.num_pairs)()
            self._check(fn(self.handle, history, C.sizeof(FrameHistory)))
            self._history = history
        pair, viewer = divmod(side, 2)
        history = self._history[pair]
        states = self.states[2 * pair:2 * pair + 2]
        if states[0].frame != states[1].frame:
            raise ValueError("public controller observations require one console clock")
        frame = int(states[0].frame)
        canonical = (1, 0, 2)
        visible, deciding = [], []
        for i, state in enumerate(states):
            colors = tuple(canonical[c] for c in state.pill)
            age = frame - int(history.spawn_frame[i])
            deciding.append(state.falling and (i == viewer or age == 0))
            phase = ("terminal" if state.terminal else "falling" if state.falling
                     else "clearing" if state.phase == 1 and state.subphase in (5, 6, 7)
                     else "settling" if state.phase == 1 else "spawn" if state.phase in (3, 5, 6)
                     else "resolving")
            visible.append(VisibleSideState(
                board=bytes(state.board), pill=colors,
                preview=tuple(canonical[c] for c in state.preview),
                active=(FallingPillView(state.x, C.c_int8(state.y_top).value,
                                       state.rotation, colors, True, age)
                        if state.falling else None),
                viruses_remaining=int(history.viruses_remaining[i]),
                animation_phase=phase, state_age_frames=0,
            ))
        events = tuple(event.public() for event in history.events[:history.count])
        if any(event.frame_id > frame for event in events) or any(
            a.frame_id > b.frame_id for a, b in zip(events, events[1:])
        ):
            raise ValueError("native public history is not causal and chronological")
        boundary = (DecisionBoundary.TERMINAL if states[0].terminal else
                    DecisionBoundary.BOTH if all(deciding) else
                    DecisionBoundary.P1 if deciding[0] else
                    DecisionBoundary.P2 if deciding[1] else DecisionBoundary.ADVANCE)
        return PublicPairState(
            frame_id=frame, viewer_side=viewer, sides=tuple(visible),
            decision_boundary=boundary, recent_events=events,
            observable_clock_delta_frames=0,
            own_controller_state=self.states[side].semantic(self.states[side ^ 1])["falling"],
        )

    def semantic(self, side, *, public_context=False):
        state = self.states[side].semantic(self.states[side ^ 1])
        # Unlike the warp pool, both boards here are from the same executed tick.
        state["vs/observation_timeline"] = "causal-settled-pair-v1"
        if public_context:
            from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
            state["public_context_schema"] = PUBLIC_CONTEXT_SCHEMA
            state["public_pair_state"] = self.public_state(side)
        return state

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
        self._history = None
        return self.progress
