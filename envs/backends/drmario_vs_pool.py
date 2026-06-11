from __future__ import annotations

"""ctypes wrapper for the in-process Dr. Mario 2-player VS pool backend.

This module loads ``game_engine/build/libdrmario_pool.{dylib,so}`` and exposes
a thin, allocation-minimizing interface around the ``drm_vspool_*`` C ABI
(see ``game_engine/drmario_pool_capi.h``).

The pool owns N *pairs* of engine instances running NES nbPlayers==2 rules on
a shared frame clock. Side indexing is flattened: side index = pair*2 + side,
side 0 = P1, side 1 = P2 (2N sides total). Unlike the 1P pool, observations
are not built natively — Python builds them from ``board_bytes``.
"""

import ctypes as C
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from envs.backends.drmario_pool import (
    GRID_H,
    GRID_W,
    MACRO_ACTIONS,
    DrMarioPoolError,
    _load_cdll,
    resolve_library_path,
)

DRMARIO_VSPOOL_PROTOCOL_VERSION = 1

# Outcome codes (per side).
VS_OUTCOME_NONE = 0
VS_OUTCOME_WIN = 1
VS_OUTCOME_LOSS = 2
VS_OUTCOME_DRAW = 3


class _DrmVsPoolConfig(C.Structure):
    _fields_ = [
        ("protocol_version", C.c_uint32),
        ("struct_size", C.c_uint32),
        ("num_pairs", C.c_uint32),
        ("max_lock_frames", C.c_uint32),
        ("max_wait_frames", C.c_uint32),
    ]


class _DrmVsResetSpec(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("level", C.c_int32 * 2),
        ("speed_setting", C.c_int32 * 2),
        ("rng_state", C.c_uint8 * 2),
        ("rng_override", C.c_uint8),
        ("_pad0", C.c_uint8),
        ("frame_counter_base", C.c_uint32),
        ("attack_colors", C.c_uint8 * 8),
    ]


class _DrmVsVolley(C.Structure):
    _fields_ = [
        ("pair", C.c_uint32),
        ("receiver", C.c_uint8),
        ("size", C.c_uint8),
        ("cols", C.c_uint8 * 4),
        ("colors", C.c_uint8 * 4),
        ("_pad", C.c_uint16),
        ("frame", C.c_uint32),
    ]


class _DrmVsPoolOutputs(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        # per-side decision outputs [2N, ...]
        ("feasible_mask", C.POINTER(C.c_uint8)),
        ("cost_to_lock", C.POINTER(C.c_uint16)),
        ("pill_colors", C.POINTER(C.c_uint8)),
        ("preview_colors", C.POINTER(C.c_uint8)),
        ("spawn_id", C.POINTER(C.c_uint8)),
        ("viruses_rem", C.POINTER(C.c_uint16)),
        ("board_bytes", C.POINTER(C.c_uint8)),
        # per-side step outputs
        ("tau_frames", C.POINTER(C.c_uint32)),
        ("side_frames", C.POINTER(C.c_uint64)),
        ("need_action", C.POINTER(C.c_uint8)),
        ("garbage_pending", C.POINTER(C.c_uint8)),
        ("garbage_sent_total", C.POINTER(C.c_uint32)),
        ("outcome", C.POINTER(C.c_uint8)),
        ("invalid_action", C.POINTER(C.c_int32)),
        # per-pair outputs [N]
        ("terminated", C.POINTER(C.c_uint8)),
        ("truncated", C.POINTER(C.c_uint8)),
        # volley event log
        ("volleys", C.POINTER(_DrmVsVolley)),
        ("volley_capacity", C.c_uint32),
        ("volley_count", C.POINTER(C.c_uint32)),
    ]


@dataclass(frozen=True)
class VsVolley:
    """One garbage release event (checkReleaseAttack on the receiver's board)."""

    pair: int
    receiver: int  # side (0/1) whose top row received the garbage
    size: int  # 2..4 half pills
    cols: Tuple[int, ...]
    colors: Tuple[int, ...]  # raw NES colors (0=Y,1=R,2=B)
    frame: int  # pair-clock frame of the release


@dataclass(slots=True)
class VsPoolBuffers:
    """Persistent numpy arrays owned by :class:`DrMarioVsPoolRunner`.

    All per-side arrays have leading dim 2N (N pairs); per-pair arrays have N.
    """

    feasible_mask: np.ndarray  # (2N,512) uint8
    cost_to_lock: np.ndarray  # (2N,512) uint16
    pill_colors: np.ndarray  # (2N,2) uint8 (canonical 0=R,1=Y,2=B)
    preview_colors: np.ndarray  # (2N,2) uint8 (canonical)
    spawn_id: np.ndarray  # (2N,) uint8
    viruses_rem: np.ndarray  # (2N,) uint16
    board_bytes: np.ndarray  # (2N,128) uint8 (NES tile encoding)

    tau_frames: np.ndarray  # (2N,) uint32
    side_frames: np.ndarray  # (2N,) uint64
    need_action: np.ndarray  # (2N,) uint8
    garbage_pending: np.ndarray  # (2N,) uint8
    garbage_sent_total: np.ndarray  # (2N,) uint32
    outcome: np.ndarray  # (2N,) uint8
    invalid_action: np.ndarray  # (2N,) int32

    terminated: np.ndarray  # (N,) uint8
    truncated: np.ndarray  # (N,) uint8


class DrMarioVsPoolRunner:
    """Owns the native VS pool handle and numpy output buffers."""

    def __init__(
        self,
        *,
        num_pairs: int,
        max_lock_frames: int = 2048,
        max_wait_frames: int = 6000,
        volley_capacity: int = 256,
        lib_path: Optional[str] = None,
    ) -> None:
        self.num_pairs = int(max(1, int(num_pairs)))
        self.num_sides = 2 * self.num_pairs

        path = resolve_library_path(lib_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DrMario pool library not found at {path}. "
                "Build it with: python -m tools.build_drmario_pool (or `make -C game_engine libdrmario_pool`)."
            )

        self._lib = _load_cdll(path)

        create = getattr(self._lib, "drm_vspool_create", None)
        destroy = getattr(self._lib, "drm_vspool_destroy", None)
        reset = getattr(self._lib, "drm_vspool_reset", None)
        step = getattr(self._lib, "drm_vspool_step", None)
        if create is None or destroy is None or reset is None or step is None:
            raise DrMarioPoolError(f"{path} does not export the required drm_vspool_* symbols")

        create.argtypes = [C.POINTER(_DrmVsPoolConfig)]
        create.restype = C.c_void_p
        destroy.argtypes = [C.c_void_p]
        destroy.restype = None
        reset.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint8),
            C.POINTER(_DrmVsResetSpec),
            C.POINTER(_DrmVsPoolOutputs),
        ]
        reset.restype = C.c_int
        step.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_uint8),
            C.POINTER(_DrmVsResetSpec),
            C.POINTER(_DrmVsPoolOutputs),
        ]
        step.restype = C.c_int

        self._destroy_fn = destroy
        self._reset_fn = reset
        self._step_fn = step

        cfg = _DrmVsPoolConfig()
        cfg.protocol_version = DRMARIO_VSPOOL_PROTOCOL_VERSION
        cfg.struct_size = C.sizeof(_DrmVsPoolConfig)
        cfg.num_pairs = self.num_pairs
        cfg.max_lock_frames = int(max(1, int(max_lock_frames)))
        cfg.max_wait_frames = int(max(1, int(max_wait_frames)))

        handle = create(C.byref(cfg))
        if not handle:
            raise DrMarioPoolError("drm_vspool_create failed (null handle)")
        self._handle = C.c_void_p(handle)

        self.volley_capacity = int(max(1, int(volley_capacity)))
        self._volley_buf = (_DrmVsVolley * self.volley_capacity)()
        self._volley_count = C.c_uint32(0)

        self.buffers = self._allocate_buffers()
        self._out = self._build_outputs_struct(self.buffers)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        try:
            self._destroy_fn(handle)
        except Exception:
            pass
        self._handle = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _allocate_buffers(self) -> VsPoolBuffers:
        S = self.num_sides
        N = self.num_pairs
        return VsPoolBuffers(
            feasible_mask=np.zeros((S, MACRO_ACTIONS), dtype=np.uint8),
            cost_to_lock=np.full((S, MACRO_ACTIONS), 0xFFFF, dtype=np.uint16),
            pill_colors=np.zeros((S, 2), dtype=np.uint8),
            preview_colors=np.zeros((S, 2), dtype=np.uint8),
            spawn_id=np.zeros((S,), dtype=np.uint8),
            viruses_rem=np.zeros((S,), dtype=np.uint16),
            board_bytes=np.zeros((S, 128), dtype=np.uint8),
            tau_frames=np.zeros((S,), dtype=np.uint32),
            side_frames=np.zeros((S,), dtype=np.uint64),
            need_action=np.zeros((S,), dtype=np.uint8),
            garbage_pending=np.zeros((S,), dtype=np.uint8),
            garbage_sent_total=np.zeros((S,), dtype=np.uint32),
            outcome=np.zeros((S,), dtype=np.uint8),
            invalid_action=np.full((S,), -1, dtype=np.int32),
            terminated=np.zeros((N,), dtype=np.uint8),
            truncated=np.zeros((N,), dtype=np.uint8),
        )

    def _build_outputs_struct(self, buffers: VsPoolBuffers) -> _DrmVsPoolOutputs:
        out = _DrmVsPoolOutputs()
        out.struct_size = C.sizeof(_DrmVsPoolOutputs)

        def _ptr(arr: np.ndarray, c_type: object) -> C._Pointer:  # type: ignore[name-defined]
            return arr.ctypes.data_as(C.POINTER(c_type))  # type: ignore[arg-type]

        out.feasible_mask = _ptr(buffers.feasible_mask, C.c_uint8)
        out.cost_to_lock = _ptr(buffers.cost_to_lock, C.c_uint16)
        out.pill_colors = _ptr(buffers.pill_colors, C.c_uint8)
        out.preview_colors = _ptr(buffers.preview_colors, C.c_uint8)
        out.spawn_id = _ptr(buffers.spawn_id, C.c_uint8)
        out.viruses_rem = _ptr(buffers.viruses_rem, C.c_uint16)
        out.board_bytes = _ptr(buffers.board_bytes, C.c_uint8)

        out.tau_frames = _ptr(buffers.tau_frames, C.c_uint32)
        out.side_frames = _ptr(buffers.side_frames, C.c_uint64)
        out.need_action = _ptr(buffers.need_action, C.c_uint8)
        out.garbage_pending = _ptr(buffers.garbage_pending, C.c_uint8)
        out.garbage_sent_total = _ptr(buffers.garbage_sent_total, C.c_uint32)
        out.outcome = _ptr(buffers.outcome, C.c_uint8)
        out.invalid_action = _ptr(buffers.invalid_action, C.c_int32)

        out.terminated = _ptr(buffers.terminated, C.c_uint8)
        out.truncated = _ptr(buffers.truncated, C.c_uint8)

        out.volleys = self._volley_buf
        out.volley_capacity = self.volley_capacity
        out.volley_count = C.pointer(self._volley_count)
        return out

    # ------------------------------------------------------------------ calls
    def reset(self, reset_mask: Optional[np.ndarray], reset_specs: Optional[object]) -> None:
        """Reset selected pairs (mask is per pair) and populate decision outputs."""

        mask_ptr = None
        mask_u8: Optional[np.ndarray] = None
        if reset_mask is not None:
            mask_u8 = np.asarray(reset_mask, dtype=np.uint8).reshape(self.num_pairs)
            mask_ptr = mask_u8.ctypes.data_as(C.POINTER(C.c_uint8))

        specs_ptr = None
        specs_arr: Optional[object] = None
        if reset_specs is not None:
            specs_arr = _build_vs_reset_spec_array(reset_specs, self.num_pairs)
            specs_ptr = C.cast(specs_arr, C.POINTER(_DrmVsResetSpec))

        rc = int(self._reset_fn(self._handle, mask_ptr, specs_ptr, C.byref(self._out)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_reset failed with rc={rc}")
        _ = specs_arr, mask_u8  # keep alive until after call

    def step(
        self,
        actions: np.ndarray,
        reset_mask: Optional[np.ndarray],
        reset_specs: Optional[object],
    ) -> None:
        """Step all pairs at decision boundaries.

        ``actions`` is [2N]: macro action (0..511) for each side parked at a
        decision (need_action==1); -1 = noop-fall (used when the feasible mask
        is empty); other negatives leave the side parked. Actions for sides
        that are not parked are ignored natively.
        """

        acts = np.asarray(actions, dtype=np.int32).reshape(self.num_sides)
        acts_ptr = acts.ctypes.data_as(C.POINTER(C.c_int32))

        mask_ptr = None
        mask_u8: Optional[np.ndarray] = None
        if reset_mask is not None:
            mask_u8 = np.asarray(reset_mask, dtype=np.uint8).reshape(self.num_pairs)
            mask_ptr = mask_u8.ctypes.data_as(C.POINTER(C.c_uint8))

        specs_ptr = None
        specs_arr: Optional[object] = None
        if reset_specs is not None:
            specs_arr = _build_vs_reset_spec_array(reset_specs, self.num_pairs)
            specs_ptr = C.cast(specs_arr, C.POINTER(_DrmVsResetSpec))

        rc = int(self._step_fn(self._handle, acts_ptr, mask_ptr, specs_ptr, C.byref(self._out)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_step failed with rc={rc}")
        _ = specs_arr, mask_u8, acts  # keep alive until after call

    def volleys(self) -> List[VsVolley]:
        """Volley events recorded during the last reset/step call."""

        n = int(min(int(self._volley_count.value), self.volley_capacity))
        out: List[VsVolley] = []
        for i in range(n):
            v = self._volley_buf[i]
            size = int(v.size)
            out.append(
                VsVolley(
                    pair=int(v.pair),
                    receiver=int(v.receiver),
                    size=size,
                    cols=tuple(int(v.cols[j]) for j in range(min(size, 4))),
                    colors=tuple(int(v.colors[j]) for j in range(min(size, 4))),
                    frame=int(v.frame),
                )
            )
        return out


def _build_vs_reset_spec_array(reset_specs: object, num_pairs: int) -> object:
    """Return a contiguous `_DrmVsResetSpec[num_pairs]` ctypes array."""

    specs_list = list(reset_specs)  # type: ignore[arg-type]
    if len(specs_list) != int(num_pairs):
        raise ValueError(f"Expected {num_pairs} reset specs, got {len(specs_list)}")

    arr_type = _DrmVsResetSpec * int(num_pairs)
    arr = arr_type()
    for i, spec in enumerate(specs_list):
        if isinstance(spec, _DrmVsResetSpec):
            arr[i] = spec
        elif isinstance(spec, dict):
            arr[i] = build_vs_reset_spec(**spec)
        else:
            raise TypeError(f"Unsupported reset spec type: {type(spec)!r}")
    return arr


def build_vs_reset_spec(
    *,
    level: Tuple[int, int] = (0, 0),
    speed_setting: Tuple[int, int] = (2, 2),
    rng_state: Tuple[int, int] = (0, 0),
    rng_override: bool = False,
    frame_counter_base: int = 0,
    attack_colors: Optional[Tuple[int, ...]] = None,
) -> _DrmVsResetSpec:
    spec = _DrmVsResetSpec()
    spec.struct_size = C.sizeof(_DrmVsResetSpec)
    spec.level[0] = int(level[0])
    spec.level[1] = int(level[1])
    spec.speed_setting[0] = int(speed_setting[0])
    spec.speed_setting[1] = int(speed_setting[1])
    spec.rng_state[0] = int(rng_state[0]) & 0xFF
    spec.rng_state[1] = int(rng_state[1]) & 0xFF
    spec.rng_override = 1 if bool(rng_override) else 0
    spec.frame_counter_base = int(frame_counter_base) & 0xFFFFFFFF
    colors = tuple(attack_colors) if attack_colors is not None else (0xFF,) * 8
    if len(colors) != 8:
        raise ValueError(f"attack_colors must have 8 entries, got {len(colors)}")
    for j, value in enumerate(colors):
        spec.attack_colors[j] = int(value) & 0xFF
    return spec


__all__ = [
    "DRMARIO_VSPOOL_PROTOCOL_VERSION",
    "VS_OUTCOME_NONE",
    "VS_OUTCOME_WIN",
    "VS_OUTCOME_LOSS",
    "VS_OUTCOME_DRAW",
    "DrMarioVsPoolRunner",
    "VsPoolBuffers",
    "VsVolley",
    "build_vs_reset_spec",
    "GRID_H",
    "GRID_W",
    "MACRO_ACTIONS",
]
