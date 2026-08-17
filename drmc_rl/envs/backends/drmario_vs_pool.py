from __future__ import annotations

"""ctypes wrapper for the in-process Dr. Mario 2-player VS pool backend.

This module loads ``vendor/drmario_native/build/libdrmario_pool.{dylib,so}`` and exposes
a thin, allocation-minimizing interface around the ``drm_vspool_*`` C ABI
(see ``vendor/drmario_native/drmario_pool_capi.h``).

The pool owns N *pairs* of engine instances running NES nbPlayers==2 rules on
a shared frame clock. Side indexing is flattened: side index = pair*2 + side,
side 0 = P1, side 1 = P2 (2N sides total). Unlike the 1P pool, observations
are not built natively — Python builds them from ``board_bytes``.
"""

import ctypes as C
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from drmc_rl.envs.backends.drmario_pool import (
    GRID_H,
    GRID_W,
    MACRO_ACTIONS,
    PLAN_STATE_DTYPE,
    DrMarioPoolError,
    PlanSolver,
    _load_cdll,
    resolve_library_path,
)

DRMARIO_VSPOOL_PROTOCOL_VERSION = 2

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
        ("defer_planning", C.c_uint8),
        ("_pad0", C.c_uint8 * 3),
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
        # Mid-game checkpoint reset (per side; see drmario_pool_capi.h).
        ("checkpoint_enabled", C.c_uint8),
        ("_pad1", C.c_uint8 * 3),
        ("checkpoint_board", (C.c_uint8 * 128) * 2),
        ("checkpoint_falling_colors", (C.c_uint8 * 2) * 2),
        ("checkpoint_preview_colors", (C.c_uint8 * 2) * 2),
        ("checkpoint_pill_counter", C.c_uint8 * 2),
        ("checkpoint_pill_counter_total", C.c_uint16 * 2),
        ("checkpoint_speed_ups", C.c_uint8 * 2),
        ("_pad2", C.c_uint8 * 2),
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
        # deferred planning
        ("plan_needed", C.POINTER(C.c_uint8)),
        ("plan_state", C.POINTER(C.c_uint8)),  # [2N] packed DrmVsPlanState (28 B)
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

    plan_needed: np.ndarray  # (2N,) uint8 (deferred planning; always 0 otherwise)
    plan_state: np.ndarray  # (2N,) PLAN_STATE_DTYPE


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
        plan_solver: Optional[PlanSolver] = None,
    ) -> None:
        self.num_pairs = int(max(1, int(num_pairs)))
        self.num_sides = 2 * self.num_pairs
        self._plan_solver = plan_solver

        path = resolve_library_path(lib_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DrMario pool library not found at {path}. "
                "Build it with: python -m tools.build_drmario_pool (or `make -C vendor/drmario_native libdrmario_pool`)."
            )

        self._lib = _load_cdll(path)

        create = getattr(self._lib, "drm_vspool_create", None)
        destroy = getattr(self._lib, "drm_vspool_destroy", None)
        reset = getattr(self._lib, "drm_vspool_reset", None)
        step = getattr(self._lib, "drm_vspool_step", None)
        step_strict = getattr(self._lib, "drm_vspool_step_strict", None)
        step_search = getattr(self._lib, "drm_vspool_step_search", None)
        snapshot = getattr(self._lib, "drm_vspool_snapshot", None)
        restore = getattr(self._lib, "drm_vspool_restore", None)
        reveal_info = getattr(self._lib, "drm_vspool_search_reveal_info", None)
        search_reveal = getattr(self._lib, "drm_vspool_search_reveal", None)
        inject = getattr(self._lib, "drm_vspool_inject_plans", None)
        if create is None or destroy is None or reset is None or step is None:
            raise DrMarioPoolError(f"{path} does not export the required drm_vspool_* symbols")
        if plan_solver is not None and inject is None:
            raise DrMarioPoolError(
                f"{path} predates deferred planning (no drm_vspool_inject_plans); "
                "rebuild it with: make -C vendor/drmario_native libdrmario_pool"
            )

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
        if step_strict is not None:
            step_strict.argtypes = step.argtypes
            step_strict.restype = C.c_int
        if step_search is not None:
            step_search.argtypes = [
                C.c_void_p,
                C.POINTER(C.c_int32),
                C.POINTER(_DrmVsPoolOutputs),
            ]
            step_search.restype = C.c_int
        if snapshot is not None:
            snapshot.argtypes = [
                C.c_void_p,
                C.c_uint32,
                C.POINTER(C.c_uint8),
                C.c_size_t,
                C.POINTER(C.c_size_t),
            ]
            snapshot.restype = C.c_int
        if restore is not None:
            restore.argtypes = [
                C.c_void_p,
                C.c_uint32,
                C.POINTER(C.c_uint8),
                C.c_size_t,
            ]
            restore.restype = C.c_int
        if reveal_info is not None:
            reveal_info.argtypes = [
                C.c_void_p,
                C.c_uint32,
                C.POINTER(C.c_uint8),
                C.POINTER(C.c_uint8),
            ]
            reveal_info.restype = C.c_int
        if search_reveal is not None:
            search_reveal.argtypes = [
                C.c_void_p,
                C.c_uint32,
                C.c_uint8,
                C.c_uint8,
                C.c_uint8,
                C.POINTER(_DrmVsPoolOutputs),
            ]
            search_reveal.restype = C.c_int
        if inject is not None:
            inject.argtypes = [
                C.c_void_p,
                C.POINTER(C.c_uint8),
                C.POINTER(C.c_uint16),
                C.POINTER(_DrmVsPoolOutputs),
            ]
            inject.restype = C.c_int

        self._destroy_fn = destroy
        self._reset_fn = reset
        self._step_fn = step
        self._step_strict_fn = step_strict
        self._step_search_fn = step_search
        self._snapshot_fn = snapshot
        self._restore_fn = restore
        self._reveal_info_fn = reveal_info
        self._search_reveal_fn = search_reveal
        self._inject_fn = inject

        cfg = _DrmVsPoolConfig()
        cfg.protocol_version = DRMARIO_VSPOOL_PROTOCOL_VERSION
        cfg.struct_size = C.sizeof(_DrmVsPoolConfig)
        cfg.num_pairs = self.num_pairs
        cfg.max_lock_frames = int(max(1, int(max_lock_frames)))
        cfg.max_wait_frames = int(max(1, int(max_wait_frames)))
        cfg.defer_planning = 1 if plan_solver is not None else 0

        handle = create(C.byref(cfg))
        if not handle:
            raise DrMarioPoolError("drm_vspool_create failed (null handle)")
        self._handle = C.c_void_p(handle)

        self.volley_capacity = int(max(1, int(volley_capacity)))
        self._volley_buf = (_DrmVsVolley * self.volley_capacity)()
        self._volley_count = C.c_uint32(0)

        self.buffers = self._allocate_buffers()
        self._out = self._build_outputs_struct(self.buffers)

        self.max_lock_frames = int(max(1, int(max_lock_frames)))
        self._inject_costs_buf = np.full((self.num_sides, MACRO_ACTIONS), 0xFFFF, dtype=np.uint16)
        self._inject_mask_buf = np.zeros((self.num_sides,), dtype=np.uint8)

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
            plan_needed=np.zeros((S,), dtype=np.uint8),
            plan_state=np.zeros((S,), dtype=PLAN_STATE_DTYPE),
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

        out.plan_needed = _ptr(buffers.plan_needed, C.c_uint8)
        out.plan_state = buffers.plan_state.ctypes.data_as(C.POINTER(C.c_uint8))
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
        self._solve_deferred()

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
        self._solve_deferred()

    def step_strict(self, actions: np.ndarray) -> None:
        """Advance exactly to the next causal pair event.

        A parked opponent is never bypassed. This API is for parity and
        offline teachers; rollout environments continue to use :meth:`step`.
        """

        if self._step_strict_fn is None:
            raise DrMarioPoolError(
                "native library predates strict VS stepping; rebuild vendor/drmario_native"
            )
        acts = np.asarray(actions, dtype=np.int32).reshape(self.num_sides)
        rc = int(
            self._step_strict_fn(
                self._handle,
                acts.ctypes.data_as(C.POINTER(C.c_int32)),
                None,
                None,
                C.byref(self._out),
            )
        )
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_step_strict failed with rc={rc}")
        self._solve_deferred()

    def step_search(self, actions: np.ndarray) -> None:
        """Advance to the next decision, terminal, or pre-reveal chance node."""

        if self._step_search_fn is None:
            raise DrMarioPoolError(
                "native library predates reveal-aware search stepping; rebuild vendor/drmario_native"
            )
        acts = np.asarray(actions, dtype=np.int32).reshape(self.num_sides)
        rc = int(
            self._step_search_fn(
                self._handle,
                acts.ctypes.data_as(C.POINTER(C.c_int32)),
                C.byref(self._out),
            )
        )
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_step_search failed with rc={rc}")
        self._solve_deferred()

    def search_reveal_info(self, pair_index: int) -> tuple[int, int] | None:
        """Return ``(side, reserve_index)`` for the next causal reveal, if any."""

        if self._reveal_info_fn is None:
            raise DrMarioPoolError(
                "native library predates reveal-aware search stepping; rebuild vendor/drmario_native"
            )
        side = C.c_uint8(0)
        reserve_index = C.c_uint8(0)
        rc = int(
            self._reveal_info_fn(
                self._handle,
                int(pair_index),
                C.byref(side),
                C.byref(reserve_index),
            )
        )
        if rc < 0:
            raise DrMarioPoolError(f"drm_vspool_search_reveal_info failed with rc={rc}")
        return None if rc == 0 else (int(side.value), int(reserve_index.value))

    def search_reveal(
        self, pair_index: int, side: int, colors_raw: tuple[int, int]
    ) -> None:
        """Choose one ordered raw-NES preview pill at a pending chance node."""

        if self._search_reveal_fn is None:
            raise DrMarioPoolError(
                "native library predates reveal-aware search stepping; rebuild vendor/drmario_native"
            )
        left, right = (int(colors_raw[0]), int(colors_raw[1]))
        if side not in (0, 1) or left not in (0, 1, 2) or right not in (0, 1, 2):
            raise ValueError("side and reveal colors are out of range")
        rc = int(
            self._search_reveal_fn(
                self._handle,
                int(pair_index),
                int(side),
                left,
                right,
                C.byref(self._out),
            )
        )
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_search_reveal failed with rc={rc}")
        self._solve_deferred()

    def snapshot(self, pair_index: int) -> bytes:
        """Return the canonical, pointer-free native snapshot for one pair."""

        if self._snapshot_fn is None:
            raise DrMarioPoolError(
                "native library predates VS snapshot support; rebuild vendor/drmario_native"
            )
        pair = int(pair_index)
        if not 0 <= pair < self.num_pairs:
            raise IndexError(pair)
        size = C.c_size_t(0)
        rc = int(self._snapshot_fn(self._handle, pair, None, 0, C.byref(size)))
        if rc not in (0, -2) or size.value <= 0:
            raise DrMarioPoolError(f"drm_vspool_snapshot size query failed with rc={rc}")
        buffer = (C.c_uint8 * int(size.value))()
        rc = int(
            self._snapshot_fn(
                self._handle, pair, buffer, len(buffer), C.byref(size)
            )
        )
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_snapshot failed with rc={rc}")
        return bytes(buffer[: int(size.value)])

    def restore(self, pair_index: int, checkpoint: bytes) -> None:
        """Restore one pair from a canonical native snapshot."""

        if self._restore_fn is None:
            raise DrMarioPoolError(
                "native library predates VS restore support; rebuild vendor/drmario_native"
            )
        pair = int(pair_index)
        if not 0 <= pair < self.num_pairs:
            raise IndexError(pair)
        payload = bytes(checkpoint)
        if not payload:
            raise ValueError("checkpoint cannot be empty")
        buffer = (C.c_uint8 * len(payload)).from_buffer_copy(payload)
        rc = int(self._restore_fn(self._handle, pair, buffer, len(payload)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_restore failed with rc={rc}")

    def _solve_deferred(self) -> None:
        """Solve plan_needed sides via the external planner and inject costs.

        Volley outputs are untouched by the inject call, so the event log from
        the preceding reset/step stays valid.
        """

        if self._plan_solver is None:
            return
        buf = self.buffers
        idx = np.flatnonzero(buf.plan_needed)
        if idx.size == 0:
            return

        costs = self._plan_solver(buf.plan_state[idx])
        costs = np.asarray(costs, dtype=np.uint16)
        if costs.shape != (idx.size, MACRO_ACTIONS):
            raise DrMarioPoolError(
                f"plan_solver returned shape {costs.shape}, expected {(idx.size, MACRO_ACTIONS)}"
            )

        full = self._inject_costs_buf
        full[idx] = costs
        mask = self._inject_mask_buf
        mask.fill(0)
        mask[idx] = 1

        rc = int(
            self._inject_fn(
                self._handle,
                mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                full.ctypes.data_as(C.POINTER(C.c_uint16)),
                C.byref(self._out),
            )
        )
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_inject_plans failed with rc={rc}")

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
    checkpoint_enabled: bool = False,
    checkpoint_board: Optional[np.ndarray] = None,  # (2,128) NES tile bytes
    checkpoint_falling_colors: Optional[np.ndarray] = None,  # (2,2) raw NES colors
    checkpoint_preview_colors: Optional[np.ndarray] = None,  # (2,2) raw NES colors
    checkpoint_pill_counter: Tuple[int, int] = (0, 0),
    checkpoint_pill_counter_total: Tuple[int, int] = (0, 0),
    checkpoint_speed_ups: Tuple[int, int] = (0, 0),
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
    spec.checkpoint_enabled = 1 if bool(checkpoint_enabled) else 0
    if checkpoint_enabled:
        board = np.ascontiguousarray(
            np.asarray(checkpoint_board, dtype=np.uint8).reshape(2, 128)
        )
        falling = (
            np.full((2, 2), 0xFF, dtype=np.uint8)
            if checkpoint_falling_colors is None
            else np.asarray(checkpoint_falling_colors, dtype=np.uint8).reshape(2, 2)
        )
        preview = (
            np.full((2, 2), 0xFF, dtype=np.uint8)
            if checkpoint_preview_colors is None
            else np.asarray(checkpoint_preview_colors, dtype=np.uint8).reshape(2, 2)
        )
        for i in range(2):
            C.memmove(spec.checkpoint_board[i], board[i].ctypes.data, 128)
            spec.checkpoint_falling_colors[i][0] = int(falling[i, 0]) & 0xFF
            spec.checkpoint_falling_colors[i][1] = int(falling[i, 1]) & 0xFF
            spec.checkpoint_preview_colors[i][0] = int(preview[i, 0]) & 0xFF
            spec.checkpoint_preview_colors[i][1] = int(preview[i, 1]) & 0xFF
            spec.checkpoint_pill_counter[i] = int(checkpoint_pill_counter[i]) & 0xFF
            spec.checkpoint_pill_counter_total[i] = (
                int(checkpoint_pill_counter_total[i]) & 0xFFFF
            )
            spec.checkpoint_speed_ups[i] = int(checkpoint_speed_ups[i]) & 0xFF
    return spec


__all__ = [
    "DRMARIO_VSPOOL_PROTOCOL_VERSION",
    "PLAN_STATE_DTYPE",
    "PlanSolver",
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
