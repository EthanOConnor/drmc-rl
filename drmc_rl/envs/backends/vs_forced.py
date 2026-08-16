"""Strict forced-lock access for parity and timing-action experiments.

The native C ABI already exposes ``drm_vspool_step_forced``.  This small driver
keeps the experimental path separate from the hot policy runner while sharing
its handle and output buffers.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from drmc_rl.envs.backends.drmario_pool import DrMarioPoolError
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, _DrmVsPoolOutputs


class _DrmVsForcedLock(C.Structure):
    _fields_ = [
        ("col", C.c_int32),
        ("row", C.c_int32),
        ("rot", C.c_int32),
        ("_pad", C.c_int32),
        ("lock_frame", C.c_int64),
    ]


@dataclass(frozen=True, slots=True)
class ForcedLock:
    column: int = 0
    row_bottom: int = 0
    rotation: int = 0
    lock_frame: int = -1

    @classmethod
    def spectator(cls) -> "ForcedLock":
        return cls(lock_frame=-2)

    def __post_init__(self) -> None:
        if self.lock_frame >= 0:
            if not 0 <= int(self.column) < 8:
                raise ValueError("forced-lock column must be in [0,7]")
            if not 0 <= int(self.row_bottom) < 16:
                raise ValueError("forced-lock bottom-origin row must be in [0,15]")
            if not 0 <= int(self.rotation) < 4:
                raise ValueError("forced-lock rotation must be in [0,3]")


def _coerce_lock(value: ForcedLock | Mapping[str, int] | None) -> ForcedLock:
    if value is None:
        return ForcedLock()
    if isinstance(value, ForcedLock):
        return value
    return ForcedLock(
        column=int(value.get("column", value.get("col", 0))),
        row_bottom=int(value.get("row_bottom", value.get("row", 0))),
        rotation=int(value.get("rotation", value.get("rot", 0))),
        lock_frame=int(value.get("lock_frame", -1)),
    )


def build_forced_lock_array(
    locks: Sequence[ForcedLock | Mapping[str, int] | None],
    *,
    num_sides: int,
):
    if len(locks) != int(num_sides):
        raise ValueError(f"expected {num_sides} forced-lock entries, got {len(locks)}")
    array_type = _DrmVsForcedLock * int(num_sides)
    array = array_type()
    for index, raw in enumerate(locks):
        lock = _coerce_lock(raw)
        array[index].col = int(lock.column)
        array[index].row = int(lock.row_bottom)
        array[index].rot = int(lock.rotation)
        array[index]._pad = 0
        array[index].lock_frame = int(lock.lock_frame)
    return array


class ForcedLockDriver:
    def __init__(self, runner: DrMarioVsPoolRunner) -> None:
        self.runner = runner
        function = getattr(runner._lib, "drm_vspool_step_forced", None)
        if function is None:
            raise DrMarioPoolError(
                "native library does not export drm_vspool_step_forced; update drmario-native"
            )
        function.argtypes = [
            C.c_void_p,
            C.POINTER(_DrmVsForcedLock),
            C.POINTER(_DrmVsPoolOutputs),
        ]
        function.restype = C.c_int
        self._step = function

    def step(self, locks: Sequence[ForcedLock | Mapping[str, int] | None]) -> None:
        array = build_forced_lock_array(locks, num_sides=self.runner.num_sides)
        rc = int(self._step(self.runner._handle, array, C.byref(self.runner._out)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_vspool_step_forced failed with rc={rc}")
        # If the pool was created with deferred planning, populate any newly
        # parked decision outputs before the experiment snapshots them.
        self.runner._solve_deferred()

    def snapshot(self) -> dict[str, np.ndarray]:
        buffers = self.runner.buffers
        return {
            "board_bytes": buffers.board_bytes.copy(),
            "pill_colors": buffers.pill_colors.copy(),
            "preview_colors": buffers.preview_colors.copy(),
            "viruses_rem": buffers.viruses_rem.copy(),
            "side_frames": buffers.side_frames.copy(),
            "need_action": buffers.need_action.copy(),
            "garbage_pending": buffers.garbage_pending.copy(),
            "garbage_sent_total": buffers.garbage_sent_total.copy(),
            "outcome": buffers.outcome.copy(),
            "terminated": buffers.terminated.copy(),
            "truncated": buffers.truncated.copy(),
        }


__all__ = [
    "ForcedLock",
    "ForcedLockDriver",
    "build_forced_lock_array",
]
