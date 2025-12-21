from __future__ import annotations

"""ctypes wrapper for the in-process Dr. Mario C++ pool backend.

This module loads ``game_engine/build/libdrmario_pool.{dylib,so}`` and exposes a
thin, allocation-minimizing Python interface around the batched C ABI.

The pool owns N engine instances + the native reachability planner and steps at
SMDP decision boundaries (pill spawns). Python consumes compact arrays:
  - observations (bitplane_bottle / bitplane_bottle_mask),
  - feasibility masks + costs,
  - event counters for reward/curriculum.
"""

import ctypes as C
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

GRID_H = 16
GRID_W = 8
MACRO_ACTIONS = 4 * GRID_H * GRID_W  # 512

DRMARIO_POOL_PROTOCOL_VERSION = 1


class DrMarioPoolError(RuntimeError):
    """Raised when the native pool backend fails."""


def _default_library_name() -> str:
    if sys.platform == "darwin":
        return "libdrmario_pool.dylib"
    if sys.platform.startswith("linux"):
        return "libdrmario_pool.so"
    if sys.platform == "win32":
        return "drmario_pool.dll"
    raise RuntimeError(f"Unsupported platform for drmario pool: {sys.platform!r}")


def default_library_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "game_engine" / "build" / _default_library_name()


def resolve_library_path(path: Optional[str] = None) -> Path:
    """Return the shared library path to load.

    Priority:
      1) explicit ``path`` argument
      2) env var ``DRMARIO_POOL_LIB``
      3) repo-local default under ``game_engine/build/``
    """

    if path:
        return Path(path).expanduser()
    env = os.environ.get("DRMARIO_POOL_LIB")
    if env:
        return Path(env).expanduser()
    return default_library_path()


def is_library_present(path: Optional[str] = None) -> bool:
    try:
        return resolve_library_path(path).is_file()
    except Exception:
        return False


class _DrmPoolConfig(C.Structure):
    _fields_ = [
        ("protocol_version", C.c_uint32),
        ("struct_size", C.c_uint32),
        ("num_envs", C.c_uint32),
        ("obs_spec", C.c_uint32),
        ("max_lock_frames", C.c_uint32),
        ("max_wait_frames", C.c_uint32),
    ]


class _DrmResetSpec(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("level", C.c_int32),
        ("speed_setting", C.c_int32),
        ("speed_ups", C.c_int32),
        ("rng_state", C.c_uint8 * 2),
        ("rng_override", C.c_uint8),
        ("intro_wait_frames", C.c_uint8),
        ("_reserved0", C.c_uint8),
        ("intro_frame_counter_lo_plus1", C.c_uint16),
        ("synthetic_virus_target", C.c_int32),
        ("synthetic_patch_counter", C.c_uint8),
        ("_reserved1", C.c_uint8 * 3),
        ("synthetic_seed", C.c_uint32),
    ]


class _DrmPoolOutputs(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        # decision outputs
        ("obs", C.POINTER(C.c_float)),
        ("feasible_mask", C.POINTER(C.c_uint8)),
        ("cost_to_lock", C.POINTER(C.c_uint16)),
        ("pill_colors", C.POINTER(C.c_uint8)),
        ("preview_colors", C.POINTER(C.c_uint8)),
        ("spawn_id", C.POINTER(C.c_uint8)),
        ("viruses_rem", C.POINTER(C.c_uint16)),
        ("board_bytes", C.POINTER(C.c_uint8)),
        # step outputs
        ("tau_frames", C.POINTER(C.c_uint32)),
        ("terminated", C.POINTER(C.c_uint8)),
        ("truncated", C.POINTER(C.c_uint8)),
        ("terminal_reason", C.POINTER(C.c_uint8)),
        ("invalid_action", C.POINTER(C.c_int32)),
        ("tiles_cleared_total", C.POINTER(C.c_uint16)),
        ("tiles_cleared_virus", C.POINTER(C.c_uint16)),
        ("tiles_cleared_nonvirus", C.POINTER(C.c_uint16)),
        ("match_events", C.POINTER(C.c_uint16)),
        ("adj_pair", C.POINTER(C.c_uint8)),
        ("adj_triplet", C.POINTER(C.c_uint8)),
        ("virus_adj_pair", C.POINTER(C.c_uint8)),
        ("virus_adj_triplet", C.POINTER(C.c_uint8)),
        ("lock_x", C.POINTER(C.c_int16)),
        ("lock_y", C.POINTER(C.c_int16)),
        ("lock_rot", C.POINTER(C.c_int16)),
    ]


_CDLL_CACHE: dict[str, C.CDLL] = {}


def _load_cdll(path: Path) -> C.CDLL:
    key = str(path)
    cached = _CDLL_CACHE.get(key)
    if cached is not None:
        return cached
    lib = C.CDLL(str(path))
    _CDLL_CACHE[key] = lib
    return lib


@dataclass(slots=True)
class PoolBuffers:
    """Persistent numpy arrays owned by :class:`DrMarioPoolRunner`."""

    obs: np.ndarray  # (N,C,16,8) float32
    feasible_mask: np.ndarray  # (N,512) uint8
    cost_to_lock: np.ndarray  # (N,512) uint16
    pill_colors: np.ndarray  # (N,2) uint8 (canonical)
    preview_colors: np.ndarray  # (N,2) uint8 (canonical)
    spawn_id: np.ndarray  # (N,) uint8
    viruses_rem: np.ndarray  # (N,) uint16
    board_bytes: Optional[np.ndarray]  # (N,128) uint8

    tau_frames: np.ndarray  # (N,) uint32
    terminated: np.ndarray  # (N,) uint8
    truncated: np.ndarray  # (N,) uint8
    terminal_reason: np.ndarray  # (N,) uint8
    invalid_action: np.ndarray  # (N,) int32

    tiles_cleared_total: np.ndarray  # (N,) uint16
    tiles_cleared_virus: np.ndarray  # (N,) uint16
    tiles_cleared_nonvirus: np.ndarray  # (N,) uint16
    match_events: np.ndarray  # (N,) uint16

    adj_pair: np.ndarray  # (N,3) uint8
    adj_triplet: np.ndarray  # (N,3) uint8
    virus_adj_pair: np.ndarray  # (N,3) uint8
    virus_adj_triplet: np.ndarray  # (N,3) uint8

    lock_x: np.ndarray  # (N,) int16
    lock_y: np.ndarray  # (N,) int16
    lock_rot: np.ndarray  # (N,) int16


class DrMarioPoolRunner:
    """Owns the native pool handle and numpy output buffers."""

    def __init__(
        self,
        *,
        num_envs: int,
        obs_spec: int,
        obs_channels: int,
        max_lock_frames: int = 2048,
        max_wait_frames: int = 6000,
        lib_path: Optional[str] = None,
        emit_board: bool = False,
    ) -> None:
        self.num_envs = int(max(1, int(num_envs)))
        self.obs_spec = int(obs_spec)
        self.obs_channels = int(max(0, int(obs_channels)))

        path = resolve_library_path(lib_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DrMario pool library not found at {path}. "
                "Build it with: python -m tools.build_drmario_pool (or `make -C game_engine libdrmario_pool`)."
            )

        self._lib = _load_cdll(path)

        create = getattr(self._lib, "drm_pool_create", None)
        destroy = getattr(self._lib, "drm_pool_destroy", None)
        reset = getattr(self._lib, "drm_pool_reset", None)
        step = getattr(self._lib, "drm_pool_step", None)
        if create is None or destroy is None or reset is None or step is None:
            raise DrMarioPoolError(f"{path} does not export the required drm_pool_* symbols")

        create.argtypes = [C.POINTER(_DrmPoolConfig)]
        create.restype = C.c_void_p
        destroy.argtypes = [C.c_void_p]
        destroy.restype = None
        reset.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.POINTER(_DrmResetSpec), C.POINTER(_DrmPoolOutputs)]
        reset.restype = C.c_int
        step.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_uint8),
            C.POINTER(_DrmResetSpec),
            C.POINTER(_DrmPoolOutputs),
        ]
        step.restype = C.c_int

        self._destroy_fn = destroy
        self._reset_fn = reset
        self._step_fn = step

        cfg = _DrmPoolConfig()
        cfg.protocol_version = DRMARIO_POOL_PROTOCOL_VERSION
        cfg.struct_size = C.sizeof(_DrmPoolConfig)
        cfg.num_envs = self.num_envs
        cfg.obs_spec = int(self.obs_spec)
        cfg.max_lock_frames = int(max(1, int(max_lock_frames)))
        cfg.max_wait_frames = int(max(1, int(max_wait_frames)))

        handle = create(C.byref(cfg))
        if not handle:
            raise DrMarioPoolError("drm_pool_create failed (null handle)")
        self._handle = C.c_void_p(handle)

        self.buffers = self._allocate_buffers(emit_board=bool(emit_board))
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

    def _allocate_buffers(self, *, emit_board: bool) -> PoolBuffers:
        N = self.num_envs
        Cch = self.obs_channels

        obs = np.zeros((N, Cch, GRID_H, GRID_W), dtype=np.float32)
        feasible_mask = np.zeros((N, MACRO_ACTIONS), dtype=np.uint8)
        cost_to_lock = np.full((N, MACRO_ACTIONS), 0xFFFF, dtype=np.uint16)
        pill_colors = np.zeros((N, 2), dtype=np.uint8)
        preview_colors = np.zeros((N, 2), dtype=np.uint8)
        spawn_id = np.zeros((N,), dtype=np.uint8)
        viruses_rem = np.zeros((N,), dtype=np.uint16)
        board_bytes = np.zeros((N, 128), dtype=np.uint8) if emit_board else None

        tau_frames = np.zeros((N,), dtype=np.uint32)
        terminated = np.zeros((N,), dtype=np.uint8)
        truncated = np.zeros((N,), dtype=np.uint8)
        terminal_reason = np.zeros((N,), dtype=np.uint8)
        invalid_action = np.full((N,), -1, dtype=np.int32)

        tiles_cleared_total = np.zeros((N,), dtype=np.uint16)
        tiles_cleared_virus = np.zeros((N,), dtype=np.uint16)
        tiles_cleared_nonvirus = np.zeros((N,), dtype=np.uint16)
        match_events = np.zeros((N,), dtype=np.uint16)

        adj_pair = np.zeros((N, 3), dtype=np.uint8)
        adj_triplet = np.zeros((N, 3), dtype=np.uint8)
        virus_adj_pair = np.zeros((N, 3), dtype=np.uint8)
        virus_adj_triplet = np.zeros((N, 3), dtype=np.uint8)

        lock_x = np.full((N,), -1, dtype=np.int16)
        lock_y = np.full((N,), -1, dtype=np.int16)
        lock_rot = np.full((N,), -1, dtype=np.int16)

        return PoolBuffers(
            obs=obs,
            feasible_mask=feasible_mask,
            cost_to_lock=cost_to_lock,
            pill_colors=pill_colors,
            preview_colors=preview_colors,
            spawn_id=spawn_id,
            viruses_rem=viruses_rem,
            board_bytes=board_bytes,
            tau_frames=tau_frames,
            terminated=terminated,
            truncated=truncated,
            terminal_reason=terminal_reason,
            invalid_action=invalid_action,
            tiles_cleared_total=tiles_cleared_total,
            tiles_cleared_virus=tiles_cleared_virus,
            tiles_cleared_nonvirus=tiles_cleared_nonvirus,
            match_events=match_events,
            adj_pair=adj_pair,
            adj_triplet=adj_triplet,
            virus_adj_pair=virus_adj_pair,
            virus_adj_triplet=virus_adj_triplet,
            lock_x=lock_x,
            lock_y=lock_y,
            lock_rot=lock_rot,
        )

    def _build_outputs_struct(self, buffers: PoolBuffers) -> _DrmPoolOutputs:
        out = _DrmPoolOutputs()
        out.struct_size = C.sizeof(_DrmPoolOutputs)

        def _ptr(arr: np.ndarray, c_type: object) -> C._Pointer:  # type: ignore[name-defined]
            return arr.ctypes.data_as(C.POINTER(c_type))  # type: ignore[arg-type]

        out.obs = _ptr(buffers.obs, C.c_float) if buffers.obs.size else C.cast(0, C.POINTER(C.c_float))
        out.feasible_mask = _ptr(buffers.feasible_mask, C.c_uint8)
        out.cost_to_lock = _ptr(buffers.cost_to_lock, C.c_uint16)
        out.pill_colors = _ptr(buffers.pill_colors, C.c_uint8)
        out.preview_colors = _ptr(buffers.preview_colors, C.c_uint8)
        out.spawn_id = _ptr(buffers.spawn_id, C.c_uint8)
        out.viruses_rem = _ptr(buffers.viruses_rem, C.c_uint16)
        if buffers.board_bytes is not None:
            out.board_bytes = _ptr(buffers.board_bytes, C.c_uint8)
        else:
            out.board_bytes = C.cast(0, C.POINTER(C.c_uint8))

        out.tau_frames = _ptr(buffers.tau_frames, C.c_uint32)
        out.terminated = _ptr(buffers.terminated, C.c_uint8)
        out.truncated = _ptr(buffers.truncated, C.c_uint8)
        out.terminal_reason = _ptr(buffers.terminal_reason, C.c_uint8)
        out.invalid_action = _ptr(buffers.invalid_action, C.c_int32)

        out.tiles_cleared_total = _ptr(buffers.tiles_cleared_total, C.c_uint16)
        out.tiles_cleared_virus = _ptr(buffers.tiles_cleared_virus, C.c_uint16)
        out.tiles_cleared_nonvirus = _ptr(buffers.tiles_cleared_nonvirus, C.c_uint16)
        out.match_events = _ptr(buffers.match_events, C.c_uint16)

        out.adj_pair = _ptr(buffers.adj_pair, C.c_uint8)
        out.adj_triplet = _ptr(buffers.adj_triplet, C.c_uint8)
        out.virus_adj_pair = _ptr(buffers.virus_adj_pair, C.c_uint8)
        out.virus_adj_triplet = _ptr(buffers.virus_adj_triplet, C.c_uint8)

        out.lock_x = _ptr(buffers.lock_x, C.c_int16)
        out.lock_y = _ptr(buffers.lock_y, C.c_int16)
        out.lock_rot = _ptr(buffers.lock_rot, C.c_int16)
        return out

    def reset(self, reset_mask: Optional[np.ndarray], reset_specs: Optional[np.ndarray]) -> None:
        """Reset selected envs and populate decision outputs."""

        mask_ptr: Optional[C.POINTER(C.c_uint8)]
        if reset_mask is None:
            mask_ptr = None
        else:
            mask_u8 = np.asarray(reset_mask, dtype=np.uint8).reshape(self.num_envs)
            mask_ptr = mask_u8.ctypes.data_as(C.POINTER(C.c_uint8))

        specs_ptr: Optional[C.POINTER(_DrmResetSpec)]
        specs_arr: Optional[object] = None
        if reset_specs is None:
            specs_ptr = None
        else:
            specs_arr = _build_reset_spec_array(reset_specs, self.num_envs)
            specs_ptr = C.cast(specs_arr, C.POINTER(_DrmResetSpec))

        rc = int(self._reset_fn(self._handle, mask_ptr, specs_ptr, C.byref(self._out)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_pool_reset failed with rc={rc}")
        _ = specs_arr  # keep alive until after call

    def step(
        self,
        actions: np.ndarray,
        reset_mask: Optional[np.ndarray],
        reset_specs: Optional[np.ndarray],
    ) -> None:
        """Step the pool once at decision boundaries for all envs."""

        acts = np.asarray(actions, dtype=np.int32).reshape(self.num_envs)
        acts_ptr = acts.ctypes.data_as(C.POINTER(C.c_int32))

        mask_ptr: Optional[C.POINTER(C.c_uint8)]
        if reset_mask is None:
            mask_ptr = None
        else:
            mask_u8 = np.asarray(reset_mask, dtype=np.uint8).reshape(self.num_envs)
            mask_ptr = mask_u8.ctypes.data_as(C.POINTER(C.c_uint8))

        specs_ptr: Optional[C.POINTER(_DrmResetSpec)]
        specs_arr: Optional[object] = None
        if reset_specs is None:
            specs_ptr = None
        else:
            specs_arr = _build_reset_spec_array(reset_specs, self.num_envs)
            specs_ptr = C.cast(specs_arr, C.POINTER(_DrmResetSpec))

        rc = int(self._step_fn(self._handle, acts_ptr, mask_ptr, specs_ptr, C.byref(self._out)))
        if rc != 0:
            raise DrMarioPoolError(f"drm_pool_step failed with rc={rc}")
        _ = specs_arr  # keep alive until after call


def _build_reset_spec_array(reset_specs: object, num_envs: int) -> object:
    """Return a contiguous `_DrmResetSpec[num_envs]` ctypes array.

    The returned array must be kept alive by the caller for the duration of the
    C call that consumes it.
    """

    specs_list = list(reset_specs)  # type: ignore[arg-type]
    if len(specs_list) != int(num_envs):
        raise ValueError(f"Expected {num_envs} reset specs, got {len(specs_list)}")

    arr_type = _DrmResetSpec * int(num_envs)
    arr = arr_type()
    for i, spec in enumerate(specs_list):
        if isinstance(spec, _DrmResetSpec):
            arr[i] = spec
        elif isinstance(spec, dict):
            arr[i] = build_reset_spec(**spec)
        else:
            raise TypeError(f"Unsupported reset spec type: {type(spec)!r}")
    return arr


def build_reset_spec(
    *,
    level: int = 0,
    speed_setting: int = 2,
    speed_ups: int = 0,
    rng_state: tuple[int, int] = (0, 0),
    rng_override: bool = False,
    intro_wait_frames: int = 0,
    intro_frame_counter_lo_plus1: int = 0,
    synthetic_virus_target: int = -1,
    synthetic_patch_counter: bool = False,
    synthetic_seed: int = 0,
) -> _DrmResetSpec:
    spec = _DrmResetSpec()
    spec.struct_size = C.sizeof(_DrmResetSpec)
    spec.level = int(level)
    spec.speed_setting = int(speed_setting)
    spec.speed_ups = int(speed_ups)
    spec.rng_state[0] = int(rng_state[0]) & 0xFF
    spec.rng_state[1] = int(rng_state[1]) & 0xFF
    spec.rng_override = 1 if bool(rng_override) else 0
    spec.intro_wait_frames = int(intro_wait_frames) & 0xFF
    spec.intro_frame_counter_lo_plus1 = int(intro_frame_counter_lo_plus1) & 0xFFFF
    spec.synthetic_virus_target = int(synthetic_virus_target)
    spec.synthetic_patch_counter = 1 if bool(synthetic_patch_counter) else 0
    spec.synthetic_seed = int(synthetic_seed) & 0xFFFFFFFF
    return spec
