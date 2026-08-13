from __future__ import annotations

"""ctypes wrapper for the in-process Dr. Mario C++ pool backend.

This module loads ``vendor/drmario_native/build/libdrmario_pool.{dylib,so}`` and exposes a
thin, allocation-minimizing Python interface around the batched C ABI.

The pool owns N engine instances + the native reachability planner and steps at
SMDP decision boundaries (pill spawns). Python consumes compact arrays:
  - observations (bitplane_bottle / bitplane_bottle_mask / connection variants),
  - feasibility masks + costs,
  - event counters for reward/curriculum.
"""

import ctypes as C
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

GRID_H = 16
GRID_W = 8
MACRO_ACTIONS = 4 * GRID_H * GRID_W  # 512

DRMARIO_POOL_PROTOCOL_VERSION = 3

# Planner inputs for one parked env (mirrors DrmPlanState; feed to
# drm_reach_bfs_v4 or a bit-exact equivalent such as reach_cuda).
PLAN_STATE_DTYPE = np.dtype(
    [
        ("cols", "<u2", (8,)),
        ("x", "u1"),
        ("y_top", "u1"),
        ("rot", "u1"),
        ("sc", "u1"),
        ("hv", "u1"),
        ("hd", "u1"),
        ("parity", "u1"),
        ("rh", "u1"),
        ("thr", "u1"),
        ("_pad", "u1", (3,)),
    ]
)
assert PLAN_STATE_DTYPE.itemsize == 28

# Batched planner: solver(plan_states[n]) -> (n, 512) u16 pose costs
# (0xFFFF = unreachable), bit-exact drm_reach_bfs_v4 semantics.
PlanSolver = Callable[[np.ndarray], np.ndarray]


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
    root = Path(__file__).resolve().parents[3]
    return root / "vendor" / "drmario_native" / "build" / _default_library_name()


def resolve_library_path(path: Optional[str] = None) -> Path:
    """Return the shared library path to load.

    Priority:
      1) explicit ``path`` argument
      2) env var ``DRMARIO_POOL_LIB``
      3) repo-local default under ``vendor/drmario_native/build/``
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
        ("lazy_decision_outputs", C.c_uint8),
        ("defer_planning", C.c_uint8),
        ("_cfg_reserved", C.c_uint8 * 2),
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
        ("checkpoint_enabled", C.c_uint8),
        ("checkpoint_board", C.c_uint8 * 128),
        ("checkpoint_falling_colors", C.c_uint8 * 2),
        ("checkpoint_preview_colors", C.c_uint8 * 2),
        ("checkpoint_pill_counter", C.c_uint8),
        ("checkpoint_pill_counter_total", C.c_uint16),
        ("checkpoint_speed_ups", C.c_uint8),
        ("checkpoint_speed_counter", C.c_uint8),
        ("checkpoint_hor_velocity", C.c_uint8),
        ("checkpoint_frame_parity", C.c_uint8),
        ("_reserved2", C.c_uint8 * 1),
        ("inject_plan", C.c_uint8),
        ("_reserved3", C.c_uint8 * 3),
        ("inject_feasible", C.c_uint8 * 512),
        ("inject_costs", C.c_uint16 * 512),
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
        # deferred planning
        ("plan_needed", C.POINTER(C.c_uint8)),
        ("plan_state", C.POINTER(C.c_uint8)),  # [N] packed DrmPlanState (28 B)
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

    plan_needed: np.ndarray  # (N,) uint8 (deferred planning; always 0 otherwise)
    plan_state: np.ndarray  # (N,) PLAN_STATE_DTYPE


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
        lazy_decision_outputs: bool = False,
        plan_solver: Optional[PlanSolver] = None,
    ) -> None:
        self.num_envs = int(max(1, int(num_envs)))
        self.obs_spec = int(obs_spec)
        self.obs_channels = int(max(0, int(obs_channels)))
        self._plan_solver = plan_solver

        path = resolve_library_path(lib_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DrMario pool library not found at {path}. "
                "Build it with: python -m tools.build_drmario_pool (or `make -C vendor/drmario_native libdrmario_pool`)."
            )

        self._lib = _load_cdll(path)

        create = getattr(self._lib, "drm_pool_create", None)
        destroy = getattr(self._lib, "drm_pool_destroy", None)
        reset = getattr(self._lib, "drm_pool_reset", None)
        step = getattr(self._lib, "drm_pool_step", None)
        inject = getattr(self._lib, "drm_pool_inject_plans", None)
        if create is None or destroy is None or reset is None or step is None:
            raise DrMarioPoolError(f"{path} does not export the required drm_pool_* symbols")
        if plan_solver is not None and inject is None:
            raise DrMarioPoolError(
                f"{path} predates deferred planning (no drm_pool_inject_plans); "
                "rebuild it with: make -C vendor/drmario_native libdrmario_pool"
            )

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
        if inject is not None:
            inject.argtypes = [
                C.c_void_p,
                C.POINTER(C.c_uint8),
                C.POINTER(C.c_uint16),
                C.POINTER(_DrmPoolOutputs),
            ]
            inject.restype = C.c_int

        self._destroy_fn = destroy
        self._reset_fn = reset
        self._step_fn = step
        self._inject_fn = inject

        cfg = _DrmPoolConfig()
        cfg.protocol_version = DRMARIO_POOL_PROTOCOL_VERSION
        cfg.struct_size = C.sizeof(_DrmPoolConfig)
        cfg.num_envs = self.num_envs
        cfg.obs_spec = int(self.obs_spec)
        cfg.max_lock_frames = int(max(1, int(max_lock_frames)))
        cfg.max_wait_frames = int(max(1, int(max_wait_frames)))
        cfg.lazy_decision_outputs = 1 if bool(lazy_decision_outputs) else 0
        cfg.defer_planning = 1 if plan_solver is not None else 0

        handle = create(C.byref(cfg))
        if not handle:
            raise DrMarioPoolError("drm_pool_create failed (null handle)")
        self._handle = C.c_void_p(handle)

        self.buffers = self._allocate_buffers(emit_board=bool(emit_board))
        self._out = self._build_outputs_struct(self.buffers)

        self.max_lock_frames = int(max(1, int(max_lock_frames)))
        self._inject_costs_buf = np.full((self.num_envs, MACRO_ACTIONS), 0xFFFF, dtype=np.uint16)
        self._inject_mask_buf = np.zeros((self.num_envs,), dtype=np.uint8)

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
            plan_needed=np.zeros((N,), dtype=np.uint8),
            plan_state=np.zeros((N,), dtype=PLAN_STATE_DTYPE),
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

        out.plan_needed = _ptr(buffers.plan_needed, C.c_uint8)
        out.plan_state = buffers.plan_state.ctypes.data_as(C.POINTER(C.c_uint8))
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
        self._solve_deferred()

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
        self._solve_deferred()

    def _solve_deferred(self) -> None:
        """Solve plan_needed envs via the external planner and inject costs.

        Loops because an injected all-infeasible plan auto-skips that spawn
        natively and the env may park at a later spawn needing another plan
        (bounded by max_wait_frames; in practice one extra round is rare).
        """

        if self._plan_solver is None:
            return
        buf = self.buffers
        for _ in range(64):  # generous cap; each round consumes >=1 spawn per env
            idx = np.flatnonzero(buf.plan_needed)
            if idx.size == 0:
                return

            costs = np.asarray(self._plan_solver(buf.plan_state[idx]), dtype=np.uint16)
            if costs.shape != (idx.size, MACRO_ACTIONS):
                raise DrMarioPoolError(
                    f"plan_solver returned shape {costs.shape}, "
                    f"expected {(idx.size, MACRO_ACTIONS)}"
                )

            self._inject_costs_buf[idx] = costs
            mask = self._inject_mask_buf
            mask.fill(0)
            mask[idx] = 1

            rc = int(
                self._inject_fn(
                    self._handle,
                    mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                    self._inject_costs_buf.ctypes.data_as(C.POINTER(C.c_uint16)),
                    C.byref(self._out),
                )
            )
            if rc != 0:
                raise DrMarioPoolError(f"drm_pool_inject_plans failed with rc={rc}")
        raise DrMarioPoolError("deferred planning did not converge (plan_needed persists)")


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
    checkpoint_enabled: bool = False,
    checkpoint_board: Sequence[int] | np.ndarray | None = None,
    checkpoint_falling_colors: tuple[int, int] = (0xFF, 0xFF),
    checkpoint_preview_colors: tuple[int, int] = (0xFF, 0xFF),
    checkpoint_pill_counter: int = 0,
    checkpoint_pill_counter_total: int = 0,
    checkpoint_speed_ups: int = 0,
    checkpoint_speed_counter: int = 0,
    checkpoint_hor_velocity: int = 0,
    checkpoint_frame_parity: int = 0xFF,
    inject_plan: bool = False,
    inject_feasible: Sequence[int] | np.ndarray | None = None,
    inject_costs: Sequence[int] | np.ndarray | None = None,
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
    spec.checkpoint_enabled = 1 if bool(checkpoint_enabled) else 0
    board = (
        np.full((128,), 0xFF, dtype=np.uint8)
        if checkpoint_board is None
        else np.ascontiguousarray(np.asarray(checkpoint_board, dtype=np.uint8).reshape(-1))
    )
    if board.size != 128:
        raise ValueError(f"checkpoint_board must have 128 entries, got {board.size}")
    C.memmove(spec.checkpoint_board, board.ctypes.data, 128)
    spec.checkpoint_falling_colors[0] = int(checkpoint_falling_colors[0]) & 0xFF
    spec.checkpoint_falling_colors[1] = int(checkpoint_falling_colors[1]) & 0xFF
    spec.checkpoint_preview_colors[0] = int(checkpoint_preview_colors[0]) & 0xFF
    spec.checkpoint_preview_colors[1] = int(checkpoint_preview_colors[1]) & 0xFF
    spec.checkpoint_pill_counter = int(checkpoint_pill_counter) & 0xFF
    spec.checkpoint_pill_counter_total = int(checkpoint_pill_counter_total) & 0xFFFF
    spec.checkpoint_speed_ups = int(checkpoint_speed_ups) & 0xFF
    spec.checkpoint_speed_counter = int(checkpoint_speed_counter) & 0xFF
    spec.checkpoint_hor_velocity = int(checkpoint_hor_velocity) & 0xFF
    spec.checkpoint_frame_parity = int(checkpoint_frame_parity) & 0xFF
    spec.inject_plan = 1 if bool(inject_plan) else 0
    if inject_plan:
        feas = np.ascontiguousarray(
            np.asarray(inject_feasible, dtype=np.uint8).reshape(-1)
        )
        costs = np.ascontiguousarray(
            np.asarray(inject_costs, dtype=np.uint16).reshape(-1)
        )
        if feas.size != 512 or costs.size != 512:
            raise ValueError("inject_feasible/inject_costs must have 512 entries")
        C.memmove(spec.inject_feasible, feas.ctypes.data, 512)
        C.memmove(spec.inject_costs, costs.ctypes.data, 1024)
    return spec
