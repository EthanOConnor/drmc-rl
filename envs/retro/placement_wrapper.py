"""High-level placement action wrapper for :class:`DrMarioRetroEnv`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
from time import perf_counter

# High-resolution timestamp origin for timing logs within this module
_LOG_T0 = perf_counter()

def _ts() -> str:
    try:
        return f"[t={perf_counter() - _LOG_T0:.6f}s]"
    except Exception:
        return "[t=0.000000s]"

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.state_core import DrMarioState
from envs.retro.placement_actions import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PLACEMENT_EDGES,
    action_count,
    opposite_actions,
)
from envs.retro.placement_planner import (
    BoardState,
    CapsuleState,
    PillSnapshot,
    PlanResult,
    PlacementPlanner,
    PlannerError,
    ControllerStep,
    iter_cells,
    snapshot_to_capsule_state,
)


@dataclass
class _ExecutionOutcome:
    last_obs: Any
    info: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    replan_required: bool
    planner_resynced: bool = False


class _ConsistencyStatus(Enum):
    MATCH = "match"
    STALLED = "stalled"
    PROGRESS = "progress"
    DIVERGED = "diverged"


@dataclass
class PlannerDebugSnapshot:
    board: BoardState
    pill: Optional[PillSnapshot]
    legal_mask: np.ndarray
    feasible_mask: np.ndarray
    plans: Tuple[PlanResult, ...]
    selected_plan: Optional[PlanResult]
    selected_action: Optional[int]
    state: Optional[np.ndarray] = None


class _InputTimingCalibrator:
    """Adaptive retry budgets for input timing mismatches."""

    def __init__(self, *, smoothing: float = 0.25, max_frames: int = 16) -> None:
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0)) if smoothing > 0 else 0.0
        self._max_frames = max(1, int(max_frames))
        # Start with conservative defaults tuned for NES acceleration: allow several
        # attempts for horizontal moves (longer acceleration) and three for soft drop.
        self._horizontal_frames = min(float(self._max_frames), 5.0)
        self._down_frames = 3.0

    @staticmethod
    def _is_horizontal(ctrl: ControllerStep) -> bool:
        return bool(ctrl.hold_left) ^ bool(ctrl.hold_right)

    @staticmethod
    def _is_soft_drop(ctrl: ControllerStep) -> bool:
        return bool(ctrl.hold_down)

    def _ema(self, current: float, observed: float) -> float:
        if self._smoothing <= 0.0:
            return observed
        return (1.0 - self._smoothing) * current + self._smoothing * observed

    def retry_budget_for(self, ctrl: ControllerStep) -> int:
        frames = 1.0
        if self._is_horizontal(ctrl):
            frames = self._horizontal_frames
        elif self._is_soft_drop(ctrl):
            frames = self._down_frames
        budget = int(math.ceil(min(self._max_frames, max(1.0, frames))) - 1)
        return max(0, budget)

    def observe_success(self, ctrl: ControllerStep, frames_used: int) -> None:
        if frames_used <= 0:
            return
        observed = float(min(self._max_frames, max(1, frames_used)))
        if self._is_horizontal(ctrl):
            self._horizontal_frames = self._ema(self._horizontal_frames, observed)
        elif self._is_soft_drop(ctrl):
            self._down_frames = self._ema(self._down_frames, observed)

    def observe_failure(self, ctrl: ControllerStep) -> None:
        increment = 1.0
        if self._is_horizontal(ctrl):
            self._horizontal_frames = min(self._max_frames, self._horizontal_frames + increment)
        elif self._is_soft_drop(ctrl):
            self._down_frames = min(self._max_frames, self._down_frames + increment)

    @property
    def max_frames(self) -> int:
        return self._max_frames

    def info(self) -> Dict[str, float]:
        return {
            "placements/timing/frames_horizontal": float(self._horizontal_frames),
            "placements/timing/frames_down": float(self._down_frames),
            "placements/timing/retry_horizontal": float(self.retry_budget_for(ControllerStep(hold_left=True))),
            "placements/timing/retry_down": float(self.retry_budget_for(ControllerStep(hold_down=True))),
        }


class PlacementTranslator:
    """Bridges emulator RAM and the placement planner."""

    def __init__(
        self,
        env: gym.Env,
        planner: Optional[PlacementPlanner] = None,
        *,
        debug: bool = False,
        fast_options_only: bool = False,
    ) -> None:
        self.env = env
        self._debug = bool(debug)
        self._planner = planner or PlacementPlanner(debug=self._debug)
        self._fast_options_only = bool(fast_options_only)
        self._offsets = getattr(env.unwrapped, "_ram_offsets", {})
        self._legal_mask = np.zeros(action_count(), dtype=np.bool_)
        self._feasible_mask = np.zeros_like(self._legal_mask)
        self._costs = np.full(action_count(), np.inf, dtype=np.float32)
        self._path_indices = np.full(action_count(), -1, dtype=np.int32)
        self._paths: Tuple[PlanResult, ...] = tuple()
        self._plan_valid_mask: np.ndarray = np.zeros(0, dtype=np.bool_)
        self._functional_mask: np.ndarray = np.ones(action_count(), dtype=np.bool_)
        self._current_snapshot: Optional[PillSnapshot] = None
        self._board: Optional[BoardState] = None
        self._last_state: Optional[np.ndarray] = None
        size_hex = self._offsets.get("falling_pill", {}).get("size_addr")
        try:
            self._falling_size_addr: Optional[int] = int(size_hex, 16) if size_hex else None
        except (TypeError, ValueError):
            self._falling_size_addr = None
        self._last_spawn_id: Optional[int] = None
        self._identical_color_pairs: Tuple[int, ...] = tuple()
        self._last_plan_latency_ms: float = 0.0
        self._last_plan_count: int = 0
        self._cached_spawn_marker: Optional[int] = None
        self._spawn_generation: int = -1
        self._options_prepared: bool = False
        self._timing = _InputTimingCalibrator()
        # Diagnostics: track last zero-feasible snapshot to avoid repeated logs
        self._last_diag_signature: Optional[Tuple[bytes, Tuple[int, int, int, Tuple[int, int], int, int, int]]] = None
        # Fast-path: avoid recomputing options if board hasn't changed for this spawn
        self._last_options_board_sig: Optional[bytes] = None
        self._planner_stats: Dict[str, int] = {}
        self._last_refresh_frame: Optional[int] = None
        self._last_forced_options_sig: Optional[Tuple[Optional[int], bytes]] = None
        self._internal_resync_limit = 6

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def refresh(self, *, force: bool = False, state: DrMarioState) -> None:
        t0 = perf_counter()
        frame_id: Optional[int]
        try:
            frame_id = int(getattr(self.env.unwrapped, "_t", None))
        except Exception:
            frame_id = None
        if not force and frame_id is not None and frame_id == self._last_refresh_frame:
            return

        ram_bytes = state.ram.bytes

        t1 = perf_counter()
        try:
            board = BoardState.from_state(state.calc.planes)
        except Exception:
            board = None
        if board is not None:
            self._board = board
        t2 = perf_counter()
        self._last_state = np.array(state.calc.planes, copy=True)
        pill = self._extract_pill(state, ram_bytes, require_mask=False)
        if pill is None:
            self._current_snapshot = None
            self._last_spawn_id = None
            self._clear_cached_options()
            if self._debug:
                try:
                    print(
                        (
                            f"{_ts()} [timing] translator.refresh read_state_ms={{(t1-t0)*1000:.3f}} board_ms={{(t2-t1)*1000:.3f}} pill_ms=0.000"
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
            self._last_refresh_frame = frame_id
            return
        t3 = perf_counter()
        self._current_snapshot = pill
        if self._board is not None:
            try:
                self._board = self._board.without_capsule(pill)
            except Exception:
                pass
        try:
            self._last_spawn_id = int(pill.spawn_id) if pill.spawn_id is not None else None
        except Exception:
            self._last_spawn_id = None
        if not self._spawn_matches_cache(pill):
            self._spawn_generation += 1
            self._clear_cached_options()
        else:
            # Cheap refresh does not trigger planning; ensure diagnostics reflect that.
            self._last_plan_latency_ms = 0.0
            self._last_plan_count = 0
        if self._debug:
            try:
                print(
                    (
                        f"{_ts()} [timing] translator.refresh read_state_ms={(t1-t0)*1000:.3f} board_ms={(t2-t1)*1000:.3f} pill_ms={(t3-t2)*1000:.3f}"
                    ),
                    flush=True,
                )
            except Exception:
                pass
        self._last_refresh_frame = frame_id

    def refresh_state_only(self) -> None:
        """Deprecated shim; :meth:`refresh` already caches duplicate calls."""

        self.refresh()

    def prepare_options(self, *, force: bool = False, force_full: Optional[bool] = None) -> None:
        """Ensure placement options are prepared for the current spawn."""

        pill = self._current_snapshot
        board = self._board
        if pill is None or board is None:
            self._clear_cached_options()
            return
        t0 = perf_counter()
        spawn_marker = self._spawn_marker_for(pill)
        # Compute lightweight signature to detect unchanged board for this spawn
        try:
            board_sig = board.columns.tobytes()
        except Exception:
            board_sig = b""
        if force_full is None:
            force_full = not self._fast_options_only
        force_sig = (spawn_marker, board_sig)
        if force and not force_full:
            if self._last_forced_options_sig == force_sig:
                return
        if (
            self._options_prepared
            and self._cached_spawn_marker == spawn_marker
            and self._last_options_board_sig == board_sig
            and not force
        ):
            return
        start = perf_counter()
        used_fast_path = False
        use_fast_path = self._fast_options_only and not bool(force_full)
        if use_fast_path:
            legal, feasible, costs, path_indices, plans, stats = self._planner.enumerate_fast_options(board, pill)
            t_fast = perf_counter()
            self._last_plan_latency_ms = (t_fast - start) * 1000.0
            self._legal_mask = legal
            self._feasible_mask = feasible
            self._paths = plans
            self._costs = costs
            self._path_indices = path_indices
            self._planner_stats = stats
            self._last_plan_count = int(feasible.sum())
            t1 = t_fast
            used_fast_path = True
        else:
            planner_out = self._planner.plan_all(board, pill)
            t1 = perf_counter()
            self._last_plan_latency_ms = (t1 - start) * 1000.0
            self._legal_mask = planner_out.legal_mask.copy()
            self._feasible_mask = planner_out.feasible_mask.copy()
            self._paths = planner_out.plans
            self._costs = planner_out.costs.copy()
            self._path_indices = planner_out.path_indices.copy()
            self._planner_stats = dict(planner_out.stats)
            self._last_plan_count = int(planner_out.plan_count)
        self._plan_valid_mask = np.ones(len(self._paths), dtype=np.bool_) if len(self._paths) else np.zeros(0, dtype=np.bool_)
        for plan_idx, plan in enumerate(self._paths):
            if not self._plan_is_physical(plan, pill):
                self._plan_valid_mask[plan_idx] = False
        if len(self._plan_valid_mask) and not np.all(self._plan_valid_mask):
            for action_idx, plan_idx in enumerate(self._path_indices):
                if plan_idx >= 0 and not bool(self._plan_valid_mask[plan_idx]):
                    self._path_indices[action_idx] = -1
                    self._feasible_mask[action_idx] = False
                    self._costs[action_idx] = np.inf
            self._last_plan_count = int(self._feasible_mask.sum())
        if self._functional_mask.shape[0] != self._feasible_mask.shape[0]:
            self._functional_mask = self._feasible_mask.astype(bool).copy()
        else:
            self._functional_mask = np.logical_and(self._functional_mask, self._feasible_mask)

        self._cached_spawn_marker = spawn_marker
        self._options_prepared = True
        self._last_options_board_sig = board_sig
        self._last_forced_options_sig = force_sig if force else None
        # Avoid duplicate masking work if no options found
        self._mask_identical_colors(pill)
        if self._debug:
            try:
                print(
                    (
                        f"{_ts()} [timing] translator.prepare_options force={int(bool(force))} force_full={int(bool(force_full))} "
                        f"pre_ms={(start-t0)*1000:.3f} "
                        f"plan_ms={(t1-start)*1000:.3f} options={int(self._last_plan_count)} expanded={int(self._planner_stats.get('expanded',0))}"
                    ),
                    flush=True,
                )
            except Exception:
                pass
        # Diagnostics: log when legal>0 but no feasible plans are found (controlled by debug flag)
        try:
            legal_n = int(self._legal_mask.sum())
            if self._debug and legal_n > 0 and self._last_plan_count == 0:
                # Compose a signature of the board + pill to avoid duplicate logs
                board_sig = (
                    b"" if self._board is None else self._board.columns.tobytes()
                )
                pill_sig = (
                    int(pill.row),
                    int(pill.col),
                    int(pill.orient),
                    (int(pill.colors[0]), int(pill.colors[1])),
                    int(pill.gravity_counter),
                    int(pill.gravity_period),
                    int(pill.lock_counter),
                    int(pill.hor_velocity),
                    int(pill.frame_parity),
                    int(pill.hold_left),
                    int(pill.hold_right),
                    int(pill.hold_down),
                )
                diag_sig = (board_sig, pill_sig)
                if diag_sig != self._last_diag_signature:
                    self._last_diag_signature = diag_sig
                    print(
                        (
                            f"[translator] spawn={getattr(pill, 'spawn_id', None)} legal={legal_n} "
                            f"feasible=0 pill(row={pill.row}, col={pill.col}, orient={pill.orient}, "
                            f"colors={getattr(pill, 'colors', None)}, grav={pill.gravity_counter}/"
                            f"{pill.gravity_period}, vel={pill.hor_velocity}, frame_parity={pill.frame_parity}, "
                            f"holds[L={int(pill.hold_left)},R={int(pill.hold_right)},D={int(pill.hold_down)}], "
                            f"lock={pill.lock_counter})"
                        ),
                        flush=True,
                    )
                    # Dump a compact board occupancy for external diagnostics
                    occ = _board_occupancy(self._board) if self._board is not None else None
                    if occ is not None:
                        try:
                            rows = []
                            for r in range(GRID_HEIGHT):
                                line = ''.join('#' if occ[r, c] else '.' for c in range(GRID_WIDTH))
                                rows.append(line)
                            print("[translator] board:", flush=True)
                            for line in rows:
                                print("[translator] ", line, flush=True)
                        except Exception:
                            pass
        except Exception:
            pass

    def info(self) -> Dict[str, Any]:
        info = {
            "placements/legal_mask": self._legal_mask.copy(),
            "placements/feasible_mask": self._feasible_mask.copy(),
            "placements/options": int(self._feasible_mask.sum()),
            "placements/costs": self._costs.copy(),
            "placements/path_indices": self._path_indices.copy(),
            "placements/functional_mask": self._functional_mask.copy(),
        }
        if self._last_spawn_id is not None:
            info["pill/spawn_id"] = int(self._last_spawn_id)
            info["placements/spawn_id"] = int(self._last_spawn_id)
        pill = self._current_snapshot
        if pill is not None:
            info.update(
                {
                    "pill/speed_counter": int(pill.gravity_counter),
                    "pill/speed_threshold": int(pill.gravity_period),
                    "pill/hor_velocity": int(pill.hor_velocity),
                    "pill/frame_parity": int(pill.frame_parity),
                    "pill/hold_left": int(pill.hold_left),
                    "pill/hold_right": int(pill.hold_right),
                    "pill/hold_down": int(pill.hold_down),
                }
            )
        if self._planner_stats:
            for key, value in self._planner_stats.items():
                info[f"placements/stats/{key}"] = int(value)
        info.update(self._timing.info())
        return info

    def get_plan(self, action: int) -> Optional[PlanResult]:
        # Lazily prepare options if they have not been computed yet for this spawn.
        self.prepare_options()
        idx = int(self._path_indices[int(action)])
        if idx < 0 or idx >= len(self._paths):
            # Try rebuilding options in case caches desynced (only meaningful in full-planner mode).
            if not self._fast_options_only:
                self.prepare_options(force=True, force_full=True)
                try:
                    idx = int(self._path_indices[int(action)])
                except Exception:
                    idx = -1
                if 0 <= idx < len(self._paths):
                    return self._paths[idx]
            # Try on-demand single-target planning for the requested action.
            if self._board is not None and self._current_snapshot is not None:
                # Micro-cache: reuse single-target plan for same spawn marker + board + action
                try:
                    board_sig = self._board.columns.tobytes()
                except Exception:
                    board_sig = b""
                # Use translated spawn counter if available; else derive
                try:
                    tinfo = self.info()
                    marker_int = int(tinfo.get("placements/spawn_id", -1))
                except Exception:
                    marker_int = -1
                cached_sig = getattr(self, "_last_single_plan_sig", None)
                cached_plan = getattr(self, "_last_single_plan", None)
                current_sig = (marker_int, board_sig, int(action))
                if cached_sig == current_sig and cached_plan is not None:
                    return cached_plan
                try:
                    # Prefer fast route synthesis when available. Do NOT fall back to heavy planner here.
                    single = self._planner.plan_action_fast(self._board, self._current_snapshot, int(action))
                except Exception:
                    single = None
                if single is not None:
                    if not self._plan_is_physical(single, self._current_snapshot):
                        single = None
                if single is not None:
                    if getattr(self._planner, "_debug", False):
                        try:
                            print(
                                f"[plan_action_ok] action={int(action)} steps={len(single.controller)}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    # Save to micro-cache on translator
                    self._last_single_plan_sig = current_sig
                    self._last_single_plan = single
                    return single
            # Fallback: synthesize a minimal locked plan so the visualizer can
            # at least show the selected target when feasibility is bypassed.
            legal = bool(self._legal_mask[int(action)])
            feasible = bool(self._feasible_mask[int(action)])
            if legal or feasible:
                if getattr(self._planner, "_debug", False):
                    try:
                        stats = getattr(self._planner, "_last_search_stats", None)
                        extra = ""
                        if isinstance(stats, dict):
                            extra = (
                                f" start_fits={stats.get('start_fits')} expanded={stats.get('expanded')}"
                                f" grounded_seen={stats.get('grounded_seen')} max_row={stats.get('max_row')}"
                            )
                        print(
                            f"[plan_action_fail] action={int(action)} legal={int(legal)} feasible={int(feasible)}"
                            f" paths={len(self._paths)}{extra}",
                            flush=True,
                        )
                    except Exception:
                        pass
                # Do not synthesize an empty controller plan; instead force a new decision
                self._functional_mask = getattr(self, "_functional_mask", np.zeros(action_count(), dtype=np.bool_))
                self._functional_mask[int(action)] = False
                return None
            return None
        plan = self._paths[idx]
        if len(self._plan_valid_mask) and not bool(self._plan_valid_mask[idx]):
            return None
        # Sync guard: if the cached plan's start state does not match the current pill,
        # rebuild options and/or compute a single-target plan for the current snapshot.
        try:
            pill_now = self._current_snapshot
            if pill_now is not None and plan.states:
                start = plan.states[0]
                if (start.row != int(pill_now.row)) or (start.col != int(pill_now.col)) or (start.orient != int(pill_now.orient)):
                    # Recompute options once for this spawn unless in fast-options mode
                    if not bool(getattr(self, "_translator", None)) or not bool(getattr(self._translator, "_fast_options_only", False)):
                        self.prepare_options(force=True)
                        idx2 = int(self._path_indices[int(action)])
                        if 0 <= idx2 < len(self._paths):
                            if not len(self._plan_valid_mask) or bool(self._plan_valid_mask[idx2]):
                                return self._paths[idx2]
                    # Fall back to single-target fast planning only
                    try:
                        fresh = self._planner.plan_action_fast(self._board, pill_now, int(action))
                        if fresh is not None and self._plan_is_physical(fresh, pill_now):
                            return fresh
                    except Exception:
                        pass
        except Exception:
            pass
        return plan

    def capture_capsule_state(self) -> Optional[CapsuleState]:
        ram_bytes = self._read_ram_bytes()
        if ram_bytes is None:
            return None
        if self._falling_size_addr is not None:
            if self._falling_size_addr >= len(ram_bytes):
                return None
            size_val = ram_bytes[self._falling_size_addr]
            if size_val < 2:
                return None
        pill = self._extract_pill(None, ram_bytes, require_mask=False)
        if pill is None:
            return None
        return snapshot_to_capsule_state(pill)

    def current_board(self) -> Optional[BoardState]:
        return self._board

    def current_pill(self) -> Optional[PillSnapshot]:
        return self._current_snapshot

    def diagnostics(self) -> Dict[str, Any]:
        diag = {
            "plan_latency_ms": self._last_plan_latency_ms,
            "plan_count": self._last_plan_count,
        }
        if self._planner_stats:
            diag.update({f"planner/{k}": int(v) for k, v in self._planner_stats.items()})
        return diag

    def retry_budget(self, ctrl: ControllerStep) -> int:
        return self._timing.retry_budget_for(ctrl)

    def record_timing_success(self, ctrl: ControllerStep, frames_used: int) -> None:
        self._timing.observe_success(ctrl, frames_used)

    def record_timing_failure(self, ctrl: ControllerStep) -> None:
        self._timing.observe_failure(ctrl)

    def max_step_frames(self) -> int:
        return self._timing.max_frames

    def replan_to_action(self, action: int) -> Optional[PlanResult]:
        """Recompute a plan for the given placement action from the current snapshot.

        Uses the fast reachability path when possible, falling back to full planning.
        """
        pill = self.current_pill()
        board = self.current_board()
        if pill is None or board is None:
            return None
        try:
            plan = self._planner.plan_action_fast(board, pill, int(action))
        except Exception:
            plan = None
        if plan is not None:
            return plan
        return None

    def internal_resync_limit(self) -> int:
        return self._internal_resync_limit

    @staticmethod
    def _states_close(a: Optional[CapsuleState], b: Optional[CapsuleState]) -> bool:
        if a is None or b is None:
            return False
        if a.orient != b.orient:
            return False
        if abs(a.col - b.col) > 2:
            return False
        if abs(a.row - b.row) > 2:
            return False
        return True

    def resume_index_for(self, plan: PlanResult, actual: Optional[CapsuleState]) -> int:
        if actual is None or not plan.states:
            return 0
        for idx, candidate in enumerate(plan.states):
            if self._states_close(candidate, actual):
                return min(idx, len(plan.controller))
        # If no close match, fall back to the start
        return 0

    @staticmethod
    def _plan_is_physical(plan: PlanResult, pill: Optional[PillSnapshot]) -> bool:
        if plan is None or not plan.states:
            return False
        start = plan.states[0]
        final = plan.states[-1]
        start_orient = start.orient & 3
        final_orient = final.orient & 3
        allowed_lift = 1 if (start_orient in (0, 2) and final_orient in (1, 3)) else 0
        if final.row + allowed_lift < start.row:
            return False
        min_row = min(state.row for state in plan.states)
        if min_row + allowed_lift < start.row:
            return False
        return True

    def alternative_action(self, attempted: Set[int]) -> Optional[int]:
        if not self._options_prepared:
            self.prepare_options()
        best_action: Optional[int] = None
        best_cost: float = float("inf")
        for action in range(action_count()):
            if action in attempted:
                continue
            if action >= self._feasible_mask.size or not bool(self._feasible_mask[action]):
                continue
            if action < self._functional_mask.size and not bool(self._functional_mask[action]):
                continue
            plan_idx = int(self._path_indices[action])
            if plan_idx < 0 or plan_idx >= len(self._paths):
                continue
            if plan_idx < len(self._plan_valid_mask) and not bool(self._plan_valid_mask[plan_idx]):
                continue
            cost_val = float(self._costs[action]) if action < self._costs.size else float("inf")
            if cost_val < best_cost:
                best_cost = cost_val
                best_action = action
        return best_action

    def debug_snapshot(
        self, selected_action: Optional[int] = None
    ) -> Optional[PlannerDebugSnapshot]:
        if self._board is None:
            return None
        board_copy = BoardState(columns=self._board.columns.copy())
        selected_plan: Optional[PlanResult] = None
        action_value: Optional[int] = None
        if selected_action is not None:
            plan_candidate = self.get_plan(int(selected_action))
            if plan_candidate is not None:
                selected_plan = plan_candidate
                action_value = int(selected_action)
        return PlannerDebugSnapshot(
            board=board_copy,
            pill=self._current_snapshot,
            legal_mask=self._legal_mask.copy(),
            feasible_mask=self._feasible_mask.copy(),
            plans=self._paths,
            selected_plan=selected_plan,
            selected_action=action_value,
            state=None if self._last_state is None else np.array(self._last_state, copy=True),
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def consistency_status(
        self,
        actual: Optional[CapsuleState],
        expected: CapsuleState,
        previous: Optional[CapsuleState],
        ctrl: ControllerStep,
    ) -> _ConsistencyStatus:
        if actual is None:
            return _ConsistencyStatus.MATCH if expected.locked else _ConsistencyStatus.DIVERGED

        if (
            actual.row == expected.row
            and actual.col == expected.col
            and actual.orient == expected.orient
        ):
            return _ConsistencyStatus.MATCH

        if actual.locked and expected.locked:
            if previous is None or (actual.col == expected.col and actual.orient == expected.orient):
                return _ConsistencyStatus.MATCH

        if previous is not None:
            if self._ahead_of_expected(actual, expected, previous):
                return _ConsistencyStatus.MATCH
            if (
                actual.row == previous.row
                and actual.col == previous.col
                and actual.orient == previous.orient
            ):
                if self._is_progressing(actual, previous, expected, ctrl):
                    return _ConsistencyStatus.PROGRESS
                return _ConsistencyStatus.STALLED

            if self._is_progressing(actual, previous, expected, ctrl):
                return _ConsistencyStatus.PROGRESS

        return _ConsistencyStatus.DIVERGED

    @staticmethod
    def _ahead_of_expected(
        actual: CapsuleState,
        expected: CapsuleState,
        previous: CapsuleState,
    ) -> bool:
        if actual.orient != expected.orient:
            return False
        row_delta = expected.row - previous.row
        col_delta = expected.col - previous.col

        if row_delta > 0:
            if actual.col == expected.col and actual.row >= expected.row:
                return True
        elif row_delta < 0:
            if actual.col == expected.col and actual.row <= expected.row:
                return True

        if col_delta > 0:
            if actual.row == expected.row and actual.col >= expected.col:
                return True
        elif col_delta < 0:
            if actual.row == expected.row and actual.col <= expected.col:
                return True

        return False

    @staticmethod
    def _is_progressing(
        actual: CapsuleState,
        previous: CapsuleState,
        expected: CapsuleState,
        ctrl: ControllerStep,
    ) -> bool:
        delta_row = expected.row - previous.row
        delta_col = expected.col - previous.col
        delta_orient = (expected.orient - previous.orient) & 0x03

        # Vertical motion (gravity / soft drop)
        if delta_row != 0 and previous.col == expected.col and previous.orient == expected.orient:
            direction = 1 if delta_row > 0 else -1
            if (
                actual.col == previous.col
                and actual.orient == previous.orient
            ):
                if (actual.row - previous.row) * direction >= 0:
                    return True
                if (
                    ctrl.hold_down
                    or actual.hold_down
                    or actual.speed_counter != previous.speed_counter
                    or actual.speed_counter != expected.speed_counter
                ):
                    return True

        # Horizontal motion (hold acceleration)
        if delta_col != 0 and previous.row == expected.row and previous.orient == expected.orient:
            direction = 1 if delta_col > 0 else -1
            if (
                actual.row == previous.row
                and actual.orient == previous.orient
            ):
                if (actual.col - previous.col) * direction >= 0:
                    return True
                if direction > 0:
                    if ctrl.hold_right or actual.hold_right or actual.hor_velocity > previous.hor_velocity:
                        return True
                else:
                    if ctrl.hold_left or actual.hold_left or actual.hor_velocity < previous.hor_velocity:
                        return True

        # Rotational motion
        if delta_orient != 0 and actual.row == previous.row and actual.col == previous.col:
            if actual.orient == expected.orient:
                return True
            if ctrl.action in (Action.ROTATE_A, Action.ROTATE_B):
                if actual.frame_parity != previous.frame_parity or actual.speed_counter != previous.speed_counter:
                    return True
        if delta_orient != 0 and actual.orient == expected.orient and ctrl.action in (Action.ROTATE_A, Action.ROTATE_B):
            if actual.col == expected.col and abs(actual.row - expected.row) <= 1:
                return True

        return False

    def _mask_identical_colors(self, pill: PillSnapshot) -> None:
        if pill.colors[0] != pill.colors[1]:
            self._identical_color_pairs = tuple()
            return
        masked = []
        for left, right in opposite_actions():
            left_idx = int(self._path_indices[left])
            right_idx = int(self._path_indices[right])
            left_cost = float(self._costs[left])
            right_cost = float(self._costs[right])
            left_legal = bool(self._legal_mask[left])
            right_legal = bool(self._legal_mask[right])

            if not (left_legal or right_legal):
                continue

            # Prefer whichever direction currently has a feasible plan. When both
            # exist we keep the lower-cost option but remain deterministic by
            # favouring the canonical left->right orientation on ties.
            drop = None
            if left_idx >= 0 and right_idx >= 0:
                if left_cost <= right_cost:
                    drop = right
                else:
                    drop = left
            elif left_idx >= 0:
                drop = right
            elif right_idx >= 0:
                drop = left
            else:
                # No feasible plan yet; fall back to canonical ordering.
                drop = right

            if drop is None:
                continue
            self._feasible_mask[drop] = False
            self._legal_mask[drop] = False
            self._path_indices[drop] = -1
            self._costs[drop] = np.inf
            masked.append(drop)

        self._identical_color_pairs = tuple(masked)
        self._last_plan_count = int(np.count_nonzero(self._path_indices >= 0))



    def _extract_pill(
        self,
        state: DrMarioState,
        ram_bytes: bytes,
        *,
        require_mask: bool = True,
    ) -> Optional[PillSnapshot]:
        if require_mask and state is not None:
            falling_mask = ram_specs.get_falling_mask(state.calc.planes)
            if falling_mask.sum() == 0:
                return None
        try:
            return PillSnapshot.from_ram_state(state, self._offsets)
        except PlannerError:
            return None

    def _spawn_marker_for(self, pill: PillSnapshot) -> int:
        if pill.spawn_id is not None:
            try:
                return int(pill.spawn_id)
            except Exception:
                pass
        return int(self._spawn_generation)

    def _spawn_matches_cache(self, pill: PillSnapshot) -> bool:
        if self._cached_spawn_marker is None:
            return False
        return self._cached_spawn_marker == self._spawn_marker_for(pill)

    def _clear_cached_options(self) -> None:
        self._legal_mask[:] = False
        self._feasible_mask[:] = False
        self._costs.fill(np.inf)
        self._path_indices.fill(-1)
        self._paths = tuple()
        self._identical_color_pairs = tuple()
        self._options_prepared = False
        self._cached_spawn_marker = None
        self._last_plan_latency_ms = 0.0
        self._last_plan_count = 0
        self._planner_stats = {}
        self._last_forced_options_sig = None
        if getattr(self, "_functional_mask", None) is not None:
            self._functional_mask[:] = True


class DrMarioPlacementEnv(gym.Wrapper):
    """Wrapper exposing the 464-way placement action space (spawn-latched)."""

    def __init__(
        self,
        env: gym.Env,
        *,
        planner: Optional[PlacementPlanner] = None,
        debug_log: bool = False,
        path_log: bool = False,
    ) -> None:
        super().__init__(env)
        self._debug = bool(debug_log)
        self._path_log = bool(path_log)
        self.action_space = gym.spaces.Discrete(action_count())
        self._translator = PlacementTranslator(env, planner, debug=debug_log, fast_options_only=True)
        # Micro-cache for single-target plans within a spawn
        self._last_single_plan_sig: Optional[Tuple[int, bytes, int]] = None
        self._last_single_plan: Optional[PlanResult] = None
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._active_plan: Optional[PlanResult] = None
        self._latched_action: Optional[int] = None
        self._latched_spawn_id: int = -1
        self._spawn_id: int = 0
        self._capsule_present: bool = False
        self._spawn_marker: Optional[int] = None
        self._step_callback: Optional[
            Callable[[Any, Dict[str, Any], Optional[int], float, bool, bool], None]
        ] = None
        self._attempted_actions: Set[int] = set()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        t0 = perf_counter()
        obs, info = self.env.reset(**kwargs)
        t1 = perf_counter()
        self._last_obs = obs
        if info is None:
            info = {}
        r0 = perf_counter()
        self._translator.refresh(state=self.env.unwrapped._state_cache)
        r1 = perf_counter()
        self._active_plan = None
        self._latched_action = None
        self._latched_spawn_id = -1
        self._capsule_present = False
        self._spawn_marker = None
        self._spawn_id = 0
        a0 = perf_counter()
        obs, info, _, _, _ = self._await_next_pill(obs, info)
        a1 = perf_counter()
        if self._debug:
            try:
                print(
                    (
                        f"{_ts()} [timing] reset env_reset_ms={(t1-t0)*1000:.3f} translator_refresh_ms={(r1-r0)*1000:.3f} "
                        f"await_next_ms={(a1-a0)*1000:.3f}"
                    ),
                    flush=True,
                )
            except Exception:
                pass
        self._last_obs = obs
        self._last_info = info
        self._capsule_present = bool(self._translator.current_pill() is not None)
        self._attempted_actions.clear()
        # Our spawn_id is purely internal and monotonic; do not overwrite from info.
        return obs, info

    def step(self, action: int):
        # Allow a deliberate override to clear the execution latch for the same spawn.
        if (
            self._active_plan is not None
            and self._latched_spawn_id == self._spawn_id
            and self._latched_action is not None
            and int(action) != self._latched_action
        ):
            self._clear_latch()

        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._last_obs
        last_info: Dict[str, Any] = {}
        replan_attempts = 0
        planner_calls = 0
        planner_latency_ms_total = 0.0
        planner_latency_ms_max = 0.0
        planner_plan_count_total = 0.0
        planner_plan_count_last = 0.0
        planner_latency_ms_last = 0.0

        def record_refresh_metrics() -> None:
            nonlocal planner_calls, planner_latency_ms_total, planner_latency_ms_max
            nonlocal planner_plan_count_total, planner_plan_count_last, planner_latency_ms_last
            diagnostics = self._translator.diagnostics() or {}
            latency = float(diagnostics.get("plan_latency_ms", 0.0) or 0.0)
            plan_count = float(diagnostics.get("plan_count", 0.0) or 0.0)
            if plan_count == 0.0 and latency == 0.0:
                return
            planner_calls += 1
            planner_latency_ms_total += latency
            planner_latency_ms_max = max(planner_latency_ms_max, latency)
            planner_plan_count_total += plan_count
            planner_plan_count_last = plan_count
            planner_latency_ms_last = latency

        record_refresh_metrics()

        def _read_spawn_marker() -> Optional[int]:
            snap = self._translator.current_pill()
            if snap is None:
                return None
            info_t = self._translator.info() or {}
            marker = info_t.get("placements/spawn_id", getattr(snap, "spawn_id", None))
            try:
                return int(marker) if marker is not None else None
            except Exception:
                return None

        def _mark_needs_action(base: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            out = dict(base or {})
            out["placements/needs_action"] = True
            out["placements/spawn_id"] = int(self._spawn_id)
            out.setdefault("pill_changed", 0)
            return out

        if self._translator.current_pill() is None:
            obs, info, reward_delta, terminated, truncated = self._await_next_pill(
                self._last_obs, self._last_info, record_refresh_metrics
            )
            total_reward += reward_delta
            self._last_obs, self._last_info = obs, info
            self._clear_latch()
            return obs, total_reward, terminated, truncated, info

        def request_new_decision(base_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            info_payload = _mark_needs_action(base_info)
            info_payload["pill_changed"] = 0
            for key in (
                "placements/legal_mask",
                "placements/feasible_mask",
                "placements/options",
                "placements/costs",
                "placements/path_indices",
            ):
                info_payload.pop(key, None)
            # Fast path: compute options only if inputs changed
            self._translator.prepare_options(force=False)
            record_refresh_metrics()
            info_payload.update(self._translator.info() or {})
            return info_payload

        attempted_actions: Set[int] = self._attempted_actions

        while True:
            self._log(
                f"loop_start spawn={self._spawn_id} action={int(action)} attempted={sorted(attempted_actions)}"
            )
            if self._active_plan is None or self._latched_spawn_id != self._spawn_id:
                if self._debug:
                    try:
                        print(f"{_ts()} [timing] pre_get_plan action={int(action)}", flush=True)
                    except Exception:
                        pass
                plan = self._translator.get_plan(int(action))
                if plan is None:
                    attempted_actions.add(int(action))
                    # Advance emulator by one NOOP frame while requesting a new decision,
                    # so the environment and viewers continue to progress.
                    self._log(
                        f"no_plan spawn={self._spawn_id} action={int(action)} -> noop"
                    )
                    outcome = self._execute_plan(None, record_refresh_metrics)
                    total_reward += outcome.reward
                    last_obs = outcome.last_obs
                    # Overlay a fresh request for options/decision on top of the latest info
                    self._log(
                        f"request_new_decision spawn={self._spawn_id} reason=no_plan"
                    )
                    last_info = request_new_decision(outcome.info)
                    terminated = outcome.terminated
                    truncated = outcome.truncated
                    self._clear_latch()
                    break
                self._active_plan = plan
                self._latched_action = int(action)
                self._latched_spawn_id = int(self._spawn_id)
                self._log_plan_details(int(action), plan)
            else:
                plan = self._active_plan
                self._log(
                    f"reusing_plan spawn={self._spawn_id} action={int(action)} latched={self._latched_action}"
                )

            outcome = self._execute_plan(plan, record_refresh_metrics)
            total_reward += outcome.reward
            last_obs = outcome.last_obs
            last_info = outcome.info
            terminated = outcome.terminated
            truncated = outcome.truncated
            # _execute_plan already refreshes the translator as it advances the emulator.
            record_refresh_metrics()

            pill_now = self._translator.current_pill() is not None

            if terminated or truncated:
                self._clear_latch()
                break
            if outcome.replan_required:
                last_info = dict(last_info)
                last_info["placements/feasible_fp"] = int(
                    last_info.get("placements/feasible_fp", 0)
                ) + 1
                failed_idx = self._latched_action if self._latched_action is not None else action
                last_info["placements/failed_action_idx"] = int(failed_idx)
                replan_attempts += 1
                failed_action = int(failed_idx)
                attempted_actions.add(failed_action)
                if failed_action < len(self._translator._functional_mask):
                    self._translator._functional_mask[failed_action] = False
                self._log(
                    f"plan_failed spawn={self._spawn_id} action={failed_action} attempts={sorted(attempted_actions)}"
                )
                self._log(
                    f"fallback_exhausted spawn={self._spawn_id} attempts={sorted(attempted_actions)}"
                )
                self._log(
                    f"request_new_decision spawn={self._spawn_id} reason=replan_failed"
                )
                last_info = request_new_decision(last_info)
                self._clear_latch()
                break

            # Detect next-spawn arrival during/after plan execution.
            marker_now = _read_spawn_marker() if pill_now else None
            new_spawn = (
                pill_now
                and marker_now is not None
                and self._spawn_marker is not None
                and marker_now != self._spawn_marker
            )
            if new_spawn:
                # Bump OUR counter; keep translator marker only for detection.
                self._spawn_marker = marker_now
                self._spawn_id += 1
                self._capsule_present = True
                self._attempted_actions.clear()
                self._log(f"new_spawn spawn={self._spawn_id} attempts_cleared=1")
                # Fast path: compute options only if inputs changed
                self._translator.prepare_options(force=False)
                record_refresh_metrics()
                info_now = _mark_needs_action(self._translator.info())
                info_now["pill_changed"] = 1
                last_info = {**last_info, **info_now}
                self._clear_latch()
                break

            # Happy path: plan executed; we're done for this env.step().
            if pill_now and marker_now is not None and self._spawn_marker is None:
                self._spawn_marker = marker_now
            self._capsule_present = pill_now
            if not outcome.replan_required:
                attempted_actions.clear()
                self._log(f"plan_success spawn={self._spawn_id} attempts_cleared=1")
            break

        self._last_obs = last_obs
        enriched_info = dict(last_info)
        planner_latency_ms_avg = (
            planner_latency_ms_total / planner_calls if planner_calls > 0 else 0.0
        )
        if planner_calls > 0:
            planner_plan_count_avg = planner_plan_count_total / planner_calls
        else:
            planner_plan_count_avg = planner_plan_count_last
        enriched_info["placements/replan_attempts"] = replan_attempts
        enriched_info["placements/plan_calls"] = planner_calls
        enriched_info["placements/plan_latency_ms_total"] = planner_latency_ms_total
        enriched_info["placements/plan_latency_ms_avg"] = planner_latency_ms_avg
        enriched_info["placements/plan_latency_ms_max"] = planner_latency_ms_max
        enriched_info["placements/plan_count_total"] = planner_plan_count_total
        enriched_info["placements/plan_count_avg"] = planner_plan_count_avg
        enriched_info["placements/plan_count_last"] = planner_plan_count_last
        enriched_info["placements/plan_latency_ms_last"] = planner_latency_ms_last
        self._last_info = enriched_info
        # Do NOT force a decision just because a pill exists; only on spawn/replan.
        if "placements/needs_action" not in enriched_info:
            enriched_info["placements/needs_action"] = False
        enriched_info.setdefault("placements/spawn_id", int(self._spawn_id))
        return last_obs, total_reward, terminated, truncated, enriched_info

    # ------------------------------------------------------------------
    # Viewer callback plumbing
    # ------------------------------------------------------------------

    def set_step_callback(
        self,
        callback: Optional[Callable[[Any, Dict[str, Any], Optional[int], float, bool, bool], None]],
    ) -> None:
        self._step_callback = callback

    def _notify_step_callback(
        self,
        obs: Any,
        info: Optional[Dict[str, Any]],
        action: Optional[int],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._step_callback is None:
            return
        payload = dict(info or {})
        try:
            self._step_callback(
                obs,
                payload,
                None if action is None else int(action),
                float(reward),
                bool(terminated),
                bool(truncated),
            )
        except Exception:
            pass

    def _clear_latch(self) -> None:
        self._active_plan = None
        self._latched_action = None
        self._latched_spawn_id = -1

    def _log(self, message: str) -> None:
        if self._path_log or getattr(self._translator, "_debug", False):
            try:
                print(f"{_ts()} [placement_debug] {message}", flush=True)
            except Exception:
                pass

    def _log_plan_details(self, action: int, plan: PlanResult) -> None:
        try:
            if self._path_log:
                path_repr = _format_compact_path(plan.states)
                ctrl_repr = _format_ctrl_sequence(plan.controller)
                print(
                    (
                        f"[placement_path] spawn={self._spawn_id} action={int(action)} "
                        f"steps={len(plan.controller)} ctrl={ctrl_repr} planned={path_repr}"
                    ),
                    flush=True,
                )
            if not getattr(self._translator, "_debug", False):
                return
            edge = PLACEMENT_EDGES[int(action)]
            pill = self._translator.current_pill()
            legal_n = int(self._legal_mask.sum()) if hasattr(self, "_legal_mask") else -1
            feas_n = int(self._feasible_mask.sum()) if hasattr(self, "_feasible_mask") else -1
            print(
                (
                    f"[planner] spawn={self._spawn_id} action={int(action)} edge={edge.origin}->{edge.dest} dir={edge.direction} "
                    f"plans={len(self._paths) if hasattr(self, '_paths') else -1} legal={legal_n} feasible={feas_n}"
                ),
                flush=True,
            )
            if isinstance(getattr(plan, 'bfs_spawn', None), tuple):
                try:
                    sx, sy, so = plan.bfs_spawn  # type: ignore[attr-defined]
                    print(f"[planner] bfs_spawn(x={int(sx)}, y={int(sy)}, o={int(so)})", flush=True)
                except Exception:
                    pass
            if pill is not None:
                print(
                    (
                        f"[planner] pill(row={pill.row}, col={pill.col}, orient={pill.orient}, colors={pill.colors}, "
                        f"speed={pill.gravity_counter}/{pill.gravity_period}, vel={pill.hor_velocity}, "
                        f"parity={pill.frame_parity}, holds(L,R,D)=({int(pill.hold_left)},{int(pill.hold_right)},{int(pill.hold_down)}), "
                        f"lock={pill.lock_counter})"
                    ),
                    flush=True,
                )
            if plan.states:
                s0 = plan.states[0]
                print(
                    f"[planner] route: steps={len(plan.controller)} cost={int(plan.cost)} start=(r={s0.row},c={s0.col},o={s0.orient})",
                    flush=True,
                )
                if pill is not None and (pill.row != s0.row or pill.col != s0.col or pill.orient != s0.orient):
                    print(
                        f"[planner] WARN: route start != pill: start=(r={s0.row},c={s0.col},o={s0.orient}) vs pill=(r={pill.row},c={pill.col},o={pill.orient})",
                        flush=True,
                    )
                    try:
                        board = self._board
                        if board is not None:
                            occ = _board_occupancy(board)
                            print("[planner] board:", flush=True)
                            for r in range(GRID_HEIGHT):
                                line = ''.join('#' if occ[r, c] else '.' for c in range(GRID_WIDTH))
                                print("[planner] ", line, flush=True)
                    except Exception:
                        pass
                max_steps = min(16, len(plan.controller))
                for i in range(max_steps):
                    st = plan.states[min(i, len(plan.states) - 1)]
                    nxt = plan.states[min(i + 1, len(plan.states) - 1)] if plan.states else st
                    ctrl = plan.controller[i]
                    print(
                        (
                            f"[planner] step[{i}] act={int(ctrl.action)} holds(L,R,D)=({int(bool(ctrl.hold_left))},"
                            f"{int(bool(ctrl.hold_right))},{int(bool(ctrl.hold_down))}) "
                            f"from(r={st.row},c={st.col},o={st.orient},speed={st.speed_counter}/{st.speed_threshold},"
                            f"vel={st.hor_velocity}) -> (r={nxt.row},c={nxt.col},o={nxt.orient},"
                            f"speed={nxt.speed_counter}/{nxt.speed_threshold},vel={nxt.hor_velocity},"
                            f"holds={int(nxt.hold_left)}/{int(nxt.hold_right)}/{int(nxt.hold_down)},lock={int(nxt.locked)})"
                        ),
                        flush=True,
                    )
                if len(plan.controller) > max_steps:
                    print(f"[planner] ... ({len(plan.controller)-max_steps} more steps)", flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_plan(
        self,
        plan: Optional[PlanResult],
        record_refresh: Optional[Callable[[], None]] = None,
    ) -> _ExecutionOutcome:
        base = self.env.unwrapped
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._last_obs
        last_info: Dict[str, Any] = {}
        replan_required = False
        planner_resynced = False
        spawn_id_at_start = int(self._spawn_id)
        actual_path_states: List[Optional[CapsuleState]] = []
        trace_events: List[Tuple[int, _ConsistencyStatus]] = []
        failure_detail: Optional[
            Tuple[int, int, Optional[CapsuleState], Optional[CapsuleState], _ConsistencyStatus]
        ] = None
        failure_ctrl: Optional[ControllerStep] = None

        if plan is None or not plan.controller:
            # No executable plan for the requested action. Advance by one NOOP frame
            # and explicitly request a new decision so the runner can reselect.
            if self._debug:
                try:
                    print(f"{_ts()} [timing] pre_env_step no-plan noop", flush=True)
                except Exception:
                    pass
            t0 = perf_counter()
            obs, reward, terminated, truncated, info = self.env.step(int(Action.NOOP))
            t1 = perf_counter()
            total_reward += float(reward)
            last_obs = obs
            last_info = dict(info or {})
            if not (terminated or truncated):
                r0 = perf_counter()
                self._translator.refresh(state=self.env.unwrapped._state_cache)
                r1 = perf_counter()
                if record_refresh is not None:
                    record_refresh()
                extra_info = self._translator.info() or {}
                if extra_info:
                    last_info.update(extra_info)
                if self._step_callback is not None:
                    info_for_cb = dict(last_info)
                    self._notify_step_callback(
                        obs,
                        info_for_cb,
                        int(Action.NOOP),
                        float(reward),
                        terminated,
                        truncated,
                    )
                if self._translator.current_pill() is None:
                    last_obs, last_info, extra_reward, terminated, truncated = (
                        self._await_next_pill(last_obs, last_info, record_refresh)
                    )
                    total_reward += extra_reward
                else:
                    # Surface fresh options and mark that a new decision is needed.
                    try:
                        p0 = perf_counter()
                        self._translator.prepare_options(force=False)
                        p1 = perf_counter()
                    except Exception:
                        pass
                    last_info["placements/needs_action"] = True
                    last_info.setdefault("pill_changed", 0)
                if self._debug:
                    try:
                        msg = (
                            f"{_ts()} [timing] exec:no-plan env_step_ms={(t1-t0)*1000:.3f} refresh_ms={(r1-r0)*1000:.3f}"
                        )
                        if 'p0' in locals():
                            msg += f" prepare_ms={(p1-p0)*1000:.3f}"
                        print(msg, flush=True)
                    except Exception:
                        pass
            self._notify_step_callback(
                last_obs,
                last_info,
                int(Action.NOOP),
                float(reward),
                terminated,
                truncated,
            )
            base._hold_left = False
            base._hold_right = False
            base._hold_down = False
            last_info.setdefault("pill_changed", 0)
            return _ExecutionOutcome(
                last_obs,
                last_info,
                total_reward,
                terminated,
                truncated,
                True,
                False,
            )

        if self._path_log:
            initial_snapshot = self._translator.current_pill()
            initial_state = (
                snapshot_to_capsule_state(initial_snapshot)
                if initial_snapshot is not None
                else None
            )
            actual_path_states.append(initial_state)

        states = plan.states
        idx = 0
        refreshed_after_frame = False
        internal_resyncs = 0
        latched_action_val = int(self._latched_action) if self._latched_action is not None else int(plan.action)
        match_complete = False
        while idx < len(plan.controller):
            ctrl = plan.controller[idx]
            retry_budget = self._translator.retry_budget(ctrl)
            frames_for_step = 0
            restart_step = False
            previous_state = states[idx]
            expected_state = states[min(idx + 1, len(states) - 1)]
            while True:
                frames_for_step += 1
                base._hold_left = bool(ctrl.hold_left)
                base._hold_right = bool(ctrl.hold_right)
                base._hold_down = bool(ctrl.hold_down)
                if self._debug:
                    try:
                        print(
                            f"{_ts()} [timing] pre_env_step exec step={idx} action={int(ctrl.action)} holds=({int(ctrl.hold_left)},{int(ctrl.hold_right)},{int(ctrl.hold_down)})",
                            flush=True,
                        )
                    except Exception:
                        pass
                commanded_action = int(ctrl.action)
                if frames_for_step > 1 and ctrl.action in (Action.ROTATE_A, Action.ROTATE_B):
                    commanded_action = int(Action.NOOP)
                s0 = perf_counter()
                obs, reward, terminated, truncated, info = self.env.step(commanded_action)
                s1 = perf_counter()
                total_reward += float(reward)
                last_obs = obs
                last_info = info or {}
                info_for_cb = dict(last_info)
                # Expose current controller step and holds for viewer overlay
                try:
                    info_for_cb["placements/exec_step"] = int(idx)
                    info_for_cb["placements/exec_total"] = int(len(plan.controller))
                    info_for_cb["placements/ctrl_action"] = int(ctrl.action)
                    info_for_cb["placements/ctrl_action_effective"] = int(commanded_action)
                    info_for_cb["placements/ctrl_left"] = int(bool(ctrl.hold_left))
                    info_for_cb["placements/ctrl_right"] = int(bool(ctrl.hold_right))
                    info_for_cb["placements/ctrl_down"] = int(bool(ctrl.hold_down))
                except Exception:
                    pass
                r0 = perf_counter()
                self._translator.refresh(state=self.env.unwrapped._state_cache)
                r1 = perf_counter()
                if record_refresh is not None:
                    record_refresh()
                extra_info = self._translator.info() or {}
                if self._step_callback is not None:
                    cb_payload = dict(info_for_cb)
                    if extra_info:
                        cb_payload.update(extra_info)
                    self._notify_step_callback(
                        obs,
                        cb_payload,
                        int(ctrl.action),
                        float(reward),
                        terminated,
                        truncated,
                    )
                if extra_info:
                    info_for_cb.update(extra_info)
                if terminated or truncated:
                    break
                actual_snapshot = self._translator.current_pill()
                actual_state = (
                    snapshot_to_capsule_state(actual_snapshot)
                    if actual_snapshot is not None
                    else None
                )
                if self._path_log:
                    actual_path_states.append(actual_state)
                # Compare against the expected next frame but accept near-lock tolerance for rotations:
                status = self._translator.consistency_status(actual_state, expected_state, previous_state, ctrl)
                trace_events.append((idx, status))
                if self._debug:
                    try:
                        print(
                            (
                                f"{_ts()} [exec_trace] step={idx} frame_attempt={frames_for_step} "
                                f"ctrl={int(ctrl.action)} holds=({int(ctrl.hold_left)},{int(ctrl.hold_right)},{int(ctrl.hold_down)}) "
                                f"expected={_format_capsule_state(expected_state)} actual={_format_capsule_state(actual_state)} "
                                f"status={status.value}"
                            ),
                            flush=True,
                        )
                    except Exception:
                        pass
                if status is _ConsistencyStatus.MATCH:
                    if frames_for_step > 0:
                        self._translator.record_timing_success(ctrl, frames_for_step)
                    if self._debug:
                        try:
                            print(
                                (
                                    f"{_ts()} [timing] exec step={idx} env_step_ms={(s1-s0)*1000:.3f} "
                                    f"refresh_ms={(r1-r0)*1000:.3f}"
                                ),
                                flush=True,
                            )
                        except Exception:
                            pass
                    break
                if status is _ConsistencyStatus.PROGRESS:
                    if frames_for_step < self._translator.max_step_frames():
                        if frames_for_step == 1:
                            self._log(
                                f"progress_wait spawn={self._spawn_id} step={idx} frames={frames_for_step} action={int(ctrl.action)} actual={_format_compact_state(actual_state)} expected={_format_compact_state(expected_state)}"
                            )
                        continue
                    self._log(
                        f"progress_timeout spawn={self._spawn_id} step={idx} frames={frames_for_step} action={int(ctrl.action)} actual={_format_compact_state(actual_state)} expected={_format_compact_state(expected_state)}"
                    )
                    if internal_resyncs >= self._translator.internal_resync_limit():
                        if self._path_log and failure_detail is None:
                            failure_detail = (
                                idx,
                                frames_for_step,
                                expected_state,
                                actual_state,
                                status,
                            )
                            failure_ctrl = ctrl
                        self._translator.record_timing_failure(ctrl)
                        replan_required = True
                        break
                    new_plan = self._translator.replan_to_action(latched_action_val)
                    if new_plan is not None and new_plan.controller:
                        resume_idx = self._translator.resume_index_for(new_plan, actual_state)
                        if resume_idx >= len(new_plan.controller):
                            plan = new_plan
                            states = plan.states
                            idx = len(plan.controller)
                            if frames_for_step > 0:
                                self._translator.record_timing_success(ctrl, frames_for_step)
                            match_complete = True
                            break
                        plan = new_plan
                        states = plan.states
                        idx = max(0, resume_idx)
                        internal_resyncs += 1
                        self._log(
                            f"progress_resync spawn={self._spawn_id} step={idx} resume_idx={resume_idx} action={int(ctrl.action)} resyncs={internal_resyncs}"
                        )
                        restart_step = True
                        break
                    if self._path_log and failure_detail is None:
                        failure_detail = (
                            idx,
                            frames_for_step,
                            expected_state,
                            actual_state,
                            status,
                        )
                        failure_ctrl = ctrl
                    self._translator.record_timing_failure(ctrl)
                    replan_required = True
                    break
                if status is _ConsistencyStatus.STALLED and retry_budget > 0:
                    retry_budget -= 1
                    continue
                # Diverged: attempt an immediate internal resync before giving up
                new_plan = self._translator.replan_to_action(latched_action_val)
                if new_plan is not None and new_plan.controller:
                    if internal_resyncs >= self._translator.internal_resync_limit():
                        new_plan = None
                    else:
                        resume_idx = self._translator.resume_index_for(new_plan, actual_state)
                        if resume_idx >= len(new_plan.controller):
                            plan = new_plan
                            states = plan.states
                            idx = len(plan.controller)
                            if frames_for_step > 0:
                                self._translator.record_timing_success(ctrl, frames_for_step)
                            match_complete = True
                            break
                        plan = new_plan
                        states = plan.states
                        idx = max(0, resume_idx)
                        internal_resyncs += 1
                        restart_step = True
                        self._log(
                            f"diverge_resync spawn={self._spawn_id} step={idx} resume_idx={resume_idx} action={int(ctrl.action)} resyncs={internal_resyncs}"
                        )
                        break
                if self._path_log and failure_detail is None:
                    failure_detail = (
                        idx,
                        frames_for_step,
                        expected_state,
                        actual_state,
                        status,
                    )
                    failure_ctrl = ctrl
                self._translator.record_timing_failure(ctrl)
                replan_required = True
                break
            if terminated or truncated or replan_required or match_complete:
                break
            if restart_step:
                continue
            idx += 1

        base._hold_left = False
        base._hold_right = False
        base._hold_down = False

        if not (terminated or truncated) and not replan_required:
            if not refreshed_after_frame:
                r0 = perf_counter()
                self._translator.refresh(state=self.env.unwrapped._state_cache)
                r1 = perf_counter()
                if record_refresh is not None:
                    record_refresh()
            else:
                r0 = r1 = 0.0
            # Wait until the current capsule locks before surfacing the next spawn.
            stall_checks = 0
            while self._translator.current_pill() is not None:
                if self._debug:
                    try:
                        print(f"{_ts()} [timing] pre_env_step stall noop", flush=True)
                    except Exception:
                        pass
                n0 = perf_counter()
                obs, reward, terminated, truncated, info = self.env.step(int(Action.NOOP))
                n1 = perf_counter()
                total_reward += float(reward)
                last_obs = obs
                last_info = info or {}
                r0 = perf_counter()
                self._translator.refresh(state=self.env.unwrapped._state_cache)
                r1 = perf_counter()
                if record_refresh is not None:
                    record_refresh()
                if self._step_callback is not None:
                    info_for_cb = dict(last_info)
                    extra = self._translator.info() or {}
                    if extra:
                        info_for_cb.update(extra)
                    self._notify_step_callback(
                        obs,
                        info_for_cb,
                        int(Action.NOOP),
                        float(reward),
                        terminated,
                        truncated,
                    )
                if terminated or truncated:
                    break
                if self._debug:
                    try:
                        print(
                            (
                                f"{_ts()} [timing] exec:post-plan noop_ms={(n1-n0)*1000:.3f} "
                                f"refresh_ms={(r1-r0)*1000:.3f}"
                            ),
                            flush=True,
                        )
                    except Exception:
                        pass
                stall_checks += 1
                if stall_checks > 180 and self._translator.current_pill() is not None:
                    replan_required = True
                    break

            if not (terminated or truncated) and not replan_required:
                last_obs, last_info, extra_reward, terminated, truncated = self._await_next_pill(
                    last_obs, last_info, record_refresh
                )
                total_reward += extra_reward

        if replan_required:
            last_info = dict(last_info)
            last_info["placements/replan_triggered"] = 1
            if not (terminated or truncated):
                # Resync and expose a fresh decision request so the runner reselects immediately.
                self._translator.refresh(state=self.env.unwrapped._state_cache)
                # Rebuilding options with force=True ensures we fall back to the full planner
                # after a failed fast-path execution.
                self._translator.prepare_options(force=True)
                planner_resynced = True
                if record_refresh is not None:
                    record_refresh()
                extra = self._translator.info() or {}
                if extra:
                    last_info.update(extra)
                last_info["placements/needs_action"] = True

        if "pill_changed" not in last_info:
            last_info = dict(last_info)
            last_info.setdefault("pill_changed", 0)

        if plan is not None and self._path_log:
            if terminated:
                status = "terminated"
            elif truncated:
                status = "truncated"
            elif replan_required:
                status = "replan"
            else:
                status = "success"
            actual_repr = _format_compact_path(actual_path_states)
            trace_repr = _format_trace(trace_events)
            extras: List[str] = []
            if trace_repr != "-":
                extras.append(f"trace={trace_repr}")
            if failure_detail is not None:
                fail_step, fail_frames, fail_expected, fail_actual, fail_status = failure_detail
                extras.append(f"fail_step={fail_step}")
                extras.append(f"attempt_frames={fail_frames}")
                extras.append(f"fail_status={fail_status.value}")
                extras.append(f"expected={_format_compact_state(fail_expected)}")
                extras.append(f"actual={_format_compact_state(fail_actual)}")
                if failure_ctrl is not None:
                    extras.append(f"ctrl={_format_ctrl_token(failure_ctrl)}")
            if planner_resynced:
                extras.append("resync=1")
            if internal_resyncs > 0:
                extras.append(f"ireplan={internal_resyncs}")
            extra_suffix = f" {' '.join(extras)}" if extras else ""
            print(
                (
                    f"[placement_path] spawn={spawn_id_at_start} action={int(plan.action)} "
                    f"status={status} observed={actual_repr}{extra_suffix}"
                ),
                flush=True,
            )

        return _ExecutionOutcome(
            last_obs,
            last_info,
            total_reward,
            terminated,
            truncated,
            replan_required,
            planner_resynced,
        )

    def _await_next_pill(
        self,
        last_obs: Any,
        last_info: Dict[str, Any],
        record_refresh: Optional[Callable[[], None]] = None,
    ) -> Tuple[Any, Dict[str, Any], float, bool, bool]:
        """Advance the emulator until a new pill snapshot becomes available."""

        total_reward = 0.0
        terminated = False
        truncated = False
        obs = last_obs
        info = dict(last_info) if last_info is not None else {}

        if self._translator.current_pill() is not None:
            if not self._capsule_present:
                self._spawn_id += 1
            self._capsule_present = True
            self._spawn_marker = None
            # For fast options mode, avoid forcing full enumeration at spawn.
            self._translator.prepare_options(force=not bool(getattr(self._translator, "_fast_options_only", False)))
            if record_refresh is not None:
                record_refresh()
            try:
                self._spawn_marker = int((self._translator.info() or {}).get(
                    "placements/spawn_id", getattr(self._translator.current_pill(), "spawn_id", None)
                ))
            except Exception:
                pass
            info.update(self._translator.info() or {})
            info["placements/spawn_id"] = int(self._spawn_id)
            info["placements/needs_action"] = True
            info.setdefault("pill_changed", 1)
            return obs, info, total_reward, terminated, truncated

        self._capsule_present = False
        while True:
            n0 = perf_counter()
            obs, reward, terminated, truncated, step_info = self.env.step(int(Action.NOOP))
            n1 = perf_counter()
            total_reward += float(reward)
            info = step_info or {}
            if terminated or truncated:
                self._notify_step_callback(
                    obs,
                    info,
                    int(Action.NOOP),
                    float(reward),
                    terminated,
                    truncated,
                )
                break
            r0 = perf_counter()
            self._translator.refresh(state=self.env.unwrapped._state_cache)
            r1 = perf_counter()
            if record_refresh is not None:
                record_refresh()
            if self._step_callback is not None:
                info = dict(info)
                info.update(self._translator.info() or {})
                self._notify_step_callback(
                    obs,
                    info,
                    int(Action.NOOP),
                    float(reward),
                    terminated,
                    truncated,
                )
            if self._debug:
                try:
                    print(
                        (
                            f"{_ts()} [timing] await_next noop_ms={(n1-n0)*1000:.3f} refresh_ms={(r1-r0)*1000:.3f}"
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
            if self._translator.current_pill() is not None:
                info = dict(info)
                self._translator.prepare_options(force=True)
                if record_refresh is not None:
                    record_refresh()
                info.update(self._translator.info() or {})
                if not self._capsule_present:
                    self._spawn_id += 1
                self._capsule_present = True
                # Capture translator marker at the moment of spawn detection.
                self._spawn_marker = None
                try:
                    self._spawn_marker = int((self._translator.info() or {}).get(
                        "placements/spawn_id", getattr(self._translator.current_pill(), "spawn_id", None)
                    ))
                except Exception:
                    pass
                info["placements/spawn_id"] = int(self._spawn_id)
                info["placements/needs_action"] = True
                info.setdefault("pill_changed", 1)
                break

        info.setdefault("pill_changed", 0)
        return obs, info, total_reward, terminated, truncated


def _board_occupancy(board: BoardState) -> np.ndarray:
    occupancy = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
    for col in range(GRID_WIDTH):
        mask = int(board.columns[col])
        if mask == 0:
            continue
        for row in range(GRID_HEIGHT):
            if mask & (1 << row):
                occupancy[row, col] = True
    return occupancy


def _blend(base: np.ndarray, color: Tuple[float, float, float], alpha: float) -> np.ndarray:
    return (1.0 - alpha) * base + alpha * np.asarray(color, dtype=np.float32)


def _format_capsule_state(state: Optional[CapsuleState]) -> str:
    if state is None:
        return "None"
    return (
        f"(row={int(state.row)}, col={int(state.col)}, orient={int(state.orient)}, "
        f"speed={int(state.speed_counter)}/{int(state.speed_threshold)}, "
        f"vel={int(state.hor_velocity)}, holds=({int(state.hold_left)},"
        f"{int(state.hold_right)},{int(state.hold_down)}), locked={int(bool(state.locked))})"
    )


def _format_compact_state(state: Optional[CapsuleState]) -> str:
    if state is None:
        return "None"
    holds = (int(bool(state.hold_left)) << 2) | (int(bool(state.hold_right)) << 1) | int(bool(state.hold_down))
    suffix = "K" if bool(state.locked) else ""
    return (
        f"r{int(state.row):02d}c{int(state.col):02d}o{int(state.orient)}"
        f"s{int(state.speed_counter):02d}v{int(state.hor_velocity):+d}h{holds}{suffix}"
    )


def _format_compact_path(states: Sequence[Optional[CapsuleState]], *, max_segments: int = 32) -> str:
    if not states:
        return "-"
    tokens = [_format_compact_state(state) for state in states]
    segments: List[Tuple[str, int]] = []
    current_token: Optional[str] = None
    run_length = 0
    for token in tokens:
        if token == current_token:
            run_length += 1
        else:
            if current_token is not None:
                segments.append((current_token, run_length))
            current_token = token
            run_length = 1
    if current_token is not None:
        segments.append((current_token, run_length))
    shown = segments[:max_segments]
    remainder = segments[max_segments:]
    entries = [
        f"{token}x{count}" if count > 1 else token
        for token, count in shown
    ]
    if remainder:
        leftover = sum(count for _, count in remainder)
        entries.append(f"...(+{leftover})")
    return ">".join(entries)


def _format_ctrl_token(ctrl: ControllerStep) -> str:
    base_map = {
        int(Action.NOOP): "·",
        int(Action.LEFT): "<",
        int(Action.RIGHT): ">",
        int(Action.DOWN): "v",
        int(Action.ROTATE_A): "A",
        int(Action.ROTATE_B): "B",
        int(Action.LEFT_HOLD): "<",
        int(Action.RIGHT_HOLD): ">",
        int(Action.DOWN_HOLD): "v",
        int(Action.BOTH_ROT): "*",
    }
    base = base_map.get(int(ctrl.action), str(int(ctrl.action)))
    holds = []
    if ctrl.hold_left:
        holds.append("<")
    if ctrl.hold_right:
        holds.append(">")
    if ctrl.hold_down:
        holds.append("v")
    if holds:
        return f"{base}[{''.join(holds)}]"
    return base


def _format_ctrl_sequence(controller: Sequence[ControllerStep], *, max_segments: int = 24) -> str:
    if not controller:
        return "-"
    tokens = [_format_ctrl_token(ctrl) for ctrl in controller]
    segments: List[Tuple[str, int]] = []
    current_token: Optional[str] = None
    run_length = 0
    for token in tokens:
        if token == current_token:
            run_length += 1
        else:
            if current_token is not None:
                segments.append((current_token, run_length))
            current_token = token
            run_length = 1
    if current_token is not None:
        segments.append((current_token, run_length))
    shown = segments[:max_segments]
    remainder = segments[max_segments:]
    entries = [
        f"{token}x{count}" if count > 1 else token
        for token, count in shown
    ]
    if remainder:
        leftover = sum(count for _, count in remainder)
        entries.append(f"...(+{leftover})")
    return ",".join(entries)


_TRACE_SYMBOLS = {
    _ConsistencyStatus.MATCH: "M",
    _ConsistencyStatus.STALLED: "S",
    _ConsistencyStatus.PROGRESS: "P",
    _ConsistencyStatus.DIVERGED: "X",
}


def _format_trace(events: Sequence[Tuple[int, _ConsistencyStatus]], *, max_segments: int = 32) -> str:
    if not events:
        return "-"
    segments: List[Tuple[Tuple[int, _ConsistencyStatus], int]] = []
    current: Optional[Tuple[int, _ConsistencyStatus]] = None
    run_length = 0
    for step_idx, status in events:
        token = (step_idx, status)
        if token == current:
            run_length += 1
        else:
            if current is not None:
                segments.append((current, run_length))
            current = token
            run_length = 1
    if current is not None:
        segments.append((current, run_length))
    shown = segments[:max_segments]
    remainder = segments[max_segments:]
    def _format_segment(segment: Tuple[Tuple[int, _ConsistencyStatus], int]) -> str:
        (step_idx, status), count = segment
        symbol = _TRACE_SYMBOLS.get(status, "?")
        return f"{step_idx}:{symbol}x{count}" if count > 1 else f"{step_idx}:{symbol}"

    entries = [_format_segment(seg) for seg in shown]
    if remainder:
        leftover = sum(length for _, length in remainder)
        entries.append(f"...(+{leftover})")
    return ">".join(entries)


def _cells_from_state(state: CapsuleState) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for row, col in iter_cells(state.row, state.col, state.orient):
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            cells.append((row, col))
    return cells


_BACKGROUND_COLOR = np.array((24.0, 24.0, 32.0), dtype=np.float32)
_OCCUPIED_FALLBACK_COLOR = np.array((118.0, 118.0, 118.0), dtype=np.float32)
_STATIC_RGB = np.array(((180.0, 0.0, 0.0), (200.0, 180.0, 0.0), (0.0, 80.0, 200.0)), dtype=np.float32)
_VIRUS_RGB = np.array(((220.0, 40.0, 40.0), (240.0, 220.0, 40.0), (40.0, 120.0, 240.0)), dtype=np.float32)
_FALLING_RGB = np.array(((255.0, 128.0, 128.0), (255.0, 255.0, 120.0), (120.0, 120.0, 255.0)), dtype=np.float32)
_FALLING_COLOR_LOOKUP = {
    0: _FALLING_RGB[1],  # yellow
    1: _FALLING_RGB[0],  # red
    2: _FALLING_RGB[2],  # blue
}
_LEGAL_ONLY_COLOR = (150.0, 235.0, 170.0)
_FEASIBLE_COLOR = (40.0, 160.0, 90.0)
_PATH_COLOR = (255.0, 160.0, 40.0)
_TARGET_COLOR = (250.0, 250.0, 250.0)
_CELL_SCALE = 6


def _board_color_grid(state: Optional[np.ndarray]) -> np.ndarray:
    colors = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
    if state is None:
        return colors
    try:
        static_planes = ram_specs.get_static_color_planes(state)
        virus_planes = ram_specs.get_virus_color_planes(state)
    except Exception:
        return colors
    for idx, rgb in enumerate(_STATIC_RGB):
        if idx >= static_planes.shape[0]:
            break
        mask = static_planes[idx] > 0.1
        if mask.any():
            colors[mask] = rgb
    for idx, rgb in enumerate(_VIRUS_RGB):
        if idx >= virus_planes.shape[0]:
            break
        mask = virus_planes[idx] > 0.1
        if mask.any():
            colors[mask] = rgb
    return colors


def _apply_falling_colors(canvas: np.ndarray, state: Optional[np.ndarray]) -> None:
    if state is None:
        return
    try:
        falling_planes = ram_specs.get_falling_color_planes(state)
    except Exception:
        return
    for idx, rgb in enumerate(_FALLING_RGB):
        if idx >= falling_planes.shape[0]:
            break
        mask = falling_planes[idx] > 0.1
        if mask.any():
            canvas[mask] = rgb


def render_planner_debug_view(
    snapshot: PlannerDebugSnapshot,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    board_mask = _board_occupancy(snapshot.board)
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
    canvas[:] = _BACKGROUND_COLOR

    board_colors = _board_color_grid(snapshot.state)
    colored_mask = board_colors.sum(axis=-1) > 0.1
    if colored_mask.any():
        canvas[colored_mask] = board_colors[colored_mask]
    fallback_mask = np.logical_and(board_mask, np.logical_not(colored_mask))
    if fallback_mask.any():
        canvas[fallback_mask] = _OCCUPIED_FALLBACK_COLOR

    legal_cells = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
    feasible_cells = np.zeros_like(legal_cells)
    for edge in PLACEMENT_EDGES:
        if edge.index >= snapshot.legal_mask.size:
            continue
        row, col = edge.origin
        if not (0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH):
            continue
        if snapshot.legal_mask[edge.index]:
            legal_cells[row, col] += 1
        if snapshot.feasible_mask[edge.index]:
            feasible_cells[row, col] += 1

    legal_mask = legal_cells > 0
    feasible_mask = feasible_cells > 0
    legal_only_mask = np.logical_and(legal_mask, np.logical_not(feasible_mask))
    if legal_only_mask.any():
        canvas[legal_only_mask] = _blend(canvas[legal_only_mask], _LEGAL_ONLY_COLOR, 0.5)
    if feasible_mask.any():
        canvas[feasible_mask] = _blend(canvas[feasible_mask], _FEASIBLE_COLOR, 0.55)

    plan_to_show: Optional[PlanResult] = snapshot.selected_plan
    plan_action: Optional[int] = snapshot.selected_action
    # Do not fall back to an unrelated plan; only draw the selected route to avoid confusion.

    final_cells: List[Tuple[int, int]] = []
    if plan_to_show is not None:
        path_cells = set()
        for state in plan_to_show.states:
            path_cells.update(_cells_from_state(state))
        if plan_to_show.states:
            final_cells = _cells_from_state(plan_to_show.states[-1])
        for row, col in path_cells:
            canvas[row, col] = _blend(canvas[row, col], _PATH_COLOR, 0.65)

    _apply_falling_colors(canvas, snapshot.state)
    if snapshot.state is None and snapshot.pill is not None:
        pill_state = snapshot_to_capsule_state(snapshot.pill)
        pill_cells = _cells_from_state(pill_state)
        for index, (row, col) in enumerate(pill_cells):
            color_value = snapshot.pill.colors[min(index, len(snapshot.pill.colors) - 1)]
            tint = _FALLING_COLOR_LOOKUP.get(int(color_value))
            if tint is None:
                continue
            canvas[row, col] = tint

    if final_cells:
        for row, col in final_cells:
            canvas[row, col] = _TARGET_COLOR

    enlarged = np.repeat(np.repeat(canvas, _CELL_SCALE, axis=0), _CELL_SCALE, axis=1)
    image = np.clip(enlarged, 0.0, 255.0).astype(np.uint8)

    stats: Dict[str, Any] = {
        "planner_debug": {
            "legal_count": int(snapshot.legal_mask.sum()),
            "feasible_count": int(snapshot.feasible_mask.sum()),
            "selected_action": None if plan_action is None else int(plan_action),
            "plan_cost": None if plan_to_show is None else int(plan_to_show.cost),
            "selected_has_plan": 1 if plan_to_show is not None else 0,
            "displayed_action": None if plan_action is None else int(plan_action),
            "plan_steps": None
            if plan_to_show is None
            else int(len(plan_to_show.controller)),
        }
    }
    if snapshot.pill is not None:
        stats["planner_debug"]["pill_row"] = int(snapshot.pill.row)
        stats["planner_debug"]["pill_col"] = int(snapshot.pill.col)
        stats["planner_debug"]["pill_orient"] = int(snapshot.pill.orient)
        stats["planner_debug"]["pill_spawn_id"] = (
            None if snapshot.pill.spawn_id is None else int(snapshot.pill.spawn_id)
        )

    return image, stats


__all__ = [
    "DrMarioPlacementEnv",
    "PlannerDebugSnapshot",
    "render_planner_debug_view",
]
