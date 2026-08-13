"""Host-neutral out-of-process human player and coach backend."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from drmc_rl.human.coach import analyze_choice
from drmc_rl.human.model import canonicalize_same_color_action
from drmc_rl.human.runtime import HumanPolicyRuntime
from drmc_rl.human.search import (
    HumanValueSearch,
    blend_human_and_search,
    competitive_scores,
)
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.annotate_replay_events import POSE_TO_ACTION

PROTOCOL_SCHEMA = "drmc-human-backend-v1"
GRID_H, GRID_W = 16, 8


def _action_to_pose() -> np.ndarray:
    result = np.full(512, -1, dtype=np.int32)
    for pose, action in enumerate(POSE_TO_ACTION):
        if action >= 0:
            result[action] = pose
    return result


ACTION_TO_POSE = _action_to_pose()


def _buttons_table() -> tuple[int, ...]:
    result = []
    for action in range(18):
        direction, sub = divmod(action, 6)
        rotation = sub % 3
        mask = 0x02 if direction == 1 else 0x01 if direction == 2 else 0
        if sub >= 3:
            mask |= 0x04
        if rotation == 1:
            mask |= 0x80
        elif rotation == 2:
            mask |= 0x40
        result.append(mask)
    return tuple(result)


ACTION_TO_BUTTONS = _buttons_table()


def _board_planes(value: Any) -> np.ndarray:
    planes = np.asarray(value, dtype=np.float32)
    if planes.shape != (8, GRID_H, GRID_W):
        raise ValueError(f"state.board_planes must be [8,16,8], got {planes.shape}")
    if not np.isfinite(planes).all() or ((planes < 0) | (planes > 1)).any():
        raise ValueError("state.board_planes values must be in [0,1]")
    if (planes[:3].sum(axis=0) > 1.0).any():
        raise ValueError("a board cell cannot have multiple colors")
    return planes


def _columns(planes: np.ndarray) -> np.ndarray:
    occupied = planes[:3].sum(axis=0) > 0
    columns = np.zeros(GRID_W, dtype=np.uint16)
    for row in range(GRID_H):
        columns |= occupied[row].astype(np.uint16) << row
    return columns


def _pair(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != (2,) or ((result < 0) | (result > 2)).any():
        raise ValueError(f"state.{name} must be two canonical colors in [0,2]")
    return result


class HumanBackend:
    """Synchronous worker intended to be supervised off the gameplay thread."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        seed: int = 0,
        max_frames: int = 2048,
    ):
        started = time.perf_counter()
        self.runtime = HumanPolicyRuntime(checkpoint, device=device, seed=seed)
        self.device = device
        self.seed = int(seed)
        self.search: HumanValueSearch | None = None
        self.planner = NativeReachabilityRunner(max_frames=max_frames)
        self.ready = True
        self.started_at = time.time()
        self.load_ms = (time.perf_counter() - started) * 1e3
        self.requests = 0
        self.errors = 0
        self.cancelled: set[int] = set()
        self.last_request_id = -1
        self.latest_frame_id = -1
        self.latencies_ms: list[float] = []

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "request_types": ["hello", "health", "decide", "coach", "cancel", "shutdown"],
            "modes": ["play", "coach"],
            "state": {
                "board_planes": "8x16x8 canonical color/virus/connectivity planes",
                "colors": "0=red, 1=yellow, 2=blue",
                "coordinates": "planner coordinates: row 0 is bottle top",
            },
            "outputs": ["placement", "controller_frames", "timing", "coach_analysis"],
            "search": {
                "available": True,
                "default_for_coach": True,
                "play_control": "search_weight >= 0; zero is pure human imitation",
            },
            "cancellation": "cooperative between requests; hosts must discard stale frame_ids",
            "model": self.runtime.identity,
        }

    def health(self) -> dict[str, Any]:
        latencies = np.asarray(self.latencies_ms[-256:], dtype=np.float64)
        return {
            "ready": self.ready,
            "uptime_s": time.time() - self.started_at,
            "load_ms": self.load_ms,
            "requests": self.requests,
            "errors": self.errors,
            "latency_ms": {
                "last": None if latencies.size == 0 else float(latencies[-1]),
                "p50": None if latencies.size == 0 else float(np.percentile(latencies, 50)),
                "p95": None if latencies.size == 0 else float(np.percentile(latencies, 95)),
            },
            "model": self.runtime.identity,
        }

    def _candidates(self, state: Mapping[str, Any]):
        planes = _board_planes(state["board_planes"])
        opponent_planes = _board_planes(state["opponent_board_planes"])
        pill = _pair(state["pill"], "pill")
        preview = _pair(state["preview"], "preview")
        falling = state.get("falling", {})
        hold = HoldDir(int(falling.get("hold_dir", 0)))
        rotation_hold = Rotation(int(falling.get("rotation_hold", 0)))
        frame = FrameState(
            x=int(falling.get("x", 3)),
            y=int(falling.get("y", 0)),
            rot=int(falling.get("rotation", 0)) & 3,
            speed_counter=int(falling.get("speed_counter", 0)),
            hor_velocity=int(falling.get("horizontal_velocity", 0)) & 0x0F,
            hold_dir=hold,
            frame_parity=int(falling.get("frame_parity", 0)) & 1,
            rot_hold=rotation_hold,
        )
        speed = int(state.get("speed", 2))
        speed_ups = int(state.get("speed_ups", 0))
        reach = self.planner.bfs_full(
            _columns(planes),
            frame,
            speed_threshold=compute_speed_threshold(speed, speed_ups),
        )
        costs = np.full(512, 0xFFFF, dtype=np.uint16)
        for pose in np.flatnonzero(reach.costs_u16 != 0xFFFF):
            action = int(POSE_TO_ACTION[pose])
            if action >= 0:
                costs[action] = reach.costs_u16[pose]
        if pill[0] == pill[1]:
            costs[256:] = 0xFFFF
        packed = pack_feasible_candidates(
            (costs != 0xFFFF).reshape(4, GRID_H, GRID_W),
            costs.reshape(4, GRID_H, GRID_W),
            max_candidates=128,
            sort_by_cost=True,
        )
        if packed.count == 0:
            raise RuntimeError("no reachable placement")
        return planes, opponent_planes, pill, preview, speed, speed_ups, reach, packed, costs

    def close(self) -> None:
        if self.search is not None:
            self.search.close()
            self.search = None
        self.planner.close()

    def _value_search(self) -> HumanValueSearch:
        if self.search is None:
            self.search = HumanValueSearch(
                self.runtime,
                device=self.device,
                seed=self.seed,
                gpu_planner=str(self.device).startswith("cuda"),
            )
        return self.search

    def _infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        state = request["state"]
        rating = float(request["target_rating"])
        temperature = float(request.get("temperature", 1.0))
        (
            planes,
            opponent_planes,
            pill,
            preview,
            speed,
            speed_ups,
            reach,
            packed,
            costs512,
        ) = self._candidates(state)
        rating_sd = float(request.get("target_rating_sd", 0.0))
        opponent_rating = state.get("opponent_rating")
        opponent_rating_sd = float(state.get("opponent_rating_sd", 0.0))
        game_phase = float(state.get("game_phase", 0.0))
        recent_decisions = state.get("recent_decisions", ())
        logits, state_value, resolved_rating, rating_clamped = self.runtime.score(
            board_planes=planes,
            opponent_board_planes=opponent_planes,
            opponent_state_age_frames=int(state.get("opponent_state_age_frames", 0)),
            rating_sd=rating_sd,
            opponent_rating=None if opponent_rating is None else float(opponent_rating),
            opponent_rating_sd=opponent_rating_sd,
            game_phase=game_phase,
            recent_decisions=recent_decisions,
            pill=pill,
            preview=preview,
            candidate_actions=packed.actions,
            candidate_costs=packed.cost,
            candidate_mask=packed.mask,
            rating=rating,
        )
        valid_actions = packed.actions[packed.mask]
        valid_logits = logits[packed.mask]
        search_info = None
        comp = None
        search_error = None
        search_weight = max(float(request.get("search_weight", 0.0)), 0.0)
        use_search = bool(request.get("search", request.get("type") == "coach")) or search_weight > 0
        if use_search:
            try:
                search_info = self._value_search().analyze(
                    board_planes=planes,
                    opponent_board_planes=opponent_planes,
                    pill=pill,
                    preview=preview,
                    feasible_mask512=costs512 != 0xFFFF,
                    cost_to_lock512=costs512,
                    speed=speed,
                    speed_ups=speed_ups,
                    level=int(state.get("level", 0)),
                    rating=resolved_rating,
                    rating_sd=rating_sd,
                    opponent_rating=None if opponent_rating is None else float(opponent_rating),
                    opponent_rating_sd=opponent_rating_sd,
                    opponent_state_age_frames=int(state.get("opponent_state_age_frames", 0)),
                    game_phase=game_phase,
                    recent_decisions=recent_decisions,
                    deadline_ms=float(request.get("search_deadline_ms", 100.0)),
                )
                comp = competitive_scores(valid_actions, search_info)
            except Exception as exc:
                if request.get("require_search"):
                    raise
                search_error = {"kind": type(exc).__name__, "message": str(exc)}
        decision_logits = (
            valid_logits
            if comp is None or search_weight <= 0
            else blend_human_and_search(valid_logits, comp, weight=search_weight)
        )
        slot = self.runtime.choose(
            decision_logits,
            np.ones(len(decision_logits), dtype=np.bool_),
            temperature=temperature,
        )
        packed_slot = int(np.flatnonzero(packed.mask)[slot])
        action = int(packed.actions[packed_slot])
        if pill[0] == pill[1]:
            action = canonicalize_same_color_action(action)
        pose_index = int(ACTION_TO_POSE[action])
        x, y, rotation = pose_index & 7, (pose_index >> 3) & 15, (pose_index >> 7) & 3
        script = reach.script_for_pose(x, y, rotation)
        if script is None:
            raise RuntimeError(f"planner returned no script for action {action}")
        timing = self.runtime.timing_prediction(
            board_planes=planes,
            rating=resolved_rating,
            rating_sd=rating_sd,
            opponent_rating=None if opponent_rating is None else float(opponent_rating),
            game_phase=game_phase,
            previous_tau_frames=float(
                recent_decisions[0].get("tau_frames", 0.0) if recent_decisions else 0.0
            ),
            chosen_cost=float(packed.cost[packed_slot]),
            speed=speed,
            speed_ups=speed_ups,
            candidate_count=packed.count,
        )
        result = {
            "target_rating": rating,
            "resolved_rating": resolved_rating,
            "rating_clamped": rating_clamped,
            "state_win_probability": float(1.0 / (1.0 + np.exp(-state_value))),
            "placement": {"action": action, "x": x, "y_top": y, "rotation": rotation},
            "controller_frames": [ACTION_TO_BUTTONS[int(value)] for value in script],
            "controller_encoding": "NES button mask: R=1 L=2 D=4 B=64 A=128",
            "timing": timing,
            "candidate_count": int(packed.count),
            "human_logits": valid_logits.tolist(),
            "candidate_actions": valid_actions.tolist(),
            "search": None
            if search_info is None
            else {
                key: value
                for key, value in search_info.items()
                if key
                in {
                    "action",
                    "stage",
                    "nodes_expanded",
                    "elapsed_ms",
                    "value_root",
                    "value_best",
                    "value_fallback",
                    "agreed_with_policy",
                }
            },
            "search_error": search_error,
            "search_weight": search_weight,
        }
        if request.get("type") == "coach":
            result["coach"] = analyze_choice(
                valid_actions,
                valid_logits,
                chosen_action=request.get("chosen_action"),
                competitive_scores=comp,
                limit=int(request.get("alternative_limit", 5)),
            )
        return result

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = int(request.get("request_id", -1))
        frame_id = int(request.get("frame_id", -1))
        response: dict[str, Any] = {
            "schema": PROTOCOL_SCHEMA,
            "request_id": request_id,
            "frame_id": frame_id,
        }
        try:
            if request.get("schema") != PROTOCOL_SCHEMA:
                raise ValueError(f"schema must be {PROTOCOL_SCHEMA!r}")
            kind = str(request.get("type", ""))
            if kind == "hello":
                response.update(type="capabilities", capabilities=self.capabilities())
                return response
            if kind == "health":
                response.update(type="health", health=self.health())
                return response
            if kind == "cancel":
                self.cancelled.add(int(request["cancel_request_id"]))
                response.update(type="cancelled", cancel_request_id=int(request["cancel_request_id"]))
                return response
            if kind not in {"decide", "coach"}:
                raise ValueError(f"unsupported request type {kind!r}")
            if request_id <= self.last_request_id:
                raise ValueError("request_id must increase monotonically")
            self.last_request_id = request_id
            if request_id in self.cancelled:
                response.update(type="cancelled")
                return response
            if frame_id < self.latest_frame_id:
                response.update(type="stale", latest_frame_id=self.latest_frame_id)
                return response
            self.latest_frame_id = frame_id
            result = self._infer(request)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            budget_ms = float(request.get("deadline_ms", float("inf")))
            if elapsed_ms > budget_ms:
                response.update(type="deadline_exceeded", elapsed_ms=elapsed_ms)
                return response
            response.update(type="result", mode=kind, result=result)
            return response
        except Exception as exc:
            self.errors += 1
            response.update(type="error", error={"kind": type(exc).__name__, "message": str(exc)})
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1e3
            self.requests += 1
            self.latencies_ms.append(elapsed_ms)
            response["elapsed_ms"] = elapsed_ms
