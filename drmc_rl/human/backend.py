"""Host-neutral out-of-process human player and coach backend."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from drmc_rl.human.afterstate_model import HUMAN_AFTERSTATE_SCHEMA
from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
from drmc_rl.human.coach import analyze_choice
from drmc_rl.execution.pace import PACES, Pace, resolve_pace
from drmc_rl.human.model import canonicalize_same_color_action
from drmc_rl.human.runtime import HumanPolicyRuntime
from drmc_rl.human.search import (
    HumanValueSearch,
    blend_human_and_search,
    competitive_scores,
)
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.planning.fast_reach import (
    FrameState,
    HoldDir,
    Rotation,
    compute_speed_threshold,
    simulate_frame,
)
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.annotate_replay_events import POSE_TO_ACTION

PROTOCOL_SCHEMA = "drmc-human-backend-v1"
GRID_H, GRID_W = 16, 8

_SEARCH_PROFILES = {
    "fast": (4, 32),
    "balanced": (8, 64),
    "deep": (16, 128),
}


@dataclass(slots=True)
class AdaptiveSearchBudget:
    """Leave measured headroom while spending otherwise-idle decision time."""

    utilization: float = 0.75
    minimum_ms: float = 8.0

    def resolve(self, remaining_ms: float, requested_ms: Any = None) -> float:
        if requested_ms is not None:
            return max(float(requested_ms), 0.0)
        available = max(float(remaining_ms), 0.0)
        if not np.isfinite(available):
            available = 100.0
        reserve = max(5.0, 0.12 * available)
        return max(0.0, (available - reserve) * self.utilization)

    def observe(self, *, deadline_exceeded: bool) -> None:
        if deadline_exceeded:
            self.utilization = max(0.35, self.utilization * 0.85)
        else:
            self.utilization = min(0.9, self.utilization + 0.005)


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


def _frame_payload(frame: FrameState) -> dict[str, int]:
    return {
        "x": frame.x,
        "y": frame.y,
        "rotation": frame.rot,
        "speed_counter": frame.speed_counter,
        "horizontal_velocity": frame.hor_velocity,
        "hold_dir": frame.hold_dir.value,
        "rotation_hold": frame.rot_hold.value,
        "frame_parity": frame.frame_parity,
    }


class HumanBackend:
    """Synchronous worker intended to be supervised off the gameplay thread."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        seed: int = 0,
        max_frames: int = 2048,
        realtime_profile: str = "auto",
        competitive_checkpoint: str | None = None,
    ):
        started = time.perf_counter()
        schema = load_checkpoint(Path(checkpoint), map_location="cpu").get("schema")
        self.afterstate_v3 = schema == HUMAN_AFTERSTATE_SCHEMA
        if self.afterstate_v3:
            self.runtime = AfterstatePolicyRuntime(checkpoint, device=device, seed=seed)
        else:
            self.runtime = HumanPolicyRuntime(checkpoint, device=device, seed=seed)
        self.device = device
        with Path(checkpoint).open("rb") as artifact:
            self.checkpoint_sha256 = hashlib.file_digest(artifact, "sha256").hexdigest()
        self.competitive = None
        self.competitive_identity = None
        if competitive_checkpoint is not None:
            from tools.vs_head_to_head import PlainPolicy

            if not self.afterstate_v3:
                raise ValueError("a competitive ceiling requires a V3 human/timing checkpoint")
            self.competitive = PlainPolicy(
                Path(competitive_checkpoint), device=device, public_only=True
            )
            with Path(competitive_checkpoint).open("rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
            self.competitive_identity = {
                "checkpoint": Path(competitive_checkpoint).name,
                "sha256": digest,
                "information_scope": "public-boards-pills-reachability-zero-aux-v1",
                "observation_encoding": "legacy-vs-horizontal-bond-mask-v1",
                "control": "quality_argmax; no absolute human rating claim",
            }
        self.seed = int(seed)
        if realtime_profile == "auto":
            realtime_profile = "balanced" if str(device).startswith("cuda") else "fast"
        if realtime_profile not in _SEARCH_PROFILES:
            raise ValueError(f"unknown realtime profile {realtime_profile!r}")
        self.realtime_profile = realtime_profile
        self.search_beam, self.search_num_sim_envs = _SEARCH_PROFILES[realtime_profile]
        self.search_budget = AdaptiveSearchBudget()
        self.search: HumanValueSearch | None = None
        self.planner = NativeReachabilityRunner(max_frames=max_frames)
        self.warmup()
        self.ready = True
        self.started_at = time.time()
        self.load_ms = (time.perf_counter() - started) * 1e3
        self.requests = 0
        self.errors = 0
        self.cancelled: set[int] = set()
        self.last_request_id = -1
        self.latest_frame_id = -1
        self.latencies_ms: deque[float] = deque(maxlen=256)

    def warmup(self) -> None:
        """Initialize inference kernels before accepting a timed game request.

        Score a legal neutral position without choosing an action or sampling
        timing, so warm-up cannot consume the player's seeded error sequence.
        """

        planes = np.zeros((8, GRID_H, GRID_W), dtype=np.float32)
        state = {"board_planes": planes, "opponent_board_planes": planes,
                 "pill": [0, 1], "preview": [2, 0], "speed": 2, "speed_ups": 0}
        _own, _opp, pill, preview, speed, speed_ups, _frame, _reach, packed, costs = self._candidates(state)
        rating = self.runtime.condition.mean
        args = dict(board_planes=planes, opponent_board_planes=planes,
                    opponent_state_age_frames=0, pill=pill, preview=preview,
                    candidate_actions=packed.actions, candidate_costs=packed.cost,
                    candidate_mask=packed.mask, rating=rating)
        if self.afterstate_v3:
            self.runtime.score(**args, speed=speed, speed_ups=speed_ups)
        else:
            self.runtime.score(**args)
        self.runtime.timing_prediction(
            board_planes=planes, rating=rating, chosen_cost=float(packed.cost[0]),
            speed=speed, speed_ups=speed_ups, candidate_count=packed.count,
        )
        if self.competitive is not None:
            feasible = (costs != 0xFFFF).reshape(4, GRID_H, GRID_W)
            masks = [feasible]
            if str(self.competitive.device).startswith("mps"):
                legal = np.ones((4, GRID_H, GRID_W), dtype=bool)
                legal[::2, :, -1] = False
                legal[1::2, 0, :] = False
                positions = np.flatnonzero(legal)
                masks = []
                for count in (32, 64, 128, 256, len(positions)):
                    mask = np.zeros(legal.size, dtype=bool)
                    mask[positions[:count]] = True
                    masks.append(mask.reshape(legal.shape))
            for mask in masks:
                observation = np.concatenate((planes, planes, mask.astype(np.float32)))
                self.competitive.score(observation[None], [{
                    "placements/feasible_mask": mask,
                    "placements/cost_to_lock": np.where(mask, 32, 0xFFFF),
                    "next_pill_colors": pill,
                    "preview_pill": {"first_color": 2, "second_color": 1},
                }])

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "request_types": ["hello", "health", "decide", "coach", "cancel", "shutdown"],
            "modes": ["play", "coach"],
            "state": {
                "board_planes": "8x16x8 canonical color/virus/connectivity planes",
                "opponent_pill": "public current/last falling colors; required by the competitive ceiling",
                "colors": "0=red, 1=yellow, 2=blue",
                "coordinates": "planner coordinates: row 0 is bottle top",
            },
            "outputs": ["placement", "controller_frames", "timing", "coach_analysis"],
            "scheduled_execution": {"version": 1, "max_delay_frames": 30},
            "strength": {
                "sample_regret": "V3 only; sample calibrated regret tails independently of imitation temperature",
                "controls": ["regret", "quality"] if self.afterstate_v3 else ["regret"],
                "competitive_ceiling": self.competitive_identity,
                "regret_decoder": "ordered-log-regret-bands-v1" if self.afterstate_v3 else None,
                "style_conditioning": "fixed population mean for play; requested rating for coaching",
            },
            "cadence": {
                "control": "named motor pace; legacy timing_scale maps to a preset",
                "validation": "independent per-frame physics and hard motor limits",
                "movement_scope": "complete constrained feasibility before strategic selection",
                "unrestricted_fallback": False,
                "profiles": [pace.to_dict() for pace in PACES],
            },
            "search": {
                "available": True,
                "exact_afterstate": self.afterstate_v3,
                "default_for_coach": True,
                "play_control": "search_weight >= 0; zero is pure human imitation",
                "realtime_profile": self.realtime_profile,
                "beam": self.search_beam,
                "num_sim_envs": self.search_num_sim_envs,
                "adaptive_deadline": True,
            },
            "cancellation": "cooperative between requests; hosts must discard stale frame_ids",
            "model": {**self.runtime.identity, "sha256": self.checkpoint_sha256},
        }

    def health(self) -> dict[str, Any]:
        latencies = np.asarray(self.latencies_ms, dtype=np.float64)
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
            "model": {**self.runtime.identity, "sha256": self.checkpoint_sha256},
            "search": {
                "realtime_profile": self.realtime_profile,
                "beam": self.search_beam,
                "num_sim_envs": self.search_num_sim_envs,
                "budget_utilization": self.search_budget.utilization,
            },
        }

    def _candidates(self, state: Mapping[str, Any], execution_delay_frames: int = 0,
                    pace: Pace | None = None):
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
        for _ in range(execution_delay_frames):
            frame = simulate_frame(
                _columns(planes),
                frame,
                0,
                speed_threshold=compute_speed_threshold(speed, speed_ups),
            )
            if frame.locked:
                raise ValueError("pill locks before scheduled execution")
        reach = self.planner.bfs_full(
            _columns(planes),
            frame,
            speed_threshold=compute_speed_threshold(speed, speed_ups),
            **({} if pace is None else pace.planner_args(execution_delay_frames)),
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
            max_candidates=max(128, int(np.count_nonzero(costs != 0xFFFF))),
            sort_by_cost=True,
        )
        if packed.count == 0:
            raise RuntimeError("no reachable placement")
        return planes, opponent_planes, pill, preview, speed, speed_ups, frame, reach, packed, costs

    def close(self) -> None:
        if self.search is not None:
            self.search.close()
            self.search = None
        self.planner.close()
        close_runtime = getattr(self.runtime, "close", None)
        if close_runtime is not None:
            close_runtime()

    def _value_search(self) -> HumanValueSearch:
        if self.search is None:
            self.search = HumanValueSearch(
                self.runtime,
                device=self.device,
                beam=self.search_beam,
                seed=self.seed,
                num_sim_envs=self.search_num_sim_envs,
                gpu_planner=str(self.device).startswith("cuda"),
            )
        return self.search

    def _infer(self, request: Mapping[str, Any], *, remaining_ms: float) -> dict[str, Any]:
        state = request["state"]
        rating = float(request["target_rating"])
        temperature = float(request.get("temperature", 1.0))
        execution_delay = int(request.get("execution_delay_frames", 0))
        pace = resolve_pace(request.get("pace"), request.get("timing_scale", 1.0))
        max_delay = max(30, pace.reaction_frames)
        if not 0 <= execution_delay <= max_delay:
            raise ValueError(f"execution_delay_frames must be in [0,{max_delay}]")
        if not np.isfinite(rating) or not np.isfinite(temperature) or temperature < 0:
            raise ValueError(
                "rating and temperature must be finite; temperature must be non-negative"
            )
        (
            planes,
            opponent_planes,
            pill,
            preview,
            speed,
            speed_ups,
            frame,
            reach,
            packed,
            costs512,
        ) = self._candidates(state, execution_delay, pace)
        rating_sd = float(request.get("target_rating_sd", 0.0))
        opponent_rating = state.get("opponent_rating")
        opponent_rating_sd = float(state.get("opponent_rating_sd", 0.0))
        game_phase = float(state.get("game_phase", 0.0))
        recent_decisions = state.get("recent_decisions", ())
        score_args = dict(
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
        competitive_only = (
            self.afterstate_v3 and self.competitive is not None
            and request.get("strength_control") == "quality" and request.get("type") != "coach"
        )
        if competitive_only:
            # Maximum play needs the public competitive policy and the cheap
            # cadence model. Only coaching needs a second full human/afterstate
            # evaluation; do not spend the live decision budget on unused heads.
            resolved_rating, rating_clamped = self.runtime.condition.resolve(rating)
            logits, state_value, details = None, None, {}
        elif self.afterstate_v3:
            style_rating = (
                self.runtime.condition.mean
                if request.get("type") != "coach" and request.get("strength_control", "regret") == "regret"
                else None
            )
            details = self.runtime.score(
                **score_args, speed=speed, speed_ups=speed_ups, style_rating=style_rating
            )
            logits = details["human_logits"]
            resolved_rating = float(details["resolved_rating"])
            rating_clamped = bool(details["rating_clamped"])
            state_value = float(np.max(details["outcome_logit"][packed.mask]))
        else:
            logits, state_value, resolved_rating, rating_clamped = self.runtime.score(**score_args)
        valid_actions = packed.actions[packed.mask]
        valid_logits = None if logits is None else logits[packed.mask]
        search_info = None
        comp = details["competitive_score"][packed.mask] if self.afterstate_v3 and not competitive_only else None
        search_error = None
        search_weight = max(float(request.get("search_weight", 0.0)), 0.0)
        use_search = (
            bool(request.get("search", request.get("type") == "coach")) or search_weight > 0
        )
        search_deadline_ms = self.search_budget.resolve(
            remaining_ms, request.get("search_deadline_ms")
        )
        if self.afterstate_v3:
            search_info = {
                "stage": "public-policy" if competitive_only else "exact-afterstate",
                "nodes_expanded": int(packed.count),
            }
        elif use_search and search_deadline_ms >= self.search_budget.minimum_ms:
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
                    deadline_ms=search_deadline_ms,
                )
                comp = competitive_scores(valid_actions, search_info)
            except Exception as exc:
                if request.get("require_search"):
                    raise
                search_error = {"kind": type(exc).__name__, "message": str(exc)}
        strength = None
        if self.afterstate_v3:
            control = str(request.get("strength_control", "regret"))
            if control not in {"regret", "quality"}:
                raise ValueError("strength_control must be regret or quality")
            if control == "quality":
                scores = details.get("competitive_score")
                if self.competitive is not None:
                    from drmc_rl.game.observation import legacy_vs_policy_boards

                    opponent_pill = _pair(state["opponent_pill"], "opponent_pill")
                    policy_boards = legacy_vs_policy_boards(
                        planes, opponent_planes, pill, opponent_pill
                    )
                    observed = np.concatenate((policy_boards,
                        (costs512 != 0xFFFF).reshape(4, 16, 8).astype(np.float32)))
                    raw_colors = (1, 0, 2)
                    actions, masks, logits = self.competitive.score(observed[None], [{
                        "placements/feasible_mask": (costs512 != 0xFFFF).reshape(4, 16, 8),
                        "placements/cost_to_lock": costs512.reshape(4, 16, 8),
                        "next_pill_colors": pill,
                        "vs/opponent_pill_colors": opponent_pill,
                        "preview_pill": {"first_color": raw_colors[int(preview[0])],
                                         "second_color": raw_colors[int(preview[1])]},
                    }])
                    if set(actions[0, masks[0]]) != set(valid_actions):
                        raise RuntimeError("competitive policy changed candidate coverage")
                    by_action = np.full(512, -np.inf, dtype=np.float32)
                    by_action[actions[0, masks[0]]] = logits[0, masks[0]]
                    scores = np.where(packed.mask, by_action[packed.actions.clip(min=0)], -np.inf)
                    comp = scores[packed.mask]
                packed_slot = self.runtime.choose_quality(scores, packed.mask)
                strength = {"control": "quality", "chosen_regret": 0.0,
                            "rating_calibrated": False,
                            "competitive_model": self.competitive_identity}
            else:
                packed_slot, strength = self.runtime.choose_strength(
                    details["competitive_score"],
                    details["human_logits"],
                    packed.mask,
                    rating=resolved_rating,
                    temperature=1.0 if request.get("sample_regret", False) else temperature,
                )
                strength["control"] = "regret"
        else:
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
        motor_audit = pace.validate(
            _columns(planes), frame, script,
            speed_threshold=compute_speed_threshold(speed, speed_ups),
            execution_delay=execution_delay,
        )
        timing.update(
            execution_profile=pace.to_dict(),
            movement={"algorithm": "constrained-frame-search-v1", "validated": True,
                      "unrestricted_fallback": False, **motor_audit},
            planner_cost_frames=int(packed.cost[packed_slot]),
            cost_semantics="duration of selected profile-valid witness",
        )
        replay = frame
        controller_states = []
        for buttons in script:
            if replay.locked:
                break
            controller_states.append(_frame_payload(replay))
            replay = simulate_frame(
                _columns(planes),
                replay,
                int(buttons),
                speed_threshold=compute_speed_threshold(speed, speed_ups),
            )
        if not replay.locked or (replay.x, replay.y, replay.rot) != (x, y, rotation):
            raise RuntimeError("controller script did not replay to selected placement")
        script = script[: len(controller_states)]
        timing["execution_frames"] = len(script)
        result = {
            "execution": {
                "start_frame": int(request["frame_id"]) + execution_delay,
                "delay_frames": execution_delay,
                "falling": _frame_payload(frame),
            },
            "controller_states": controller_states,
            "target_rating": rating,
            "resolved_rating": resolved_rating,
            "rating_clamped": rating_clamped,
            "state_win_probability": None if state_value is None else float(1.0 / (1.0 + np.exp(-state_value))),
            "placement": {"action": action, "x": x, "y_top": y, "rotation": rotation},
            "controller_frames": [ACTION_TO_BUTTONS[int(value)] for value in script],
            "controller_encoding": "NES button mask: R=1 L=2 D=4 B=64 A=128",
            "timing": timing,
            "candidate_count": int(packed.count),
            "human_logits": None if valid_logits is None else valid_logits.tolist(),
            "candidate_actions": valid_actions.tolist(),
            "competitive_scores": None if comp is None else comp.tolist(),
            "strength": strength,
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
            "search_profile": self.realtime_profile,
            "search_deadline_ms": search_deadline_ms if use_search else None,
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
                response.update(
                    type="cancelled", cancel_request_id=int(request["cancel_request_id"])
                )
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
            budget_ms = float(request.get("deadline_ms", float("inf")))
            elapsed_before_infer = (time.perf_counter() - started) * 1e3
            result = self._infer(request, remaining_ms=budget_ms - elapsed_before_infer)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            if elapsed_ms > budget_ms:
                self.search_budget.observe(deadline_exceeded=True)
                response.update(type="deadline_exceeded", elapsed_ms=elapsed_ms)
                return response
            self.search_budget.observe(deadline_exceeded=False)
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
