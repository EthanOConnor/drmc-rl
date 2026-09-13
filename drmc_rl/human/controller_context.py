"""Policy input boundary shared by paced controller training and evaluation."""

from __future__ import annotations

import numpy as np

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, PUBLIC_CONTEXT_DIMS, PublicExecutionContext
from drmc_rl.human.anticipation import public_policy_inputs
from drmc_rl.planning.fast_reach import compute_speed_threshold
from drmc_rl.search.public_policy import policy_request


def live_controller_state(state):
    """Decode the shared desktop/browser wire view, without accessing emulator RAM."""
    from drmc_rl.game.pair_state import (
        DecisionBoundary, FallingPillView, PairEvent, PairEventKind, PublicPairState,
        VisibleSideState, audit_public_mapping,
    )
    from drmc_rl.human.search import semantic_planes_to_nes_board

    live = state.get("public_live_context")
    if not isinstance(live, dict) or live.get("schema") != "public-controller-history-v1":
        raise ValueError("this competitive core requires live public history from the trainer")
    audit_public_mapping(live)
    if live.get("viewer_side") != 1 or len(live.get("sides", ())) != 2:
        raise ValueError("the live controller wire view must use the scheduler's P2 perspective")
    visible = []
    for side, metadata in enumerate(live["sides"]):
        active = metadata.get("active")
        board = semantic_planes_to_nes_board(np.asarray(
            state["board_planes" if side == 1 else "opponent_board_planes"]
        ))
        visible.append(VisibleSideState(
            board=bytes(board), pill=metadata["pill"], preview=metadata["preview"],
            active=None if active is None else FallingPillView(**active),
            viruses_remaining=metadata["viruses_remaining"],
            animation_phase=metadata["animation_phase"],
            state_age_frames=metadata["state_age_frames"],
        ))
    if (visible[1].pill != tuple(state["pill"]) or visible[1].preview != tuple(state["preview"])
            or visible[0].pill != tuple(state["opponent_pill"])):
        raise ValueError("live history and the observed controller pills disagree")
    frame = int(live["frame_id"])
    events = []
    for event in live["recent_events"]:
        payload = dict(event["public_payload"])
        if event["kind"] == "volley":
            size = int(payload["garbage_size"])
            if not 2 <= size <= 4:
                raise ValueError("invalid observed garbage volley")
            payload["columns"] = payload["columns"][:size]
            payload["colors"] = payload["colors"][:size]
        events.append(PairEvent(PairEventKind(event["kind"]), int(event["frame_id"]),
                                int(event["side"]), payload))
    if any(e.frame_id > frame for e in events) or any(a.frame_id > b.frame_id for a,b in zip(events,events[1:])):
        raise ValueError("live event history must be causal and chronological")
    own = visible[1].active
    falling = state["falling"]
    if own is None or (own.column, own.row_top, own.rotation) != (
        falling["x"], falling["y"], falling["rotation"]
    ):
        raise ValueError("live history and the actual falling controller pose disagree")
    opponent = visible[0].active
    public = PublicPairState(
        frame_id=frame, viewer_side=1, sides=tuple(visible), recent_events=tuple(events),
        decision_boundary=(DecisionBoundary.BOTH if opponent and opponent.age_frames == 0 else DecisionBoundary.P2),
        observable_clock_delta_frames=0, own_controller_state=falling,
    )
    return {**state, "public_pair_state": public, "public_context_schema": PUBLIC_CONTEXT_SCHEMA,
            "vs/observation_timeline": "causal-settled-pair-v1"}


def uses_public_context(policy):
    return getattr(policy, "aux_spec", None) in PUBLIC_CONTEXT_DIMS


def controller_policy_inputs(policy, candidate, state, pace, delay, compute_frames):
    """Condition the actual complete motor-feasible frontier on its causal view.

    The public pose is the observation before computation. Costs/witnesses start
    after the charged delay; execution context makes that distinction explicit.
    """
    if not uses_public_context(policy):
        from drmc_rl.human.backend import _pair
        observations, infos = public_policy_inputs(
            candidate[0], candidate[1], candidate[2], _pair(state["opponent_pill"], "opponent_pill"),
            candidate[-1], [state["preview"]],
        )
        if "vs/observation_timeline" in state:
            infos[0]["vs/observation_timeline"] = state["vs/observation_timeline"]
        from drmc_rl.execution.pace import strategy_context
        infos[0]["pace/id"] = pace.id
        infos[0]["pace/context"] = strategy_context(pace, state, delay)
        return observations, infos
    if (state.get("public_context_schema") != PUBLIC_CONTEXT_SCHEMA
            or state.get("vs/observation_timeline") != "causal-settled-pair-v1"):
        raise ValueError("controller context requires an explicitly causal live public view")
    public = state["public_pair_state"]
    costs = np.asarray(candidate[-1], dtype=np.uint16).reshape(512)
    legal = np.flatnonzero(costs != 0xFFFF).tolist()
    execution = PublicExecutionContext(
        reaction_frames=pace.reaction_frames, edge_interval=pace.edge_interval,
        motion_interval=pace.motion_interval, max_buttons=pace.max_buttons,
        # The ROM falls when its counter exceeds the table threshold. The
        # public feature names the period, including the one-frame fast limit.
        gravity_frames=int(compute_speed_threshold(state["speed"], state["speed_ups"])) + 1,
        speed_ups=int(state["speed_ups"]), decision_delay_frames=int(delay),
        compute_frames=int(compute_frames),
    )
    observation, info = policy_request(
        public, public.viewer_side, legal, costs[legal].tolist(),
        context_schema=PUBLIC_CONTEXT_SCHEMA, execution=execution,
    )
    if policy.aux_spec != PUBLIC_CONTEXT_SCHEMA:
        info["public_context_schema"] = policy.aux_spec
        info["public_progress"] = dict(
            level=int(state["level"]), pill_counter_bcd=int(state["pill_counter_total"]),
            speed=int(state["speed"]), speed_ups=int(state["speed_ups"]),
        )
    info["vs/observation_timeline"] = state["vs/observation_timeline"]
    # Preserve the exact own controller boundary for later deterministic motor
    # labels. This is archival metadata, not another network input or a native
    # restore snapshot. Candidate costs alone cannot recover DAS/parity/counter.
    info["public_controller_geometry"] = {
        **{key: int(state[key]) for key in ("speed", "speed_ups", "pill_counter_total")},
        "decision_delay_frames": int(delay), "compute_frames": int(compute_frames),
        **{key: int(state["falling"][key]) for key in (
            "x", "y", "rotation", "speed_counter", "horizontal_velocity",
            "hold_dir", "rotation_hold", "frame_parity",
        )},
    }
    return observation[None], [info]
