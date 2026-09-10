"""Public-information preparation shared by the trainer and latency arena."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drmc_rl.execution.pace import Pace
from drmc_rl.game.observation import legacy_vs_policy_boards


# Ordered canonical colors. These are contingent answers, not chance weights.
PREVIEW_BRANCHES = tuple((left, right) for left in range(3) for right in range(3))
CANON_TO_RAW = (1, 0, 2)


def own_board_only(state):
    """Frozen-model ablation: no opponent board or pill enters the actor.

    The opponent pill only selects the legacy bond masking on that opponent
    board. With those channels zero it has no remaining model contribution;
    normalize it too so it cannot spuriously invalidate prepared work.
    """
    return {**state, "opponent_board_planes": np.zeros((8, 16, 8), dtype=np.float32),
            "opponent_pill": [0, 0]}


def execution_for_action(candidate, action, pace, *, delay=0, frame_id=0):
    """Materialize and independently verify the exact controller witness."""
    from drmc_rl.human.backend import ACTION_TO_POSE, ACTION_TO_BUTTONS, _columns, _frame_payload
    from drmc_rl.planning.fast_reach import simulate_frame, compute_speed_threshold

    own, _, _, _, speed, ups, start, reach, _, _ = candidate
    pose = int(ACTION_TO_POSE[int(action)])
    x, y, rot = pose % 8, pose // 8 % 16, pose // 128
    script = reach.script_for_pose(x, y, rot)
    if script is None:
        raise ValueError("chosen placement has no controller witness")
    columns = _columns(own)
    threshold = compute_speed_threshold(speed, ups)
    audit = pace.validate(columns, start, script, speed_threshold=threshold, execution_delay=delay)
    frame, trace = start, []
    for buttons in script:
        trace.append(_frame_payload(frame))
        frame = simulate_frame(columns, frame, int(buttons), speed_threshold=threshold)
    if not frame.locked or (frame.x, frame.y, frame.rot) != (x, y, rot):
        raise ValueError("prepared controller witness did not reach its placement")
    return {
        "placement": {"action": int(action), "x": x, "y_top": y, "rotation": rot},
        "controller_frames": [ACTION_TO_BUTTONS[int(a)] for a in script],
        "controller_states": trace,
        "execution": {"falling": _frame_payload(start), "start_frame": frame_id + delay,
                      "delay_frames": delay},
        "timing": {"execution_profile": pace.to_dict(), "execution_frames": len(script),
                   "movement": {"validated": True, "unrestricted_fallback": False, **audit}},
        "lock_state": _frame_payload(frame),
    }


class NextTurnPredictor:
    """Predict our settled bottle from a committed move.

    The one-placement simulator receives only public tiles/colors. Its reserve
    is irrelevant: we read only the resulting bottle. Garbage and opponent
    changes are handled by validating the prepared context at the real spawn.
    """

    def __init__(self, planner, *, lib_path=None):
        from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
        self.planner = planner
        self.sim = NativeAfterstateSimulator(num_envs=1, lib_path=lib_path)

    def close(self):
        self.sim.close()

    def predict(self, state, execution):
        from drmc_rl.game.observation import board_bytes_to_semantic_planes
        from drmc_rl.human.search import semantic_planes_to_nes_board
        from drmc_rl.human.backend import ACTION_TO_BUTTONS, _columns
        from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, simulate_frame, compute_speed_threshold

        action = execution["placement"]["action"]
        pill, preview = np.asarray(state["pill"]), np.asarray(state["preview"])
        raw = np.asarray(CANON_TO_RAW, dtype=np.uint8)
        after = self.sim.simulate_packed(
            fields=semantic_planes_to_nes_board(state["board_planes"])[None],
            pills=raw[pill][None], previews=raw[preview][None],
            candidate_actions=np.array([[action]]),
            candidate_costs=np.array([[len(execution["controller_frames"])]]),
            candidate_count=np.array([1]), speed=np.array([state["speed"]]),
            speed_ups=np.array([state["speed_ups"]]),
        )
        if after.invalid[0] or after.terminal_reason[0]:
            return None
        last = execution["controller_states"][-1]
        f = FrameState(x=last["x"], y=last["y"], rot=last["rotation"],
            speed_counter=last["speed_counter"], hor_velocity=last["horizontal_velocity"],
            hold_dir=HoldDir(last["hold_dir"]), rot_hold=Rotation(last["rotation_hold"]),
            frame_parity=last["frame_parity"])
        end = simulate_frame(_columns(np.asarray(state["board_planes"])), f,
            ACTION_TO_BUTTONS.index(execution["controller_frames"][-1]),
            speed_threshold=compute_speed_threshold(state["speed"], state["speed_ups"]))
        if not end.locked:
            raise ValueError("preparation requires a complete committed controller script")
        # generateNextPill increments the visible BCD count; gravity accelerates
        # at each new multiple of ten. Horizontal DAS survives the lock/clear.
        ups = min(49, int(state["speed_ups"]) + ((int(state["pill_counter_total"]) & 15) == 9))
        return {**state, "board_planes": board_bytes_to_semantic_planes(after.fields[0]),
            "pill": preview.tolist(), "preview": [0, 0], "speed_ups": ups,
            "falling": {"x": 3, "y": 0, "rotation": 0, "speed_counter": 0,
                "horizontal_velocity": end.hor_velocity, "hold_dir": 0,
                "rotation_hold": 0, "frame_parity": 0}}


class NextTurnPreparer(NextTurnPredictor):
    """Legacy scored branches; context actors must score fresh public inputs."""

    def __init__(self, policy, planner, *, lib_path=None):
        from drmc_rl.human.controller_context import uses_public_context
        if uses_public_context(policy):
            raise ValueError("context actors require geometry preparation with fresh late context; legacy scored branches are incompatible")
        super().__init__(planner, lib_path=lib_path)
        self.policy = policy

    def prepare(self, state, execution, pace):
        from drmc_rl.human.backend import plan_candidates
        predicted = self.predict(state, execution)
        if predicted is None:
            return None
        candidates, observations, infos = [], [], []
        for parity in (0, 1):
            branch_state = {**predicted, "falling": {**predicted["falling"], "frame_parity": parity}}
            try:
                candidate = plan_candidates(self.planner, branch_state, 0, pace)
            except (ValueError, RuntimeError):
                return None
            candidates.append(candidate)
            obs, info = public_policy_inputs(candidate[0], candidate[1], candidate[2],
                predicted["opponent_pill"], candidate[-1], PREVIEW_BRANCHES)
            from drmc_rl.execution.pace import strategy_context
            for item in info:
                item["pace/id"] = pace.id
                item["pace/context"] = strategy_context(pace, branch_state, 0)
            observations.append(obs)
            infos.extend(info)
        scores = score_public_inputs(self.policy, np.concatenate(observations), infos)
        plans, branches = [], []
        for parity, candidate in enumerate(candidates):
            materialized = {}
            row = []
            for action in scores[parity*9:(parity+1)*9].argmax(axis=1):
                action = int(action)
                if action not in materialized:
                    materialized[action] = len(plans)
                    plans.append(execution_for_action(candidate, action, pace))
                row.append(materialized[action])
            branches.append(row)
        return {"state": predicted, "plans": plans, "branches": branches}


@dataclass(frozen=True)
class PreparedGeometry:
    """Complete own feasibility only: no saved policy scores or public history."""

    state: dict
    candidates: tuple
    pace: Pace
    delay: int

    def select(self, state, pace, delay):
        from drmc_rl.human.backend import _board_planes, _pair

        if pace != self.pace or delay != self.delay:
            return None, "execution_profile"
        if state.get("public_context_schema") != self.state.get("public_context_schema"):
            return None, "observation_contract"
        for name in ("board_planes", "pill", "speed", "speed_ups"):
            if not np.array_equal(self.state[name], state[name]):
                return None, "own_state"
        falling = state.get("falling", {})
        for name, value in self.state["falling"].items():
            if name != "frame_parity" and falling.get(name) != value:
                return None, "microstate"
        parity = falling.get("frame_parity")
        if parity not in (0, 1):
            return None, "microstate"
        candidate = self.candidates[int(parity)]
        if candidate is None:
            return None, "unreachable"
        # Only geometry survives preparation. All board/pill model inputs come
        # from the real request; the caller separately encodes its fresh history.
        return (_board_planes(state["board_planes"]),
                _board_planes(state["opponent_board_planes"]),
                _pair(state["pill"], "pill"), _pair(state["preview"], "preview"),
                *candidate[4:]), "hit"


class NextTurnGeometryPreparer(NextTurnPredictor):
    """Prepare both spawn parities without guessing future opponent/history."""

    def prepare(self, state, execution, pace, delay):
        from drmc_rl.human.backend import NoReachablePlacement, plan_candidates

        if not 0 <= delay <= max(30, pace.reaction_frames):
            raise ValueError("invalid prepared execution delay")
        predicted = self.predict(state, execution)
        if predicted is None:
            return None
        # Explicit whitelist: the predictor's input can contain a decoded
        # PublicPairState, but it is never a future context or a cached model input.
        expected = {k: predicted[k] for k in
                    ("board_planes", "pill", "speed", "speed_ups", "falling")}
        expected["public_context_schema"] = state.get("public_context_schema")
        geometry = {**expected, "opponent_board_planes": np.zeros((8, 16, 8), np.float32),
                    "preview": [0, 0]}
        candidates = []
        for parity in (0, 1):
            branch = {**geometry, "falling": {**expected["falling"], "frame_parity": parity}}
            try:
                candidates.append(plan_candidates(self.planner, branch, delay, pace))
            except NoReachablePlacement:
                candidates.append(None)
        if all(c is None for c in candidates):
            return None
        return PreparedGeometry(expected, tuple(candidates), pace, delay)


def select_prepared(prepared, state, *, strict_opponent=True):
    """A cache miss is ordinary: never repair or force a speculative script."""
    if prepared is None:
        return None, "unavailable"
    expected = prepared["state"]
    for name in ("board_planes", "pill", "speed", "speed_ups"):
        if not np.array_equal(expected[name], state[name]):
            return None, "own_state"
    for name, value in expected["falling"].items():
        if name != "frame_parity" and state["falling"].get(name) != value:
            return None, "microstate"
    opponent_changed = any(not np.array_equal(expected[k], state[k])
                           for k in ("opponent_board_planes", "opponent_pill"))
    if strict_opponent and opponent_changed:
        return None, "opponent"
    left, right = state["preview"]
    index = prepared["branches"][state["falling"]["frame_parity"]][3*left+right]
    return prepared["plans"][index], (
        "stale_opponent" if opponent_changed else "hit")


def public_policy_inputs(own, opponent, pill, opponent_pill, costs, previews):
    """Keep the installed actor's observation encoding and zero-aux boundary."""
    costs = np.asarray(costs, dtype=np.uint16).reshape(512)
    feasible = costs != 0xFFFF
    if not feasible.any():
        raise ValueError("cannot score a position without a legal placement")
    boards = legacy_vs_policy_boards(own, opponent, pill, opponent_pill)
    observation = np.concatenate((boards, feasible.reshape(4, 16, 8).astype(np.float32)))
    infos = [{
        "placements/feasible_mask": feasible.reshape(4, 16, 8),
        "placements/cost_to_lock": costs.reshape(4, 16, 8),
        "next_pill_colors": pill,
        "vs/opponent_pill_colors": opponent_pill,
        "preview_pill": {"first_color": CANON_TO_RAW[left], "second_color": CANON_TO_RAW[right]},
    } for left, right in previews]
    return np.repeat(observation[None], len(infos), axis=0), infos


def score_preview_branches(policy, own, opponent, pill, opponent_pill, costs, *, batch_size=9):
    """Return nine complete action-score rows without observing the next reveal."""
    if batch_size not in (1, 3, 9):
        raise ValueError("preview batch size must be 1, 3, or 9")
    obs, infos = public_policy_inputs(own, opponent, pill, opponent_pill, costs, PREVIEW_BRANCHES)
    return np.concatenate([score_public_inputs(policy, obs[start:start+batch_size], infos[start:start+batch_size])
                           for start in range(0, 9, batch_size)])


def score_public_inputs(policy, observations, infos):
    scores = np.full((len(infos), 512), -np.inf, np.float32)
    actions, masks, logits = policy.score(observations, infos)
    for row, info in enumerate(infos):
        legal = set(np.flatnonzero(np.asarray(info["placements/feasible_mask"]).reshape(512)))
        selected, values = actions[row, masks[row]], logits[row, masks[row]]
        if set(selected) != legal or len(selected) != len(legal) or not np.isfinite(values).all():
            raise RuntimeError("public scoring changed legal coverage or produced invalid scores")
        scores[row, selected] = values
    return scores
