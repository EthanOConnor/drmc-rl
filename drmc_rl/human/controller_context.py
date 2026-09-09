"""Policy input boundary shared by paced controller training and evaluation."""

from __future__ import annotations

import numpy as np

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, PublicExecutionContext
from drmc_rl.human.anticipation import public_policy_inputs
from drmc_rl.planning.fast_reach import compute_speed_threshold
from drmc_rl.search.public_policy import policy_request


def uses_public_context(policy):
    return getattr(policy, "aux_spec", None) == PUBLIC_CONTEXT_SCHEMA


def controller_policy_inputs(policy, candidate, state, pace, delay, compute_frames):
    """Condition the actual complete motor-feasible frontier on its causal view.

    The public pose is the observation before computation. Costs/witnesses start
    after the charged delay; execution context makes that distinction explicit.
    """
    if not uses_public_context(policy):
        observations, infos = public_policy_inputs(
            candidate[0], candidate[1], candidate[2], state["opponent_pill"],
            candidate[-1], [state["preview"]],
        )
        if "vs/observation_timeline" in state:
            infos[0]["vs/observation_timeline"] = state["vs/observation_timeline"]
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
        gravity_frames=int(compute_speed_threshold(state["speed"], state["speed_ups"])),
        speed_ups=int(state["speed_ups"]), decision_delay_frames=int(delay),
        compute_frames=int(compute_frames),
    )
    observation, info = policy_request(
        public, public.viewer_side, legal, costs[legal].tolist(),
        context_schema=PUBLIC_CONTEXT_SCHEMA, execution=execution,
    )
    info["vs/observation_timeline"] = state["vs/observation_timeline"]
    return observation[None], [info]
