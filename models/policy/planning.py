"""One-step planning wrapper using the evaluator.

Given a policy (that proposes an action distribution) and an evaluator (predicting T_clear stats),
choose the action whose next state minimizes a chosen statistic (mean/quantile/CVaR).

This version uses a side-channel 'peek_step' to simulate one step without committing.
If your env does not support that, clone state via savestates (Retro) or copy() for your C++ sim.
"""
from typing import Literal
import numpy as np

PlanMode = Literal["mean", "quantile", "cvar"]


def plan_one_step(env, policy, evaluator, risk_tau: float = 0.5,
                  mode: PlanMode = "quantile", alpha: float = 0.25) -> int:
    """Return best action index via one-step lookahead.

    Args:
        env: your environment exposing snapshot()/restore()/peek_step(a)
        policy: object with .legal_actions(obs) -> List[int] or fallback to range(env.action_space.n)
        evaluator: object with methods: mean_time(state), quantile_time(state, tau), cvar_time(state, alpha)
        risk_tau: quantile level for 'quantile' mode
        mode: which statistic to minimize
        alpha: CVaR alpha when mode='cvar'
    """
    obs = getattr(env, "last_obs", None)
    acts = policy.legal_actions(obs) if hasattr(policy, "legal_actions") else list(range(env.action_space.n))
    best_a, best_val = None, float("inf")
    # snapshot env
    snapshot = env.snapshot()
    for a in acts:
        env.restore(snapshot)
        obs_next, r, term, trunc, info = env.peek_step(a)
        # select statistic
        if mode == "mean":
            val = float(evaluator.mean_time(obs_next))
        elif mode == "quantile":
            val = float(evaluator.quantile_time(obs_next, risk_tau))
        else:
            val = float(evaluator.cvar_time(obs_next, alpha))
        if val < best_val:
            best_val, best_a = val, a
    # restore
    env.restore(snapshot)
    return best_a if best_a is not None else int(np.random.choice(acts))

