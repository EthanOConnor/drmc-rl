"""APPO/PPO bootstrap with evaluator predictions (skeleton).

This shows how to blend evaluator value into n-step returns. Integrate into your SF version:
- Import this function into the place where GAE / bootstrapping is computed.
- Set config flags to control mean/quantile/CVaR and blend λ.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class EvalAdapter:
    def __init__(self, evaluator, mode: str = 'mean', tau: float = 0.5, alpha: float = 0.25):
        self.evaluator = evaluator
        self.mode = mode
        self.tau = tau
        self.alpha = alpha

    def v(self, s):
        if self.mode == 'mean':
            return self.evaluator.mean_time(s)
        if self.mode == 'quantile':
            return self.evaluator.quantile_time(s, self.tau)
        return self.evaluator.cvar_time(s, self.alpha)


def compute_returns_with_eval_bootstrap(
    rewards,
    dones,
    values,
    last_obs,
    cfg: Dict,
    eval_adapter: EvalAdapter,
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
):
    """Blend evaluator bootstrap into targets.

    returns_t = R_t^{gae} with V_bootstrap = (1-λb)*V_model + λb*V_eval
    cfg:
        lambda_bootstrap: 0..1 blend between model and evaluator bootstrap
    Note: 'values' is the model critic estimate; 'last_obs' is the state for bootstrap at t+n.
    """
    T = len(rewards)
    lam_b = float(cfg.get('lambda_bootstrap', 1.0))
    # evaluator bootstrap
    v_eval = float(eval_adapter.v(last_obs))
    v_bootstrap = (1.0 - lam_b) * float(values[-1]) + lam_b * v_eval

    # Standard GAE with modified bootstrap
    adv = 0.0
    returns = [0.0] * T
    next_value = v_bootstrap
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * nonterminal * next_value - float(values[t])
        adv = delta + gamma * gae_lambda * nonterminal * adv
        next_value = float(values[t])
        returns[t] = adv + float(values[t])
    return np.array(returns, dtype=np.float32)

