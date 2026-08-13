"""Potential-based reward shaping helpers for Dr. Mario RL.

Implements a safe potential function per Ng et al. 1999:

    r_shape = kappa * (gamma * Phi(s_next) - Phi(s))

where Phi(s) is derived from an evaluator. To preserve policy optimality the
discount `gamma` must match the learner's discount factor.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol


class Evaluator(Protocol):
    def predict_mean_time(self, state: Any) -> float:
        """Return E[T_clear | state]. Implemented by your trained evaluator."""


class PotentialShaper:
    """
    Safe potential-based shaping:

        r_shape = kappa * (gamma * Phi(s_next) - Phi(s))

    NOTE: `gamma` must equal the learner's discount for policy invariance.
    When `learner_discount` is provided and mismatched, a ValueError is raised.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        potential_gamma: float = 0.99,
        kappa: float = 250.0,
        learner_discount: Optional[float] = None,
    ) -> None:
        self._eval = evaluator
        self._gamma = float(potential_gamma)
        self._kappa = float(kappa)
        if learner_discount is not None and abs(self._gamma - float(learner_discount)) > 1e-6:
            raise ValueError(
                f"Potential shaping gamma ({self._gamma}) must equal learner discount ({learner_discount})."
            )

    def phi(self, state: Optional[Any]) -> float:
        """Return Phi(s); conventionally Phi(terminal)=0."""
        if state is None:
            return 0.0
        t_mean = float(self._eval.predict_mean_time(state))
        return -t_mean / self._kappa

    def potential_delta(self, s_prev: Optional[Any], s_next: Optional[Any]) -> float:
        # Terminal convention: Phi(None) = 0 (prevents leakage at episode ends)
        phi_prev = 0.0 if s_prev is None else self.phi(s_prev)
        phi_next = 0.0 if s_next is None else self.phi(s_next)
        return self._gamma * phi_next - phi_prev


__all__ = ["PotentialShaper"]
