"""Potential-based reward shaping helpers for Dr. Mario RL.

Implements r_shape = gamma * Phi(s_next) - Phi(s), where Phi(s) is derived from the evaluator.
Safe in the sense of Ng et al. 1999 (preserves optimal policies).

Usage:
    from envs.reward_shaping import PotentialShaper

    shaper = PotentialShaper(evaluator, gamma=0.997, kappa=250.0)
    r_shape = shaper.potential_delta(s, s_next)
    r_total = r_env + r_shape
"""

from dataclasses import dataclass
from typing import Protocol, Any, Optional
import numpy as np


class Evaluator(Protocol):
    def predict_mean_time(self, state: Any) -> float:
        """Return E[T_clear | state]. Implemented by your trained evaluator."""


@dataclass
class PotentialShaper:
    evaluator: Evaluator
    gamma: float = 0.997
    kappa: float = 250.0

    def phi(self, state: Optional[Any]) -> float:
        # Convention: terminal state has Phi=0
        if state is None:
            return 0.0
        t_mean = float(self.evaluator.predict_mean_time(state))
        return -t_mean / self.kappa

    def potential_delta(self, s: Optional[Any], s_next: Optional[Any]) -> float:
        return self.gamma * self.phi(s_next) - self.phi(s)
