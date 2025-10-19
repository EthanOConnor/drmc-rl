import numpy as np
import pytest

from envs.reward_shaping import PotentialShaper


class DummyEval:
    def __init__(self, f):
        self.f = f

    def predict_mean_time(self, s):
        return self.f(s)


def test_potential_shaping_identity():
    # Phi(s) = -E[T]/kappa; here E[T]=s (identity) for simplicity.
    ev = DummyEval(lambda s: float(s))
    shaper = PotentialShaper(ev, potential_gamma=0.99, kappa=100.0, learner_discount=0.99)
    s, s_next = 100.0, 80.0  # expected time drops by 20
    r_shape = shaper.potential_delta(s, s_next)
    # gamma*(-80/100) - (-100/100) = -0.8*0.99 + 1 = 1 - 0.792 = 0.208
    assert np.isclose(r_shape, 0.208, atol=1e-6)


def test_gamma_mismatch_raises():
    ev = DummyEval(lambda s: float(s))
    with pytest.raises(ValueError):
        PotentialShaper(ev, potential_gamma=0.99, kappa=100.0, learner_discount=0.95)
