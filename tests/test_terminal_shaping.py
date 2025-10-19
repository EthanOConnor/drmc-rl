import numpy as np

from envs.reward_shaping import PotentialShaper


class DummyEval:
    def __init__(self, f):
        self.f = f

    def predict_mean_time(self, s):
        return self.f(s)


def test_terminal_phi_zero():
    # Convention: Phi(terminal) = 0 to avoid last-step shaping artifacts.
    # This test documents the convention; env should set Phi=0 when terminated=True.
    ev = DummyEval(lambda s: float(s))
    shaper = PotentialShaper(ev, potential_gamma=0.99, kappa=100.0, learner_discount=0.99)
    s = 50.0
    s_terminal = 0.0  # by convention Phi(terminal)=0 => E[T_clear | terminal] = 0
    r_shape = shaper.potential_delta(s, s_terminal)
    # gamma*Phi(terminal) - Phi(s) = 0 - (-0.5) = +0.5
    assert np.isclose(r_shape, 0.5, atol=1e-6)
