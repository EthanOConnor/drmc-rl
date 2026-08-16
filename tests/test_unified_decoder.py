import numpy as np

from drmc_rl.execution.profile import BUTTON_RIGHT
from drmc_rl.human.decoder import (
    CandidateOption,
    DecisionContext,
    FixedCadence,
    ProductMode,
    UnifiedDecisionDecoder,
)


class Regret:
    def sample(self, rating, opportunity, rng):
        return 1.0

    def parameters(self, rating, opportunity):
        return 1.0, 0.05


def candidates():
    return [
        CandidateOption(0, 0.8, 0.0, (np.array([BUTTON_RIGHT, 0], dtype=np.uint8),), np.array([0.0, 1.0])),
        CandidateOption(1, 0.6, 2.0, (np.array([0, 0, BUTTON_RIGHT, 0], dtype=np.uint8),), np.array([1.0, 0.0])),
    ]


def test_unrestricted_chooses_quality_argmax() -> None:
    decoder = UnifiedDecisionDecoder(mode=ProductMode.UNRESTRICTED)
    result = decoder.choose(candidates(), context=DecisionContext(rating=2000))
    assert result.action == 0
    assert result.regret_win_logit == 0


def test_trainer_strength_then_style() -> None:
    decoder = UnifiedDecisionDecoder(
        mode=ProductMode.TRAINER,
        regret=Regret(),
        cadence=FixedCadence(2),
        style_vector=(1.0, 0.0),
        seed=2,
    )
    result = decoder.choose(candidates(), context=DecisionContext(rating=1200))
    assert result.action == 1
    assert result.regret_win_logit > 0
    assert result.diagnostics["strength_mechanism"] == "calibrated_win_logit_regret"
