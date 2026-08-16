import numpy as np
import torch

from drmc_rl.game.pair_state import PairEvent, PairEventKind
from drmc_rl.models.policy.event_belief import (
    EVENT_FEATURE_DIM,
    PublicEventBeliefEncoder,
    PublicEventHistory,
)


def test_public_event_history_is_bounded_and_semantic() -> None:
    history = PublicEventHistory(max_events=2)
    history.append(PairEvent(PairEventKind.SPAWN, 10, 0, {"column": 3}))
    history.append(PairEvent(PairEventKind.VOLLEY, 12, 1, {"size": 3}))
    history.append(PairEvent(PairEventKind.LOCK, 14, 0, {"row_top": 8, "rotation": 1}))
    features, mask = history.features(20)
    assert features.shape == (2, EVENT_FEATURE_DIM)
    assert mask.tolist() == [True, True]
    assert np.isfinite(features).all()


def test_event_belief_handles_empty_and_nonempty_rows() -> None:
    net = PublicEventBeliefEncoder(16)
    features = torch.zeros(2, 3, EVENT_FEATURE_DIM)
    mask = torch.tensor([[False, False, False], [False, True, True]])
    features[1, 1:, 0] = 1.0
    output = net(features, mask)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
