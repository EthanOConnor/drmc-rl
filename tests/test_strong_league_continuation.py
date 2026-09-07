from __future__ import annotations

import pytest
import numpy as np
import torch
from collections import OrderedDict
from types import SimpleNamespace

from drmc_rl.search.strong_league import DavidsonCalibration, FrozenStrongLeagueMixture


def test_davidson_calibration_is_normalized_and_monotone() -> None:
    calibration = DavidsonCalibration(
        slope=1.7,
        bias=-0.1,
        draw_logit=-2.5,
        artifact_sha256="test",
    )
    low = calibration.wdl(-0.8)
    middle = calibration.wdl(0.0)
    high = calibration.wdl(0.8)
    assert low.win < middle.win < high.win
    assert low.loss > middle.loss > high.loss
    for value in (low, middle, high):
        assert value.win + value.draw + value.loss == pytest.approx(1.0)


@pytest.mark.parametrize("root_side", [0, 1])
def test_value_uses_the_acting_perspective_and_reverses_calibrated_wdl(root_side) -> None:
    mixture = object.__new__(FrozenStrongLeagueMixture)
    mixture.calibration = DavidsonCalibration(1.7, -.2, -2.5, "test")
    inferred_sides = []

    def infer(state, side):
        inferred_sides.append(side)
        return {0: 1.0}, .8

    mixture._infer = infer
    opponent = 1 - root_side
    need = [False, False]
    need[opponent] = True
    state = SimpleNamespace(privileged=SimpleNamespace(need_action=tuple(need)))
    value = mixture.evaluate(state, root_side)
    expected = mixture.calibration.wdl(.8)
    assert inferred_sides == [opponent]
    assert value.win == expected.loss
    assert value.draw == expected.draw
    assert value.loss == expected.win

    need[root_side] = True
    state.privileged.need_action = tuple(need)
    assert mixture.evaluate(state, root_side) == expected
    assert inferred_sides[-1] == root_side


def test_resolving_state_is_rejected_by_the_decision_critic() -> None:
    mixture = object.__new__(FrozenStrongLeagueMixture)
    state = SimpleNamespace(
        privileged=SimpleNamespace(need_action=(False, False)),
        legal_actions_by_side=((), ()),
    )
    with pytest.raises(ValueError, match="actionable decision"):
        mixture.evaluate(state, 0)
    with pytest.raises(ValueError, match="acting side with legal candidates"):
        mixture._infer(state, 0)


def test_batched_mixture_preserves_probabilities_and_complete_frontiers(monkeypatch):
    import drmc_rl.search.strong_league as module

    class Net(torch.nn.Module):
        def forward(self, obs, pill, preview, actions, costs, mask, aux=None):
            logits = actions.float()*.01-costs*.001+obs[:, 0, 0, 0, None]
            # Exercise the mask in candidate-set pooling as real actors do.
            value = (logits*mask).sum(1)/mask.sum(1)
            return logits, value

    member = object.__new__(module._FrozenMember)
    member.net, member.device, member.aux_dim = Net(), "cpu", 0
    mixture = object.__new__(FrozenStrongLeagueMixture)
    mixture.members, mixture.weights = (member,), np.asarray([1.])
    mixture._cache, mixture.cache_size = OrderedDict(), 8
    requests = []
    for count in (5, 129, 257):
        state = SimpleNamespace(
            count=count, privileged=SimpleNamespace(
                need_action=(True, False), engine_checkpoint=str(count).encode()),
            legal_actions_by_side=(tuple(range(count)), ()))
        requests.append((state, 0))

    def inputs(state, side):
        count = state.count
        return (np.zeros((16, 16, 8), np.float32), np.zeros(2, np.int64),
                np.zeros(2, np.int64), np.arange(count, dtype=np.int32),
                np.full(count, 20, np.float32), np.ones(count, bool),
                np.zeros(72, np.float32))

    monkeypatch.setattr(module, "_policy_inputs", inputs)
    batched = mixture.infer_batch(requests)
    singles = [mixture._infer(state, side) for state, side in requests]
    for (actual, value), (expected, single_value) in zip(batched, singles, strict=True):
        assert list(actual) == list(expected)
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=2e-6)
        assert sum(actual.values()) == pytest.approx(1.)
        assert value == pytest.approx(single_value)
    assert len(batched[-1][0]) == 257
    assert mixture.infer_batch([]) == []
