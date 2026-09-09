"""Finite-difference and update-state checks for actual trainer loss code."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drmc_rl.models.policy.pace_adapter import PaceAdapter
from drmc_rl.training.episodic_objective import (
    clipped_surrogate,
    normalize_advantages,
    normalized_weights,
    objective_contract,
    validate_resume_objective,
)
from tools.train_pace_strategy import update_adapter


def expected_update(reduction, *, length=100, baseline=0.0, normalization="episode_center_scale"):
    # Exactly enumerate .5*(.8 win/.2 loss) for A, .5*(.6/.4) for B.
    # Later steps have a single forced action and zero score derivative.
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    logs = torch.stack(
        (torch.nn.functional.logsigmoid(theta), torch.nn.functional.logsigmoid(-theta))
    )
    ratios, returns, inverse = [], [], []
    for action, wins, count in ((0, 8, length), (1, 6, 1)):
        for game in range(10):
            ratios += [logs[action] - logs[action].detach()] + [theta * 0] * (count - 1)
            reward = 1 if game < wins else -1
            returns += [reward - baseline] + [reward - (2 * wins / 10 - 1)] * (count - 1)
            inverse += [1 / count] * count
    advantage, _, _ = normalize_advantages(returns, inverse, normalization)
    loss = clipped_surrogate(
        torch.stack(ratios),
        torch.tensor(advantage),
        torch.tensor(normalized_weights(inverse, reduction)),
        0.15,
    )
    loss.backward()
    return -theta.grad.item(), len(returns) / 20


def test_corrected_loss_does_not_reverse_the_better_long_action():
    assert expected_update("decision_mean")[0] > 0
    assert expected_update("episode_mean")[0] < 0


@pytest.mark.parametrize("length", [1, 10, 100])
@pytest.mark.parametrize("baseline", [-0.8, 0.0, 0.4, 0.9])
def test_episode_score_sum_matches_expected_win_gradient_and_cancels_baseline(length, baseline):
    gradient, common_scale = expected_update(
        "decision_mean", length=length, baseline=baseline, normalization="none"
    )

    def utility(theta):
        p = 1 / (1 + np.exp(-theta))
        return p * (2 * 0.8 - 1) + (1 - p) * (2 * 0.6 - 1)

    finite = (utility(1e-5) - utility(-1e-5)) / 2e-5
    assert gradient * common_scale == pytest.approx(finite, abs=1e-10)


def test_legacy_resume_is_explicit_and_new_objective_cannot_silently_resume():
    with pytest.raises(ValueError, match="init_adapter"):
        validate_resume_objective({}, {})
    assert (
        validate_resume_objective({}, {"objective": {"actor": "episode_mean"}})["actor"]
        == "episode_mean"
    )
    assert objective_contract({})["actor"] == "decision_mean"
    with pytest.raises(ValueError):
        objective_contract({"objective": {"invented": True}})


def update_fixture():
    torch.manual_seed(610)
    adapter = PaceAdapter(width=8, hidden=8)
    records = []
    for i in range(12):
        records.append(
            dict(
                candidate=np.random.default_rng(i).normal(size=(3, 8)).astype(np.float32),
                context=np.zeros(8, np.float32),
                motor=np.ones(8, np.float32),
                base_logits=np.zeros(3, np.float32),
                base_value=0.0,
                slot=i % 3,
                old_logprob=-np.log(3),
                old_value=0.0,
                **{"return": 1.0 if i % 3 == 0 else -1.0, "weight": 1.0},
            )
        )
    return SimpleNamespace(adapter=adapter, device="cpu"), records


def test_post_update_kl_rejection_restores_parameters_and_adam_moments():
    actor, records = update_fixture()
    before = {k: v.clone() for k, v in actor.adapter.state_dict().items()}
    optimizer = torch.optim.AdamW(actor.adapter.parameters(), lr=0.001)
    result = update_adapter(
        actor,
        optimizer,
        records,
        dict(epochs=1, minibatch=12, max_update_kl=1e-12, kl_backtracks=0),
        17,
    )
    assert result["kl_backtracks"] == 1 and result["early_kl_stop"]
    assert result["optimizer_steps"] == 0 and result["update_kl"] == 0
    for key, value in actor.adapter.state_dict().items():
        assert torch.equal(value, before[key])
    assert not optimizer.state


def test_collection_likelihood_mismatch_fails_before_update():
    actor, records = update_fixture()
    records[0]["old_logprob"] = 0.0
    with pytest.raises(RuntimeError, match="collection likelihood"):
        update_adapter(actor, torch.optim.AdamW(actor.adapter.parameters()), records, {}, 17)
