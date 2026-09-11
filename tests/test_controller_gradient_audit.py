import numpy as np
import pytest
import torch
from torch.nn import functional as F

from drmc_rl.training.episodic_objective import categorical_kl, clipped_surrogate
from drmc_rl.training.gradient_diagnostics import loss_gradient_geometry
from tools.train_pace_strategy import (
    prepare_training_records, training_loss_terms, weighted_training_terms,
)


def test_gradient_geometry_detects_conflict_without_writing_weights_or_grad():
    shared = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))
    policy_only = torch.nn.Parameter(torch.tensor([4.], dtype=torch.float64))
    parameters = [('shared', shared), ('policy', policy_only)]
    values = [p.detach().clone() for _, p in parameters]
    terms = dict(policy_loss=shared[0] + 2*policy_only[0], value_loss=-3*shared[0],
                 entropy=0*shared.sum(), parent_kl=2*shared[1])
    result = loss_gradient_geometry(parameters, terms)
    all_params = result['groups']['all']
    np.testing.assert_allclose(all_params['gram'], [[5,-3,0,0],[-3,9,0,0],[0,0,0,0],[0,0,0,4]])
    assert all_params['combined_norm'] == pytest.approx(np.sqrt(12))
    assert result['groups']['shared_actor_value']['cosine'][0][1] == -1
    assert all_params['cosine'][2][2] is None  # Zero is not perfect agreement.
    for (_, p), before in zip(parameters, values):
        assert p.grad is None
        torch.testing.assert_close(p, before, rtol=0, atol=0)


def test_loss_refactor_preserves_formula_and_complete_collection_weights():
    rows = [dict(weight=1., **{'return': 1.}, old_value=.3),
            *[dict(weight=1/3, **{'return': -1.}, old_value=-.2) for _ in range(3)]]
    config = dict(value_coefficient=.7, entropy=.04, parent_kl=.03, clip=.12)
    prepare_training_records(rows, config)
    assert [r['actor_weight'] for r in rows] == [1.]*4
    np.testing.assert_allclose([r['value_weight'] for r in rows], [2., 2/3, 2/3, 2/3])
    logits = torch.tensor([[.2, -.1], [.5, -.4], [.3, .7], [-.2, .8]], requires_grad=True)
    values = torch.tensor([.1, -.2, -.3, -.4], requires_grad=True)
    class Actor:
        def training_forward(self, _):
            return logits, values
    data = {k: torch.tensor([r[k] for r in rows]) for k in
            ('return','advantage','actor_weight','value_weight','entropy_weight','parent_kl_weight')}
    data.update(slot=torch.tensor([0,1,0,1]), old_logprob=torch.tensor([-.5,-1.,-.9,-.2]),
                parent_logp=torch.zeros(4,2).log_softmax(-1))
    logs, terms = training_loss_terms(Actor(), None, data, config)
    expected_policy = clipped_surrogate(logs.gather(1,data['slot'][:,None]).squeeze(1)-data['old_logprob'],
                                        data['advantage'], data['actor_weight'], .12)
    expected_value = (data['value_weight']*F.smooth_l1_loss(values,data['return'],reduction='none')).mean()
    expected_entropy = -(data['entropy_weight']*(logs.exp()*logs).sum(-1)).mean()
    expected_kl = (data['parent_kl_weight']*categorical_kl(data['parent_logp'],logs)).mean()
    old = expected_policy + .7*expected_value - .04*expected_entropy + .03*expected_kl
    new = sum(weighted_training_terms(terms,config).values())
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    old_grad=torch.autograd.grad(old,(logits,values),retain_graph=True)
    new_grad=torch.autograd.grad(new,(logits,values))
    for a,b in zip(old_grad,new_grad):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
