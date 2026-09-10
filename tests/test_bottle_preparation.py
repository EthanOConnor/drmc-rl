from copy import deepcopy

import pytest
import torch

from drmc_rl.game.public_context import PUBLIC_CONTEXT_DIM, PUBLIC_CONTEXT_SCHEMA
from drmc_rl.models.policy.bottle_preparation import prepare_bottle
from drmc_rl.models.policy.candidate_policy_g5 import G5CandidatePlacementPolicyNet


def model(**overrides):
    config = dict(in_channels=20, board_channels=16, encoder_blocks=2, d_model=32,
        pill_embed_dim=16, aux_dim=PUBLIC_CONTEXT_DIM, public_context_schema=PUBLIC_CONTEXT_SCHEMA,
        cand_hidden_dim=48, transformer_heads=4, cross_layers=1, interaction_layers=1,
        patch_kernel=3, value_atoms=21, conditioned_trunk=False)
    config.update(overrides)
    return G5CandidatePlacementPolicyNet(**config).eval()


def inputs(batch=1):
    return dict(obs=torch.rand(batch, 20, 16, 8),
        pill_colors=torch.tensor([[0, 1]]).expand(batch, -1),
        preview_pill_colors=torch.tensor([[0, 1]]).expand(batch, -1),
        cand_actions=torch.arange(32).expand(batch, -1),
        cand_cost=torch.rand(batch, 32)*40,
        cand_mask=torch.arange(32).expand(batch, -1)<24,
        aux=torch.rand(batch, PUBLIC_CONTEXT_DIM))


@pytest.mark.parametrize("both", [False, True])
def test_nine_previews_share_only_bottle_features_with_fresh_context_and_candidates(both):
    torch.manual_seed(407)
    net = model()
    x = inputs(9)
    x["obs"][:, :8] = x["obs"][0, :8].clone()
    x["preview_pill_colors"] = torch.cartesian_prod(torch.arange(3), torch.arange(3))
    if both:
        x["obs"][:, 8:16] = x["obs"][0, 8:16].clone()
    calls = []
    hook = net.bottle.register_forward_hook(lambda module, args, output: calls.append(len(args[0])))
    with torch.inference_mode():
        reference = net(**x, return_aux=True)
        calls.clear()
        own = prepare_bottle(net, x["obs"][:1, :8])
        opponent = prepare_bottle(net, x["obs"][:1, 8:16]) if both else None
        actual = net(**x, prepared_bottles=(own, opponent), return_aux=True)
        assert len(calls) == 2  # Two preparations, or own preparation + fresh opponent.
        assert sum(calls) == (2 if both else 10)
        for a, b in zip(actual[:2], reference[:2], strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        for key in reference[2]:
            torch.testing.assert_close(actual[2][key], reference[2][key], atol=1e-5, rtol=1e-5)
        # The late path is still sensitive to actual changed public context.
        changed = dict(x, aux=x["aux"]+1)
        changed_result = net(**changed, prepared_bottles=(own, opponent))
        assert not torch.allclose(changed_result[0], actual[0])
    hook.remove()


@pytest.mark.parametrize("change", ["board", "weights", "model", "features", "training", "autocast"])
def test_stale_or_incompatible_preparations_are_rejected(change):
    net, x = model(), inputs()
    with torch.no_grad():
        cached = prepare_bottle(net, x["obs"][:, :8])
        if change == "board":
            x["obs"][0, 0, 0, 0] += 1
        elif change == "weights":
            net.bottle.stem.weight.add_(.01)
        elif change == "model":
            net = deepcopy(net)
        elif change == "features":
            cached._features.add_(.01)
        elif change == "training":
            net.train()
        if change == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
                net(**x, prepared_bottles=(cached, None))
        else:
            with pytest.raises(ValueError):
                net(**x, prepared_bottles=(cached, None))


def test_conditioned_models_cannot_silently_drop_preview_or_history():
    net, x = model(conditioned_trunk=True), inputs()
    with torch.no_grad(), pytest.raises(ValueError, match="conditioned encoder"):
        prepare_bottle(net, x["obs"][:, :8])


def test_encoder_update_during_preparation_is_rejected():
    net, x = model(), inputs()
    def update(module, args, result):
        module.stem.weight.add_(.01)
    hook = net.bottle.register_forward_hook(update)
    with torch.no_grad(), pytest.raises(ValueError, match="changed during"):
        prepare_bottle(net, x["obs"][:, :8])
    hook.remove()


def test_preparation_cannot_detach_training_and_source_input_is_not_aliased():
    net, x = model(), inputs()
    with pytest.raises(ValueError, match="inference-only"):
        prepare_bottle(net, x["obs"][:, :8])
    with torch.no_grad():
        cached = prepare_bottle(net, x["obs"][:, :8])
        original = x["obs"].clone()
        x["obs"].zero_()
        # Mutating the source has not changed the cached bytes or features.
        net(**dict(x, obs=original), prepared_bottles=(cached, None))
    with pytest.raises(ValueError, match="inference-only"):
        net(**dict(x, obs=original), prepared_bottles=(cached, None))
    net.train()
    logits, value = net(**dict(x, obs=original))
    (logits[:, :24].sum()+value.sum()).backward()
    assert net.bottle.stem.weight.grad.abs().sum() > 0
