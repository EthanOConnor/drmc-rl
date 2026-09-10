from copy import deepcopy

import pytest
import torch

from drmc_rl.training.model_growth import grow_bottle_encoder
from drmc_rl.training.quality_supervision import upgrade_public_model
from tests.test_quality_supervision import checkpoint
from tools.eval_policy import _build_net_from_cfg


def inputs(aux_dim):
    mask = torch.arange(17)[None, :] < torch.tensor([5, 11, 17])[:, None]
    return dict(
        obs=torch.rand(3, 20, 16, 8),
        pill_colors=torch.tensor([[0, 1], [2, 0], [1, 2]]),
        preview_pill_colors=torch.tensor([[2, 2], [1, 1], [0, 0]]),
        cand_actions=torch.arange(17).expand(3, -1),
        cand_cost=torch.rand(3, 17) * 50,
        cand_mask=mask,
        aux=torch.randn(3, aux_dim),
        return_aux=True,
    )


@pytest.mark.parametrize("channels,blocks", [(24, 3), (32, 2), (16, 3)])
def test_grown_encoder_preserves_all_heads_and_new_capacity_learns(channels, blocks):
    torch.set_num_threads(1)
    torch.manual_seed(371)
    source, _ = checkpoint()
    parent, cfg = upgrade_public_model(source, mode="combined", device="cpu")
    with torch.no_grad():
        parent.state_wdl_head.weight.normal_(std=0.1)
        parent.candidate_wdl_head.weight.normal_(std=0.1)
    source = dict(cfg=cfg, state_dict=deepcopy(parent.state_dict()))
    snapshot = deepcopy(source)
    rng = torch.random.get_rng_state()
    grown, grown_cfg, metadata = grow_bottle_encoder(
        source,
        channels=channels,
        blocks=blocks,
        seed=29,
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    assert metadata["token_width"] == 16
    assert metadata["parameters"] > metadata["parent_parameters"]
    request = inputs(parent.aux_dim)
    expected, actual = parent(**request), grown(**request)
    for a, b in zip(expected[:2], actual[:2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
    for key in ("candidate_context", "candidate_wdl_logits", "state_wdl_logits"):
        torch.testing.assert_close(expected[2][key], actual[2][key], rtol=1e-5, atol=2e-6)
    optimizer = torch.optim.AdamW(grown.parameters(), lr=1e-3)
    loss = actual[0][:, 0].mean() + actual[1].square().mean()
    loss.backward()
    assert grown.bottle.blocks[-1].conv2.weight.grad.abs().sum() > 0
    if channels > 16:
        assert grown.bottle_projection.weight.grad[:, 16:].abs().sum() > 0
        assert grown.bottle.blocks[0].conv2.weight.grad[:16, 16:].abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    grown(**request)[0][:, 0].mean().backward()
    if channels > 16:
        assert grown.bottle.stem.weight.grad[16:].abs().sum() > 0
    reloaded, _, _ = _build_net_from_cfg(grown_cfg, 20, "cpu")
    reloaded.load_state_dict(grown.state_dict(), strict=True)
    torch.testing.assert_close(reloaded(**request)[0], grown(**request)[0], rtol=0, atol=0)
    for key, value in snapshot["state_dict"].items():
        torch.testing.assert_close(source["state_dict"][key], value, rtol=0, atol=0)
    assert source["cfg"] == snapshot["cfg"]


def test_repeated_growth_retains_learned_projection_and_normalization_partitions():
    torch.manual_seed(721)
    source, _ = checkpoint()
    first, cfg, _ = grow_bottle_encoder(source, channels=24, blocks=2, seed=7)
    with torch.no_grad():
        first.bottle_projection.weight.normal_(std=0.1)
        first.bottle_projection.bias.normal_(std=0.01)
    second, cfg2, _ = grow_bottle_encoder(
        dict(cfg=cfg, state_dict=first.state_dict()),
        channels=32,
        blocks=3,
        seed=8,
    )
    assert cfg2["candidate_bottle_norm_partitions"] == [16, 8, 8]
    request = inputs(first.aux_dim)
    request["aux"].zero_()
    for a, b in zip(first(**request)[:2], second(**request)[:2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)


def test_growth_seeds_change_new_features_without_changing_parent_subnetwork(monkeypatch):
    source, _ = checkpoint()

    def unexpected_cuda_seed(*args, **kwargs):
        raise AssertionError("CPU growth must not modify any CUDA generator")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_cuda_seed)
    a, _, _ = grow_bottle_encoder(source, channels=24, blocks=2, seed=3)
    b, _, _ = grow_bottle_encoder(source, channels=24, blocks=2, seed=4)
    assert torch.equal(a.bottle.stem.weight[:16], b.bottle.stem.weight[:16])
    assert not torch.equal(a.bottle.stem.weight[16:], b.bottle.stem.weight[16:])
    request = inputs(a.aux_dim)
    request["aux"].zero_()
    for x, y in zip(a(**request)[:2], b(**request)[:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for channels, blocks in ((8, 2), (24, 0), (24.0, 2)):
        with pytest.raises(ValueError):
            grow_bottle_encoder(source, channels=channels, blocks=blocks, seed=5)
