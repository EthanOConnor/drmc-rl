"""Grow the dense bottle encoder without changing its initial public policy.

The token width stays fixed. Existing channels retain their GroupNorm groups;
new channels have separate groups and initially cannot feed existing channels.
A learned projection keeps the attention/candidate interface unchanged. Added
residual blocks start as identities. New random features feed zero-initialized
readouts, which receive gradients immediately rather than leaving a dead branch.
"""

from __future__ import annotations

from copy import deepcopy

import torch


def grow_bottle_encoder(checkpoint, *, channels, blocks, seed, device="cpu"):
    from tools.eval_policy import _build_net_from_cfg

    cfg = deepcopy(checkpoint["cfg"])
    sp = cfg.get("smdp_ppo", cfg)
    if (
        sp.get("candidate_architecture") != "g5"
        or sp.get("candidate_bottle_block", "dense") != "dense"
    ):
        raise ValueError("bottle growth requires the dense G5 architecture")
    token_width = int(sp.get("candidate_d_model", 128))
    old_width = int(sp.get("candidate_bottle_channels") or token_width)
    old_blocks = int(sp.get("encoder_blocks", 0))
    if type(channels) is not int or type(blocks) is not int:
        raise ValueError("encoder width and depth must be integers")
    if channels < old_width or blocks < old_blocks or min(channels, blocks) < 1:
        raise ValueError("bottle growth cannot shrink its parent")
    if channels > old_width:
        partitions = list(sp.get("candidate_bottle_norm_partitions") or [old_width])
        if sum(partitions) != old_width:
            raise ValueError("parent normalization partitions do not match its width")
        sp["candidate_bottle_norm_partitions"] = [*partitions, channels - old_width]
    sp.update(candidate_bottle_channels=channels, encoder_blocks=blocks)
    # Construction must not change caller/sampler RNG state, including CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    original = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    state = net.state_dict()
    consumed = set()

    def take(name):
        consumed.add(name)
        return original[name].detach().cpu()

    with torch.no_grad():
        for name, value in state.items():
            if not name.startswith(("bottle.", "bottle_projection.")):
                old = take(name)
                if old.shape != value.shape:
                    raise ValueError(f"growth would alter the fixed token/head interface: {name}")
                value.copy_(old)
        for suffix in ("weight", "bias"):
            name = "bottle.stem." + suffix
            state[name][:old_width].copy_(take(name))
        for index in range(old_blocks):
            prefix = f"bottle.blocks.{index}."
            for norm in ("norm1", "norm2"):
                for suffix in ("weight", "bias"):
                    name = prefix + norm + "." + suffix
                    state[name][:old_width].copy_(take(name))
            for conv in ("conv1", "conv2"):
                name = prefix + conv + ".weight"
                state[name][:old_width].zero_()
                state[name][:old_width, :old_width].copy_(take(name))
                name = prefix + conv + ".bias"
                state[name][:old_width].copy_(take(name))
            for suffix in ("weight", "bias"):
                name = prefix + "film." + suffix
                # FiLM concatenates scale/shift for both convolutions. A plain
                # prefix copy would mix the four learned fields when widening.
                state[name].reshape(4, channels, -1)[:, :old_width].copy_(
                    take(name).reshape(4, old_width, -1)
                )
        for index in range(old_blocks, blocks):
            for suffix in ("weight", "bias"):
                state[f"bottle.blocks.{index}.conv2.{suffix}"].zero_()
        if channels != token_width:
            projection = state["bottle_projection.weight"]
            projection.zero_()
            state["bottle_projection.bias"].zero_()
            if old_width == token_width:
                index = torch.arange(token_width)
                projection[index, index, 0, 0] = 1
            else:
                projection[:, :old_width].copy_(take("bottle_projection.weight"))
                state["bottle_projection.bias"].copy_(take("bottle_projection.bias"))
        if consumed != set(original):
            raise ValueError(
                f"growth would discard learned tensors: {sorted(set(original) - consumed)}"
            )
    net.load_state_dict(state, strict=True)
    metadata = dict(
        method="partitioned-dense-bottle-growth-v1",
        seed=int(seed),
        parent_channels=old_width,
        parent_blocks=old_blocks,
        channels=channels,
        blocks=blocks,
        token_width=token_width,
        normalization_partitions=sp.get("candidate_bottle_norm_partitions", [channels]),
        parent_parameters=sum(v.numel() for v in original.values()),
        parameters=sum(p.numel() for p in net.parameters()),
        policy_preservation="algebraic initialization; verify FP32 outputs on target hardware",
    )
    return net.to(device).eval(), cfg, metadata
