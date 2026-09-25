"""Build the arm-C initialization: the champion G5 core plus a zero-output afterstate branch.

Every champion tensor is loaded unchanged; only the ``afterstate.`` branch is
new, and its output projection is zero. The script verifies on recorded public
replay decisions that the new model's logits, values and auxiliary heads equal
the champion's (fp32), that host-computed and explicit afterstates agree, and
that the zero projection still receives a gradient, then writes the checkpoint
and a JSON audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def replay_batches(shard, rows, batch):
    z = np.load(shard)
    offsets = z["offsets"]
    picks = np.linspace(0, len(offsets) - 2, rows).astype(int)
    for start in range(0, len(picks), batch):
        chosen = picks[start:start + batch]
        sizes = [int(offsets[i + 1] - offsets[i]) for i in chosen]
        width = max(32, max(sizes))
        actions = np.full((len(chosen), width), -1, np.int64)
        costs = np.zeros((len(chosen), width), np.float32)
        mask = np.zeros((len(chosen), width), bool)
        for j, (i, n) in enumerate(zip(chosen, sizes)):
            actions[j, :n] = z["actions"][offsets[i]:offsets[i + 1]]
            costs[j, :n] = z["costs"][offsets[i]:offsets[i + 1]]
            mask[j, :n] = True
        inputs = tuple(torch.from_numpy(x) for x in (
            z["observation"][chosen].astype(np.float32), z["pill"][chosen].astype(np.int64),
            z["preview"][chosen].astype(np.int64), actions, costs, mask))
        yield inputs, torch.from_numpy(z["public_context"][chosen].astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True,
                        help="drmc-public-controller-replay-v2 .npz (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256, help="decisions per shard")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    from drmc_rl.models.policy.afterstate_full_core import BRANCH_PREFIX, afterstate_full_config, from_g5
    from drmc_rl.training.utils.checkpoint_io import load_checkpoint
    from tools.vs_head_to_head import PlainPolicy

    torch.manual_seed(args.seed)
    payload = load_checkpoint(args.champion, map_location="cpu")
    champion = PlainPolicy(args.champion, "cpu", public_only=True).net.eval()
    cfg = afterstate_full_config(payload["cfg"])
    net = from_g5(champion, cfg["smdp_ppo"])
    state = net.state_dict()
    for key, value in champion.state_dict().items():
        if not torch.equal(state[key], value):
            raise AssertionError(f"champion tensor changed: {key}")
    branch = {k: v for k, v in state.items() if k.startswith(BRANCH_PREFIX)}
    assert torch.count_nonzero(state["afterstate.out.weight"]) == 0
    assert torch.count_nonzero(state["afterstate.out.bias"]) == 0

    worst = dict(logits=0.0, value=0.0, value_logits=0.0, candidate_wdl=0.0, state_wdl=0.0)
    decisions = candidates = 0
    bitwise = True
    for shard in args.shard:
        for inputs, aux in replay_batches(shard, args.rows, 32):
            with torch.inference_mode():
                ref_logits, ref_value, ref_extra = champion(*inputs, aux=aux, return_aux=True)
                logits, value, extra = net(*inputs, aux=aux, return_aux=True)
                after = net.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
                explicit, _ = net.forward_features(*inputs, aux, *after)
            mask = inputs[5]
            bitwise &= bool(torch.equal(logits, ref_logits) and torch.equal(value, ref_value))
            for name, a, b in (("logits", logits[mask], ref_logits[mask]), ("value", value, ref_value),
                               ("value_logits", extra["value_logits"], ref_extra["value_logits"]),
                               ("candidate_wdl", extra["candidate_wdl_logits"][mask],
                                ref_extra["candidate_wdl_logits"][mask]),
                               ("state_wdl", extra["state_wdl_logits"], ref_extra["state_wdl_logits"])):
                worst[name] = max(worst[name], float((a - b).abs().max()))
            assert torch.equal(explicit, logits)
            decisions += int(mask.shape[0]); candidates += int(mask.sum())
    if max(worst.values()) > 1e-6:
        raise AssertionError(f"initial model differs from the champion: {worst}")

    # The zero projection must learn: its gradient is the branch features times dL/dtoken.
    inputs, aux = next(replay_batches(args.shard[0], 8, 8))
    net.train(False)
    logits, value = net(*inputs, aux=aux)
    (logits.masked_fill(~inputs[5], 0).log_softmax(-1)[:, 0].sum() + value.sum()).backward()
    out_grad = float(net.afterstate.out.weight.grad.abs().sum())
    stem_grad = net.afterstate.stem.weight.grad
    assert out_grad > 0, "zero-initialized projection receives no gradient"
    net.zero_grad(set_to_none=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".next")
    torch.save(dict(cfg=cfg, state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                    parent_sha256=hashlib.sha256(args.champion.read_bytes()).hexdigest(),
                    initialization="champion-weights-plus-zero-output-afterstate-branch",
                    branch_seed=args.seed), temporary)
    temporary.replace(args.out)
    reloaded = PlainPolicy(args.out, "cpu", public_only=True).net.eval()
    inputs, aux = next(replay_batches(args.shard[0], 32, 32))
    with torch.inference_mode():
        assert torch.equal(reloaded(*inputs, aux=aux)[0], champion(*inputs, aux=aux)[0])
    audit = dict(
        schema="drmc-afterstate-full-init-audit-v1",
        champion=str(args.champion), champion_sha256=hashlib.sha256(args.champion.read_bytes()).hexdigest(),
        init=str(args.out), init_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        shards=[str(s) for s in args.shard], decisions=decisions, candidates=candidates,
        max_abs_difference=worst, bitwise_identical_logits_and_values=bitwise, tolerance=1e-6,
        champion_tensors_unchanged=len(champion.state_dict()),
        parameters=dict(champion=sum(p.numel() for p in champion.parameters()),
                        full=sum(p.numel() for p in net.parameters()),
                        branch=sum(v.numel() for v in branch.values())),
        zero_projection_gradient_l1=out_grad,
        upstream_stem_gradient_l1=float(stem_grad.abs().sum()) if stem_grad is not None else 0.0,
    )
    args.audit.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
