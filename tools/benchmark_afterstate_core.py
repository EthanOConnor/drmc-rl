"""Batch-1 cost of the afterstate core versus a G5 public core on recorded decisions.

Reports parameters, FLOPs (torch.utils.flop_counter), host afterstate time,
PyTorch CPU latency and, when onnx/onnxruntime are importable, the latency of
an fp16-weight ONNX export on the CPU execution provider (the browser's WASM
path runs the same graph). Models alternate within each repetition so shared
host load affects both equally; absolute numbers still depend on that load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch


def recorded_inputs(shard, widths):
    z = np.load(shard)
    off = z["offsets"]
    sizes = off[1:] - off[:-1]
    rows = []
    for width in widths:
        rows.append(int(np.argmin(np.abs(sizes - width))))
    out = []
    for i in rows:
        k = int(sizes[i])
        width = max(32, k)
        actions = np.full((1, width), -1, np.int64)
        costs = np.zeros((1, width), np.float32)
        mask = np.zeros((1, width), bool)
        actions[0, :k] = z["actions"][off[i]:off[i + 1]]
        costs[0, :k] = z["costs"][off[i]:off[i + 1]]
        mask[0, :k] = True
        tensors = tuple(torch.from_numpy(x) for x in (
            z["observation"][i:i + 1].astype(np.float32), z["pill"][i:i + 1].astype(np.int64),
            z["preview"][i:i + 1].astype(np.int64), actions, costs, mask))
        out.append((k, tensors, torch.from_numpy(z["public_context"][i:i + 1])))
    return out


class _PortableAttention(torch.nn.Module):
    """nn.MultiheadAttention algebra with symbolic sequence length for ONNX."""

    def __init__(self, original):
        super().__init__()
        self.w, self.b, self.out = original.in_proj_weight, original.in_proj_bias, original.out_proj
        self.heads, self.dim = original.num_heads, original.embed_dim

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, **_):
        batch, head = query.shape[0], self.dim // self.heads
        q, k, v = [torch.nn.functional.linear(x, self.w[i * self.dim:(i + 1) * self.dim],
                                              self.b[i * self.dim:(i + 1) * self.dim])
                   .reshape(batch, -1, self.heads, head).transpose(1, 2)
                   for i, x in enumerate((query, key, value))]
        scores = torch.matmul(q, k.transpose(-2, -1)) * head**-0.5
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        out = torch.matmul(scores.softmax(-1), v).transpose(1, 2).reshape(batch, -1, self.dim)
        return self.out(out), None


def _portable(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.MultiheadAttention):
            setattr(module, name, _PortableAttention(child))
        else:
            _portable(child)


def _fp16_weights(path):
    import onnx
    import onnx.helper
    import onnx.numpy_helper

    model = onnx.load(str(path))
    casts = []
    for tensor in model.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        values = onnx.numpy_helper.to_array(tensor)
        if values.size < 1024:
            continue
        name = tensor.name
        tensor.CopyFrom(onnx.numpy_helper.from_array(values.astype(np.float16), f"{name}.fp16"))
        casts.append(onnx.helper.make_node("Cast", [f"{name}.fp16"], [name], to=onnx.TensorProto.FLOAT))
    nodes = casts + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.save(model, str(path))
    return Path(path).stat().st_size


def export_onnx(net, example, afterstate, path, threads):
    import copy

    import onnxruntime as ort

    model = copy.deepcopy(net).eval()
    _portable(model)
    is_after = hasattr(model, "forward_features")

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            if is_after:
                return self.inner.forward_features(*args)
            return self.inner(*args[:6], aux=args[6])

    args = (*example[1], example[2], *afterstate) if is_after else (*example[1], example[2])
    names = ["board", "pill", "preview", "actions", "costs", "mask", "aux"] + (["after_tiles", "facts"] if is_after else [])
    axes = {n: {1: "candidates"} for n in ("actions", "costs", "mask", "after_tiles", "facts") if n in names}
    import warnings
    # Tracing records the batch-1 bottle stack (2 bottles) as a constant; the
    # browser also runs batch 1.
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(Wrapper(model), args, str(path), input_names=names, output_names=["logits", "value"],
                          opset_version=18, dynamic_axes=axes, dynamo=False)
    size = _fp16_weights(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session, names, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, help="a drmc-public-controller-replay-v2 .npz")
    parser.add_argument("--student", help="afterstate checkpoint (default: untrained default config)")
    parser.add_argument("--teacher", required=True, help="G5 public-context checkpoint")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--widths", default="32,57")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from drmc_rl.models.policy.afterstate_core import AfterstateCorePolicyNet, afterstate_core_config
    from tools.eval_policy import _build_net_from_cfg
    from tools.vs_head_to_head import PlainPolicy

    teacher = PlainPolicy(Path(args.teacher), "cpu", public_only=True).net.eval()
    if args.student:
        student = PlainPolicy(Path(args.student), "cpu", public_only=True).net.eval()
    else:
        student, _, _ = _build_net_from_cfg(afterstate_core_config(), 20, "cpu")
        student.eval()
    report = dict(threads=args.threads, repeats=args.repeats,
                  parameters=dict(afterstate=sum(p.numel() for p in student.parameters()),
                                  teacher=sum(p.numel() for p in teacher.parameters())),
                  widths={})
    from torch.utils.flop_counter import FlopCounterMode

    for k, inputs, aux in recorded_inputs(args.shard, [int(w) for w in args.widths.split(",")]):
        afterstate = AfterstateCorePolicyNet.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
        entry = dict(candidates=k)
        for name, net, call in (("afterstate", student, lambda: student(*inputs, aux=aux, afterstate=afterstate)),
                                ("teacher", teacher, lambda: teacher(*inputs, aux=aux))):
            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                call()
            entry[f"{name}_gflops"] = counter.get_total_flops() / 1e9
        timings = {"afterstate": [], "teacher": [], "host_afterstate": []}
        with torch.inference_mode():
            for _ in range(3):
                student(*inputs, aux=aux, afterstate=afterstate); teacher(*inputs, aux=aux)
            for _ in range(args.repeats):
                t = time.perf_counter(); AfterstateCorePolicyNet.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
                timings["host_afterstate"].append(time.perf_counter() - t)
                t = time.perf_counter(); student(*inputs, aux=aux, afterstate=afterstate)
                timings["afterstate"].append(time.perf_counter() - t)
                t = time.perf_counter(); teacher(*inputs, aux=aux)
                timings["teacher"].append(time.perf_counter() - t)
        for key, values in timings.items():
            entry[f"torch_{key}_ms_median"] = 1000 * statistics.median(values)
        if args.onnx:
            with tempfile.TemporaryDirectory() as tmp:
                sessions = {}
                for name, net in (("afterstate", student), ("teacher", teacher)):
                    session, names, size = export_onnx(net, (k, inputs, aux), afterstate, Path(tmp) / f"{name}.onnx", args.threads)
                    values = [*[t.numpy() for t in inputs], aux.numpy()]
                    if name == "afterstate":
                        values += [afterstate[0].numpy(), afterstate[1].numpy()]
                    feed = dict(zip(names, values))
                    reference = (net.forward_features(*inputs, aux, *afterstate) if name == "afterstate"
                                 else net(*inputs, aux=aux))[0]
                    got = session.run(["logits"], feed)[0]
                    valid = inputs[5].numpy()
                    entry[f"onnx_{name}_fp16w_mb"] = size / 1e6
                    entry[f"onnx_{name}_max_logit_error"] = float(np.abs(got[valid] - reference.detach().numpy()[valid]).max())
                    sessions[name] = (session, feed)
                ort_times = {n: [] for n in sessions}
                for _ in range(3):
                    for session, feed in sessions.values():
                        session.run(None, feed)
                for _ in range(args.repeats):
                    for name, (session, feed) in sessions.items():
                        t = time.perf_counter(); session.run(None, feed); ort_times[name].append(time.perf_counter() - t)
                for name, values in ort_times.items():
                    entry[f"onnx_{name}_ms_median"] = 1000 * statistics.median(values)
        report["widths"][str(k)] = entry
        print(json.dumps(entry), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
