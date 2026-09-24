"""Distil a public-context teacher core into the afterstate core.

``build`` samples exact public model inputs from controller replay shards
(``drmc-public-controller-replay-v2``), adds every candidate's exact afterstate
and scores the complete frontier with the frozen teacher. ``train`` fits the
student to the teacher's candidate distribution and value distribution on
whole-seed training games and reports agreement on held-out reset seeds.

The teacher sees its original inputs; the student sees the same inputs plus
afterstates computed from them. Replay outcomes are stored for diagnostics
only; distillation targets are teacher outputs.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import glob
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

DATASET_SCHEMA = "drmc-afterstate-distillation-v1"
PACES = ("sloth", "relaxed", "normal", "fast", "top_humans", "super_human", "frame_perfect")
FACT_OFFSET = 32  # height_change is stored with this offset in uint8 facts


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _afterstate_chunk(args):
    from drmc_rl.game.afterstate import FACT_NAMES, resolve_placement

    fields, pills, offsets, actions = args
    change = FACT_NAMES.index("height_change")
    facts = np.zeros((len(actions), len(FACT_NAMES)), np.uint8)
    counts = np.zeros(len(actions), np.int32)
    cells, values = [], []
    for s in range(len(fields)):
        root = fields[s]
        for c in range(offsets[s], offsets[s + 1]):
            after, raw = resolve_placement(root, pills[s], int(actions[c]))
            raw = raw.copy()
            raw[change] += FACT_OFFSET
            if raw.min() < 0 or raw.max() > 255:
                raise ValueError("fact outside the uint8 storage range")
            facts[c] = raw.astype(np.uint8)
            after = np.frombuffer(after, np.uint8)
            changed = np.flatnonzero(after != root)
            counts[c] = len(changed)
            cells.append(changed.astype(np.uint8))
            values.append(after[changed])
    return facts, counts, np.concatenate(cells), np.concatenate(values)


def _sample_shard(path, fraction, rng):
    from drmc_rl.game.afterstate import planes_to_fields

    with np.load(path) as z:
        meta = json.loads(str(z["metadata"]))
        if meta["schema"] != "drmc-public-controller-replay-v2" or meta["observation_schema"] != "public_pair_context_v3":
            raise ValueError(f"unexpected replay contract in {path}")
        take = np.flatnonzero(rng.random(len(z["pill"])) < fraction)
        if not len(take):
            return None
        obs = z["observation"][take]
        off = z["offsets"]
        index = np.concatenate([np.arange(off[i], off[i + 1]) for i in take])
        return dict(
            own=planes_to_fields(obs[:, :8]), opp=planes_to_fields(obs[:, 8:16]),
            pill=z["pill"][take], preview=z["preview"][take], game_seed=z["game_seed"][take],
            port=z["learner_port"][take], ctx=z["public_context"][take].astype(np.float16),
            ret=z["return"][take].astype(np.float32),
            pace=np.full(len(take), PACES.index(meta["pace"]), np.int8),
            update=np.full(len(take), meta["update"], np.int16),
            counts=(off[take + 1] - off[take]).astype(np.int32),
            actions=z["actions"][index], costs=z["costs"][index], obs_check=obs[:2],
        )


def build(args):
    """Stream replay shards into self-contained dataset parts (bounded host memory)."""
    import torch

    from tools.vs_head_to_head import PlainPolicy

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(args.shards))
    if not shards:
        raise SystemExit("no replay shards")
    sizes = []
    for path in shards:
        with np.load(path) as z:
            sizes.append(len(z["pill"]))
    fraction = min(1.0, args.max_states / int(sum(sizes)))
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    teacher = PlainPolicy(Path(args.teacher), args.device, public_only=True)
    net = teacher.net.eval()
    teacher_sha = sha256(args.teacher)
    started = time.monotonic()
    parts_written = []
    with get_context("spawn").Pool(args.workers) as pool:
        for part, s0 in enumerate(range(0, len(shards), args.shards_per_part)):
            path = out / f"part-{part:03d}.npz"
            group = shards[s0:s0 + args.shards_per_part]
            sampled = [x for x in (_sample_shard(p, fraction, rng) for p in group) if x is not None]
            if path.exists():  # resumable: sampling above keeps the RNG stream identical
                parts_written.append(path.name)
                continue
            data = {k: np.concatenate([x[k] for x in sampled]) for k in sampled[0]}
            obs_check = data.pop("obs_check")
            _check_reconstruction(obs_check)
            n = len(data["pill"])
            offsets = np.concatenate(([0], np.cumsum(data["counts"]))).astype(np.int64)
            jobs = []
            for a in range(0, n, 1024):
                b = min(n, a + 1024)
                jobs.append((data["own"][a:b], data["pill"][a:b], offsets[a:b + 1] - offsets[a],
                             data["actions"][offsets[a]:offsets[b]]))
            results = pool.map(_afterstate_chunk, jobs, chunksize=1)
            data["facts"] = np.concatenate([r[0] for r in results])
            delta_counts = np.concatenate([r[1] for r in results])
            data["delta_offsets"] = np.concatenate(([0], np.cumsum(delta_counts))).astype(np.int64)
            data["delta_cells"] = np.concatenate([r[2] for r in results])
            data["delta_values"] = np.concatenate([r[3] for r in results])
            del results, jobs
            data["offsets"] = offsets
            loader = _Batcher(data, offsets, args.device)
            logits = np.zeros(offsets[-1], np.float32)
            value_logits = np.zeros((n, net.value_atoms), np.float16)
            for a in range(0, n, args.teacher_batch):
                rows = np.arange(a, min(n, a + args.teacher_batch))
                inputs, aux, valid = loader.inputs(rows, width_floor=32)
                with torch.inference_mode():
                    lg, _v, extra = net(*inputs, aux=aux, return_aux=True)
                logits[offsets[rows[0]]:offsets[rows[-1] + 1]] = lg.float().cpu().numpy()[valid.cpu().numpy()]
                value_logits[rows] = extra["value_logits"].float().cpu().numpy().astype(np.float16)
            del loader
            data["teacher_logits"] = logits
            data["teacher_value_logits"] = value_logits
            tmp = path.with_suffix(".next.npz")
            np.savez(tmp, **data)
            tmp.replace(path)
            parts_written.append(path.name)
            print(json.dumps(dict(part=part, states=n, candidates=int(offsets[-1]),
                                  mean_changed=float(delta_counts.mean()), max_changed=int(delta_counts.max()),
                                  seconds=round(time.monotonic() - started))), flush=True)
            del data
    meta = dict(schema=DATASET_SCHEMA, created_at=datetime.now(UTC).isoformat(),
                teacher=str(args.teacher), teacher_sha256=teacher_sha,
                shards=len(shards), shard_glob=args.shards, sampled_fraction=fraction,
                selection_seed=args.seed, tf32=bool(args.tf32), fact_offset=FACT_OFFSET, paces=PACES,
                parts=parts_written, reconstruction_check="semantic planes round-trip exactly on two rows per shard")
    (out / "dataset.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


def _check_reconstruction(observations):
    from drmc_rl.game.afterstate import planes_to_fields

    tables = _plane_table()
    for obs in observations:
        for side in (0, 8):
            tiles = planes_to_fields(obs[side:side + 8])
            planes = tables[tiles].T.reshape(8, 16, 8)
            if not np.array_equal(planes.astype(np.uint8), obs[side:side + 8]):
                raise ValueError("semantic planes do not round-trip through tile bytes")
    return dict(observations=int(len(observations)), exact=True)


def _plane_table():
    from drmc_rl.models.policy.afterstate_core import _tile_plane_table

    return _tile_plane_table().numpy()


class _Batcher:
    """Rebuild exact model inputs (and afterstates) for ragged dataset rows on one device."""

    def __init__(self, data, offsets, device):
        import torch

        self.device = device

        def t(a, dtype=None):
            if not isinstance(a, torch.Tensor):
                a = torch.from_numpy(np.ascontiguousarray(a))
            return a.to(device=device, dtype=dtype)

        self.own = t(data["own"])
        self.opp = t(data["opp"])
        self.pill = t(data["pill"], torch.long)
        self.preview = t(data["preview"], torch.long)
        self.ctx = t(data["ctx"])
        self.offsets = t(offsets, torch.long)
        self.actions = t(data["actions"], torch.int32)
        self.costs = t(data["costs"], torch.int32)
        self.table = t(_plane_table())
        if "facts" in data:
            from drmc_rl.game.afterstate import _FACT_SCALE, FACT_NAMES

            self.facts = t(data["facts"])
            shift = np.zeros(len(FACT_NAMES), np.float32)
            shift[FACT_NAMES.index("height_change")] = FACT_OFFSET
            self.fact_shift = t(shift)
            self.fact_scale = t(_FACT_SCALE)
            self.delta_offsets = t(data["delta_offsets"], torch.long)
            self.delta_cells = t(data["delta_cells"], torch.long)
            self.delta_values = t(data["delta_values"])
        for key in ("teacher_logits", "teacher_value_logits"):
            if key in data:
                setattr(self, key, t(data[key]))


    def candidates(self, rows, width_floor=32):
        import torch

        rows = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        start = self.offsets[rows]
        count = self.offsets[rows + 1] - start
        width = max(int(width_floor), int(count.max()))
        slot = torch.arange(width, device=self.device)
        valid = slot.unsqueeze(0) < count.unsqueeze(1)
        index = torch.where(valid, start.unsqueeze(1) + slot, torch.zeros_like(start).unsqueeze(1))
        return rows, index, valid

    def inputs(self, rows, width_floor=32, afterstate=False):
        import torch

        rows, index, valid = self.candidates(rows, width_floor)
        batch, width = valid.shape
        actions = torch.where(valid, self.actions[index].long(), torch.full_like(index, -1))
        costs = torch.where(valid, self.costs[index].float(), torch.zeros_like(index, dtype=torch.float32))
        own, opp = self.own[rows].long(), self.opp[rows].long()
        planes = torch.cat((self.table[own], self.table[opp]), dim=-1)  # [B,128,16]
        planes = planes.transpose(1, 2).reshape(batch, 16, 16, 8)
        feasible = torch.zeros((batch, 512), device=self.device)
        feasible.scatter_(1, actions.clamp_min(0), valid.float())
        obs = torch.cat((planes, feasible.reshape(batch, 4, 16, 8)), dim=1)
        inputs = (obs, self.pill[rows], self.preview[rows], actions, costs, valid)
        aux = self.ctx[rows].float()
        if not afterstate:
            return inputs, aux, valid
        after = own.to(torch.uint8).unsqueeze(1).expand(-1, width, -1).clone()
        flat_candidates = index[valid]
        d0 = self.delta_offsets[flat_candidates]
        dn = self.delta_offsets[flat_candidates + 1] - d0
        owner = torch.repeat_interleave(torch.arange(len(flat_candidates), device=self.device), dn)
        position = torch.arange(int(dn.sum()), device=self.device) - torch.repeat_interleave(
            torch.cumsum(dn, 0) - dn, dn) + d0[owner]
        b_index, k_index = valid.nonzero(as_tuple=True)
        after[b_index[owner], k_index[owner], self.delta_cells[position]] = self.delta_values[position]
        facts = torch.zeros((batch, width, self.facts.shape[1]), device=self.device)
        facts[valid] = (self.facts[flat_candidates].float() - self.fact_shift) / self.fact_scale
        return inputs, aux, valid, (after, facts), index


def load_parts(directory, device):
    """Concatenate dataset parts on the device, rebasing ragged offsets."""
    import torch

    meta = json.loads((Path(directory) / "dataset.json").read_text())
    ragged = ("offsets", "delta_offsets")
    pieces, host = {}, {"game_seed": [], "pace": []}
    base = {"offsets": 0, "delta_offsets": 0}
    for name in meta["parts"]:
        with np.load(Path(directory) / name) as z:
            for key in z.files:
                value = z[key]
                if key in ragged:
                    value = value[:-1] + base[key]
                    base[key] += int(z[key][-1])
                if key in host:
                    host[key].append(value)
                pieces.setdefault(key, []).append(torch.from_numpy(np.ascontiguousarray(value)).to(device))
    data = {k: torch.cat(v) for k, v in pieces.items()}
    for key in ragged:
        data[key] = torch.cat((data[key], torch.tensor([base[key]], device=device, dtype=data[key].dtype)))
    return meta, data, {k: np.concatenate(v) for k, v in host.items()}


def _split(data, holdout_mod):
    """Train on non-held-out seeds outside the evaluation reserve; validate on the held-out ones."""
    from drmc_rl.program.seed_reserve import load_reserve

    seeds = data["game_seed"].astype(np.int64)
    held = (seeds % holdout_mod) == 0
    reserved = np.isin(seeds, np.fromiter(load_reserve().blocked, dtype=np.int64))
    return np.flatnonzero(~held & ~reserved), np.flatnonzero(held)


def evaluate(net, batcher, rows, batch=256):
    import torch

    totals = dict(states=0, agree=0.0, kl=0.0, value_abs=0.0, teacher_top_prob=0.0)
    per_pace = {}
    net.eval()
    for s0 in range(0, len(rows), batch):
        part = rows[s0:s0 + batch]
        inputs, aux, valid, after, index = batcher.inputs(part, afterstate=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=batcher.device.startswith("cuda")):
            logits, value, extra = net(*inputs, aux=aux, afterstate=after, return_aux=True)
        stats = _distill_terms(net, logits.float(), extra["value_logits"].float(), batcher, index, valid, part)
        for key in ("agree", "kl", "value_abs", "teacher_top_prob"):
            totals[key] += float(stats[key].sum())
        totals["states"] += len(part)
        pace = batcher.pace_np[part]
        for p in np.unique(pace):
            m = torch.as_tensor(pace == p, device=stats["agree"].device)
            entry = per_pace.setdefault(PACES[int(p)], dict(states=0, agree=0.0))
            entry["states"] += int(m.sum())
            entry["agree"] += float(stats["agree"][m].sum())
    n = max(1, totals.pop("states"))
    result = {k: v / n for k, v in totals.items()}
    result["states"] = n
    result["per_pace_agree"] = {p: v["agree"] / max(1, v["states"]) for p, v in per_pace.items()}
    return result


def _distill_terms(net, logits, value_logits, batcher, index, valid, rows):
    import torch

    teacher = torch.where(valid, batcher.teacher_logits[index], torch.full_like(logits, -1e9))
    target = teacher.log_softmax(-1)
    student = logits.masked_fill(~valid, -1e9).log_softmax(-1)
    p = target.exp()
    kl = (p * (target - student)).masked_fill(~valid, 0.0).sum(-1)
    agree = (student.argmax(-1) == target.argmax(-1)).float()
    tv = batcher.teacher_value_logits[torch.as_tensor(rows, device=logits.device)].float()
    support = net.value_support.float()
    teacher_value = (tv.softmax(-1) * support).sum(-1)
    student_value = (value_logits.softmax(-1) * support).sum(-1)
    value_ce = -(tv.softmax(-1) * value_logits.log_softmax(-1)).sum(-1)
    return dict(kl=kl, agree=agree, value_ce=value_ce, value_abs=(teacher_value - student_value).abs(),
                teacher_top_prob=p.max(-1).values)


def train(args):
    import torch

    from drmc_rl.models.policy.afterstate_core import afterstate_core_config
    from tools.eval_policy import _build_net_from_cfg

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, data, host = load_parts(args.dataset, args.device)
    train_rows, held_rows = _split(host, args.holdout_mod)
    batcher = _Batcher(data, data["offsets"], args.device)
    batcher.pace_np = host["pace"]
    del data
    cfg = afterstate_core_config()
    net, _aux, _ = _build_net_from_cfg(cfg, 20, args.device)
    start_step = 0
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
    steps = args.steps
    rng = np.random.default_rng(args.seed + start_step)
    eval_rows = held_rows if len(held_rows) <= args.eval_states else np.sort(
        np.random.default_rng(1).choice(held_rows, args.eval_states, replace=False))
    log = (out / "log.jsonl").open("a")
    params = sum(p.numel() for p in net.parameters())
    identity = dict(schema="drmc-afterstate-distillation-run-v1", dataset=str(args.dataset),
                    dataset_meta=meta, train_states=int(len(train_rows)), holdout_states=int(len(held_rows)),
                    holdout_rule=f"game_seed % {args.holdout_mod} == 0", parameters=params, args=vars(args))
    (out / "run.json").write_text(json.dumps(identity, indent=1, default=str) + "\n")
    print(json.dumps(dict(parameters=params, train=len(train_rows), holdout=len(held_rows))), flush=True)
    began = time.monotonic()

    def save(name, step, metrics):
        payload = dict(schema="drmc-afterstate-core-distilled-v1", cfg=cfg,
                       state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                       optimizer=optimizer.state_dict(), step=step, metrics=metrics,
                       teacher_sha256=meta["teacher_sha256"], observation_schema="public_pair_context_v3",
                       distillation=identity)
        tmp = out / (name + ".next")
        torch.save(payload, tmp)
        tmp.replace(out / name)

    metrics = None
    for step in range(start_step + 1, steps + 1):
        net.train()
        lr = args.lr * min(1.0, step / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))) * 0.95 + 0.05)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = np.sort(rng.choice(train_rows, args.batch, replace=False))
        inputs, aux, valid, after, index = batcher.inputs(rows, afterstate=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            logits, _value, extra = net(*inputs, aux=aux, afterstate=after, return_aux=True)
        terms = _distill_terms(net, logits.float(), extra["value_logits"].float(), batcher, index, valid, rows)
        loss = terms["kl"].mean() + args.value_coefficient * terms["value_ce"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        if step % args.log_every == 0:
            row = dict(step=step, lr=lr, loss=float(loss), kl=float(terms["kl"].mean()),
                       agree=float(terms["agree"].mean()), value_ce=float(terms["value_ce"].mean()),
                       grad=float(grad), seconds=time.monotonic() - began)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == steps:
            metrics = dict(step=step, holdout=evaluate(net, batcher, eval_rows), seconds=time.monotonic() - began)
            log.write(json.dumps(metrics) + "\n"); log.flush()
            print(json.dumps(metrics), flush=True)
            save("student-latest.pt", step, metrics)
    if metrics is not None:
        final = dict(metrics, holdout_full=evaluate(net, batcher, held_rows))
        save("student-final.pt", steps, final)
        inference = torch.load(out / "student-final.pt", map_location="cpu", weights_only=False)
        inference.pop("optimizer")
        torch.save(inference, out / "student-final-inference.pt")
        (out / "final.json").write_text(json.dumps(final, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--shards", required=True)
    b.add_argument("--teacher", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--max-states", type=int, default=1_500_000)
    b.add_argument("--seed", type=int, default=20260924)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--device", default="cuda")
    b.add_argument("--teacher-batch", type=int, default=512)
    b.add_argument("--tf32", type=int, default=0)
    b.add_argument("--shards-per-part", type=int, default=64)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True, help="directory holding dataset.json and its parts")
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--value-coefficient", type=float, default=0.5)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--holdout-mod", type=int, default=50)
    t.add_argument("--eval-states", type=int, default=20000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=20260924)
    t.add_argument("--resume")
    args = parser.parse_args()
    build(args) if args.command == "build" else train(args)


if __name__ == "__main__":
    main()
