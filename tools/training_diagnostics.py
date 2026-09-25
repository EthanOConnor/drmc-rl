"""Offline diagnostics of the controller-retention PPO recipe (afterstate core, arm A).

Every measurement reads the trainer's own persisted data: on-policy collections
(`public-replay/update-*.npz`), the game journal, the per-update log and saved
checkpoints. Batch loading and loss rebuilding reuse `tools/gradient_noise_scale`
(validated against the trainer's advantages and log-probabilities).

Subcommands
  replay      numpy: value explained variance by update, pace, level and game phase;
              advantage variance by pace; allocation inputs (cost per game).
  gradients   torch: per-term and per-pace gradient norms/cosines restricted to the
              shared trunk, value head and policy head, plus retention by pace.
  value-fit   torch: offline value fitting on saved batches (frozen trunk head-only
              at several learning rates, fresh MLP probe, full network).
  epochs      torch: replays one update's optimizer from a full checkpoint on a
              game-level train split; held-out surrogate gain and KL after each epoch.
  counterfactual  torch+engine: stochastic branch rollouts for advantage SNR.
  curve       numpy: learning-curve fit from the rating pool report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from tools.gradient_noise_scale import load_collection, natural_games
except ImportError:  # scratch copy next to this file on the GPU host
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gradient_noise_scale import load_collection, natural_games

PACES = ("sloth", "relaxed", "normal", "fast", "top_humans", "super_human", "frame_perfect")
# Strength-weighted objective (adoption addendum 3).
PACE_WEIGHTS = dict(frame_perfect=3, super_human=3, top_humans=2, fast=1.5, normal=1, relaxed=.5, sloth=.5)


def ev(y, v, w=None):
    """Explained variance 1 - Var(y - v) / Var(y) under optional weights."""
    w = np.ones_like(y) if w is None else w
    my = np.average(y, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    r = y - v
    vr = np.average((r - np.average(r, weights=w)) ** 2, weights=w)
    return float(1 - vr / vy) if vy > 0 else float("nan")


def boot_ci(fn, groups, boots=300, seed=0):
    """Bootstrap over whole games: `groups` is a list of index arrays."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(boots):
        pick = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(fn(idx))
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


# ------------------------------------------------------------------------- replay

def replay(args):
    run = Path(args.run)
    replay_dir = run / "public-replay"
    journal = [json.loads(line) for line in (run / "training-games.jsonl").open()]
    updates = sorted({int(p.name[7:12]) for p in replay_dir.glob("update-*.npz")})
    if args.updates:
        updates = [u for u in updates if u in set(args.updates)]
    rows = []  # per decision: update, pace, level, game key, index, length, return, value
    for u in updates:
        for pace in PACES:
            path = replay_dir / f"update-{u:05d}-{pace}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            meta = json.loads(str(z["metadata"]))
            seed, port = z["game_seed"], z["learner_port"]
            key = seed.astype(np.int64) * 2 + port
            # Decisions of a game are contiguous and in play order.
            change = np.r_[True, key[1:] != key[:-1]]
            gid = np.cumsum(change) - 1
            start = np.flatnonzero(change)
            length = np.diff(np.r_[start, len(key)])
            index = np.arange(len(key)) - start[gid]
            rows.append(dict(update=np.full(len(key), u), pace=np.full(len(key), PACES.index(pace)),
                             level=np.full(len(key), meta["level"]), game=gid + 1000000 * (u * 8 + PACES.index(pace)),
                             index=index, length=length[gid], ret=z["return"], value=z["old_value"],
                             frame=z["observed_frame"]))
        print(f"loaded update {u}", file=sys.stderr, flush=True)
    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    w_ep = 1 / d["length"]
    out = dict(run=str(run), updates=[int(updates[0]), int(updates[-1])], decisions=int(len(d["ret"])))

    def games_of(mask):
        g = d["game"][mask]
        idx = np.flatnonzero(mask)
        order = np.argsort(g, kind="stable")
        _, first = np.unique(g[order], return_index=True)
        return np.split(idx[order], first[1:])

    def stats(mask, boots=200):
        y, v, w = d["ret"][mask], d["value"][mask], w_ep[mask]
        res = dict(decisions=int(mask.sum()), games=int(len(np.unique(d["game"][mask]))),
                   win_rate=float(np.average((y + 1) / 2, weights=w)),
                   ev_decision=ev(y, v), ev_episode=ev(y, v, w),
                   mse=float(np.mean((y - v) ** 2)), value_mean=float(v.mean()), return_mean=float(y.mean()),
                   value_std=float(v.std()), return_std=float(y.std()))
        # Calibration: regression slope of the return on the value prediction.
        cv = np.cov(v, y)
        res["calibration_slope"] = float(cv[0, 1] / cv[0, 0]) if cv[0, 0] > 0 else float("nan")
        res["advantage_std_episode"] = float(np.sqrt(np.average((y - v - np.average(y - v, weights=w)) ** 2, weights=w)))
        if boots:
            groups = games_of(mask)
            y_all, v_all = d["ret"], d["value"]
            res["ev_decision_ci"] = boot_ci(lambda i: ev(y_all[i], v_all[i]), groups, boots)
        return res

    # Trend: EV per update (all paces) and per pace for update windows.
    out["by_update"] = {int(u): stats(d["update"] == u, boots=0) for u in updates}
    late = d["update"] >= updates[-1] - args.window + 1
    early = d["update"] <= updates[0] + args.window - 1
    out["window"] = args.window
    out["late_overall"] = stats(late)
    out["early_overall"] = stats(early)
    out["late_by_pace"] = {p: stats(late & (d["pace"] == i)) for i, p in enumerate(PACES)}
    out["early_by_pace"] = {p: stats(early & (d["pace"] == i), boots=0) for i, p in enumerate(PACES)}
    out["late_by_level"] = {int(l): stats(late & (d["level"] == l), boots=0) for l in np.unique(d["level"][late])}
    # Game phase: thirds of each game's own decisions.
    phase = np.minimum((3 * d["index"] / d["length"]).astype(int), 2)
    names = ("early", "mid", "late")
    out["late_by_phase"] = {names[k]: stats(late & (phase == k)) for k in range(3)}
    out["late_by_pace_phase"] = {p: {names[k]: stats(late & (d["pace"] == i) & (phase == k), boots=0)
                                     for k in range(3)} for i, p in enumerate(PACES)}
    # Absolute decision index bins (the first decisions of a game are nearly unpredictable).
    bins = [0, 5, 10, 20, 40, 80, 10000]
    out["late_by_index"] = {f"{a}-{b - 1}": stats(late & (d["index"] >= a) & (d["index"] < b), boots=0)
                            for a, b in zip(bins[:-1], bins[1:]) if (late & (d["index"] >= a) & (d["index"] < b)).any()}

    # Baseline variants for the Monte Carlo advantage R - b(s), fitted on the updates before
    # the late window and evaluated on it (out of sample). Lower mean square = lower PG noise.
    fit = (d["update"] < updates[-1] - args.window + 1) & (d["update"] >= updates[-1] - 3 * args.window + 1)
    cells = [(i, k) for i in range(len(PACES)) for k in range(3)]
    b_recal = np.zeros_like(d["ret"]); b_const = np.zeros_like(d["ret"]); b_shrink = np.zeros_like(d["ret"])
    for i, k in cells:
        cf = fit & (d["pace"] == i) & (phase == k)
        ce = late & (d["pace"] == i) & (phase == k)
        if cf.sum() < 50 or not ce.any():
            continue
        y, v = d["ret"][cf], d["value"][cf]
        slope, intercept = np.polyfit(v, y, 1)
        b_recal[ce] = intercept + slope * d["value"][ce]
        b_const[ce] = y.mean()
        # Best pure shrink toward the cell mean: b = m + c (V - m), c in [0, 1].
        m = y.mean(); c = np.clip(np.dot(v - v.mean(), y - m) / max(np.dot(v - v.mean(), v - v.mean()), 1e-9), 0, 1)
        b_shrink[ce] = m + c * (d["value"][ce] - v.mean())
    y = d["ret"][late]
    variants = dict(current_value=d["value"][late], collection_mean=np.full(late.sum(), y.mean()),
                    pace_phase_constant=b_const[late], pace_phase_recalibrated=b_recal[late],
                    pace_phase_shrunk=b_shrink[late])
    base_ms = {name: float(np.mean((y - b) ** 2)) for name, b in variants.items()}
    out["baselines_late"] = dict(mse=base_ms, relative_to_current={k: v / base_ms["current_value"] for k, v in base_ms.items()},
        by_phase={names[k]: {name: float(np.mean((d["ret"][late & (phase == k)] - b[phase[late] == k]) ** 2))
                             for name, b in variants.items()} for k in range(3)},
        by_pace={p: {name: float(np.mean((d["ret"][late & (d["pace"] == i)] - b[d["pace"][late] == i]) ** 2))
                     for name, b in variants.items()} for i, p in enumerate(PACES)})

    # Journal-derived cost and allocation inputs per pace (late window).
    journal_updates = sorted({r["update"] for r in journal if updates[-1] - args.window + 1 <= r["update"] <= updates[-1]})
    jl = [r for r in journal if r["update"] in set(journal_updates)]
    args.window = len(journal_updates)
    cost = {}
    for i, p in enumerate(PACES):
        games = [r for r in jl if r["pace"] == p]
        if not games:
            continue
        cost[p] = dict(games_per_update=len(games) / args.window,
                       timeouts=sum(r["reason"] == "timeout" for r in games) / len(games),
                       frames_per_game=float(np.mean([r["frames"] for r in games])),
                       learner_decisions_per_game=float(np.mean([r["a_stats"].get("decisions", 0) for r in games])),
                       total_decisions_per_game=float(np.mean([r["a_stats"].get("decisions", 0) + r["b_stats"].get("decisions", 0) for r in games])),
                       score=float(np.mean([r["score"] for r in games])),
                       draws=float(np.mean([r["score"] == .5 for r in games])))
    out["cost_by_pace"] = cost
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("late_overall", "early_overall")}, indent=1))


# -------------------------------------------------------------------------- torch

def _setup(run, checkpoint, device):
    import torch
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy
    from drmc_rl.training.episodic_objective import objective_contract
    config = json.loads((Path(run) / "config.json").read_text())
    config["objective"] = objective_contract(config)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actor = ControllerCorePolicy(config["checkpoint"], device, resume=checkpoint, training=True,
                                 seed=int(config["seed"]))
    return config, actor


def _retention(actor, config, progress):
    from drmc_rl.training.controller_retention import PaceRetention
    retention = PaceRetention(actor, config["anchor_banks"], excluded_seeds=config["holdout_seeds"],
        paces=config["paces"], max_kl_increase=config.get("max_anchor_kl_increase", .03),
        coefficient=config.get("retention_coefficient", .1),
        batch_size=config.get("retention_batch_size", 64),
        pressure_strength=config.get("retention_pressure_strength", 0.))
    retention.baseline = progress["retention_baseline"]
    return retention


def _prepared(run, update, config):
    from drmc_rl.training.controller_retention import balance_pace_credit
    from tools.train_pace_strategy import prepare_training_records
    records, completed = load_collection(Path(run) / "public-replay", update, config["paces"],
                                         natural_games(Path(run) / "training-games.jsonl"))
    prepare_training_records(records, config)
    balance_pace_credit(records, completed)
    return records, completed


def _groups(net):
    """Parameter groups: value head, policy head (candidate path) and shared trunk."""
    value, policy, trunk = [], [], []
    candidate_prefixes = ("row_embed", "col_embed", "orient_embed", "cost_mlp", "root_projection",
                          "after_", "facts", "candidate", "policy")
    unused = ("state_wdl_head", "candidate_wdl_head")
    for name, p in net.named_parameters():
        if not p.requires_grad or name.startswith(unused):
            continue
        if name.startswith("value_head"):
            value.append(name)
        elif name.startswith(candidate_prefixes):
            policy.append(name)
        else:
            trunk.append(name)
    return dict(value=value, policy=policy, trunk=trunk)


def gradients(args):
    """Per-term gradient geometry at a checkpoint on its own next collection."""
    import torch
    import torch.nn.functional as F
    from drmc_rl.training.episodic_objective import categorical_kl
    run = Path(args.run)
    progress = json.loads((run / "training.json").read_text())
    config, actor = _setup(run, args.checkpoint, args.device)
    net = actor.net
    names = dict(net.named_parameters())
    groups = _groups(net)
    params = [names[n] for g in ("trunk", "value", "policy") for n in groups[g]]
    sizes = [p.numel() for p in params]
    bounds = np.cumsum([0] + [sum(names[n].numel() for n in groups[g]) for g in ("trunk", "value", "policy")])
    sl = dict(trunk=slice(bounds[0], bounds[1]), value=slice(bounds[1], bounds[2]), policy=slice(bounds[2], bounds[3]))
    adam = None
    if args.adam:
        import torch as T
        payload = T.load(args.checkpoint, map_location="cpu", weights_only=False)
        state, group = payload["optimizer"]["state"], payload["optimizer"]["param_groups"][0]
        ordered = [n for n, p in net.named_parameters()]
        index = {n: i for i, n in enumerate(ordered)}
        beta2, eps = group["betas"][1], group["eps"]
        vs = []
        for n in [n for g in ("trunk", "value", "policy") for n in groups[g]]:
            s = state[index[n]]
            vs.append((1 / ((s["exp_avg_sq"] / (1 - beta2 ** float(s["step"]))).sqrt() + eps)).reshape(-1))
        adam = T.cat(vs).to(args.device)
        del payload, state

    def flat(loss):
        g = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])

    retention = _retention(actor, config, progress)
    measured = retention.measure()
    retention.set_pressure(measured)
    results = dict(checkpoint=str(args.checkpoint), groups={k: int(bounds[i + 1] - bounds[i]) for i, k in enumerate(("trunk", "value", "policy"))},
                   retention_kl=measured, retention_baseline=progress["retention_baseline"],
                   retention_pressure=retention.pressure, collections=[])
    coef = dict(policy=1.0, value=config.get("value_coefficient", .5), entropy=-config.get("entropy", .003),
                parent_kl=config.get("parent_kl", .02))
    P = sum(sizes)
    # Exact retention gradient by pace (the expectation of the trainer's per-step term).
    G_ret = {}
    for pace in retention.paces:
        rows, weights = retention.by_pace[pace], retention.weights[pace]
        g = torch.zeros(P, device=args.device)
        for start in range(0, len(rows), 64):
            kl = retention._kl(rows[start:start + 64])
            w = torch.as_tensor(weights[start:start + 64], device=args.device, dtype=torch.float32)
            scale = retention.coefficient * retention.pressure[pace] / len(retention.paces)
            g += flat(scale * (w * kl).sum())
        G_ret[pace] = g
    for u in args.updates:
        records, completed = _prepared(run, u, config)
        N = len(records)
        clip = config.get("clip", .15)
        # Per-pace, per-term collection-mean gradients (sum over the pace / N):
        # the trainer's full-batch loss is the sum over paces of these.
        G = {p: {t: torch.zeros(P, device=args.device) for t in coef} for p in config["paces"]}
        by_pace = defaultdict(list)
        for r in records:
            by_pace[r["pace"]].append(r)
        for pace, rows in by_pace.items():
            for start in range(0, len(rows), args.chunk):
                part = rows[start:start + args.chunk]
                features, data = actor.training_batch(part)
                logits, values = actor.training_forward(features)
                logp = logits.log_softmax(-1)
                chosen = logp.gather(1, data["slot"][:, None]).squeeze(1)
                ratio = (chosen - data["old_logprob"]).exp()
                adv = data["advantage"]
                terms = dict(
                    policy=-(data["actor_weight"] * torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)).sum(),
                    value=(data["value_weight"] * F.smooth_l1_loss(values, data["return"], reduction="none")).sum(),
                    entropy=-(data["entropy_weight"] * (logp.exp() * logp).sum(-1)).sum(),
                    parent_kl=(data["parent_kl_weight"] * categorical_kl(data["parent_logp"], logp)).sum())
                for t, loss in terms.items():
                    G[pace][t] += flat(coef[t] * loss / N)
            print(f"update {u} pace {pace} rows {len(rows)}", file=sys.stderr, flush=True)
        results["collections"].append(summarize_gradients(G, G_ret, sl, adam, completed, u, N))
        del records
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results["collections"][-1]["overall"], indent=1))


def summarize_gradients(G, G_ret, sl, adam, completed, update, N):
    import torch

    def norm(v, s=None, metric=None):
        v = (v if s is None else v[s]).double()
        if metric is not None:
            m = (metric if s is None else metric[s]).double()
            return float(torch.sqrt((v * v * m).sum()))
        return float(v.norm())

    def cos(a, b, s=None, metric=None):
        a = (a if s is None else a[s]).double()
        b = (b if s is None else b[s]).double()
        if metric is not None:
            m = (metric if s is None else metric[s]).double()
            return float((a * b * m).sum() / torch.sqrt((a * a * m).sum() * (b * b * m).sum()).clamp_min(1e-300))
        return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))

    paces = list(G)
    tot = {t: sum(G[p][t] for p in paces) for t in G[paces[0]]}
    ret_total = sum(G_ret.values())
    ppo = sum(tot.values())
    out = dict(update=update, decisions=N, natural_games=completed)
    metrics = [("euclid", None)] + ([("adam", adam)] if adam is not None else [])
    overall = {}
    for mname, m in metrics:
        o = {}
        for part in ("trunk", "value", "policy", None):
            key = part or "all"
            s = sl[part] if part else None
            o[key] = dict({f"norm_{t}": norm(v, s, m) for t, v in tot.items()},
                          norm_retention=norm(ret_total, s, m), norm_ppo=norm(ppo, s, m),
                          cos_policy_value=cos(tot["policy"], tot["value"], s, m),
                          cos_policy_retention=cos(tot["policy"], ret_total, s, m),
                          cos_ppo_retention=cos(ppo, ret_total, s, m),
                          cos_policy_parent_kl=cos(tot["policy"], tot["parent_kl"], s, m),
                          cos_policy_entropy=cos(tot["policy"], tot["entropy"], s, m))
        o["value_to_policy_trunk_ratio"] = o["trunk"]["norm_value"] / o["trunk"]["norm_policy"]
        o["value_share_of_trunk_norm2"] = o["trunk"]["norm_value"] ** 2 / sum(o["trunk"][f"norm_{t}"] ** 2 for t in tot)
        per_pace = {}
        for p in paces:
            pol = G[p]["policy"]; r = G_ret[p]
            full = sum(G[p].values())
            per_pace[p] = dict(norm_policy=norm(pol, None, m), norm_retention=norm(r, None, m),
                               retention_to_policy=norm(r, None, m) / max(norm(pol, None, m), 1e-300),
                               cos_policy_retention=cos(pol, r, None, m),
                               cos_ppo_retention=cos(full, r, None, m),
                               cos_retention_with_total_policy=cos(tot["policy"], r, None, m),
                               # Share of the pace's own policy step cancelled by its retention term.
                               projection_cancel=float(-(pol.double() @ r.double()) / (pol.double() @ pol.double()).clamp_min(1e-300)) if m is None else None,
                               norm_value=norm(G[p]["value"], None, m),
                               trunk_norm_policy=norm(pol, sl["trunk"], m),
                               trunk_norm_retention=norm(r, sl["trunk"], m),
                               trunk_cos_policy_retention=cos(pol, r, sl["trunk"], m))
        o["per_pace"] = per_pace
        overall[mname] = o
    out["overall"] = overall
    return out


# ---------------------------------------------------------------------- value fit

def value_fit(args):
    """How far can explained variance move on held-out games with more value optimization?"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    run = Path(args.run)
    config, actor = _setup(run, args.checkpoint, args.device)
    net = actor.net
    # 1. Features of the frozen trunk (value context) for train and held-out updates.
    feats, targets, meta = {}, {}, {}
    context = {}

    def hook(_m, inp, _out):
        context["x"] = inp[0]
    handle = net.value_head.register_forward_hook(hook)
    for u in sorted(set(args.train_updates) | set(args.test_updates)):
        records, _ = load_collection(run / "public-replay", u, config["paces"],
                                     natural_games(run / "training-games.jsonl"))
        xs, vs = [], []
        with torch.no_grad():
            for start in range(0, len(records), 256):
                rows = records[start:start + 256]
                for r in rows:
                    r.setdefault("advantage", 0.)
                features, data = actor.training_batch(rows)
                _, value = actor.training_forward(features)
                xs.append(context["x"].float().cpu()); vs.append(value.float().cpu())
        feats[u] = torch.cat(xs)
        targets[u] = torch.as_tensor([r["return"] for r in records], dtype=torch.float32)
        meta[u] = dict(pace=np.asarray([config["paces"].index(r["pace"]) for r in records]),
                       game=[r["game"] for r in records], length=np.asarray([r["episode_length"] for r in records]),
                       old_value=np.asarray([r["old_value"] for r in records]), current=torch.cat(vs).numpy())
        print(f"features update {u}: {len(records)}", file=sys.stderr, flush=True)
        del records
    handle.remove()
    # Held-out: every test update entirely, plus a game split of the train updates' games is
    # unnecessary (different updates are independent rollouts).
    Xtr = torch.cat([feats[u] for u in args.train_updates]).to(args.device)
    ytr = torch.cat([targets[u] for u in args.train_updates]).to(args.device)
    wtr = torch.as_tensor(np.concatenate([1 / meta[u]["length"] for u in args.train_updates]), dtype=torch.float32, device=args.device)
    wtr = wtr / wtr.mean()
    Xte = torch.cat([feats[u] for u in args.test_updates]).to(args.device)
    yte = torch.cat([targets[u] for u in args.test_updates]).numpy()
    pte = np.concatenate([meta[u]["pace"] for u in args.test_updates])
    lte = np.concatenate([meta[u]["length"] for u in args.test_updates])
    cur = np.concatenate([meta[u]["current"] for u in args.test_updates])
    old = np.concatenate([meta[u]["old_value"] for u in args.test_updates])
    support = net.value_support.float()

    def report(pred):
        pred = np.asarray(pred, dtype=np.float64)
        out = dict(ev=ev(yte, pred), ev_episode=ev(yte, pred, 1 / lte), mse=float(np.mean((yte - pred) ** 2)))
        out["by_pace"] = {p: ev(yte[pte == i], pred[pte == i]) for i, p in enumerate(config["paces"]) if (pte == i).any()}
        return out

    results = dict(checkpoint=str(args.checkpoint), train_updates=args.train_updates, test_updates=args.test_updates,
                   train_rows=int(len(ytr)), test_rows=int(len(yte)),
                   baseline_current=report(cur), baseline_behavior=report(old), runs=[])
    print("baseline", json.dumps(results["baseline_current"]), flush=True)

    def fit(module, params, lr, steps, loss_kind, label, batch=args.batch):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=.001)
        gen = torch.Generator(device="cpu").manual_seed(0)
        curve = []
        for step in range(1, steps + 1):
            idx = torch.randint(0, len(ytr), (batch,), generator=gen).to(args.device)
            v = module(Xtr[idx])
            if loss_kind == "huber":
                loss = (wtr[idx] * F.smooth_l1_loss(v, ytr[idx], reduction="none")).mean()
            else:
                loss = (wtr[idx] * (v - ytr[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step in args.eval_at or step == steps:
                with torch.no_grad():
                    pred = torch.cat([module(Xte[i:i + 4096]) for i in range(0, len(Xte), 4096)]).cpu().numpy()
                r = report(pred); r["step"] = step
                curve.append(r)
                print(label, lr, step, round(r["ev"], 4), flush=True)
        results["runs"].append(dict(label=label, lr=lr, steps=steps, loss=loss_kind, batch=batch, curve=curve))

    import copy
    for lr in args.head_lrs:
        head = copy.deepcopy(net.value_head).float().train()
        module = lambda x, h=head: (h(x).softmax(-1) * support).sum(-1)  # noqa: E731
        fit(module, head.parameters(), lr, args.steps, "huber", "head_only")
    for lr in args.probe_lrs:
        probe = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], 512), nn.SiLU(),
                              nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1)).to(args.device)
        module = lambda x, h=probe: torch.tanh(h(x).squeeze(-1))  # noqa: E731
        fit(module, probe.parameters(), lr, args.steps, "mse", "fresh_mlp_probe")
    Path(args.out).write_text(json.dumps(results, indent=1))


def value_full(args):
    """Full-network value-only fitting (trunk trainable) at two learning rates."""
    import torch
    import torch.nn.functional as F
    run = Path(args.run)
    results = dict(checkpoint=str(args.checkpoint), runs=[])
    config, actor0 = _setup(run, args.checkpoint, args.device)
    base_state = {k: v.clone() for k, v in actor0.net.state_dict().items()}
    natural = natural_games(run / "training-games.jsonl")
    train = []
    for u in args.train_updates:
        records, _ = load_collection(run / "public-replay", u, config["paces"], natural)
        for r in records:
            r.setdefault("advantage", 0.)
        train.extend(records)
    test, _ = load_collection(run / "public-replay", args.test_updates[0], config["paces"], natural)
    for r in test:
        r.setdefault("advantage", 0.)
    yte = np.asarray([r["return"] for r in test]); lte = np.asarray([r["episode_length"] for r in test])
    wtr = np.asarray([1 / r["episode_length"] for r in train]); wtr = wtr / wtr.mean()

    def evaluate():
        actor0.net.eval()
        with torch.no_grad():
            pred = []
            for start in range(0, len(test), 256):
                features, _ = actor0.training_batch(test[start:start + 256])
                pred.append(actor0.training_forward(features)[1].float().cpu().numpy())
        actor0.net.train(False)
        pred = np.concatenate(pred)
        return dict(ev=ev(yte, pred), ev_episode=ev(yte, pred, 1 / lte))

    results["baseline"] = evaluate()
    print("baseline", results["baseline"], flush=True)
    for lr in args.full_lrs:
        actor0.net.load_state_dict(base_state)
        opt = torch.optim.AdamW(actor0.net.parameters(), lr=lr, weight_decay=.001)
        rng = np.random.default_rng(0)
        curve = []
        for step in range(1, args.steps + 1):
            idx = rng.choice(len(train), args.batch, replace=False)
            rows = [train[i] for i in idx]
            features, data = actor0.training_batch(rows)
            _, values = actor0.training_forward(features)
            w = torch.as_tensor(wtr[idx], device=args.device, dtype=torch.float32)
            loss = .5 * (w * F.smooth_l1_loss(values, data["return"], reduction="none")).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor0.net.parameters(), args.clip)
            opt.step()
            if step in args.eval_at or step == args.steps:
                r = evaluate(); r["step"] = step; curve.append(r)
                print("full", lr, step, round(r["ev"], 4), flush=True)
        results["runs"].append(dict(label="full_network_value_only", lr=lr, steps=args.steps, batch=args.batch,
                                    clip=args.clip, curve=curve))
    Path(args.out).write_text(json.dumps(results, indent=1))


# ------------------------------------------------------------------------ epochs

def epochs(args):
    """Replay the trainer's optimizer for one update on a game-level train split."""
    import copy
    import torch
    from drmc_rl.training.episodic_objective import categorical_kl
    from tools.train_pace_strategy import training_loss_terms, weighted_training_terms
    run = Path(args.run)
    progress = json.loads((run / "training.json").read_text())
    config, actor = _setup(run, args.checkpoint, args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    update = int(payload["update"]) + 1
    opt_state = payload["optimizer"]
    del payload
    records, completed = _prepared(run, update, config)
    # Game-level split, stratified by pace.
    rng = np.random.default_rng(args.seed)
    games = sorted({r["game"] for r in records})
    held = {g for g in games if rng.random() < args.holdout}
    train = [r for r in records if r["game"] not in held]
    test = [r for r in records if r["game"] in held]
    retention = _retention(actor, config, progress)
    base = copy.deepcopy(actor.net.state_dict())

    @torch.no_grad()
    def snapshot(rows):
        out = []
        for start in range(0, len(rows), 512):
            features, data = actor.training_batch(rows[start:start + 512])
            out.append(actor.training_forward(features)[0].log_softmax(-1).cpu())
        return out

    ref_train, ref_test = snapshot(train), snapshot(test)

    @torch.no_grad()
    def evaluate(rows, ref):
        clip = config.get("clip", .15)
        sums = defaultdict(float)
        pace_gain = defaultdict(float)
        for k, start in enumerate(range(0, len(rows), 512)):
            part = rows[start:start + 512]
            features, data = actor.training_batch(part)
            logits, values = actor.training_forward(features)
            logp = logits.log_softmax(-1)
            chosen = logp.gather(1, data["slot"][:, None]).squeeze(1)
            ratio = (chosen - data["old_logprob"]).exp()
            adv, w = data["advantage"], data["actor_weight"]
            unclipped = (w * (ratio - 1) * adv)
            clipped = (w * (torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv) - adv))
            old = ref[k].to(logp.device)
            kl = categorical_kl(old[:, :logp.shape[1]] if old.shape[1] >= logp.shape[1] else
                                torch.nn.functional.pad(old, (0, logp.shape[1] - old.shape[1]), value=-1e9), logp)
            sums["surrogate_gain"] += float(unclipped.sum())
            sums["clipped_surrogate_gain"] += float(clipped.sum())
            sums["kl"] += float(kl.sum())
            sums["clip_fraction"] += float(((ratio - 1).abs() > clip).float().sum())
            sums["value_se"] += float(((values - data["return"]) ** 2).sum())
            for p, g in zip([r["pace"] for r in part], unclipped.tolist()):
                pace_gain[p] += g
        n = len(rows)
        out = {k: v / n for k, v in sums.items()}
        out["surrogate_gain_by_pace"] = {p: v / n for p, v in pace_gain.items()}
        return out

    results = dict(checkpoint=str(args.checkpoint), update=update, train_rows=len(train), test_rows=len(test),
                   held_games=len(held), variants=[])
    for variant in args.variants:
        size, n_epochs, lr_mult = (int(x) if i < 2 else float(x) for i, x in enumerate(variant.split(":")))
        actor.net.load_state_dict(base)
        opt = torch.optim.AdamW(actor.net.parameters(), lr=config["lr"], weight_decay=.001)
        opt.load_state_dict(opt_state)
        for group in opt.param_groups:
            group["lr"] = config["lr"] * lr_mult
        retention.set_pressure(retention.measure())
        order_rng = np.random.default_rng(config["seed"] + update)
        retention_rng = np.random.default_rng((config["seed"] + update) ^ 0x71A90)
        curve = [dict(epoch=0, train=evaluate(train, ref_train), test=evaluate(test, ref_test))]
        steps = 0
        for epoch in range(1, n_epochs + 1):
            indices = order_rng.permutation(len(train))
            norms = []
            for start in range(0, len(indices), size):
                rows = [train[i] for i in indices[start:start + size]]
                features, data = actor.training_batch(rows)
                _, terms = training_loss_terms(actor, features, data, config)
                loss = sum(weighted_training_terms(terms, config).values()) + retention.loss(retention_rng)
                opt.zero_grad(set_to_none=True); loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(actor.net.parameters(), args.clip)))
                opt.step(); steps += 1
            ret = retention.measure()
            retention.set_pressure(ret)
            curve.append(dict(epoch=epoch, steps=steps, grad_norm_median=float(np.median(norms)),
                              train=evaluate(train, ref_train), test=evaluate(test, ref_test),
                              retention_increase={p: ret[p] - retention.baseline[p] for p in ret},
                              retention_accepts=bool(retention.accepts(ret))))
            print(variant, epoch, json.dumps({k: round(v, 6) for k, v in curve[-1]["test"].items() if not isinstance(v, dict)}), flush=True)
        results["variants"].append(dict(variant=variant, minibatch=size, epochs=n_epochs, lr_multiplier=lr_mult, curve=curve))
        Path(args.out).write_text(json.dumps(results, indent=1))


# ----------------------------------------------------------------- counterfactual

def counterfactual(args):
    """Stochastic branch rollouts under the trainer's own behaviour.

    Base game: the learner (checkpoint, sampled exactly as in collection) plays side 0
    against the champion (argmax), with the trainer's delay, inputs and planner. At
    sampled learner decisions, each of the top-K candidates by behaviour probability
    is forced and the game continues M times with fresh learner sampling. The prefix is
    replayed by forcing the recorded learner actions; the board at the branch point is
    checked against the base game. Every learner value along each continuation is kept
    so any lambda-return can be evaluated offline.
    """
    import torch
    from drmc_rl.envs.backends.vs_frames import EventVsPool
    from drmc_rl.execution.pace import resolve_pace, strategy_context
    from drmc_rl.human.anticipation import execution_for_action, score_public_inputs
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
    from drmc_rl.human.controller_context import controller_policy_inputs
    from drmc_rl.human.early_decision import network_execution_frames
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from tools.trainer_arena_cache import MemoPlanner
    from tools.vs_head_to_head import PlainPolicy
    from drmc_rl.training.public_league import PublicOpponentPool

    run = Path(args.run)
    config, actor = _setup(run, args.checkpoint, args.device)
    actor.rng.manual_seed(args.seed)
    parent = PlainPolicy(Path(config["opponent_parent"]), args.device, public_only=True)
    opponent = PublicOpponentPool(config["opponent_pool"], parent, config["opponent_parent"], args.device).load(args.opponent)
    pace = resolve_pace(args.pace)
    params = {"delay": 4}
    delay = max(int(params["delay"]), pace.reaction_frames)
    delay_input, compute_input = network_execution_frames(params, pace, delay)
    planner = MemoPlanner(NativeReachabilityRunner())
    out = Path(args.out)

    def play(seeds, plans):
        """plans[pair] = (prefix actions, branch index, forced action) or None (sample all)."""
        n = len(seeds)
        counts = [0] * n
        log = [[] for _ in range(n)]
        valid = [True] * n
        with EventVsPool(n, lib_path=config.get("native_library")) as pool:
            pool.reset(list(seeds), level=args.level)
            while True:
                progress = pool.advance(args.max_frames)
                ready = [s for s, p in enumerate(progress) if p.needs_action]
                if not ready:
                    break
                rows = []
                for side in ready:
                    pair, physical = divmod(side, 2)
                    actor_policy = actor if physical == 0 else opponent
                    state = pool.semantic(side, public_context=True)
                    try:
                        candidate = plan_candidates(planner, state, delay, pace)
                    except NoReachablePlacement:
                        pool.install(side)
                        continue
                    obs, info = controller_policy_inputs(actor_policy, candidate, state, pace, delay, compute_input,
                                                         decision_delay_frames=delay_input)
                    info[0]["pace/context"] = strategy_context(pace, state, delay)
                    rows.append((side, pair, physical, candidate, obs, info))
                for physical, policy in ((0, actor), (1, opponent)):
                    part = [r for r in rows if r[2] == physical]
                    for start in range(0, len(part), args.batch):
                        chunk = part[start:start + args.batch]
                        obs = np.concatenate([r[4] for r in chunk])
                        infos = [i for r in chunk for i in r[5]]
                        scores = score_public_inputs(policy, obs, infos)
                        records = actor.learning_records if physical == 0 else None
                        for j, (side, pair, _, candidate, _, _) in enumerate(chunk):
                            action = int(scores[j].argmax())
                            if physical == 0:
                                rec = records[j]
                                plan = plans[pair]
                                probs = np.exp(rec["behavior_logp"])
                                entry = dict(i=counts[pair], v=rec["old_value"], a=action)
                                if plan is not None:
                                    prefix, branch, forced = plan
                                    if counts[pair] < branch:
                                        action = prefix[counts[pair]]
                                    elif counts[pair] == branch:
                                        if bytes(pool.states[side].board) != plan_boards[pair] or forced not in rec["actions"]:
                                            valid[pair] = False
                                        action = forced
                                    if not np.isfinite(scores[j, action]):
                                        valid[pair] = False
                                        action = int(scores[j].argmax())
                                    entry["a"] = action
                                else:
                                    order = np.argsort(-probs, kind="stable")[:args.k]
                                    entry.update(board=bytes(pool.states[side].board),
                                                 top=[int(rec["actions"][o]) for o in order],
                                                 p=[float(probs[o]) for o in order], n=int(len(probs)))
                                log[pair].append(entry)
                                counts[pair] += 1
                            pool.install(side, execution_for_action(candidate, action, pace, delay=delay), delay=delay)
            outcomes = []
            for pair in range(n):
                end = pool.states[2 * pair]
                outcomes.append(None if not end.terminal else 1.0 if end.outcome == 1 else 0.0 if end.outcome == 2 else 0.5)
        return log, outcomes, valid

    rng = np.random.default_rng(args.seed)
    reserved = set(config["holdout_seeds"])
    seeds = [int(s) for s in rng.choice(np.setdiff1d(np.arange(1, 65536), list(reserved)), args.games, replace=False)]
    for batch_start in range(0, len(seeds), args.games_per_batch):
        batch = seeds[batch_start:batch_start + args.games_per_batch]
        t0 = time.monotonic()
        plan_boards = None
        base_log, base_out, _ = play(batch, [None] * len(batch))
        points = []
        for g, seed in enumerate(batch):
            L = len(base_log[g])
            if base_out[g] is None or L < 2:
                continue
            every = max(1, L // args.points)
            picks = [i for i in range(every // 2, L, every) if len(base_log[g][i]["top"]) >= 2][:args.points]
            for d in picks:
                points.append((g, seed, d))
        jobs, plans, plan_boards = [], [], []
        for g, seed, d in points:
            entry = base_log[g][d]
            prefix = [e["a"] for e in base_log[g][:d]]
            for k, action in enumerate(entry["top"]):
                for m in range(args.m):
                    jobs.append((g, seed, d, k, m))
                    plans.append((prefix, d, action))
                    plan_boards.append(entry["board"])
        t1 = time.monotonic()
        roll_log, roll_out, roll_valid = play([j[1] for j in jobs], plans) if jobs else ([], [], [])
        with out.open("a") as stream:
            for g, seed, d in points:
                entry = base_log[g][d]
                rolls = []
                for idx, (gg, _, dd, k, m) in enumerate(jobs):
                    if gg == g and dd == d:
                        tail = [(e["i"], e["v"]) for e in roll_log[idx] if e["i"] > d]
                        rolls.append(dict(k=k, m=m, outcome=roll_out[idx], valid=roll_valid[idx],
                                          values=[v for _, v in tail]))
                stream.write(json.dumps(dict(pace=args.pace, level=args.level, seed=seed, decision=d,
                    length=len(base_log[g]), base_outcome=base_out[g], value=entry["v"],
                    base_values=[e["v"] for e in base_log[g]], top=entry["top"], p=entry["p"],
                    legal=entry["n"], rollouts=rolls)) + "\n")
        print(json.dumps(dict(batch=batch_start // args.games_per_batch, games=len(batch), points=len(points),
                              rollouts=len(jobs), invalid=int(len(jobs) - sum(roll_valid)),
                              base_seconds=round(t1 - t0, 1), rollout_seconds=round(time.monotonic() - t1, 1))), flush=True)


def cf_analyze(args):
    """Variance components and advantage-estimator quality from branch rollouts."""
    points = [json.loads(l) for p in args.inputs for l in open(p)]
    lambdas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0]
    rng = np.random.default_rng(0)
    by_pace = defaultdict(list)
    for pt in points:
        by_pace[pt["pace"]].append(pt)
    by_pace["all"] = points

    def estimators(pt, r):
        """A_lambda from V(s_d) along one continuation; terminal reward only, gamma 1."""
        R = 2 * r["outcome"] - 1
        vs = [pt["value"]] + r["values"]           # V at d, d+1, ... T-1
        T = len(vs)
        out = {}
        for lam in lambdas:
            # A_lam = sum_j lam^j delta_{d+j}; delta_t = V_{t+1}-V_t, last delta = R - V_{T-1}.
            nxt = vs[1:] + [R]
            deltas = np.asarray(nxt) - np.asarray(vs)
            out[lam] = float(np.sum(deltas * lam ** np.arange(T)))
        return out

    def components(pts):
        rows = []
        for pt in pts:
            rs = [r for r in pt["rollouts"] if r["valid"] and r["outcome"] is not None]
            K = len(pt["top"])
            groups = [[r for r in rs if r["k"] == k] for k in range(K)]
            if any(len(g) < 4 for g in groups):
                continue
            p = np.asarray(pt["p"]); p = p / p.sum()
            est = [[estimators(pt, r) for r in g] for g in groups]
            # Split halves: "truth" from even m, estimator samples from odd m, and vice versa.
            rows.append(dict(pt=pt, groups=groups, est=est, p=p))
        return rows

    def summarize(rows, boots=args.boots):
        def stats(sample):
            acc = defaultdict(list)
            for row in sample:
                groups, est, p = row["groups"], row["est"], row["p"]
                R = [np.asarray([2 * r["outcome"] - 1 for r in g]) for g in groups]
                M = np.asarray([len(x) for x in R])
                means = np.asarray([x.mean() for x in R])
                within = np.asarray([x.var(ddof=1) for x in R])
                mu = p @ means
                between_raw = p @ (means - mu) ** 2
                noise_in_means = p @ (within / M) * (1 - p @ p) if False else p @ (within / M) - (p ** 2) @ (within / M)
                tau2 = between_raw - noise_in_means            # policy-weighted Var of true Q over candidates
                sigma2 = p @ within                            # Var(R | s, a)
                acc["tau2"].append(tau2); acc["sigma2"].append(sigma2)
                acc["q_range"].append(means.max() - means.min())
                acc["v_minus_mu"].append((row["pt"]["value"] - mu) ** 2)
                for lam in lambdas:
                    e = [np.asarray([x[lam] for x in g]) for g in est]
                    em = np.asarray([x.mean() for x in e])
                    ew = np.asarray([x.var(ddof=1) for x in e])
                    # Covariance of the estimator's candidate means with independent truth halves.
                    ev_ = [x[1::2].mean() for x in e]; tr_ = [x[0::2].mean() for x in R]
                    ev2 = [x[0::2].mean() for x in e]; tr2 = [x[1::2].mean() for x in R]
                    cov = 0.5 * (p @ ((np.asarray(ev_) - p @ ev_) * (np.asarray(tr_) - p @ tr_)) +
                                 p @ ((np.asarray(ev2) - p @ ev2) * (np.asarray(tr2) - p @ tr2)))
                    acc[f"cov_{lam}"].append(cov)
                    acc[f"var_{lam}"].append(p @ ew + p @ (em - p @ em) ** 2)   # single-sample variance
                    acc[f"bvar_{lam}"].append(p @ (em - p @ em) ** 2 - (p @ (ew / np.asarray([len(x) for x in e])) - (p ** 2) @ (ew / np.asarray([len(x) for x in e]))))
                    acc[f"within_{lam}"].append(p @ ew)
            s = {k: float(np.mean(v)) for k, v in acc.items()}
            res = dict(tau2=s["tau2"], sigma2=s["sigma2"], signal_fraction=s["tau2"] / (s["tau2"] + s["sigma2"]),
                       mean_q_range=s["q_range"], baseline_error2=s["v_minus_mu"])
            for lam in lambdas:
                # Squared correlation of a single estimator sample with the true action value,
                # across candidates within a decision (policy-weighted, pooled over decisions).
                res[f"corr2_{lam}"] = s[f"cov_{lam}"] ** 2 / (max(s["tau2"], 1e-12) * s[f"var_{lam}"])
                res[f"between_share_{lam}"] = s[f"bvar_{lam}"] / s[f"var_{lam}"]
                res[f"cov_{lam}"] = s[f"cov_{lam}"]
                res[f"var_{lam}"] = s[f"var_{lam}"]
            return res
        base = stats(rows)
        draws = [stats([rows[i] for i in rng.integers(len(rows), size=len(rows))]) for _ in range(boots)]
        ci = {k: [float(x) for x in np.percentile([d[k] for d in draws], [2.5, 97.5])] for k in base}
        return dict(decisions=len(rows), point=base, ci95=ci)

    result = {}
    for pace, pts in by_pace.items():
        rows = components(pts)
        if len(rows) >= 5:
            result[pace] = summarize(rows)
            r = result[pace]["point"]
            print(pace, len(rows), "tau2=%.4f sigma2=%.3f signal=%.4f" % (r["tau2"], r["sigma2"], r["signal_fraction"]),
                  " ".join(f"corr2[{l}]={r[f'corr2_{l}']:.3f}" for l in lambdas))
    Path(args.out).write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------------- curve

def curve(args):
    """Rating-vs-frames fit for one lineage from `tools.rating_pool` report.json."""
    report = json.loads(Path(args.report).read_text())
    cs = next(c for c in report["condition_sets"] if c["name"] == args.condition_set)

    def trajectory(rows):
        for r in rows:
            if r.get("lineage") == args.lineage and r.get("trajectory"):
                pts = []
                for t in r["trajectory"]:
                    se = t.get("se") or (t["ci95"][1] - t["ci95"][0]) / (2 * 1.96)
                    pts.append((float(t["step"]) / 1e6, float(t["rating"]), float(se), int(t["games"])))
                return pts
        return []

    def fits(pts):
        pts = [p for p in pts if p[3] >= args.min_games]
        if len(pts) < 3:
            return None
        f, r, se = (np.asarray([p[i] for p in pts]) for i in range(3))
        w = 1 / se ** 2
        X = np.stack([np.ones_like(f), f / 100], 1)
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
        beta = cov @ X.T @ (w * r)
        resid = r - X @ beta
        chi2 = float(np.sum(w * resid ** 2))
        dof = len(r) - 2
        out = dict(points=[dict(frames_m=a, rating=b, se=c, games=d) for a, b, c, d in pts],
                   slope_per_100m=float(beta[1]), slope_se=float(np.sqrt(cov[1, 1])),
                   slope_se_inflated=float(np.sqrt(cov[1, 1] * max(1, chi2 / max(dof, 1)))),
                   intercept=float(beta[0]), chi2=chi2, dof=dof)
        # Saturating fit R = Rinf - (Rinf - R0) exp(-f / tau) on a tau grid (weighted LS in R0, Rinf).
        best = None
        for tau in np.geomspace(5, 2000, 400):
            Z = np.stack([np.exp(-f / tau), 1 - np.exp(-f / tau)], 1)
            c2 = np.linalg.inv(Z.T @ (w[:, None] * Z))
            b = c2 @ Z.T @ (w * r)
            s = float(np.sum(w * (r - Z @ b) ** 2))
            if best is None or s < best[0]:
                best = (s, tau, b, c2)
        s, tau, b, c2 = best
        out["saturating"] = dict(tau_m=float(tau), r0=float(b[0]), plateau=float(b[1]), plateau_se=float(np.sqrt(c2[1, 1])),
                                 chi2=s, peak_observed=float(r.max()), peak_frames_m=float(f[r.argmax()]))
        return out

    result = dict(condition_set=args.condition_set, lineage=args.lineage,
                  pooled=fits(trajectory(cs["pooled"])), pooled_equal=fits(trajectory(cs["pooled_equal"])), by_pace={})
    for p in cs["paces"]:
        fp = fits(trajectory(p["ratings"]))
        if fp:
            result["by_pace"][p["pace"]] = fp
    Path(args.out).write_text(json.dumps(result, indent=1))
    for k in ("pooled", "pooled_equal"):
        r = result[k]
        print(k, "slope/100M = %.1f +- %.1f (inflated %.1f); chi2=%.2f dof=%d; plateau %.0f +- %.0f tau=%.0fM; peak %.0f at %.0fM" % (
            r["slope_per_100m"], r["slope_se"], r["slope_se_inflated"], r["chi2"], r["dof"],
            r["saturating"]["plateau"], r["saturating"]["plateau_se"], r["saturating"]["tau_m"],
            r["saturating"]["peak_observed"], r["saturating"]["peak_frames_m"]))
    for p, r in result["by_pace"].items():
        print(f"  {p:14s} slope/100M = {r['slope_per_100m']:6.1f} +- {r['slope_se']:.1f}  points={[round(x['rating']) for x in r['points']]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--run", required=True); r.add_argument("--out", required=True)
    r.add_argument("--updates", type=int, nargs="*"); r.add_argument("--window", type=int, default=10)
    g = sub.add_parser("gradients")
    g.add_argument("--run", required=True); g.add_argument("--checkpoint", required=True)
    g.add_argument("--out", required=True); g.add_argument("--updates", type=int, nargs="+", required=True)
    g.add_argument("--chunk", type=int, default=64); g.add_argument("--adam", action="store_true")
    g.add_argument("--device", default="cuda")
    v = sub.add_parser("value-fit")
    v.add_argument("--run", required=True); v.add_argument("--checkpoint", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--train-updates", type=int, nargs="+", required=True)
    v.add_argument("--test-updates", type=int, nargs="+", required=True)
    v.add_argument("--head-lrs", type=float, nargs="*", default=[3e-6, 3e-5, 3e-4, 1e-3])
    v.add_argument("--probe-lrs", type=float, nargs="*", default=[1e-3])
    v.add_argument("--steps", type=int, default=4000); v.add_argument("--batch", type=int, default=256)
    v.add_argument("--eval-at", type=int, nargs="*", default=[100, 300, 1000, 2000])
    v.add_argument("--device", default="cuda")
    f = sub.add_parser("value-full")
    f.add_argument("--run", required=True); f.add_argument("--checkpoint", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--train-updates", type=int, nargs="+", required=True)
    f.add_argument("--test-updates", type=int, nargs="+", required=True)
    f.add_argument("--full-lrs", type=float, nargs="+", default=[3e-6, 3e-5])
    f.add_argument("--steps", type=int, default=600); f.add_argument("--batch", type=int, default=128)
    f.add_argument("--clip", type=float, default=0.7)
    f.add_argument("--eval-at", type=int, nargs="*", default=[100, 300])
    f.add_argument("--device", default="cuda")
    e = sub.add_parser("epochs")
    e.add_argument("--run", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--variants", nargs="+", default=["128:2:1", "128:4:1", "512:4:1", "1024:4:1", "1024:4:4"],
                   help="minibatch:epochs:lr_multiplier")
    e.add_argument("--holdout", type=float, default=.25); e.add_argument("--seed", type=int, default=7)
    e.add_argument("--clip", type=float, default=0.7)
    e.add_argument("--device", default="cuda")
    x = sub.add_parser("counterfactual")
    x.add_argument("--run", required=True); x.add_argument("--checkpoint", required=True)
    x.add_argument("--out", required=True); x.add_argument("--pace", required=True)
    x.add_argument("--opponent", default="champion"); x.add_argument("--level", type=int, default=14)
    x.add_argument("--games", type=int, default=8); x.add_argument("--games-per-batch", type=int, default=4)
    x.add_argument("--points", type=int, default=5); x.add_argument("--k", type=int, default=4)
    x.add_argument("--m", type=int, default=8); x.add_argument("--batch", type=int, default=128)
    x.add_argument("--max-frames", type=int, default=120000); x.add_argument("--seed", type=int, default=924)
    x.add_argument("--device", default="cuda")
    xa = sub.add_parser("cf-analyze")
    xa.add_argument("inputs", nargs="+"); xa.add_argument("--out", required=True)
    xa.add_argument("--boots", type=int, default=300)
    c = sub.add_parser("curve")
    c.add_argument("--report", required=True); c.add_argument("--out", required=True)
    c.add_argument("--condition-set", default="l14-spawn"); c.add_argument("--lineage", default="armA-ppo-v1")
    c.add_argument("--min-games", type=int, default=100)
    args = parser.parse_args()
    {"replay": replay, "gradients": gradients, "value-fit": value_fit, "value-full": value_full,
     "epochs": epochs, "counterfactual": counterfactual, "cf-analyze": cf_analyze, "curve": curve}[args.command](args)


if __name__ == "__main__":
    main()
